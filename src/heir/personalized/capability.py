"""Separate analysis of a completed study; never modify or promote its predictions."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .cohort import discover_cohort, prepare_cohort_inputs
from .cohort_cli import _config, _subset, qualified_targets
from .consistency import measurement_consistency, protein_pathway_diagnostics
from .diagnostics import _region_rows
from .error_decomposition import aggregate_error_decompositions, error_decomposition
from .evaluate import score
from .reconstruction import (
    fit_reference_mixtures,
    predict_reconstruction,
    save_reconstruction,
)
from .schema import identifiers
from .serialization import load_prediction
from .study import load_pooled_reference, patient_summary
from .train import FittedModel

_FULL = (
    "he",
    "he_rna",
    "he_rna_protein",
    "constant_reference",
    "generic_reference",
    "rna_transfer_trained",
)
_EXPORTS = _FULL + tuple(
    f"{arm}.transfer"
    for arm in ("he_rna", "he_rna_protein", "constant_reference", "generic_reference")
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _table(path, rows):
    if not rows:
        return
    with Path(path).open("x", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _linear_score(scaffold, expression, targets, operator, genes, normal):
    """Substitute linear expression for scoring only; never export this scaffold."""
    identifiers(scaffold.cell_ids, "reconstruction scaffold region IDs")
    identifiers(operator.cell_ids, "reconstruction operator region IDs")
    columns = _region_rows(operator)
    region_ids = tuple(operator.cell_ids[column] for column in columns)
    values = np.asarray(expression)
    if len(set(region_ids)) != len(region_ids) or values.shape != (
        len(columns),
        len(scaffold.gene_ids),
    ):
        raise ValueError("reconstruction scores require unique regions and the saved gene axis")
    unknown = sorted(set(operator.cell_ids) - set(scaffold.cell_ids))
    if unknown:
        raise ValueError(f"reconstruction operator contains unknown scaffold region IDs: {unknown}")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("invalid linear reconstruction values")
    scaffold_rows = {region: row for row, region in enumerate(scaffold.cell_ids)}
    destination = [scaffold_rows[region] for region in region_ids]
    linear = torch.zeros_like(scaffold.expression)
    linear[destination] = torch.as_tensor(values, dtype=linear.dtype, device=linear.device)
    prediction = replace(scaffold, expression=linear)
    result = score(prediction, targets, operator, genes, normalization_gene_ids=normal)
    result.update(
        evaluation_role="target_informed_reference_reconstruction",
        independent_prediction=False,
        biological_claim="diagnostic reconstruction; no deployable prediction or certified bound",
    )
    return result


def _reconstruction_summary(records):
    """Equal directions/replicates within patient, then equal patients; retain missingness."""
    specimens = []
    for patient in sorted({r["patient_id"] for r in records}):
        selected = [r for r in records if r["patient_id"] == patient]
        correlations = [r["scores"]["median_gene_spearman"] for r in selected]
        errors = [r["scores"]["normalized_log_mse"] for r in selected]
        specimens.append(
            {
                "patient_id": patient,
                "specimen_id": selected[0]["specimen_id"],
                "mean_replicate_median_gene_spearman": float(np.mean(correlations))
                if all(v is not None for v in correlations)
                else None,
                "normalized_log_mse": float(np.mean(errors))
                if all(v is not None for v in errors)
                else None,
            }
        )
    return (
        patient_summary(specimens)
        if specimens
        else {
            "n_patients": 0,
            "mean_patient_spearman": None,
            "mean_patient_normalized_log_mse": None,
            "patients": [],
        }
    )


def reconstruct_patient(
    inputs, predictions, targets, operator, groups, references, settings, output
):
    """Fit each source replicate independently, then score both shared-region directions."""
    genes = tuple(predictions["he_rna"].gene_ids)
    normal = tuple(settings["normalization_gene_ids"])
    labels = np.asarray(groups)
    replicates = {
        label: _subset(targets, operator, labels == label) for label in sorted(set(groups))
    }
    fits, within, cross, direct = {}, [], [], []
    for reference_name, reference in references.items():
        reference_path = output / f"{reference_name}-reference.npz"
        with reference_path.open("xb") as handle:
            np.savez_compressed(
                handle,
                gene_ids=np.asarray(reference.gene_ids),
                state_ids=np.asarray(reference.state_ids),
                expression=reference.expression,
                proportions=reference.proportions,
            )
        for label, (source, source_operator) in replicates.items():
            print(
                f"capability: {inputs.specimen_id} {reference_name}, source replicate {label}",
                flush=True,
            )
            fit = fit_reference_mixtures(
                reference,
                source,
                gene_ids=genes,
                normalization_gene_ids=normal,
                steps=settings["reconstruction"]["steps"],
                learning_rate=settings["reconstruction"]["learning_rate"],
                seed=settings["reconstruction"]["seed"],
            )
            save_reconstruction(fit, output / f"{reference_name}-replicate-{label}.npz")
            fits[(reference_name, label)] = fit
            values = predict_reconstruction(
                fit,
                source_operator,
                source_operator,
                patient_id=inputs.patient_id,
                specimen_id=inputs.specimen_id,
            )
            within.append(
                {
                    "patient_id": inputs.patient_id,
                    "specimen_id": inputs.specimen_id,
                    "reference": reference_name,
                    "source_replicate": label,
                    "n_regions": len(source.observation_ids),
                    "reference_sha256": _sha(reference_path),
                    "scores": _linear_score(
                        predictions["he_rna"], values, source, source_operator, genes, normal
                    ),
                }
            )
    for source_label, evaluation_label in itertools.permutations(replicates, 2):
        source, source_operator = replicates[source_label]
        evaluation, evaluation_operator = replicates[evaluation_label]
        source_columns = _region_rows(source_operator)
        evaluated_columns = _region_rows(evaluation_operator)
        keep = np.isin(evaluated_columns, source_columns)
        if int(keep.sum()) < settings["cross_replicate"]["minimum_shared_regions"]:
            cross.append(
                {
                    "patient_id": inputs.patient_id,
                    "specimen_id": inputs.specimen_id,
                    "source_replicate": source_label,
                    "target_replicate": evaluation_label,
                    "status": "insufficient_shared_regions",
                    "n_shared_regions": int(keep.sum()),
                }
            )
            continue
        evaluation, evaluation_operator = _subset(evaluation, evaluation_operator, keep)
        region_ids = [evaluation_operator.cell_ids[i] for i in _region_rows(evaluation_operator)]
        base_record = {
            "patient_id": inputs.patient_id,
            "specimen_id": inputs.specimen_id,
            "source_replicate": source_label,
            "target_replicate": evaluation_label,
            "status": "exploratory_registered_pixel_cross_replicate",
            "n_shared_regions": len(region_ids),
            "region_ids": region_ids,
        }
        for reference_name in references:
            values = predict_reconstruction(
                fits[(reference_name, source_label)],
                source_operator,
                evaluation_operator,
                patient_id=inputs.patient_id,
                specimen_id=inputs.specimen_id,
            )
            cross.append(
                {
                    **base_record,
                    "reference": reference_name,
                    "scores": _linear_score(
                        predictions["he_rna"],
                        values,
                        evaluation,
                        evaluation_operator,
                        genes,
                        normal,
                    ),
                }
            )
        source_index = {source_operator.cell_ids[c]: i for i, c in enumerate(source_columns)}
        source_rows = [source_index[region] for region in region_ids]
        values = source.expression[np.ix_(source_rows, [source.gene_ids.index(g) for g in genes])]
        direct.append(
            {
                **base_record,
                "scores": _linear_score(
                    predictions["he_rna"], values, evaluation, evaluation_operator, genes, normal
                ),
                "interpretation": "direct_source_assay_reference_not_a_universal_bound",
            }
        )
    return {
        "within_replicate": within,
        "cross_replicate": cross,
        "direct_replicate_transfer": direct,
        "cross_replicate_status": "available" if direct else "no_qualifying_replicate_pair",
        "target_informed": True,
        "correction": False,
        "he_mapper": False,
    }


def run_capability(root, study_directory, protocol_path, output_directory):
    """Analyze the bound v0.6 study and reconstruct fixed references without a new mapper fit."""
    root, study, output = map(Path, (root, study_directory, output_directory))
    if output.resolve().is_relative_to(root.resolve()) or output.resolve().is_relative_to(
        study.resolve()
    ):
        raise ValueError("capability output must be outside the immutable data and completed study")
    protocol = json.loads(Path(protocol_path).read_text())
    if (
        protocol.get("schema") != "heir.reference_capability.v1"
        or protocol.get("main_v060_study_unchanged") is not True
        or protocol.get("new_deployable_model_fitting") is not False
        or protocol.get("reconstruction", {}).get("initializations")
        != ["uniform", "reference_proportions"]
    ):
        raise ValueError("unsupported reference-capability protocol")
    for name, digest in protocol["study_files_sha256"].items():
        if Path(name).name != name or _sha(study / name) != digest:
            raise ValueError("completed-study identity differs from the capability protocol")
    metadata_path = root / "metadata/selected_13_biopsies.tsv"
    if _sha(metadata_path) != protocol["patient_metadata_sha256"]:
        raise ValueError("patient metadata differs from the frozen diagnostic groups")
    with metadata_path.open(newline="") as handle:
        metadata = {row["biopsy_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    config, model = _config(study / "config.json")
    if (
        tuple(protocol["outcome_gene_ids"]) != model.gene_ids
        or tuple(protocol["normalization_gene_ids"]) != model.normalization_gene_ids
    ):
        raise ValueError("capability panels differ from the completed study")
    previous = json.loads((study / "summary.json").read_text())
    cohort = discover_cohort(root)
    test_roles = {
        patient: selection["outer_fold"]
        for selection in previous["selections"]
        for patient, role in selection["fold_roles"].items()
        if role == "test"
    }
    if set(test_roles) != {r["patient_id"] for r in cohort.values()}:
        raise ValueError("completed study and current cohort patient axes differ")
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_bytes(Path(protocol_path).read_bytes())
    package = Path(__file__).resolve().parents[1]
    _write(
        output / "identity.json",
        {
            "protocol_sha256": _sha(protocol_path),
            "source_file_sha256": {
                str(p.relative_to(package.parent)): _sha(p) for p in sorted(package.rglob("*.py"))
            },
            "completed_study": str(study.resolve()),
            "new_mapper_fitting": False,
        },
    )
    errors = {name: [] for name in _EXPORTS}
    consistency, proteins, reconstructions, gene_rows = {}, {}, {}, []
    for biopsy, entry in sorted(cohort.items()):
        print(
            f"capability: loading saved test artifacts and qualified targets for {biopsy}",
            flush=True,
        )
        directory = output / biopsy
        directory.mkdir()
        inputs, provenance = prepare_cohort_inputs(
            root, biopsy, gene_ids=model.gene_ids, **config["preparation"]
        )
        fold = study / f"fold-{test_roles[inputs.patient_id]}"
        completion = json.loads((fold / "test_predictions_complete.json").read_text())
        predictions = {}
        for name in _EXPORTS:
            path = fold / f"test-{biopsy}-{name}.npz"
            if _sha(path) != completion["prediction_sha256"][path.name]:
                raise ValueError(
                    "saved prediction differs from its pre-evaluation completion receipt"
                )
            prediction, record = load_prediction(path)
            for key in ("sources", "grid", "image_features", "rna", "protein"):
                if provenance[key] != record["cohort_inputs"][key]:
                    raise ValueError(f"rehydrated {key} differs from the completed prediction")
            predictions[name] = prediction
        target, operator, groups, coverage = qualified_targets(
            root, biopsy, inputs, provenance, model
        )
        if coverage != json.loads((fold / f"test-{biopsy}-coverage.json").read_text()):
            raise ValueError("diagnostic target coverage differs from the completed study")
        all_ids = set(target.observation_ids) | set(
            coverage["excluded_zero_or_incomplete_exposure"]
        )
        coverage["all_observation_ids"] = sorted(
            all_ids,
            key=lambda name: (name.split(":", 2)[1], inputs.cell_ids.index(name.split(":", 2)[2])),
        )
        patient_errors = {}
        for name, prediction in predictions.items():
            report = error_decomposition(
                prediction, target, operator, groups, model.gene_ids, model.normalization_gene_ids
            )
            errors[name].append(report)
            patient_errors[name] = report
            gene_rows.extend({"arm": name, **row} for row in report["per_gene"])
        _write(directory / "error_decomposition.json", patient_errors)
        consistency[biopsy] = measurement_consistency(
            inputs,
            predictions,
            target,
            operator,
            groups,
            gene_ids=model.gene_ids,
            normalization_gene_ids=model.normalization_gene_ids,
            target_metadata=coverage,
            provenance=provenance,
            patient_metadata=metadata[biopsy],
        )
        _write(directory / "consistency.json", consistency[biopsy])
        fitted = FittedModel.load(fold / "he_rna_protein.bundle")
        proteins[biopsy] = protein_pathway_diagnostics(
            fitted,
            inputs,
            scored_operator=operator,
            expected_prediction=predictions["he_rna_protein"],
            permutation_seed=protocol["protein_diagnostics"]["permutation_seed"],
        )
        _write(directory / "protein_pathway.json", proteins[biopsy])
        matched = FittedModel.load(fold / "he_rna.bundle").predict(inputs)
        if not torch.allclose(
            matched.transfer, predictions["he_rna"].transfer, atol=1e-6, rtol=1e-6
        ):
            raise ValueError(
                "rehydrated matched reference fails the original model transfer replay"
            )
        references = {
            "matched": inputs.reference,
            "generic": load_pooled_reference(fold / "generic_reference.bundle"),
        }
        reconstructions[biopsy] = reconstruct_patient(
            inputs, predictions, target, operator, groups, references, protocol, directory
        )
        _write(directory / "reconstruction.json", reconstructions[biopsy])
    aggregates = {name: aggregate_error_decompositions(values) for name, values in errors.items()}
    for name, result in aggregates.items():
        old = (
            previous["postfit_transfer"][name.removesuffix(".transfer")]
            if name.endswith(".transfer")
            else previous["arms"][name]
        )
        expected = old["mean_patient_normalized_log_mse"]
        if not np.isclose(result["normalized_log_mse"], expected, atol=1e-5, rtol=1e-6):
            raise ValueError("decomposed error fails reconciliation with the completed study")
        result["original_study_mse"] = expected
        result["rounding_difference"] = result["normalized_log_mse"] - expected
    _table(output / "patient_gene_errors.tsv", gene_rows)
    reconstruction_summary = {}
    for kind in ("within_replicate", "cross_replicate"):
        reconstruction_summary[kind] = {
            reference: _reconstruction_summary(
                [
                    row
                    for report in reconstructions.values()
                    for row in report[kind]
                    if row.get("reference") == reference and "scores" in row
                ]
            )
            for reference in ("matched", "generic")
        }
    reconstruction_summary["direct_replicate_transfer"] = _reconstruction_summary(
        [row for report in reconstructions.values() for row in report["direct_replicate_transfer"]]
    )
    summary = {
        "schema": "heir.reference_capability_result.v1",
        "protocol_sha256": _sha(protocol_path),
        "error_decomposition": aggregates,
        "reconstruction": reconstruction_summary,
        "patient_consistency": consistency,
        "protein_pathways": proteins,
        "new_mapper_fitting": False,
        "target_informed_reconstruction": True,
        "independent_confirmation": False,
        "scientific_qualification": False,
    }
    _write(output / "summary.json", summary)
    print("capability: completed all thirteen patients", flush=True)
    return summary
