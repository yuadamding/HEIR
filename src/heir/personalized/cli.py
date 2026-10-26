"""Small shared prepare/train/predict/evaluate commands; no remote acquisition."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from .data import load_spatial_targets, prepare_sample, specimen_eligibility
from .evaluate import score
from .geometry import write_overlay
from .model import ModelConfig
from .observation import ObservationOperator
from .serialization import load_prediction, save_prediction
from .train import FittedModel, TrainingConfig, TrainingSample, fit, read_fold_roles


def _json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _source_path(manifest_path, key):
    value = Path(_json(manifest_path)["paths"][key])
    return value if value.is_absolute() else Path(manifest_path).resolve().parent / value


def preparation_contract(manifest):
    """Bind shared feature meanings and scale choices, excluding specimen-local paths."""
    reference = manifest["reference"]
    return {
        "feature_provenance": manifest["feature_provenance"],
        "rna_embedding_gene_ids": sorted(reference["embedding_gene_ids"]),
        "rna_library_size": reference.get("library_size", 10000),
        "max_states": reference.get("max_states", 64),
        "graph": manifest.get("graph", {"radius_um": 100.0, "max_neighbors": 32}),
        "proteins": manifest.get("proteins", {}),
    }


def prepare_command(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = specimen_eligibility([args.manifest])
    _write_json(output / "eligibility.json", rows)
    with (output / "eligibility.tsv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    if not rows[0]["prediction_eligible"]:
        raise ValueError("specimen preparation failed; see eligibility.json")
    inputs = prepare_sample(args.manifest)
    # Target-free geometry inspection. Full H&E/protein/ST overlay is a separate
    # training/evaluation inspection, never an implicit prediction-input read.
    write_overlay(output / "nuclei_regions.svg", inputs.coordinates_um)
    _write_json(
        output / "preparation.json",
        {
            "manifest": str(Path(args.manifest).resolve()),
            "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
            "input_contract": preparation_contract(_json(args.manifest)),
            "patient_id": inputs.patient_id,
            "specimen_id": inputs.specimen_id,
            "n_cells": len(inputs.cell_ids),
            "n_states": len(inputs.reference.state_ids),
            "gene_ids": list(inputs.reference.gene_ids),
            "target_payload_read": False,
            "scientific_qualification": False,
        },
    )


def train_command(args):
    roles = read_fold_roles(args.folds, args.outer_fold)
    manifests = [(path, _json(path)) for path in args.manifests]
    if any(m["patient_id"] not in roles for _, m in manifests):
        raise ValueError("all supplied patients must appear in the frozen fold table")
    selected = [(path, m) for path, m in manifests if roles[m["patient_id"]] == "train"]
    if not selected or {m["patient_id"] for _, m in selected} != {
        p for p, role in roles.items() if role == "train"
    }:
        raise ValueError("manifests must cover every training patient in the chosen fold")
    contract = preparation_contract(selected[0][1])
    if any(preparation_contract(m) != contract for _, m in selected):
        raise ValueError("training specimens disagree on shared feature/preparation settings")
    preparation = {
        "input_contract": contract,
        "fold_roles": roles,
        "outer_fold": args.outer_fold,
        "training_manifests": [
            {
                "patient_id": manifest["patient_id"],
                "specimen_id": manifest["specimen_id"],
                "manifest_path": str(Path(path).resolve()),
                "manifest_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "manifest": manifest,
            }
            for path, manifest in sorted(
                selected, key=lambda item: (item[1]["patient_id"], item[1]["specimen_id"])
            )
        ],
    }
    samples = [
        TrainingSample(
            prepare_sample(path),
            load_spatial_targets(path),
            ObservationOperator.load(_source_path(path, "observation")),
        )
        for path, _ in selected
    ]
    fitted = fit(
        samples,
        ModelConfig.from_dict(_json(args.config)),
        TrainingConfig(
            epochs=args.epochs, learning_rate=args.learning_rate, seed=args.seed, device=args.device
        ),
        baseline=FittedModel.load(args.baseline) if args.baseline else None,
        preparation=preparation,
    )
    fitted.save(args.output)


def predict_command(args):
    fitted = FittedModel.load(args.bundle, device=args.device)
    manifest = _json(args.manifest)
    if preparation_contract(manifest) != fitted.preparation.get("input_contract"):
        raise ValueError("prediction preparation settings differ from the inference bundle")
    if args.transfer_only:
        if fitted.model.config.variant == "he":
            raise ValueError("H&E-only prediction has no reference transfer")
        fitted.model.config = replace(fitted.model.config, transfer_only=True)
    prediction = fitted.predict(prepare_sample(args.manifest))
    save_prediction(
        prediction,
        args.output,
        provenance={
            "training_patient_ids": list(fitted.training_patient_ids),
            "fold_roles": fitted.preparation.get("fold_roles", {}),
            "model_config": fitted.model.config.to_dict(),
            "bundle_manifest_sha256": hashlib.sha256(
                (Path(args.bundle) / "bundle.json").read_bytes()
            ).hexdigest(),
            "target_payload_read": False,
        },
    )


def evaluate_command(args):
    prediction, provenance = load_prediction(args.prediction)
    targets = load_spatial_targets(args.manifest)
    roles = provenance.get("fold_roles", {})
    independent = roles.get(prediction.patient_id) == "test"
    if independent and prediction.patient_id in provenance.get("training_patient_ids", []):
        raise ValueError("test patient appears in the fitted model training set")
    result = score(
        prediction,
        targets,
        ObservationOperator.load(_source_path(args.manifest, "observation")),
        _json(args.genes),
        independent=independent,
    )
    result["evaluation_role"] = roles.get(prediction.patient_id, "development")
    result["qualification"] = (
        "synthetic_or_developmental; fold role is not independent cohort validation"
    )
    _write_json(args.output, result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="validate allowed specimen inputs without target access"
    )
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--output", required=True)
    prepare.set_defaults(run=prepare_command)
    train = commands.add_parser("train", help="fit whole training specimens from a patient fold")
    train.add_argument("--manifests", nargs="+", required=True)
    train.add_argument("--folds", required=True)
    train.add_argument("--outer-fold", type=int, required=True)
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--baseline", help="fitted protein baseline required by context variants")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="cpu", help="cpu or a local cuda device")
    train.set_defaults(run=train_command)
    predict = commands.add_parser("predict", help="predict from allowed specimen evidence only")
    predict.add_argument("--manifest", required=True)
    predict.add_argument("--bundle", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--transfer-only", action="store_true")
    predict.add_argument("--device", default="cpu", help="cpu or a local cuda device")
    predict.set_defaults(run=predict_command)
    evaluate = commands.add_parser("evaluate", help="join saved predictions to separate targets")
    evaluate.add_argument("--prediction", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--genes", required=True, help="JSON list of prespecified shared genes")
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(run=evaluate_command)
    demo = commands.add_parser(
        "demo", help="run a small synthetic file-to-score integration example"
    )
    demo.add_argument("--output", required=True)
    demo.set_defaults(run=demo_command)
    args = parser.parse_args(argv)
    args.run(args)
    return 0


def demo_command(args):
    from .synthetic import create_fixture

    root = Path(args.output)
    manifests = create_fixture(root)
    main(["prepare", "--manifest", str(manifests[0]), "--output", str(root / "prepared")])
    main(
        [
            "train",
            "--manifests",
            *map(str, manifests),
            "--folds",
            str(root / "folds.csv"),
            "--outer-fold",
            "0",
            "--config",
            str(root / "model.json"),
            "--epochs",
            "5",
            "--output",
            str(root / "bundle"),
        ]
    )
    main(
        [
            "predict",
            "--manifest",
            str(manifests[2]),
            "--bundle",
            str(root / "bundle"),
            "--output",
            str(root / "prediction.npz"),
        ]
    )
    main(
        [
            "evaluate",
            "--prediction",
            str(root / "prediction.npz"),
            "--manifest",
            str(manifests[2]),
            "--genes",
            str(root / "genes.json"),
            "--output",
            str(root / "scores.json"),
        ]
    )
    print(f"Synthetic engineering example completed: {root.resolve() / 'scores.json'}")


if __name__ == "__main__":
    raise SystemExit(main())
