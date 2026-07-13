#!/usr/bin/env python3
"""Freeze spatially disjoint reference/evaluation rows for the oracle ridge probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from heir.data import MorphologyRidgeDatasetArtifact
from heir.utils import reject_output_input_collisions, sha256_file

PLAN_SCHEMA = "heir.morphology_ridge_preparation_plan.v1"


def _load_plan(path: Path, source: Path) -> Mapping[str, object]:
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("morphology-ridge plan is not valid JSON") from error
    required = {
        "schema",
        "source_schema",
        "source_observations_sha256",
        "development_donors",
        "locked_test_donors",
        "type_names",
        "gene_ids",
        "type_marker_gene_ids",
        "technical_covariate_names",
        "feature_space_id",
        "feature_checkpoint_sha256",
        "molecular_space_id",
        "label_source_sha256",
        "registration_source_sha256",
        "exclusion_policy_sha256",
        "registration_method",
        "encoder_name",
        "crop_scale",
        "cohort_id",
        "cohort_release",
        "assay",
        "observation_level",
        "target_construction",
        "reference_mode",
    }
    if not isinstance(plan, Mapping) or not required.issubset(plan):
        raise ValueError("morphology-ridge preparation plan is incomplete")
    if plan["schema"] != PLAN_SCHEMA:
        raise ValueError("morphology-ridge preparation plan schema is unsupported")
    if plan["source_observations_sha256"] != sha256_file(source):
        raise ValueError("morphology-ridge source observations differ from the frozen plan")
    if plan["reference_mode"] != "simulated_spatially_disjoint_unpaired_rna":
        raise ValueError(
            "only the prespecified spatially disjoint simulated reference is supported"
        )
    development = tuple(str(value) for value in plan["development_donors"])
    locked = tuple(str(value) for value in plan["locked_test_donors"])
    if len(set(development)) < 5 or len(set(locked)) < 5 or set(development) & set(locked):
        raise ValueError("the frozen plan needs at least five disjoint donors in each role")
    return plan


def _write_npz(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".npz.tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _row_digest(identities: np.ndarray, values: np.ndarray, role: str) -> str:
    digest = hashlib.sha256(role.encode("utf-8"))
    for identity in identities.astype(str):
        digest.update(identity.encode("utf-8"))
        digest.update(b"\0")
    digest.update(np.ascontiguousarray(values).view(np.uint8))
    return digest.hexdigest()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--source-observations", type=Path, required=True)
    parser.add_argument("--development-output", type=Path, required=True)
    parser.add_argument("--locked-test-output", type=Path, required=True)
    args = parser.parse_args(argv)

    plan_path = args.plan.expanduser().resolve()
    source_path = args.source_observations.expanduser().resolve()
    development_path = args.development_output.expanduser().resolve()
    locked_path = args.locked_test_output.expanduser().resolve()
    if not plan_path.is_file() or not source_path.is_file() or plan_path == source_path:
        raise ValueError("plan and source must be distinct existing files")
    reject_output_input_collisions(
        (development_path, locked_path),
        (plan_path, source_path),
        label="morphology-ridge preparation",
    )
    if development_path == locked_path:
        raise ValueError("development and locked-test outputs must be distinct")
    plan = _load_plan(plan_path, source_path)
    with np.load(source_path, allow_pickle=False) as archive:
        if "schema_version" not in archive.files:
            raise ValueError("source observation artifact has no schema identity")
        source_schema = str(np.asarray(archive["schema_version"]).reshape(()).item())
        if source_schema != str(plan["source_schema"]):
            raise ValueError("source observation schema differs from the frozen plan")
        required_arrays = {
            "observation_ids",
            "donor_ids",
            "block_ids",
            "roi_ids",
            "pool_roles",
            "type_labels",
            "frozen_features",
            "molecular_targets",
            "coordinate_features",
            "stain_features",
            "stain_feature_names",
            "composition_features",
            "composition_feature_names",
            "technical_covariates",
            "registration_is_one_to_one",
        }
        missing = sorted(required_arrays - set(archive.files))
        if missing:
            raise ValueError("source cell artifact is missing: %s" % ", ".join(missing))
        row_arrays = required_arrays - {"stain_feature_names", "composition_feature_names"}
        values = {name: np.array(archive[name], copy=True) for name in row_arrays}
        stain_feature_names = np.asarray(archive["stain_feature_names"]).astype(str)
        composition_feature_names = np.asarray(archive["composition_feature_names"]).astype(str)

    cells = len(values["observation_ids"])
    if any(len(array) != cells for array in values.values()):
        raise ValueError("source cell rows are misaligned")
    for matrix_name, names in (
        ("stain_features", stain_feature_names),
        ("composition_features", composition_feature_names),
    ):
        matrix = np.asarray(values[matrix_name])
        if (
            matrix.ndim != 2
            or names.ndim != 1
            or matrix.shape[1] != len(names)
            or len(set(names.tolist())) != len(names)
            or any(not name.strip() for name in names.tolist())
        ):
            raise ValueError("source %s differ from their names" % matrix_name)
    if len(set(values["observation_ids"].astype(str).tolist())) != cells:
        raise ValueError("source observation identities are not unique")
    if not np.asarray(values["registration_is_one_to_one"], dtype=np.bool_).all():
        raise ValueError("source contains a non-one-to-one registration")
    pool_roles = values["pool_roles"].astype(str)
    if set(pool_roles.tolist()) != {"evaluation", "reference"}:
        raise ValueError("source pool_roles must contain evaluation and reference")
    donors = values["donor_ids"].astype(str)
    blocks = values["block_ids"].astype(str)
    labels = np.asarray(values["type_labels"], dtype=np.int64)
    num_types = len(tuple(plan["type_names"]))
    if np.any(labels < 0) or np.any(labels >= num_types):
        raise ValueError("source labels exceed the frozen RNA-only ontology")

    outputs = (
        (
            "development",
            tuple(str(value) for value in plan["development_donors"]),
            development_path,
        ),
        ("locked_test", tuple(str(value) for value in plan["locked_test_donors"]), locked_path),
    )
    for role, role_donors, output in outputs:
        donor_mask = np.isin(donors, np.asarray(role_donors))
        evaluation = donor_mask & (pool_roles == "evaluation")
        reference = donor_mask & (pool_roles == "reference")
        if set(donors[evaluation].tolist()) != set(role_donors):
            raise ValueError("every frozen donor needs evaluation cells")
        reference_means = np.zeros_like(values["molecular_targets"][evaluation], dtype=np.float64)
        evaluation_donors = donors[evaluation]
        evaluation_labels = labels[evaluation]
        for donor in role_donors:
            reference_blocks = set(blocks[reference & (donors == donor)].tolist())
            evaluation_blocks = set(blocks[evaluation & (donors == donor)].tolist())
            if (
                not reference_blocks
                or not evaluation_blocks
                or reference_blocks & evaluation_blocks
            ):
                raise ValueError("reference and evaluation blocks are not spatially disjoint")
            for type_index in sorted(set(evaluation_labels[evaluation_donors == donor].tolist())):
                source_selected = reference & (donors == donor) & (labels == type_index)
                target_selected = (evaluation_donors == donor) & (evaluation_labels == type_index)
                if not np.any(source_selected):
                    raise ValueError("an evaluated donor/type lacks an independent reference pool")
                reference_means[target_selected] = values["molecular_targets"][
                    source_selected
                ].mean(axis=0)
        payload = {
            "schema_version": np.asarray(MorphologyRidgeDatasetArtifact.SCHEMA),
            "observation_ids": values["observation_ids"][evaluation],
            "donor_ids": donors[evaluation],
            "block_ids": blocks[evaluation],
            "roi_ids": values["roi_ids"][evaluation],
            "type_labels": labels[evaluation],
            "type_names": np.asarray(plan["type_names"]),
            "frozen_features": values["frozen_features"][evaluation],
            "molecular_targets": values["molecular_targets"][evaluation],
            "reference_means": reference_means,
            "coordinate_features": values["coordinate_features"][evaluation],
            "stain_features": values["stain_features"][evaluation],
            "stain_feature_names": stain_feature_names,
            "composition_features": values["composition_features"][evaluation],
            "composition_feature_names": composition_feature_names,
            "technical_covariates": values["technical_covariates"][evaluation],
            "technical_covariate_names": np.asarray(plan["technical_covariate_names"]),
            "gene_ids": np.asarray(plan["gene_ids"]),
            "type_marker_gene_ids": np.asarray(plan["type_marker_gene_ids"]),
            "feature_space_id": np.asarray(plan["feature_space_id"]),
            "feature_checkpoint_sha256": np.asarray(plan["feature_checkpoint_sha256"]),
            "molecular_space_id": np.asarray(plan["molecular_space_id"]),
            "reference_source_sha256": np.asarray(
                _row_digest(
                    values["observation_ids"][reference],
                    values["molecular_targets"][reference],
                    role + "_reference",
                )
            ),
            "label_source_sha256": np.asarray(plan["label_source_sha256"]),
            "target_source_sha256": np.asarray(
                _row_digest(
                    values["observation_ids"][evaluation],
                    values["molecular_targets"][evaluation],
                    role + "_evaluation",
                )
            ),
            "registration_source_sha256": np.asarray(plan["registration_source_sha256"]),
            "exclusion_policy_sha256": np.asarray(plan["exclusion_policy_sha256"]),
            "registration_method": np.asarray(plan["registration_method"]),
            "encoder_name": np.asarray(plan["encoder_name"]),
            "crop_scale": np.asarray(plan["crop_scale"]),
            "cohort_id": np.asarray(plan["cohort_id"]),
            "cohort_release": np.asarray(plan["cohort_release"]),
            "assay": np.asarray(plan["assay"]),
            "observation_level": np.asarray(plan["observation_level"]),
            "target_construction": np.asarray(plan["target_construction"]),
            "reference_pool_independent": np.asarray(True),
            "labels_independent_of_images": np.asarray(True),
            "registration_is_one_to_one": np.asarray(True),
        }
        _write_npz(output, payload)
        MorphologyRidgeDatasetArtifact.load_npz(output, role=role)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
