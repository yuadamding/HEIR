#!/usr/bin/env python3
"""Record target H&E OOD-score telemetry without changing the detector.

The Mahalanobis location and precision, feature-space identity, training donors,
training-source hashes, and the development-calibrated rejection threshold are
copied unchanged.  A target-score quantile is descriptive telemetry only; it
never becomes a training target or prediction threshold. No target expression
artifact is accepted by this command.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from heir.data import HistologyBag
from heir.uncertainty import MahalanobisOOD

CALIBRATION_CONTRACT = "heir.target_histology_ood_calibration"
CALIBRATION_VERSION = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name,
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(value), allow_pickle=False)
    return buffer.getvalue()


def _atomic_deterministic_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    """Write a byte-stable compressed NPZ and replace the destination atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % path.name,
        suffix=".tmp.npz",
        dir=str(path.parent),
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name, value in payload.items():
                info = zipfile.ZipInfo(filename=name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, _npy_bytes(value), compress_type=zipfile.ZIP_DEFLATED)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _score_in_batches(
    detector: MahalanobisOOD,
    features: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("score_batch_size must be positive")
    scores = np.empty(features.shape[0], dtype=np.float32)
    for start in range(0, features.shape[0], batch_size):
        stop = min(start + batch_size, features.shape[0])
        scores[start:stop] = detector.score(features[start:stop])
    return scores


def calibrate(
    *,
    base_ood_path: Path,
    histology_path: Path,
    sample_id: str,
    quantile: float,
    output_path: Path,
    provenance_path: Path,
    score_batch_size: int,
) -> Mapping[str, object]:
    base_ood_path = base_ood_path.expanduser().resolve()
    histology_path = histology_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    provenance_path = provenance_path.expanduser().resolve()
    sample_id = sample_id.strip()
    if not sample_id:
        raise ValueError("sample_id cannot be empty")
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    if output_path == provenance_path:
        raise ValueError("OOD and provenance outputs must differ")
    existing = [path for path in (output_path, provenance_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "calibration outputs already exist; validate or choose new outputs: %s"
            % ", ".join(str(path) for path in existing)
        )

    base = MahalanobisOOD.from_npz(base_ood_path)
    bag = HistologyBag.load_npz(histology_path)
    if bag.sample_id != sample_id:
        raise ValueError(
            "HistologyBag sample_id %r differs from requested sample %r"
            % (bag.sample_id, sample_id)
        )
    if bag.feature_space_id != base.feature_space_id:
        raise ValueError("HistologyBag and base OOD detector use different feature spaces")
    if base.mean is None or base.precision is None:
        raise ValueError("base OOD detector has no fitted Mahalanobis parameters")
    if base.threshold is None:
        raise ValueError("base OOD detector has no development-calibrated threshold")
    if bag.features.shape[1] != base.mean.shape[0]:
        raise ValueError("HistologyBag width differs from the base OOD detector")

    scores = _score_in_batches(base, bag.features, score_batch_size)
    target_quantile_value = float(np.quantile(scores.astype(np.float64), quantile))
    threshold = float(base.threshold)
    score_stats = {
        "count": int(scores.size),
        "minimum": float(scores.min()),
        "maximum": float(scores.max()),
        "mean": float(scores.astype(np.float64).mean()),
        "standard_deviation": float(scores.astype(np.float64).std()),
        "median": float(np.quantile(scores.astype(np.float64), 0.5)),
        "descriptive_target_quantile": float(quantile),
        "descriptive_target_quantile_value": target_quantile_value,
    }

    calibrated = MahalanobisOOD(
        mean=np.array(base.mean, dtype=np.float64, copy=True),
        precision=np.array(base.precision, dtype=np.float64, copy=True),
        threshold=threshold,
        quantile=float(base.quantile),
        training_donors=tuple(base.training_donors),
        source_sha256=tuple(base.source_sha256),
        feature_space_id=base.feature_space_id,
    )
    calibrated._validate_loaded()

    base_sha256 = _sha256(base_ood_path)
    histology_sha256 = _sha256(histology_path)
    payload = {
        "__contract__": np.asarray(calibrated.CONTRACT, dtype=np.dtype("U")),
        "__version__": np.asarray(calibrated.CONTRACT_VERSION, dtype=np.int64),
        "mean": calibrated.mean,
        "precision": calibrated.precision,
        "threshold": np.asarray(calibrated.threshold, dtype=np.float64),
        "quantile": np.asarray(calibrated.quantile, dtype=np.float64),
        "training_donors": np.asarray(calibrated.training_donors, dtype=np.dtype("U")),
        "source_sha256": np.asarray(calibrated.source_sha256, dtype=np.dtype("U")),
        "feature_space_id": np.asarray(calibrated.feature_space_id, dtype=np.dtype("U")),
        "calibration_contract": np.asarray(CALIBRATION_CONTRACT, dtype=np.dtype("U")),
        "calibration_version": np.asarray(CALIBRATION_VERSION, dtype=np.int64),
        "base_ood_sha256": np.asarray(base_sha256, dtype=np.dtype("U")),
        "histology_sha256": np.asarray(histology_sha256, dtype=np.dtype("U")),
        "sample_id": np.asarray(sample_id, dtype=np.dtype("U")),
        "threshold_source": np.asarray("development_detector", dtype=np.dtype("U")),
        "target_score_quantile": np.asarray(quantile, dtype=np.float64),
        "target_score_quantile_value": np.asarray(target_quantile_value, dtype=np.float64),
        "target_expression_accessed": np.asarray(False, dtype=np.bool_),
        "score_count": np.asarray(score_stats["count"], dtype=np.int64),
        "score_minimum": np.asarray(score_stats["minimum"], dtype=np.float64),
        "score_maximum": np.asarray(score_stats["maximum"], dtype=np.float64),
        "score_mean": np.asarray(score_stats["mean"], dtype=np.float64),
        "score_standard_deviation": np.asarray(score_stats["standard_deviation"], dtype=np.float64),
        "score_median": np.asarray(score_stats["median"], dtype=np.float64),
    }
    _atomic_deterministic_npz(output_path, payload)
    output_sha256 = _sha256(output_path)
    provenance: Mapping[str, object] = {
        "schema": CALIBRATION_CONTRACT + ".v2",
        "sample_id": sample_id,
        "target_expression_accessed": False,
        "calibration_input_modality": "development_threshold_plus_target_histology_telemetry",
        "threshold_source": "development_detector",
        "descriptive_target_quantile": float(quantile),
        "descriptive_target_quantile_value": target_quantile_value,
        "threshold": threshold,
        "score_stats": score_stats,
        "copied_training_provenance": {
            "training_donors": list(calibrated.training_donors),
            "source_sha256": list(calibrated.source_sha256),
            "feature_space_id": calibrated.feature_space_id,
        },
        "inputs": {
            "base_ood": {
                "path": str(base_ood_path),
                "sha256": base_sha256,
                "threshold": float(base.threshold),
                "quantile": float(base.quantile),
            },
            "histology": {
                "path": str(histology_path),
                "sha256": histology_sha256,
                "sample_id": bag.sample_id,
                "nuclei": bag.n_nuclei,
                "feature_width": int(bag.features.shape[1]),
                "feature_space_id": bag.feature_space_id,
            },
        },
        "output": {
            "path": str(output_path),
            "sha256": output_sha256,
            "contract": calibrated.CONTRACT,
            "contract_version": calibrated.CONTRACT_VERSION,
        },
    }
    _atomic_json(provenance_path, provenance)
    return provenance


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-ood", type=Path, required=True)
    parser.add_argument("--histology", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--quantile", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument("--score-batch-size", type=int, default=2048)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    provenance = calibrate(
        base_ood_path=args.base_ood,
        histology_path=args.histology,
        sample_id=args.sample_id,
        quantile=args.quantile,
        output_path=args.output,
        provenance_path=args.provenance_output,
        score_batch_size=args.score_batch_size,
    )
    print(json.dumps(provenance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
