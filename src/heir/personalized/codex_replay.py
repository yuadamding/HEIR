"""Replay retained CODEX geometry and inventory the declared H&E image chain.

Coordinate agreement checks provenance. It is not an independent landmark test,
physical scanner calibration, or evidence that assay sections are interchangeable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .author_replay import (
    read_coordinate_source,
    registration_parameters,
    residual_summary,
    source_identity,
)
from .cohort import _verified_source_path
from .transform_replay import (
    TransformStep,
    author_rigid_step,
    observation_join,
    replay_coordinates,
    replay_piecewise,
)


def _codex_geometry(coordinates, regions, counts, parameters, *, biopsy_id):
    """Use source QC and declared transforms, without estimating their parameters."""
    xy = np.asarray(coordinates, dtype=np.float64)
    regions = np.asarray(regions)
    if (
        xy.ndim != 2
        or xy.shape[1] != 2
        or not len(xy)
        or not np.isfinite(xy).all()
        or regions.shape != (len(xy),)
        or not np.isfinite(regions).all()
        or np.any(regions < 1)
        or np.any(regions != regions.astype(int))
        or counts.shape[0] != len(xy)
    ):
        raise ValueError("CODEX replay requires finite coordinates and positive integer regions")
    qc = (np.asarray(counts.sum(axis=1)).ravel() >= 1) & (
        np.asarray((counts > 0).sum(axis=1)).ravel() >= 1
    )
    factor = 0.386 * (2 if biopsy_id == "514-6760" else 1)
    matrix = np.eye(3)
    matrix[:2, :2] *= factor
    matrix[:2, 2] = -xy.min(axis=0) * factor
    native_step = TransformStep(
        "native_minimum_subtraction_and_author_CODEX_scale",
        matrix,
        "author_CODEX_native_pixel",
        "author_analysis_coordinate",
    )
    original = replay_coordinates(xy, [native_step], input_units=native_step.input_units)
    pre_rotation = qc.copy()
    if biopsy_id == "514-6760":
        pre_rotation &= ~(
            ((regions == 3) & (original[:, 0] > 3600)) | ((regions == 2) & (original[:, 0] < 450))
        )
    pieces, origins = {}, {}
    for region in np.unique(regions):
        label = str(int(region))
        selected = pre_rotation & (regions == region)
        if not selected.any():
            raise ValueError("CODEX fragment has no observations in its rotation population")
        origin = original[selected].mean(axis=0)
        values = {}
        for key in ("degrees", "shift_x", "shift_y"):
            value = parameters[key]
            if isinstance(value, (list, tuple)):
                if int(region) > len(value):
                    raise ValueError("CODEX fragment lacks a declared transform")
                value = value[int(region) - 1]
            values[key] = value
        # spatial_utils.aligner ignores flip for the multi-fragment branch.
        flip = parameters.get("flip") if len(np.unique(regions)) == 1 else None
        reflection_maximum = None
        if flip in ("v", "h"):
            reflection_maximum = original[selected, 0 if flip == "v" else 1].max()
        rigid = author_rigid_step(
            "author_fragment_rotation_and_translation",
            rotation_origin=origin,
            degrees=values["degrees"],
            shift=[values["shift_x"], values["shift_y"]],
            input_units=native_step.output_units,
            output_units="registered_pixel",
            flip=flip,
            reflection_maximum=reflection_maximum,
        )
        pieces[label] = [rigid]
        origins[label] = {
            "rotation_origin": origin.tolist(),
            "rotation_population_n": int(selected.sum()),
            "degrees": values["degrees"],
            "shift": [values["shift_x"], values["shift_y"]],
            "steps": [rigid.to_dict()],
        }
    replayed = replay_piecewise(
        original, regions.astype(int).astype(str), pieces, input_units=native_step.output_units
    )
    retained = pre_rotation.copy()
    overlap_line = None
    if biopsy_id == "213-6752":
        fourth = replayed[(regions == 4) & qc]
        if not len(fourth):
            raise ValueError("213 overlap replay requires CODEX fragment 4")
        xmin, ymin = fourth.min(axis=0)
        x = fourth[fourth[:, 1] == ymin, 0].max()
        y = fourth[fourth[:, 0] == xmin, 1].min()
        if xmin == x:
            raise ValueError("213 author overlap line is degenerate")
        slope = (y - ymin) / (xmin - x)
        intercept = y - slope * xmin
        retained &= ~((regions == 3) & (replayed[:, 1] > slope * replayed[:, 0] + intercept))
        overlap_line = {"slope": float(slope), "intercept": float(intercept)}
    return {
        "original": original,
        "replayed": replayed,
        "qc": qc,
        "pre_rotation": pre_rotation,
        "retained_before_HE_mask": retained,
        "chain": {
            "native_step": native_step.to_dict(),
            "native_scale_factor": factor,
            "fragments": origins,
            "initial_QC": "sum(counts)>=1 and number_of_positive_markers>=1",
            "centering_population": "initial_QC_then_514_pre_rotation_overlap_removal",
            "post_rotation_overlap_line_213": overlap_line,
            "physical_calibration_independently_verified": False,
        },
    }


def _image_dimensions(path):
    from PIL import Image

    # Header inspection only; no image is decoded or allocated here.
    previous_limit = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as image:
            return list(image.size)
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit


def _image_chains(root, entry, parameters, *, verification_cache):
    result = []
    for replicate in ("1", "2"):
        found = {}
        for source in entry["sources"]:
            parts = Path(source["working_path"]).parts
            if source["modality"] != "HE" or source["replicate_id"] != replicate:
                continue
            if "registered" in parts:
                found["processed"] = source
            elif "unregistered" in parts and "HE_highres" in parts[-1]:
                found["native_highres"] = source
        if set(found) != {"processed", "native_highres"}:
            raise ValueError("H&E replay metadata requires native highres and processed images")
        images = {}
        for role, source in found.items():
            path = _verified_source_path(root, source, verification_cache=verification_cache)
            images[role] = {
                "path": str(path),
                "sha256": source["sha256"],
                "bytes": int(source["bytes"]),
                "dimensions_width_height": _image_dimensions(path),
            }
        param = parameters[f"HE {replicate}"]
        result.append(
            {
                "image_id": f"HE {replicate}",
                "images": images,
                "declared_operations": [
                    {"operation": "matplotlib.image.thumbnail", "scale": 0.4},
                    {
                        "operation": "skimage.transform.rotate",
                        "degrees": param["degrees"],
                        "resize": True,
                        "mode": "edge",
                    },
                    {
                        "operation": "crop_or_pad_before_rescale",
                        "left": param["crop_left"],
                        "right": param["crop_right"],
                        "first_row": param["crop_bottom"],
                        "last_row": param["crop_top"],
                        "negative_left_or_first_row_means_padding": True,
                    },
                    {"operation": "skimage.transform.rescale", "scale": param["scale"]},
                    {"operation": "clip_0_1_then_matplotlib_JPEG_save"},
                ],
                "shared_frame": "same_registered_xy_on_HE1_and_HE2",
                "array_index_convention": "image[y,x]; first_stored_row_is_y0",
                "display_origin": "lower",
                "display_origin_adds_no_array_flip": True,
                "raster_replay_performed": False,
                "independent_landmark_residuals_available": False,
                "physical_calibration_independently_verified": False,
            }
        )
    return result


def replay_codex(root, biopsy_id, entry, *, pipeline_root=None, verification_cache=None):
    """Return per-replicate reports, source chains, and stable-ID diagnostic arrays."""
    root = Path(root)
    pipeline = root / "reference/HTAPP-Pipelines" if pipeline_root is None else Path(pipeline_root)
    spatial = (
        pipeline
        if (pipeline / "spatial_utils.py").is_file()
        else pipeline / "HTAPP_MBC/03_spatial_analysis"
    )
    sample = "HTAPP-" + biopsy_id.replace("-", "-SMP-", 1)
    notebook = spatial / "00.1_processing" / f"{sample}_processing.ipynb"
    params, notebook_evidence = registration_parameters(notebook, "codex")
    he_params, _ = registration_parameters(notebook, "HE")
    image_chains = _image_chains(root, entry, he_params, verification_cache=verification_cache)
    registered_source = entry["role_sources"]["protein"]
    registered_path = _verified_source_path(
        root, registered_source, verification_cache=verification_cache
    )
    stored = read_coordinate_source(
        registered_path,
        coordinate_columns=("x", "y"),
        obs_fields=("x_orig", "y_orig", "region", "replicate"),
    )
    reports, chains, arrays = [], [], {}
    for replicate in registered_source["replicate_id"].split(";"):
        candidates = [
            s
            for s in entry["sources"]
            if s["modality"] == "codex"
            and s["replicate_id"] == replicate
            and s["working_path"].endswith(".h5ad")
            and "unregistered" in Path(s["working_path"]).parts
        ]
        if len(candidates) != 1:
            raise ValueError("CODEX replay requires exactly one native source per replicate")
        native_source = candidates[0]
        native_path = _verified_source_path(
            root, native_source, verification_cache=verification_cache
        )
        native = read_coordinate_source(native_path, obs_fields=("region",), counts=True)
        geometry = _codex_geometry(
            native["coordinates"],
            native["obs"]["region"],
            native["counts"],
            params[f"codex {replicate}"],
            biopsy_id=biopsy_id,
        )
        selected = np.asarray(stored["obs"]["replicate"]).astype(str) == replicate
        target_ids = np.asarray(stored["ids"])[selected]
        indices = observation_join(native["ids"], target_ids, target_suffix="-" + replicate)
        if not np.array_equal(
            np.asarray(native["obs"]["region"])[indices],
            np.asarray(stored["obs"]["region"])[selected],
        ):
            raise ValueError("CODEX native and registered fragment labels disagree")
        actual = stored["coordinates"][selected]
        original_actual = np.column_stack(
            [np.asarray(stored["obs"][k])[selected] for k in ("x_orig", "y_orig")]
        )
        replay = geometry["replayed"][indices]
        error = np.linalg.norm(replay - actual, axis=1)
        original_error = np.linalg.norm(geometry["original"][indices] - original_actual, axis=1)
        expected_ids = set(np.asarray(native["ids"])[geometry["retained_before_HE_mask"]])
        actual_ids = set(np.asarray(native["ids"])[indices])
        report = {
            "patient_id": entry["patient_id"],
            "specimen_id": entry["specimen_id"],
            "biopsy_id": biopsy_id,
            "assay": "codex",
            "replicate": replicate,
            "n_native": len(native["ids"]),
            "n_registered": len(target_ids),
            "n_matched": len(indices),
            "source_ids_complete": True,
            "max_replay_error_pixels": float(error.max()),
            "max_original_coordinate_error_pixels": float(original_error.max()),
            "pre_qc_n": len(native["ids"]),
            "post_qc_n": int(geometry["qc"].sum()),
            "post_pre_rotation_selection_n": int(geometry["pre_rotation"].sum()),
            "post_overlap_before_HE_mask_n": len(expected_ids),
            "registered_subset_of_replayed_selection": actual_ids <= expected_ids,
            "registered_selection_exact_before_HE_mask": actual_ids == expected_ids,
            "remaining_author_HE_mask_exclusions_n": len(expected_ids - actual_ids),
            "HE_mask_selection_replayed": False,
            "coordinate_replay_passed": bool(error.max() < 1e-7 and original_error.max() < 1e-7),
            "biological_alignment_validated": False,
            "independent_landmark_residuals_available": False,
        }
        reports.append(report)
        chains.append(
            {
                **{
                    k: report[k]
                    for k in ("patient_id", "specimen_id", "biopsy_id", "assay", "replicate")
                },
                "notebook": notebook_evidence,
                "spatial_utils": source_identity(spatial / "spatial_utils.py"),
                "upstream_assembly_notebook": source_identity(spatial / "00_matchSpatial.ipynb"),
                "upstream_assembly_note": (
                    "cell57 stacks native fragments in y before retained native export"
                ),
                "sources": {
                    "native": {"path": str(native_path), "sha256": native_source["sha256"]},
                    "registered": {
                        "path": str(registered_path),
                        "sha256": registered_source["sha256"],
                    },
                },
                "geometry": geometry["chain"],
                "residuals": residual_summary(replay, actual),
                "image_chains": image_chains,
                "interpretation": "source_chain_reproduction_only_not_biological_alignment",
                "cached_notebook_rerun_caveat": (
                    "saved adjust=False cells may load prior processed files"
                ),
            }
        )
        prefix = f"{biopsy_id}__codex__{replicate}__"
        for key, value in {
            "native_ids": np.asarray(native["ids"], dtype=str),
            "registered_ids": target_ids.astype(str),
            "registered_to_native": indices,
            "native_coordinates": native["coordinates"],
            "native_region": native["obs"]["region"],
            "initial_QC_mask": geometry["qc"],
            "pre_rotation_mask": geometry["pre_rotation"],
            "retained_before_HE_mask": geometry["retained_before_HE_mask"],
            "registered_coordinates": actual,
            "replayed_coordinates": replay,
            "original_coordinate_residual": geometry["original"][indices] - original_actual,
        }.items():
            arrays[prefix + key] = value
    return reports, chains, arrays
