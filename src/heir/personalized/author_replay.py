"""Replay the published HTAPP Slide-seq coordinate chain from named native beads.

The count filters below reproduce the author's preprocessing population, whose
mean fixes each rotation pivot. They do not select genes by agreement, estimate
transforms, or qualify image/biological correspondence.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .cohort import _verified_source_path
from .schema import identifiers
from .transform_replay import TransformStep, author_rigid_step, observation_join, replay_coordinates


def source_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read_coordinate_source(
    path, *, coordinate_columns=("xcoord", "ycoord"), obs_fields=(), counts=False
):
    """Read only coordinate/identity fields and optionally the native sparse measurement matrix."""
    import h5py
    from scipy.sparse import csc_matrix, csr_matrix

    with h5py.File(path, "r") as handle:

        def column(name):
            node = handle["obs"][name]
            if isinstance(node, h5py.Group) or "categories" in node.attrs:
                categories = (
                    node["categories"]
                    if isinstance(node, h5py.Group)
                    else handle[node.attrs["categories"]]
                )
                codes = node["codes"][:] if isinstance(node, h5py.Group) else node[:]
                values = categories.asstr()[:] if categories.dtype.kind in "OS" else categories[:]
                if np.any((codes < 0) | (codes >= len(values))):
                    raise ValueError("coordinate source categorical values include missing codes")
                return values[codes]
            return node.asstr()[:] if node.dtype.kind in "OS" else node[:]

        index = handle["obs"].attrs.get("_index", "_index")
        ids = tuple(map(str, column(index)))
        identifiers(ids, "source observation IDs")
        xy = np.column_stack([column(name) for name in coordinate_columns]).astype(np.float64)
        if xy.shape != (len(ids), 2) or not np.isfinite(xy).all():
            raise ValueError("source coordinates must be finite and follow the observation axis")
        result = {"ids": ids, "coordinates": xy, "obs": {name: column(name) for name in obs_fields}}
        if counts:
            node = handle["X"]
            encoding = node.attrs.get("encoding-type")
            if encoding not in ("csr_matrix", "csc_matrix"):
                raise ValueError("native measurements require explicit CSR/CSC encoding")
            constructor = csr_matrix if encoding == "csr_matrix" else csc_matrix
            matrix = constructor(
                (node["data"][:], node["indices"][:], node["indptr"][:]),
                shape=tuple(node.attrs["shape"]),
            ).tocsr()
            matrix.sum_duplicates()
            matrix.eliminate_zeros()
            index = handle["var"].attrs.get("_index", "_index")
            genes = tuple(handle["var"][index].asstr()[:])
            identifiers(genes, "native measured gene IDs")
            if (
                matrix.shape != (len(ids), len(genes))
                or not np.isfinite(matrix.data).all()
                or np.any(matrix.data < 0)
            ):
                raise ValueError("native measurements must align to IDs and be finite/nonnegative")
            result.update(counts=matrix, gene_ids=genes)
    return result


def _numeric_literal(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str, type(None))):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_numeric_literal(value) for value in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return (-1 if isinstance(node.op, ast.USub) else 1) * _numeric_literal(node.operand)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        left, right = _numeric_literal(node.left), _numeric_literal(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        return left * right if isinstance(node.op, ast.Mult) else left / right
    raise ValueError("registration parameter is not a declarative numeric literal")


def registration_parameters(notebook_path, assay):
    """Extract declared registration.loc assignments without executing notebook code."""
    notebook = json.loads(Path(notebook_path).read_text())
    parameters, cells = {}, set()
    for cell_index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        try:
            tree = ast.parse("".join(cell.get("source", [])))
        except SyntaxError:
            continue  # Unrelated notebook magics are never executed.
        for statement in tree.body:
            if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
                continue
            target = statement.targets[0]
            if not (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Attribute)
                and isinstance(target.value.value, ast.Name)
                and target.value.value.id == "registration"
                and target.value.attr == "loc"
                and isinstance(target.slice, ast.Tuple)
                and len(target.slice.elts) == 2
            ):
                continue
            label, field = (_numeric_literal(value) for value in target.slice.elts)
            if not isinstance(label, str) or not re.fullmatch(re.escape(assay) + r" \d+", label):
                continue
            parameters.setdefault(label, {})[field] = _numeric_literal(statement.value)
            cells.add(cell_index)
    if not parameters:
        raise ValueError(f"no declared {assay} registration parameters in notebook")
    return parameters, {
        **source_identity(notebook_path),
        "source_cell_indices_zero_based": sorted(cells),
    }


def residual_summary(replayed, stored, *, units="registered_pixel"):
    left, right = np.asarray(replayed, float), np.asarray(stored, float)
    if (
        left.shape != right.shape
        or left.ndim != 2
        or left.shape[1] != 2
        or not np.isfinite([left, right]).all()
    ):
        raise ValueError("coordinate residuals need finite aligned N-by-2 arrays")
    distance = np.linalg.norm(left - right, axis=1)
    maximum = float(distance.max()) if len(distance) else None
    return {
        "n": len(distance),
        "units": units,
        "max_euclidean": maximum,
        "rms_euclidean": float(np.sqrt(np.mean(distance**2))) if len(distance) else None,
        "p95_euclidean": float(np.quantile(distance, 0.95)) if len(distance) else None,
        "max_fraction_100_pixels": maximum / 100 if maximum is not None else None,
        "max_fraction_200_pixels": maximum / 200 if maximum is not None else None,
    }


def density_supported_mask(coordinates, *, radius=50.0, min_samples=20):
    """Exact DBSCAN non-noise membership: core points and points within radius of a core.

    Cluster labels are unnecessary because the author only discards label -1.
    The radius includes its boundary; min_samples includes the point itself.
    """
    from scipy.spatial import cKDTree

    xy = np.asarray(coordinates, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("density coordinates must be finite N-by-2 values")
    if (
        not np.isfinite(radius)
        or radius <= 0
        or isinstance(min_samples, bool)
        or not isinstance(min_samples, int)
        or min_samples <= 0
    ):
        raise ValueError("density radius and integer min_samples must be positive")
    core = cKDTree(xy).query_ball_point(xy, radius, return_length=True) >= min_samples
    return cKDTree(xy[core]).query(xy)[0] <= radius if core.any() else np.zeros(len(xy), bool)


def author_slide_seq_qc(native):
    """Return pre-alignment masks and the author's shared adaptive count cutoff.

    ``native`` maps assay labels to dictionaries containing counts, gene_ids and
    density_keep. All native replicates participate, even a later excluded one.
    """
    genes = sorted({gene for source in native.values() for gene in source["gene_ids"]})
    lookup, detected = {g: i for i, g in enumerate(genes)}, np.zeros(len(genes), dtype=int)
    initial, raw_counts = {}, {}
    for label, source in native.items():
        counts = source["counts"]
        if not np.allclose(counts.data, np.round(counts.data), atol=1e-6, rtol=0):
            raise ValueError("Slide-seq QC requires raw integer RNA counts")
        totals = np.asarray(counts.sum(axis=1)).ravel()
        present = np.asarray((counts > 0).sum(axis=1)).ravel()
        initial[label] = source["density_keep"] & (totals >= 30) & (present >= 30)
        raw_counts[label] = totals
        columns = [lookup[g] for g in source["gene_ids"]]
        detected[columns] += np.asarray((counts[initial[label]] > 0).sum(axis=0)).ravel()
    all_counts = np.concatenate([raw_counts[label][keep] for label, keep in initial.items()])
    cutoff = 30
    while True:
        above = all_counts[all_counts > cutoff]
        if not len(above):
            raise ValueError("author adaptive QC has no eligible high-count population")
        if np.mean(above >= 100) > 0.65:
            break
        cutoff += 1
    selected = {}
    for label, source in native.items():
        columns = [detected[lookup[g]] >= 3 for g in source["gene_ids"]]
        totals = np.asarray(source["counts"][:, columns].sum(axis=1)).ravel()
        selected[label] = initial[label] & (totals >= cutoff)
    return (
        selected,
        initial,
        {
            "min_counts_initial": 30,
            "min_genes": 30,
            "min_cells_per_gene_across_native_replicates": 3,
            "adaptive_count_cutoff": cutoff,
            "high_count_boundary": 100,
            "criterion": (
                "fraction_of_initial_counts_strictly_above_cutoff_that_are_at_least100_gt0.65"
            ),
            "final_filter": "counts_after_gene_filter_greater_than_or_equal_to_cutoff",
            "native_replicate_ids_in_shared_qc": sorted(native),
            "n_retained_genes_across_native_replicates": int((detected >= 3).sum()),
        },
    )


def replay_slide_seq(root, biopsy_id, entry, *, pipeline_root=None, verification_cache=None):
    """Return (reports, chains, numeric arrays); the caller publishes external artifacts."""
    root = Path(root)
    pipeline = (
        Path(pipeline_root) if pipeline_root is not None else root / "reference/HTAPP-Pipelines"
    )
    if not (pipeline / "spatial_utils.py").is_file():
        pipeline = pipeline / "HTAPP_MBC/03_spatial_analysis"
    participant, sample = biopsy_id.split("-")
    stem = f"HTAPP-{participant}-SMP-{sample}"
    params, notebook = registration_parameters(
        pipeline / f"00.1_processing/{stem}_processing.ipynb", "slide_seq"
    )
    utilities = source_identity(pipeline / "spatial_utils.py")
    registered_source = entry["role_sources"]["targets"]
    registered_path = _verified_source_path(
        root, registered_source, verification_cache=verification_cache
    )
    registered = read_coordinate_source(
        registered_path, coordinate_columns=("x", "y"), obs_fields=("replicate", "x_orig", "y_orig")
    )
    registered_labels = np.asarray(registered["obs"]["replicate"]).astype(str)
    native, source_records = {}, {}
    for source in entry["sources"]:
        if (
            source["modality"] != "slide_seq"
            or not source["working_path"].startswith("working/unregistered/")
            or not source["working_path"].endswith("_slide_seq.h5ad")
        ):
            continue
        label = source["replicate_id"]
        if label in native:
            raise ValueError("multiple native Slide-seq files claim the same replicate")
        path = _verified_source_path(root, source, verification_cache=verification_cache)
        data = read_coordinate_source(path, counts=True)
        xy = data["coordinates"]
        native_origin = xy.min(axis=0)
        scaled = (xy - native_origin) / (5000 / 3000)
        keep = density_supported_mask(scaled)
        if not keep.any():
            raise ValueError("author spatial density filter has no supported native beads")
        post_origin = scaled[keep].min(axis=0)
        data.update(
            native_origin=native_origin,
            post_origin=post_origin,
            density_keep=keep,
            analysis_coordinates=scaled - post_origin,
        )
        native[label], source_records[label] = (
            data,
            {key: source[key] for key in ("working_path", "bytes", "sha256")},
        )
    if not native or not set(registered_labels) <= set(native):
        raise ValueError("registered Slide-seq replicate has no native source")
    selected, initial, qc = author_slide_seq_qc(native)
    reports, chains, arrays = [], [], {}
    for label, data in sorted(native.items()):
        prefix = f"{biopsy_id}__slide_seq__{label}__"
        for name, value in {
            "native_observation_ids": np.asarray(data["ids"]),
            "native_coordinates": data["coordinates"],
            "density_keep": data["density_keep"],
            "initial_qc_keep": initial[label],
            "alignment_qc_keep": selected[label],
            "analysis_coordinates": data["analysis_coordinates"],
        }.items():
            arrays[prefix + name] = value
        mask = registered_labels == label
        metadata = {
            "biopsy_id": biopsy_id,
            "patient_id": entry["patient_id"],
            "specimen_id": entry["specimen_id"],
            "assay": "slide_seq",
            "replicate": label,
            "n_native": len(data["ids"]),
            "n_density_excluded": int((~data["density_keep"]).sum()),
            "n_initial_qc_excluded_after_density": int(
                (data["density_keep"] & ~initial[label]).sum()
            ),
            "n_alignment_qc": int(selected[label].sum()),
            "n_registered": int(mask.sum()),
            "alignment_cutoff": qc["adaptive_count_cutoff"],
            "correspondence_qualified": False,
            "retention_selection_replayed": False,
            "retention_note": "retained coordinate replay; later H&E-mask selection not replayed",
        }
        chain = {
            **metadata,
            "source": source_records[label],
            "registered_source": {
                key: registered_source[key] for key in ("working_path", "bytes", "sha256")
            },
            "all_native_qc_sources": source_records,
            "notebook": notebook,
            "utilities": utilities,
            "qc": qc,
            "native_units": "slide_seq_native_pixel",
            "analysis_units": "author_analysis_coordinate",
            "registered_units": "registered_pixel",
            "executed_native_scale": 0.6,
            "publication_native_scale_um": 0.65,
            "physical_calibration_verified": False,
            "counterfactual_scale_applied": False,
            "display_origin": "lower; no coordinate flip for display",
            "pivot_population": (
                "native_density_and_initial_author_count_gene_QC_before_later_HE_mask_and_QC"
            ),
        }
        if not mask.any():
            reports.append(
                {
                    **metadata,
                    "status": "native_replicate_not_retained_in_registered_bundle",
                    "max_coordinate_error_pixels": None,
                    "max_replay_error_pixels": None,
                    "source_ids_complete": None,
                    "n_matched": 0,
                }
            )
            chains.append({**chain, "steps": [], "status": "contributes_to_shared_author_QC_only"})
            continue
        ids = tuple(np.asarray(registered["ids"])[mask])
        joined = observation_join(data["ids"], ids, target_suffix="-" + label)
        if not selected[label][joined].all():
            raise ValueError("retained registered beads violate replayed pre-alignment QC")
        declared = params[f"slide_seq {label}"]
        origin = data["analysis_coordinates"][selected[label]].mean(axis=0)
        steps = []
        for name, scale, shift, before, after in (
            (
                "subtract_native_minimum",
                1.0,
                -data["native_origin"],
                "slide_seq_native_pixel",
                "slide_seq_native_pixel",
            ),
            (
                "executed_native_scale",
                0.6,
                np.zeros(2),
                "slide_seq_native_pixel",
                "author_analysis_coordinate",
            ),
            (
                "subtract_density_survivor_minimum",
                1.0,
                -data["post_origin"],
                "author_analysis_coordinate",
                "author_analysis_coordinate",
            ),
        ):
            matrix = np.diag([scale, scale, 1.0])
            matrix[:2, 2] = shift
            steps.append(TransformStep(name, matrix, before, after))
        flip = declared.get("flip")
        maximum = (
            data["analysis_coordinates"][selected[label], 0 if flip == "v" else 1].max()
            if flip
            else None
        )
        steps.append(
            author_rigid_step(
                "author_reflection_rotation_translation",
                rotation_origin=origin,
                degrees=declared["degrees"],
                shift=[declared["shift_x"], declared["shift_y"]],
                input_units="author_analysis_coordinate",
                output_units="registered_pixel",
                flip=flip,
                reflection_maximum=maximum,
            )
        )
        replayed = replay_coordinates(
            data["coordinates"][joined], steps, input_units="slide_seq_native_pixel"
        )
        stored = registered["coordinates"][mask]
        stored_orig = np.column_stack(
            (registered["obs"]["x_orig"][mask], registered["obs"]["y_orig"][mask])
        )
        residuals = residual_summary(replayed, stored)
        original_residuals = residual_summary(
            data["analysis_coordinates"][joined], stored_orig, units="author_analysis_coordinate"
        )
        reports.append(
            {
                **metadata,
                "status": "replayed",
                "max_coordinate_error_pixels": residuals["max_euclidean"],
                "max_replay_error_pixels": residuals["max_euclidean"],
                "source_ids_complete": True,
                "n_matched": len(joined),
                "coordinate_residuals": residuals,
                "intermediate_residuals": original_residuals,
                "n_alignment_qc_not_retained": int(selected[label].sum() - mask.sum()),
            }
        )
        chains.append(
            {
                **chain,
                "steps": [step.to_dict() for step in steps],
                "rotation_origin": origin.tolist(),
                "declared_registration": declared,
                "native_origin": data["native_origin"].tolist(),
                "post_density_origin": data["post_origin"].tolist(),
            }
        )
        for name, value in {
            "registered_observation_ids": np.asarray(ids),
            "joined_native_rows": joined,
            "stored_registered_xy": stored,
            "replayed_registered_xy": replayed,
            "residual_xy": replayed - stored,
            "stored_analysis_xy": stored_orig,
        }.items():
            arrays[prefix + name] = value
    return reports, chains, arrays
