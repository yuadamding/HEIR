"""Offline visual review of an H&E-selected regional pilot and later target coverage."""

from __future__ import annotations

import base64
import csv
import html
import io
import json
from pathlib import Path

import numpy as np

from .cohort import (
    _restore_grid,
    _source_path,
    _verified_source_path,
    load_cohort_targets,
    prepare_cohort_inputs,
)
from .regions import point_to_grid

_COLORS = {
    "retained": "#16834a",
    "protein_complete": "#7952ba",
    "protein_partial": "#da9414",
    "excluded": "#d52f35",
    "uncovered": "#717b87",
}


def write_measurement_overlay(output, he_path, grid, bead_xy, bead_replicates, common_fine_indices):
    """Write morphology/support panels without loading models or expression values.

    Image and points are flipped together only for lower-origin display. At most
    8,000 evenly spaced in-image source rows are drawn per replicate; the caller
    retains all bead positions in the numeric measurement artifact.
    """
    from PIL import Image, ImageDraw

    xy = np.asarray(bead_xy, dtype=float)
    replicates = np.asarray(bead_replicates).astype(str)
    common = np.asarray(common_fine_indices)
    if replicates.ndim != 1 or xy.shape != (len(replicates), 2):
        raise ValueError("overlay bead coordinates and replicate labels must align")
    if not len(replicates) or any(not label.strip() for label in replicates):
        raise ValueError("overlay requires nonempty assay replicate labels")
    if common.ndim != 1 or common.dtype.kind not in "iu":
        raise ValueError("common fine-region indices must be an integer vector")
    if len(set(common)) != len(common) or np.any((common < 0) | (common >= len(grid.region_ids))):
        raise ValueError("common fine-region indices must be unique and within the grid")
    if grid.settings["coordinate_units"] != "registered_pixel":
        raise ValueError("measurement overlays require registered-pixel geometry")
    mapped = point_to_grid(grid, xy)
    height, width = grid.settings["image_shape"]
    inside = ((xy >= 0) & (xy < [width, height])).all(1)
    with Image.open(he_path) as source:
        if source.size != (width, height):
            raise ValueError("H&E image dimensions differ from the measurement grid")
        thumbnail = source.convert("RGB")
        thumbnail.thumbnail((640, 640), Image.Resampling.LANCZOS)
        thumbnail = thumbnail.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    thumb_width, thumb_height = thumbnail.size
    sx, sy = thumb_width / width, thumb_height / height
    panel_width, image_top = max(320, thumb_width + 20), 48
    labels = sorted(set(replicates))
    canvas = Image.new("RGB", (panel_width * (len(labels) + 1), thumb_height + 90), "white")
    text = ImageDraw.Draw(canvas)
    panels = []
    size = grid.settings["bin_size_pixels"]
    for column, label in enumerate([*labels, None]):
        image_left = column * panel_width + (panel_width - thumb_width) // 2
        image = thumbnail.copy().convert("RGBA")
        overlay = Image.new("RGBA", image.size)
        drawing = ImageDraw.Draw(overlay)
        if label is None:
            for index in common:
                x, y = grid.bin_keys[index] * size
                right, top = min(x + size, width), min(y + size, height)
                drawing.rectangle(
                    (x * sx, (height - top) * sy, right * sx, (height - y) * sy),
                    fill=(20, 165, 100, 65),
                    outline=(0, 130, 80, 220),
                )
            heading = f"Shared {size}-pixel support: {len(common)} regions"
            detail = "Green: caller-declared common fine regions"
        else:
            selected = replicates == label
            visible = np.flatnonzero(selected & inside)
            sampled = visible[np.linspace(0, len(visible) - 1, min(len(visible), 8000), dtype=int)]
            for index in sampled:
                x, y = xy[index]
                px, py = x * sx, (height - y) * sy
                color = (15, 110, 180, 130) if mapped[index] >= 0 else (220, 0, 125, 150)
                drawing.ellipse((px - 0.8, py - 0.8, px + 0.8, py + 0.8), fill=color)
            heading = f"Slide-seq replicate {label}: {int(selected.sum())} beads"
            detail = "Blue: selected tissue bin; magenta: other in-image"
            panels.append(
                {
                    "replicate": label,
                    "source_beads": int(selected.sum()),
                    "inside_image": int((selected & inside).sum()),
                    "outside_image": int((selected & ~inside).sum()),
                    "in_selected_tissue_regions": int((selected & (mapped >= 0)).sum()),
                    "in_common_fine_regions": int((selected & np.isin(mapped, common)).sum()),
                    "displayed_beads": len(sampled),
                    "image_rectangle_pixels": [image_left, image_top, thumb_width, thumb_height],
                }
            )
        canvas.paste(Image.alpha_composite(image, overlay).convert("RGB"), (image_left, image_top))
        text.text((column * panel_width + 10, 9), heading, fill="black")
        text.text((column * panel_width + 10, 27), detail, fill="black")
    text.text(
        (10, thumb_height + 60),
        "Lower origin; registered pixels. "
        "Section linkage, physical scale and registration unqualified.",
        fill="black",
    )
    with Path(output).open("xb") as handle:
        canvas.save(handle, format="PNG")
    return {
        "schema": "heir.measurement_overlay.v1",
        "coordinate_units": "registered_pixel",
        "display_origin": "lower",
        "display_flip": "image_and_geometry_together",
        "source_coordinates_modified": False,
        "image_shape": [height, width],
        "region_size_pixels": size,
        "common_fine_region_ids": [grid.region_ids[i] for i in common],
        "common_support_definition": "caller_declared_geometry_only",
        "point_display_policy": "at_most_8000_evenly_spaced_in_image_source_rows_per_replicate",
        "panels": panels,
        "registration_qualified": False,
        "physical_scale_verified": False,
        "section_linkage": "unknown",
        "manual_review_status": "not_recorded",
    }


def _coordinates(path, source, grid):
    """Read observation coordinates only; do not materialize expression matrices again."""
    import h5py

    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        xy = np.column_stack((obs["x"][:], obs["y"][:]))
        node = obs["replicate"]
        if isinstance(node, h5py.Group) or "categories" in node.attrs:
            if isinstance(node, h5py.Group):
                category_node, codes = node["categories"], node["codes"][:]
            else:
                # Published HTAPP objects use the legacy dataset + HDF5 category reference.
                category_node, codes = handle[node.attrs["categories"]], node[:]
            categories = (
                category_node.asstr()[:]
                if category_node.dtype.kind in "OS"
                else category_node[:].astype(str)
            )
            if np.any((codes < 0) | (codes >= len(categories))):
                raise ValueError("assay replicate categories contain missing or invalid labels")
            replicates = categories[codes]
        else:
            replicates = node.asstr()[:] if node.dtype.kind in "OS" else node[:].astype(str)
    if xy.shape != (len(replicates), 2) or set(replicates) != set(
        source["replicate_id"].split(";")
    ):
        raise ValueError(
            "inspection coordinates or replicate labels differ from the source manifest"
        )
    mapped = point_to_grid(grid, xy)
    height, width = grid.settings["image_shape"]
    inside = ((xy >= 0) & (xy < [width, height])).all(1)
    return xy, replicates, mapped, inside


def _thumbnail(path, shape):
    from PIL import Image

    with Image.open(path) as source:
        if source.size != tuple(shape[::-1]):
            raise ValueError("H&E image dimensions differ from the prepared region grid")
        image = source.convert("RGB")
        image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
    # Do not flip these pixels: the SVG flips image and coordinates together.
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _record(grid, index, assay, replicate, counts):
    x, y = grid.bin_keys[index]
    size = grid.settings["bin_size_pixels"]
    height, width = grid.settings["image_shape"]
    return {
        "assay": assay,
        "replicate": replicate,
        "region_id": grid.region_ids[index],
        "observation_id": "",
        "x_min": int(x * size),
        "y_min": int(y * size),
        "x_max": int(min((x + 1) * size, width)),
        "y_max": int(min((y + 1) * size, height)),
        "tissue_fraction": float(grid.tissue_fraction[index]),
        "n_source_observations": int(counts[index]),
        "status": "uncovered",
        "reason": "no_assay_observations_in_selected_region",
        "normalization_complete": None,
        "target_exposure": None,
        "normalization_counts": {},
        "available_markers": [],
        "missing_markers": [],
    }


def _records(inputs, grid, targets, target_operator, target_metadata, normalization_genes, clouds):
    records, panels = [], []
    protein = inputs.proteins
    protein_rows = {name: i for i, name in enumerate(protein.operator.observation_ids)}
    target_rows = {name: i for i, name in enumerate(targets.observation_ids)}
    norm = [targets.gene_ids.index(gene) for gene in normalization_genes]
    for assay, (xy, replicates, mapped, inside) in clouds.items():
        for replicate in sorted(set(replicates)):
            selected = replicates == replicate
            counts = np.bincount(mapped[selected & (mapped >= 0)], minlength=len(grid.region_ids))
            panel_rows = []
            for index, name in enumerate(grid.region_ids):
                row = _record(grid, index, assay, replicate, counts)
                observation = f"{assay}:{replicate}:{name}"
                if assay == "codex" and observation in protein_rows:
                    position = protein_rows[observation]
                    mask = protein.observed[position]
                    row.update(
                        observation_id=observation,
                        available_markers=[
                            m for m, valid in zip(protein.marker_ids, mask) if valid
                        ],
                        missing_markers=[
                            m for m, valid in zip(protein.marker_ids, mask) if not valid
                        ],
                        status="protein_complete" if mask.all() else "protein_partial",
                        reason="all_markers_observed"
                        if mask.all()
                        else "incomplete_marker_coverage",
                    )
                elif assay == "slide_seq" and observation in target_rows:
                    position = target_rows[observation]
                    if (
                        target_operator.columns[position] != index
                        or counts[index] != target_metadata["region_bead_counts"][position]
                    ):
                        raise ValueError(
                            "inspection target mapping differs from prepared aggregation"
                        )
                    mask = targets.observed[position, norm]
                    values = targets.expression[position, norm]
                    complete = bool(mask.all())
                    exposure = float(values.sum()) if complete else None
                    retained = complete and exposure > 0
                    row.update(
                        observation_id=observation,
                        normalization_complete=complete,
                        target_exposure=exposure,
                        normalization_counts={
                            gene: float(value) if valid else None
                            for gene, value, valid in zip(normalization_genes, values, mask)
                        },
                        status="retained" if retained else "excluded",
                        reason="positive_complete_fixed_panel"
                        if retained
                        else (
                            "zero_normalization_exposure"
                            if complete
                            else "incomplete_normalization_panel"
                        ),
                    )
                panel_rows.append(row)
            if any(
                row["n_source_observations"] > 0 and not row["observation_id"] for row in panel_rows
            ):
                raise ValueError("inspection source coverage differs from prepared evidence")
            records.extend(panel_rows)
            panels.append(
                {
                    "key": f"{assay}:{replicate}",
                    "assay": assay,
                    "replicate": str(replicate),
                    "source_observations": int(selected.sum()),
                    "outside_image": int((selected & ~inside).sum()),
                    "outside_selected_tissue": int((selected & inside & (mapped < 0)).sum()),
                    "status_counts": {
                        status: sum(r["status"] == status for r in panel_rows) for status in _COLORS
                    },
                }
            )
    return records, panels


def _svg(grid, thumbnail, records, panels, clouds):
    height, width = grid.settings["image_shape"]
    size = grid.settings["bin_size_pixels"]
    parts = [
        f'<svg id="map" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        'role="img" aria-label="H&E regional inspection, source lower origin">',
        f'<g id="lower-origin" transform="translate(0 {height}) scale(1 -1)">',
        f'<image href="{thumbnail}" width="{width}" height="{height}"/>',
    ]
    parts.append('<g id="selected-grid" fill="none" stroke="#00a6bd" stroke-width="2">')
    empty_counts = np.zeros(len(grid.region_ids), dtype=int)
    for index in range(len(grid.region_ids)):
        row = _record(grid, index, "he", "1", empty_counts)
        parts.append(
            f'<rect x="{row["x_min"]}" y="{row["y_min"]}" '
            f'width="{row["x_max"] - row["x_min"]}" height="{row["y_max"] - row["y_min"]}">'
            f"<title>{html.escape(row['region_id'])}; tissue fraction "
            f"{row['tissue_fraction']:.4f}</title></rect>"
        )
    parts.append("</g>")
    for panel in panels:
        key = panel["key"]
        rows = [r for r in records if f"{r['assay']}:{r['replicate']}" == key]
        max_exposure = max([r["target_exposure"] or 0 for r in rows] + [1])
        parts.append(f'<g data-layer="{html.escape(key, quote=True)}" style="display:none">')
        for row in rows:
            opacity = 0.2 + 0.45 * np.log1p(row["target_exposure"] or 0) / np.log1p(max_exposure)
            if row["assay"] == "codex":
                opacity = 0.2 + 0.4 * len(row["available_markers"]) / max(
                    1, len(row["available_markers"]) + len(row["missing_markers"])
                )
            title = "; ".join(f"{k}: {v}" for k, v in row.items())
            parts.append(
                f'<rect x="{row["x_min"]}" y="{row["y_min"]}" '
                f'width="{row["x_max"] - row["x_min"]}" '
                f'height="{row["y_max"] - row["y_min"]}" '
                f'fill="{_COLORS[row["status"]]}" fill-opacity="{opacity:.3f}" '
                f'stroke="{_COLORS[row["status"]]}" stroke-width="2">'
                f"<title>{html.escape(title)}</title></rect>"
            )
        xy, replicates, mapped, inside = clouds[panel["assay"]]
        visible = np.flatnonzero((replicates == panel["replicate"]) & inside)
        selected = visible[np.linspace(0, len(visible) - 1, min(len(visible), 8000), dtype=int)]
        panel["displayed_points"] = len(selected)
        panel["point_display_policy"] = "at_most_8000_evenly_spaced_source_rows_inside_image"
        parts.append('<g class="source-points" fill-opacity="0.55">')
        for index in selected:
            x, y = xy[index]
            color = "#171717" if mapped[index] >= 0 else "#dc00a5"
            parts.append(
                f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{max(0.5, size * 0.018):g}" fill="{color}"/>'
            )
        parts.append("</g></g>")
    parts.append("</g></svg>")
    return "".join(parts)


def _page(summary, records, svg):
    payload = json.dumps({"summary": summary, "regions": records}, allow_nan=False).replace(
        "<", "\\u003c"
    )
    options = '<option value="he">H&E-selected grid</option>' + "".join(
        f'<option value="{html.escape(p["key"], quote=True)}">'
        f"{html.escape(p['assay'])} replicate {html.escape(p['replicate'])}</option>"
        for p in summary["panels"]
    )
    return f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HTAPP {html.escape(summary["biopsy_id"])} regional inspection</title>
<style>
body{{font:15px system-ui,sans-serif;max-width:1400px;margin:24px auto;
padding:0 16px;color:#182330}}
h1{{font-size:25px}}button,select,input{{font:inherit;padding:6px;margin:3px}}
.viewer{{height:76vh;overflow:auto;background:#e9e9e9;border:1px solid #aaa}}
svg{{height:74vh;display:block;margin:auto;max-width:none}}
.controls{{position:sticky;top:0;background:white;padding:6px;z-index:1}}
.legend span{{display:inline-block;margin:4px 12px 4px 0}}
.swatch{{padding-left:8px;border-left:10px solid}}
table{{border-collapse:collapse;width:100%;font-size:12px}}
th,td{{border:1px solid #ddd;padding:5px;text-align:left}}
thead{{background:#eef2f5}}.table{{max-height:500px;overflow:auto}}
pre{{white-space:pre-wrap;font-size:12px}}
</style><h1>HTAPP {html.escape(summary["biopsy_id"])} · regional inspection</h1>
<p>{summary["n_selected_regions"]} H&E-selected regions; {summary["region_size_pixels"]}-pixel bins.
Coordinates are <b>registered image pixels</b>. Image array row 0 is displayed at the bottom,
matching the authors’ lower-origin convention. Geometry and feature extraction are not flipped.</p>
<p>Review tissue boundaries, assay coverage, missing markers, and fixed-panel exposure.
Published coarse registration is unvalidated; this report does not establish physical calibration,
same-cell correspondence, registration accuracy, or scientific qualification.</p>
<div class="controls"><select id="layer">{options}</select>
<label><input id="grid" type="checkbox" checked>Grid</label>
<label><input id="points" type="checkbox" checked>Source points</label>
<button id="in">Zoom +</button><button id="out">Zoom −</button>
<button id="reset">Reset</button></div>
<div class="legend"><span class="swatch" style="border-color:#00a6bd">H&E grid</span>
<span class="swatch" style="border-color:#7952ba">All protein markers observed</span>
<span class="swatch" style="border-color:#da9414">Incomplete markers</span>
<span class="swatch" style="border-color:#16834a">Retained target</span>
<span class="swatch" style="border-color:#d52f35">Excluded target</span>
<span class="swatch" style="border-color:#717b87">No assay observations</span></div>
<p>Green shading follows log fixed-panel exposure; purple shading follows marker availability.
Black points lie in selected tissue; magenta points lie outside it. Each assay replicate displays
at most 8,000 source points; all region counts and exclusions below are exact.
Hover a region for details.</p>
<div class="viewer">{svg}</div><p id="panel"></p>
<h2>Exact regional decisions</h2>
<p>Includes uncovered selected regions separately for every assay replicate.
An unobserved region is not a measured zero.
RNA counts are summed within each region and replicate.</p>
<input id="search" placeholder="Filter by region ID, status, or reason" size="45">
<button id="all">Show all matching rows</button><button id="download">Download exact TSV</button>
<p id="row-count"></p><div class="table"><table>
<thead id="head"></thead><tbody id="rows"></tbody></table></div>
<h2>Evidence and scope</h2><pre id="evidence"></pre>
<script type="application/json" id="data">{payload}</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
const get=id=>document.getElementById(id);let limit=200,zoom=1;
const columns=['assay','replicate','region_id','status','reason','n_source_observations',
 'target_exposure','normalization_counts','available_markers','missing_markers'];
const show=v=>v===null?'not observed':typeof v==='object'?JSON.stringify(v):String(v);
const head=document.createElement('tr');
columns.forEach(c=>{{let h=document.createElement('th');h.textContent=c;head.append(h)}});
get('head').append(head);
function update(){{
 let key=get('layer').value;
 document.querySelectorAll('[data-layer]').forEach(
  e=>e.style.display=e.dataset.layer===key?'':'none');
 get('selected-grid').style.display=get('grid').checked?'':'none';
 document.querySelectorAll('.source-points').forEach(
  e=>e.style.display=get('points').checked?'':'none');
 let query=get('search').value.toLowerCase();
 let rows=data.regions.filter(r=>(key==='he'||r.assay+':'+r.replicate===key)
  &&JSON.stringify(r).toLowerCase().includes(query));
 get('rows').replaceChildren();
 rows.slice(0,limit).forEach(r=>{{let tr=document.createElement('tr');
  columns.forEach(c=>{{let td=document.createElement('td');
   td.textContent=show(r[c]);tr.append(td)}});get('rows').append(tr)}});
 get('row-count').textContent='Showing '+Math.min(limit,rows.length)
  +' of '+rows.length+' matching rows.';
 let p=data.summary.panels.find(p=>p.key===key);
 get('panel').textContent=p?JSON.stringify(p):
  'H&E selection used image pixels only, before spatial RNA was read.';
}}
['layer','grid','points'].forEach(id=>get(id).onchange=update);get('search').oninput=update;
if([...get('layer').options].some(o=>o.value===location.hash.slice(1)))
 get('layer').value=location.hash.slice(1);
get('layer').onchange=()=>{{location.hash=get('layer').value;update()}};
get('all').onclick=()=>{{limit=Infinity;update()}};
function resize(){{get('map').style.height=(74*zoom)+'vh'}}
get('in').onclick=()=>{{zoom*=1.5;resize()}};get('out').onclick=()=>{{zoom=Math.max(.5,zoom/1.5);resize()}};
get('reset').onclick=()=>{{zoom=1;resize()}};
get('download').onclick=()=>{{
 let keys=Object.keys(data.regions[0]);
 const cell=v=>v===null?'':typeof v==='object'?JSON.stringify(v):String(v);
 const quote=v=>'"'+cell(v).replace(/"/g,'""')+'"';
 let text=[keys.join('\\t'),...data.regions.map(
  r=>keys.map(k=>quote(r[k])).join('\\t'))].join('\\n');
 let a=document.createElement('a');
 a.href=URL.createObjectURL(new Blob([text],{{type:'text/tab-separated-values'}}));
 a.download='regions.tsv';a.click();URL.revokeObjectURL(a.href);
}};
get('evidence').textContent=JSON.stringify(data.summary,null,2);update();
</script></html>"""


def write_pilot_inspection(
    root,
    biopsy_id,
    output_directory,
    *,
    preparation,
    gene_ids,
    normalization_gene_ids,
    prepared=None,
):
    """Write a self-contained HTML review plus exact TSV/JSON; never modify source data.

    ``prepared=(inputs, provenance)`` can reuse a previously prepared feature arm;
    its H&E grid is verified against the target loader before any overlay is built.
    Spatial RNA is explicitly read only after the input/grid preparation completes.
    """
    output, root = Path(output_directory).resolve(), Path(root).resolve()
    if output.is_relative_to(root):
        raise ValueError("inspection output must be outside the immutable dataset root")
    genes, normal = tuple(gene_ids), tuple(normalization_gene_ids)
    if len(normal) < 2 or len(set(normal)) != len(normal) or not set(normal) <= set(genes):
        raise ValueError(
            "inspection requires at least two fixed normalization genes in its gene panel"
        )
    if output.exists():
        raise FileExistsError(output)
    inputs, provenance = (
        prepared
        if prepared is not None
        else prepare_cohort_inputs(root, biopsy_id, gene_ids=genes, **preparation)
    )
    grid = _restore_grid(inputs, provenance)
    targets, operator, target_record = load_cohort_targets(
        root, biopsy_id, inputs, provenance, gene_ids=genes
    )
    sources = {**provenance["sources"], "targets": target_record["source"]}
    clouds = {
        "codex": _coordinates(
            _verified_source_path(root, sources["protein"]), sources["protein"], grid
        ),
        "slide_seq": _coordinates(_source_path(root, sources["targets"]), sources["targets"], grid),
    }
    records, panels = _records(inputs, grid, targets, operator, target_record, normal, clouds)
    thumbnail = _thumbnail(_verified_source_path(root, sources["he"]), grid.settings["image_shape"])
    svg = _svg(grid, thumbnail, records, panels, clouds)
    total_bins = int(
        np.prod(
            np.ceil(np.asarray(grid.settings["image_shape"]) / grid.settings["bin_size_pixels"])
        )
    )
    summary = {
        "schema": "heir.pilot_inspection.v1",
        "biopsy_id": biopsy_id,
        "patient_id": inputs.patient_id,
        "specimen_id": inputs.specimen_id,
        "prediction_unit": "tissue_region",
        "coordinate_units": "registered_pixel",
        "grid_selection": "H&E_pixels_only_before_target_loading",
        "n_selected_regions": len(grid.region_ids),
        "n_image_bins": total_bins,
        "n_bins_excluded_by_image_tissue_rule": total_bins - len(grid.region_ids),
        "region_size_pixels": grid.settings["bin_size_pixels"],
        "grid_settings": grid.settings,
        "evaluation_gene_ids": list(genes),
        "normalization_gene_ids": list(normal),
        "panels": panels,
        "retained_target_observation_ids": [
            r["observation_id"] for r in records if r["status"] == "retained"
        ],
        "excluded_target_observations": [
            {"observation_id": r["observation_id"], "reason": r["reason"]}
            for r in records
            if r["assay"] == "slide_seq" and r["status"] == "excluded"
        ],
        "uncovered_regions": {
            panel["key"]: [
                r["region_id"]
                for r in records
                if r["status"] == "uncovered" and f"{r['assay']}:{r['replicate']}" == panel["key"]
            ]
            for panel in panels
        },
        "protein_quality_policy": provenance["protein"]["quality_policy"],
        "source_files": {
            role: {key: source[key] for key in ("working_path", "sha256", "bytes")}
            for role, source in sources.items()
        },
        "source_integrity": "consumed_source_SHA256_checks_in_preparation_and_target_loading",
        "registration": provenance["registration"],
        "scientific_qualification": False,
        "manual_review_status": "not_recorded",
        "target_payload_read": True,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "index.html").write_text(_page(summary, records, svg), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (output / "regions.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(
            {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()}
            for row in records
        )
    return summary
