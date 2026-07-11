"""Bridge Space Ranger `nucleus_segmentations.geojson` -> HEIR inputs.

Emits (1) a canonical nucleus CSV (centroid + geometric morphology) and (2) a
feature NPZ (per-nucleus shape-descriptor vector) whose nucleus_ids match the
CSV source ids, ready for `heir prepare-histology`. Feature space is documented
as geometry-morphology; a pathology foundation-model embedding is the intended
upgrade for testing HEIR's novel morphology->state claim.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def polygon_geometry(coords):
    """Return dict of shape descriptors from a polygon ring (Nx2)."""
    p = np.asarray(coords, dtype=np.float64)
    if p.shape[0] < 3:
        return None
    x, y = p[:, 0], p[:, 1]
    # shoelace area + perimeter
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    area = 0.5 * abs(np.sum(x * y2 - x2 * y))
    perim = float(np.sum(np.hypot(x2 - x, y2 - y)))
    if area <= 0 or perim <= 0:
        return None
    cx, cy = x.mean(), y.mean()
    ux, uy = x - cx, y - cy
    # central second moments (vertex-based ellipse approximation)
    mxx = np.mean(ux * ux)
    myy = np.mean(uy * uy)
    mxy = np.mean(ux * uy)
    cov = np.array([[mxx, mxy], [mxy, myy]])
    ev = np.linalg.eigvalsh(cov)
    ev = np.clip(ev, 1e-9, None)
    major = 4.0 * np.sqrt(ev[1])
    minor = 4.0 * np.sqrt(ev[0])
    ecc = float(np.sqrt(max(0.0, 1.0 - (ev[0] / ev[1]))))
    orient = float(0.5 * np.arctan2(2 * mxy, (mxx - myy)))
    circ = float(min(1.0, 4.0 * np.pi * area / (perim * perim)))
    bbox = (x.max() - x.min()) * (y.max() - y.min())
    extent = float(area / bbox) if bbox > 0 else 0.0
    equiv_diam = float(np.sqrt(4.0 * area / np.pi))
    # convex hull -> solidity
    solidity = 1.0
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(p)
        solidity = float(min(1.0, area / hull.volume)) if hull.volume > 0 else 1.0
    except Exception:
        pass
    return dict(
        area=area,
        perimeter=perim,
        circularity=circ,
        eccentricity=ecc,
        major_axis_length=major,
        minor_axis_length=minor,
        orientation=orient,
        solidity=solidity,
        extent=extent,
        equiv_diameter=equiv_diam,
        cx=cx,
        cy=cy,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson", required=True)
    ap.add_argument("--out-nuclei", required=True)
    ap.add_argument("--out-features", required=True)
    ap.add_argument("--min-area-px", type=float, default=8.0)
    args = ap.parse_args()

    with open(args.geojson) as fh:
        gj = json.load(fh)
    feats = gj["features"]
    print(f"parsing {len(feats)} polygons ...", flush=True)

    MORPH = [
        "area",
        "perimeter",
        "circularity",
        "eccentricity",
        "major_axis_length",
        "minor_axis_length",
        "orientation",
        "solidity",
        "extent",
        "equiv_diameter",
    ]
    ids, rows, fvec = [], [], []
    skipped = 0
    for ft in feats:
        geom = ft.get("geometry", {})
        if geom.get("type") != "Polygon":
            skipped += 1
            continue
        ring = geom["coordinates"][0]
        g = polygon_geometry(ring)
        if g is None or g["area"] < args.min_area_px:
            skipped += 1
            continue
        prop = ft.get("properties", {})
        cid = prop.get("cell_id")
        cent = prop.get("nucleus_centroid")
        cx, cy = (float(cent[0]), float(cent[1])) if cent else (g["cx"], g["cy"])
        ids.append(str(cid))
        rows.append((cx, cy) + tuple(g[m] for m in MORPH))
        fvec.append([g[m] for m in MORPH])
    n = len(ids)
    print(f"kept {n} nuclei (skipped {skipped})", flush=True)
    arr = np.asarray(rows, dtype=np.float64)
    ids_arr = np.asarray(ids)
    assert len(set(ids)) == n, "cell_id collisions"

    # nucleus CSV
    header = ["nucleus_id", "x", "y"] + MORPH
    with open(args.out_nuclei, "w") as fh:
        fh.write(",".join(header) + "\n")
        for i in range(n):
            fh.write(ids[i] + "," + ",".join(f"{v:.6g}" for v in arr[i]) + "\n")

    # feature NPZ (log/standardized geometry vector); ids must match CSV source ids
    F = np.asarray(fvec, dtype=np.float64)
    # robust standardize
    med = np.median(F, axis=0)
    mad = np.median(np.abs(F - med), axis=0) + 1e-6
    Fz = ((F - med) / (1.4826 * mad)).astype(np.float32)
    np.savez(
        args.out_features,
        nucleus_ids=ids_arr.astype(str),
        features=Fz,
        feature_names=np.asarray(MORPH),
    )
    print(
        f"wrote {args.out_nuclei} and {args.out_features} (feat dim {Fz.shape[1]})",
        flush=True,
    )


if __name__ == "__main__":
    main()
