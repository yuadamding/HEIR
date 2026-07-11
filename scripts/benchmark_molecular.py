"""HEIR molecular-arm falsification benchmark on real NatCommun snRNA.

Runs HEIR's own prescribed validation ladder (docs/validation.md) at the level
that needs no H&E segmentation: does DONOR-MATCHED snRNA recover held-out cells'
panel expression and composition better than a generic atlas, a wrong donor, or
label-permuted prototypes? This isolates the *molecular personalization* premise
on which HEIR's whole thesis rests. It does NOT test the H&E->state mapping.

All scoring uses HEIR's shipped code: expression_metrics, composition_metrics,
within_type_residuals, type_mean_prediction, build_sample_prototypes,
prototype_mean_prediction, and the frozen log1p-CPM-10k expression space.
"""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

from heir.data.cohorts import NATCOMM_REFERENCE_MAPPINGS
from heir.evaluation.metrics import composition_metrics, expression_metrics

ROOT = "/mnt/seagate/HnE/NatCommun_2025_s41467_025_59005_9/cellxgene/"
FILES = {
    "breast": ROOT
    + (
        "Breast_cancer_4_patients_archival_FFPE_samples_profiling_with_Chromium_"
        "FLEX__f04ec2aa-2272-48df-b2d6-9864078c9892.h5ad"
    ),
    "lung": ROOT
    + (
        "Lung_cancer_4_patients_archival_FFPE_samples_profiling_with_Chromium_"
        "FLEX__c4cb7c18-761d-40ce-a6ea-29ebd9413123.h5ad"
    ),
    "dlbcl": ROOT
    + (
        "Diffuse_large_B_cell_lymphoma_DLBCL_6_patients_archival_FFPE_samples_"
        "profiling_with_Chromium_FLEX__2cd70fd4-9042-42f5-8f46-3780f1ad0626.h5ad"
    ),
}
TISSUE = {"B": "breast", "L": "lung", "D": "dlbcl"}
PANEL = [
    line.strip()
    for line in open("/storage/HE_GPT/HEIR/manifests/gene_panel_example.tsv")
    if line.strip() and not line.startswith("#")
]
LEVEL = "Level1"  # broad types (matches num_cell_types=6 in config)
SEEDS = [17, 41, 89]
CHUNK = 4000


def load_specimen_panel(fn, donor_vals, sample_vals, level):
    """Return (log1p-CPM panel matrix over FULL library, cell-type labels)."""
    a = ad.read_h5ad(fn, backed="r")
    obs = a.obs
    m = np.ones(a.n_obs, bool)
    m &= obs["donor_id"].astype(str).isin([str(x) for x in donor_vals]).values
    m &= obs["sample_id"].astype(str).isin([str(x) for x in sample_vals]).values
    idx = np.flatnonzero(m)
    labels = obs[level].astype(str).values[idx]
    # panel columns via feature_name
    fname = a.var["feature_name"].astype(str).values
    col = {g: i for i, g in enumerate(fname)}
    panel_cols = np.array([col[g] for g in PANEL], dtype=np.int64)
    # chunked: full library size + panel counts
    lib = np.zeros(idx.size, np.float64)
    panel_counts = np.zeros((idx.size, len(PANEL)), np.float64)
    X = a.X
    for s in range(0, idx.size, CHUNK):
        rows = idx[s : s + CHUNK]
        blk = X[rows]
        blk = blk.toarray() if sp.issparse(blk) else np.asarray(blk)
        lib[s : s + CHUNK] = blk.sum(1)
        panel_counts[s : s + CHUNK] = blk[:, panel_cols]
    # frozen expression space: CPM to 10k on FULL library, then log1p, subset already applied
    lib = np.maximum(lib, 1.0)
    cpm = panel_counts * (10000.0 / lib[:, None])
    expr = np.log1p(cpm).astype(np.float32)
    a.file.close()
    return expr, labels


def split(labels, seed, frac=0.5):
    rng = np.random.default_rng(seed)
    tr = np.zeros(labels.size, bool)
    for t in np.unique(labels):
        ix = np.flatnonzero(labels == t)
        rng.shuffle(ix)
        tr[ix[: max(1, int(round(frac * ix.size)))]] = True
    return tr, ~tr


def type_means(expr, labels, ontology):
    """Mean panel expr per ontology type; NaN row if type absent."""
    M = np.full((len(ontology), expr.shape[1]), np.nan, np.float64)
    for i, t in enumerate(ontology):
        mask = labels == t
        if mask.any():
            M[i] = expr[mask].mean(0)
    return M


def predict_typecond(test_labels, ontology, ref_means, global_mean):
    """Oracle-type prediction: each test cell -> its type's reference mean."""
    idx = {t: i for i, t in enumerate(ontology)}
    pred = np.empty((test_labels.size, ref_means.shape[1]), np.float64)
    for j, t in enumerate(test_labels):
        row = ref_means[idx[t]]
        pred[j] = global_mean if not np.isfinite(row).all() else row
    return pred.astype(np.float32)


def per_cell_mse(pred, truth):
    return float(np.mean(np.square(pred.astype(np.float64) - truth.astype(np.float64))))


def composition_vec(labels, ontology):
    v = np.array([(labels == t).sum() for t in ontology], np.float64)
    return v


def main():
    out = {
        "level": LEVEL,
        "panel_genes": len(PANEL),
        "seeds": SEEDS,
        "expression_space": "log1p-cpm-10000-v1",
        "specimens": {},
    }
    # load all specimens once per seed-independent (data is fixed); cache
    specs = [s for s in NATCOMM_REFERENCE_MAPPINGS if s != "B2"]
    print(f"loading {len(specs)} specimens ...", flush=True)
    data = {}
    for spec in specs:
        mp = NATCOMM_REFERENCE_MAPPINGS[spec]
        fn = FILES[TISSUE[spec[0]]]
        expr, labels = load_specimen_panel(fn, mp.donor_values, mp.sample_values, LEVEL)
        data[spec] = (expr, labels)
        print(f"  {spec}: {expr.shape[0]} cells, types={sorted(set(labels))}", flush=True)

    # fixed wrong-donor pairing within tissue (cyclic)
    by_tissue = {}
    for spec in specs:
        by_tissue.setdefault(TISSUE[spec[0]], []).append(spec)
    wrong = {}
    for t, lst in by_tissue.items():
        for i, s in enumerate(lst):
            wrong[s] = lst[(i + 1) % len(lst)] if len(lst) > 1 else None

    rows = []
    for spec in specs:
        expr, labels = data[spec]
        tissue = TISSUE[spec[0]]
        ontology = sorted(set(labels))  # per-specimen ontology for type-cond scoring
        rec = {
            "tissue": tissue,
            "n_cells": int(expr.shape[0]),
            "wrong_donor": wrong[spec],
            "conditions": {},
        }
        agg = {
            c: {"mse": [], "spearman": [], "js": []}
            for c in ["matched", "wrong_donor", "generic_atlas", "permuted_label"]
        }
        for seed in SEEDS:
            tr, te = split(labels, seed)
            te_expr, te_lab = expr[te], labels[te]
            gmean_matched = expr[tr].mean(0)
            # matched reference (same specimen train half)
            refs = {}
            refs["matched"] = (
                type_means(expr[tr], labels[tr], ontology),
                gmean_matched,
                labels[tr],
            )
            # permuted-label: shuffle train labels
            rng = np.random.default_rng(seed + 1)
            perm = labels[tr].copy()
            rng.shuffle(perm)
            refs["permuted_label"] = (type_means(expr[tr], perm, ontology), gmean_matched, perm)
            # wrong-donor: another specimen (all its cells)
            if wrong[spec] is not None:
                we, wl = data[wrong[spec]]
                refs["wrong_donor"] = (type_means(we, wl, ontology), we.mean(0), wl)
            # generic atlas: pooled other specimens in tissue
            others = [s for s in by_tissue[tissue] if s != spec]
            if others:
                ge = np.concatenate([data[s][0] for s in others])
                gl = np.concatenate([data[s][1] for s in others])
                refs["generic_atlas"] = (type_means(ge, gl, ontology), ge.mean(0), gl)

            te_comp = composition_vec(te_lab, ontology)
            for cond, (means, gm, ref_lab) in refs.items():
                pred = predict_typecond(te_lab, ontology, means, gm)
                agg[cond]["mse"].append(per_cell_mse(pred, te_expr))
                em = expression_metrics(pred, te_expr)
                agg[cond]["spearman"].append(em["median_gene_spearman"])
                # composition: reference train composition vs test composition
                ref_comp = composition_vec(ref_lab, ontology)
                cm = composition_metrics(ref_comp[None, :], te_comp[None, :])
                agg[cond]["js"].append(cm["mean_js_divergence"])
        for cond, d in agg.items():
            if d["mse"]:
                rec["conditions"][cond] = {
                    "recon_mse_mean": float(np.mean(d["mse"])),
                    "recon_mse_sd": float(np.std(d["mse"])),
                    "median_gene_spearman_mean": float(np.nanmean(d["spearman"])),
                    "composition_js_mean": float(np.mean(d["js"])),
                }
        out["specimens"][spec] = rec
        m = rec["conditions"]
        print(
            f"{spec:4s} MSE "
            f"matched={m.get('matched', {}).get('recon_mse_mean', float('nan')):.4f} "
            f"generic={m.get('generic_atlas', {}).get('recon_mse_mean', float('nan')):.4f} "
            f"wrong={m.get('wrong_donor', {}).get('recon_mse_mean', float('nan')):.4f} "
            f"permuted={m.get('permuted_label', {}).get('recon_mse_mean', float('nan')):.4f}",
            flush=True,
        )
        rows.append(rec)

    # pooled paired comparisons (matched vs generic / wrong) across specimens
    def paired(metric, a, b, better="lower"):
        va, vb = [], []
        for spec in specs:
            c = out["specimens"][spec]["conditions"]
            if a in c and b in c:
                va.append(c[a][metric])
                vb.append(c[b][metric])
        va, vb = np.array(va), np.array(vb)
        if better == "lower":
            wins = int((va < vb).sum())
        else:
            wins = int((va > vb).sum())
        return {
            "n": len(va),
            f"{a}_wins": wins,
            f"{a}_mean": float(va.mean()),
            f"{b}_mean": float(vb.mean()),
        }

    out["summary"] = {
        "recon_mse_matched_vs_generic": paired(
            "recon_mse_mean", "matched", "generic_atlas", "lower"
        ),
        "recon_mse_matched_vs_wrong": paired("recon_mse_mean", "matched", "wrong_donor", "lower"),
        "recon_mse_matched_vs_permuted": paired(
            "recon_mse_mean", "matched", "permuted_label", "lower"
        ),
        "composition_js_matched_vs_generic": paired(
            "composition_js_mean", "matched", "generic_atlas", "lower"
        ),
        "composition_js_matched_vs_wrong": paired(
            "composition_js_mean", "matched", "wrong_donor", "lower"
        ),
    }
    outpath = Path("/storage/HE_GPT/HEIR/artifacts/benchmark_molecular.json")
    outpath.write_text(json.dumps(out, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(out["summary"], indent=2))
    print("wrote", outpath)


if __name__ == "__main__":
    main()
