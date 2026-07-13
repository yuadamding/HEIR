# HEIR

HEIR is currently a compact falsification project, not a full inference pipeline. It retains only
the experiments needed to answer two questions:

1. Given the correct RNA-derived broad type, do frozen H&E features predict within-type molecular
   state beyond an independent matched-donor/type mean, a refitted donor/type/ROI shuffle, and a
   coordinate-only ridge?
2. If that succeeds, does a matched molecular reference outperform equal-sized, type-balanced
   wrong-donor and generic banks?

UOT, graph learning, unknown-mass tuning, neural residual heads, M-steps, and refinement are outside
the repository until both premises are supported.

## Current decision

No biological gate has run. Full HEIR development is not authorized.

The only completed historical cohort was snPATHO (4066, 4399, 4411), using frozen
`omiclip-loki-coca-vit-l-14` features with checkpoint SHA-256
`fc38e84f8b6f916cce87650cc096ebe6ad5cfa648a53a9a82e99fd231ca2f042`. That experiment is not the
new primary test: snPATHO has three context-confounded donors, opened Visium truth, and no
one-to-one nucleus RNA target. OmiCLIP/Loki is cross-modal and is excluded from the primary
morphology-only probe.

Local inventory confirms:

- All 195 HESCAPE lung Parquet shards are downloaded at the pinned release revision.
- The matching 20-section HEST lung payload is downloaded: H&E WSIs, aligned transcripts, native
  Xenium cell/nucleus boundaries, and CellViT boundaries.
- Corrected GSE250346 RNA annotations are downloaded. Their source-study identities show that the
  20 sections represent **15 biological donors**, not the 19 pseudo-patients in HESCAPE metadata.
- UNI2-h is approved and downloaded as a frozen replication encoder. H-Optimus-1 still returns
  manual-approval HTTP 403; H0-mini has not been requested.
- snPATHO and NatCommun/MOSAIC remain development/reference-sensitivity material only.

See [the readiness decision](reports/morphology_ridge_readiness.json) and
[cohort ledger](manifests/morphology_ridge_cohorts.tsv).

## Frozen study design

HESCAPE `human-lung-healthy-panel` is suitable only for an exploratory regional test. The paper's
supplement states that each target is a simulated 55-µm pseudo-Visium spot created by sum-pooling
Xenium transcripts. The release has no nucleus/cell identity, segmentation, or cell-level target.
It therefore cannot decide the nucleus-level hypothesis. Broad *niche* labels must be RNA-only and
their marker genes removed from evaluation. [HESCAPE dataset card](https://huggingface.co/datasets/Peng-AI/hescape-pyarrow)

The official HESCAPE split is not donor-safe. Corrected GSE250346 metadata identifies paired
less-/more-affected sections from the same lungs (`VUILD96`, `VUILD91`, `VUILD78`, and `TILD117`),
while HESCAPE assigns them different pseudo-patient IDs and sometimes different splits. HEIR uses a
pinned 10-development/5-locked split over the 15 true source-study donors and keeps every section
from a donor together. [GSE250346](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE250346)

Encoder for that regional sensitivity: frozen `bioptimus/H-optimus-1`, 224 px at 0.5 µm/px,
producing 1,536 features. The authenticated account currently lacks its required manual approval.
[H-Optimus-1 model card](https://huggingface.co/bioptimus/H-optimus-1)

Available strong replication: frozen `MahmoodLab/UNI2-h`, 224-pixel input and 1,536-dimensional
direct feature. Its canonical `pytorch_model.bin` is pinned outside Git. It cannot repair
pseudo-spot targets; it only changes the image encoder. [UNI2-h model card](https://huggingface.co/MahmoodLab/UNI2-h)

Required lightweight replication: frozen `bioptimus/H0-mini`, using the recommended 768-dimensional
CLS feature. It is gated and non-commercial. [H0-mini model card](https://huggingface.co/bioptimus/H0-mini)

Cell-level primary gate: the raw HEST representation of the same 20 sections supplies aligned H&E,
native Xenium nuclei, nucleus-assigned transcripts, and one-to-one RNA-derived broad lineages from
the corrected source-study Seurat object. This can test general morphology-to-state association,
but it is **not** a non-overlapping confirmation cohort and has no matched dissociated snRNA bank.
[HEST dataset card](https://huggingface.co/datasets/MahmoodLab/hest)

Consequently, HEST can evaluate the direct oracle-lineage ridge premise. HESCAPE can provide a
regional sensitivity. Neither can, alone or together, validate personalized matched-snRNA value or
external-cohort generalization.

## Minimal method

For each broad type, development donors alone define technical-covariate correction, a low-rank RNA
basis, feature normalization, rank, and ridge penalty. The held-out donor supplies only an
independent reference-pool mean. The primary score is equal over types within donor and then equal
over donors.

The gate requires:

- macro residual-coordinate R² ≥ 0.05;
- matched-minus-refitted-shuffle R² ≥ 0.03 with empirical p < 0.01;
- 100 training-set refits for each of three frozen permutation seeds;
- at least 70% positive supported donor/type strata;
- positive donor consistency and no donor contributing over half the improvement;
- improvement over a coordinate/ROI-only ridge;
- at least 5% molecular error reduction over the independent reference mean; and
- an adequate low-rank ceiling.

A component pass still reports `authorizes_full_heir=false`. Encoder replication, a genuinely
non-overlapping cell-resolved cohort, and the separate matched-reference-specificity gate remain
mandatory before full HEIR development.

## Run

Install:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

After gated data/checkpoints are available, freeze source rows with a reviewed plan:

```bash
python scripts/prepare_morphology_ridge_artifacts.py \
  --plan /external/frozen_plan.json \
  --source-observations /external/observations_and_frozen_features.npz \
  --development-output /external/ridge_development.npz \
  --locked-test-output /external/ridge_locked_test.npz
```

Run the primary probe:

```bash
python scripts/benchmark_morphology_state_gate.py \
  --development-data /external/ridge_development.npz \
  --locked-test-data /external/ridge_locked_test.npz \
  --experiment-role regional_hescape_hoptimus1 \
  --report-output /external/hoptimus1_small_report.json
```

Run the same artifact construction and benchmark for `replication_h0mini`, then use
`scripts/benchmark_reference_specificity.py` only after morphology and confirmation pass.

## Repository contents

```text
src/heir/data/                 strict registered-observation artifact
src/heir/evaluation/           oracle ridge and reference-specificity tests
scripts/                       preparation and two benchmark entry points
manifests/                     local cohort readiness ledger
reports/                       no-run readiness decision
tests/                         deterministic scientific-contract tests
```

Raw data, extracted features, checkpoints, and run outputs must remain outside Git.
