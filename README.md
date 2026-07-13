# HEIR

HEIR is currently a compact falsification project, not a full inference pipeline. It retains only
the experiments needed to answer two questions:

1. Given the correct RNA-derived broad type, do frozen H&E features predict within-type molecular
   state beyond an independent matched-donor/section/type mean, a refitted donor/type/ROI shuffle, and a
   coordinate-only ridge?
2. If that succeeds, does a matched molecular reference outperform equal-sized, type-balanced
   wrong-donor and generic banks?

UOT, graph learning, unknown-mass tuning, neural residual heads, M-steps, and refinement are outside
the repository until both premises are supported.

## Current decision

No biological gate has run. Full HEIR development is not authorized.

The next scientific action is development-only H-MEAS. H-CELL remains blocked until H-MEAS emits
its frozen target/type receipt, label–target independence is proven by an exact annotation receipt,
an outcome-free donor/section/type support topology is frozen, and an exact-gate calibration is
bound to that completed design. The five reserved HEST donors must remain unopened until those
conditions are met.

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
- UNI2-h is approved, downloaded, and frozen as the **preregistered primary encoder**.
  H-Optimus-1 is replication 1 but still returns manual-approval HTTP 403; H0-mini is replication 2
  and has not been materialized. These roles must not change after a locked outcome is opened.
- snPATHO and NatCommun/MOSAIC remain development/reference-sensitivity material only.

See [the readiness decision](reports/morphology_ridge_readiness.json) and
[cohort ledger](manifests/morphology_ridge_cohorts.tsv).

## Frozen study design

HESCAPE `human-lung-healthy-panel` is suitable only for an exploratory, development-donor regional
test. It is analyzed by development-donor cross-validation and cannot issue a validated-regional
authorization. The paper's
supplement states that each target is a simulated 55-µm pseudo-Visium spot created by sum-pooling
Xenium transcripts. The release has no nucleus/cell identity, segmentation, or cell-level target.
It therefore cannot decide the nucleus-level hypothesis. Broad *niche* labels must be RNA-only and
their marker genes removed from evaluation. [HESCAPE dataset card](https://huggingface.co/datasets/Peng-AI/hescape-pyarrow)

The official HESCAPE split is not donor-safe. Corrected GSE250346 metadata identifies paired
less-/more-affected sections from the same lungs (`VUILD96`, `VUILD91`, `VUILD78`, and `TILD117`),
while HESCAPE assigns them different pseudo-patient IDs and sometimes different splits. HEIR uses a
pinned 10-development/5-locked split over the 15 true source-study donors and keeps every section
from a donor together. [GSE250346](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE250346)

Primary encoder: frozen `MahmoodLab/UNI2-h`, 224-pixel input and 1,536-dimensional direct feature.
Its canonical `pytorch_model.bin` is pinned outside Git. It cannot repair pseudo-spot targets; it
only changes the image encoder. [UNI2-h model card](https://huggingface.co/MahmoodLab/UNI2-h)

Replication 1: frozen `bioptimus/H-optimus-1`, 224 px at 0.5 µm/px, producing 1,536 features.
The authenticated account currently lacks its required manual approval.
[H-Optimus-1 model card](https://huggingface.co/bioptimus/H-optimus-1)

Replication 2: frozen `bioptimus/H0-mini`, using the recommended 768-dimensional CLS feature. It is
gated and non-commercial. [H0-mini model card](https://huggingface.co/bioptimus/H0-mini)

Cell-level primary gate: the raw HEST representation of the same 20 sections supplies aligned H&E,
native Xenium nuclei, nucleus-assigned transcripts, and one-to-one RNA-derived broad lineages from
the corrected source-study Seurat object. This can test general morphology-to-state association,
but it is **not** a non-overlapping confirmation cohort and has no matched dissociated snRNA bank.
[HEST dataset card](https://huggingface.co/datasets/MahmoodLab/hest)

Consequently, HEST can evaluate the direct oracle-lineage ridge premise. Its five reserved donors
form a stringent internal go/no-go falsification gate: even a pass supports only the decision to
seek external confirmation, not population inference or external generalization. HESCAPE can
provide only development-stage regional sensitivity. Neither can, alone or together, validate
personalized matched-snRNA value or external-cohort generalization.

## Minimal method

For each broad type, development donors alone define technical-covariate correction, a low-rank RNA
basis, feature normalization, rank, and ridge penalty. Each held-out donor/section/type evaluation
stratum uses only its spatially disjoint same-section reference-pool mean. The primary score is equal
over types within donor and then equal over donors.

Section and batch indicators are development-fold controls. Because locked sections and batches
are unseen categories, their one-hot columns do not fully adjust arbitrary held-out section or batch
effects; interpretation instead relies on measured stain/quality covariates, spatial controls,
section-balanced metrics, balance audits, and alternate reference splits.

The gate requires:

- macro residual-coordinate R² ≥ 0.05;
- matched-minus-refitted-shuffle R² ≥ 0.03 with empirical p ≤ 0.01;
- exactly 333 unique training-set refits from each of three frozen permutation streams;
- at least 80% positive supported donor/type strata;
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

H-MEAS source construction is fail-closed until the locked protocol names a hashed
`measurement_development_annotation_export` containing only development sections; the combined
GSE250346 export is not accepted because a CSV reader would materialize locked rows. Confirmatory
source construction additionally requires `--annotation-receipt` and `--annotation-predictions`;
their bytes, row order, feature set, donor-training scope, and cross-fitting folds must match the
non-pending label-target contract.

Calibration finalization additionally requires a separate
`heir.confirmatory_stratum_topology.v1` JSON artifact created before H-CELL opening. It must list the
ordered donor/section/fine-type strata, the frozen minimum evaluation-cell count for each stratum,
the H-CELL analysis-plan hash, and `locked_outcomes_used=false`. A topology inferred after loading
locked molecular outcomes is not admissible.

Do not launch a long calibration from a notebook or service process. The calibration CLI marks a
dedicated child process, pins it to one logical CPU by default, checkpoints every trial, stops at a
16-GiB RSS ceiling, and applies a separate 64-GiB address-space ceiling so CUDA virtual mappings are
not confused with resident memory. The checked-in runner is still preliminary and non-authorizing;
no production calibration should run until the six-condition truth-matrix generator and per-trial
report attestation are complete.

When such a run is eventually authorized, start the dedicated CLI under `tmux`, `screen`, or a
service manager so an SSH disconnect does not terminate the resumable worker. Aggregate pass counts
alone are not an authorizing calibration receipt; the remaining implementation must preserve and
cryptographically bind the individual actual-gate reports used to recompute those counts.

Only after H-MEAS, independent-label, calibration, and manifest-opening receipts exist, prepare the
confirmatory source rows with the exact receipt-bound inputs:

```bash
python scripts/prepare_morphology_ridge_artifacts.py \
  --study-manifest /external/hest_cell.opened.json \
  --measurement-report /external/h_meas_report.json \
  --plan /external/frozen_plan.json \
  --source-observations /external/observations_and_frozen_features.npz \
  --development-output /external/ridge_development.npz \
  --locked-test-output /external/ridge_locked_test.npz
```

Run the primary probe:

```bash
python scripts/benchmark_morphology_state_gate.py \
  --study-manifest /external/hest_cell.opened.json \
  --measurement-report /external/h_meas_report.json \
  --development-data /external/ridge_development.npz \
  --locked-test-data /external/ridge_locked_test.npz \
  --calibration-receipt /external/exact_gate_calibration_receipt.json \
  --report-output /external/hest_uni2h_internal_gate.json
```

If checkpoints become available, run the same frozen HEST estimand for H-Optimus-1 replication 1
and H0-mini replication 2. Use `scripts/benchmark_reference_specificity.py` only after morphology
and independent external confirmation pass.

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
