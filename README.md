# HEIR

HEIR is currently a compact falsification project, not a full inference pipeline. It retains only
the experiments needed to answer two questions:

1. Given a receipt-bound, independently derived fine type, do frozen H&E features predict
   within-type molecular state beyond an independent matched-donor/section/type mean, a refitted
   donor/type/ROI shuffle, and a coordinate-only ridge?
2. If that succeeds, does a matched molecular reference outperform equal-sized, type-balanced
   wrong-donor and generic banks?

UOT, graph learning, unknown-mass tuning, neural residual heads, M-steps, and refinement are outside
the repository until both premises are supported.

## Current decision

No biological gate capable of testing the current H-MEAS/H-CELL hypothesis has run. Full HEIR
development is not authorized, and the primary hypothesis is
**untested, not failed**.

The next scientific action is to prepare the protected development export and independent labels,
then run development-only H-MEAS. H-MEAS cannot select its target/type receipt from `final_CT` labels
whose target dependence is unresolved: both annotation inputs and the training-label ontology must
first have target-independent provenance. H-CELL additionally requires an outcome-free
donor/section/type support topology and an exact six-condition calibration bound to the completed
design.

The five previously designated HEST test donors (`THD0008`, `THD0011`, `TILD117`, `VUILD78`, and
`VUILD96`) are not a prospective lock. A historical run at commit `28c6fff` materialized their
molecular and image outcomes and reloaded the locked artifact for schema validation. No endpoint
report or evidence of metric-guided tuning was found, but materialization alone invalidates the
prospective designation and it cannot be restored. Those donors are restricted to retrospective
internal/exploratory use. The words `locked_test` and `reserved` in the non-executable historical
draft are legacy partition identifiers, not a claim that a prospective lock remains valid. A
genuinely unexposed registered cell-resolved cohort is required for a prospective H-CELL test. See
[the lock-exposure audit](reports/hest_lock_exposure_audit.json).

The only completed historical cohort was snPATHO (4066, 4399, 4411), using frozen
`omiclip-loki-coca-vit-l-14` features with checkpoint SHA-256
`fc38e84f8b6f916cce87650cc096ebe6ad5cfa648a53a9a82e99fd231ca2f042`. That experiment is not the
new primary test: snPATHO has three context-confounded donors, opened Visium truth, and no
one-to-one nucleus RNA target. OmiCLIP/Loki is cross-modal and is excluded from the primary
morphology-only probe.

Local inventory confirms:

- Only 4 of the expected 195 HESCAPE lung Parquet shards are complete locally; a fifth is truncated
  and 190 are absent (95,102,132,905 bytes remain). No full-cohort H-REGIONAL run is ready.
- The matching 20-section HEST lung payload is downloaded: H&E WSIs, aligned transcripts, native
  Xenium cell/nucleus boundaries, and CellViT boundaries.
- Corrected GSE250346 RNA annotations are downloaded. Their source-study identities show that the
  20 sections represent **15 biological donors**, not the 19 pseudo-patients in HESCAPE metadata.
- UNI2-h is approved, downloaded, and frozen as the **designated primary encoder**.
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
10-development/5-test donor grouping over the 15 true source-study donors and keeps every section
from a donor together. That grouping remains useful for development and retrospective analyses,
but the five test donors are no longer prospectively locked. [GSE250346](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE250346)

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
the corrected source-study Seurat object. Once their independent provenance is established, these
labels can support a retrospective broad-lineage upper-bound analysis; they cannot by themselves test
the fine-type H-CELL endpoint. HEST is **not** a non-overlapping confirmation cohort and has no matched
dissociated snRNA bank.
[HEST dataset card](https://huggingface.co/datasets/MahmoodLab/hest)

Consequently, HEST can support development and retrospective evaluation of the direct oracle-lineage
ridge premise. Its five historical test donors cannot issue a prospective go/no-go decision. Any
retrospective signal would be hypothesis-generating only; absence of a benchmark report means the
hypothesis has not yet been tested. HESCAPE can provide only development-stage regional sensitivity.
Neither can, alone or together, validate prospective H-CELL, personalized matched-snRNA value, or
external-cohort generalization.

## Minimal method

For each supported fine type, development donors alone define technical-covariate correction, a low-rank RNA
basis, feature normalization, rank, and ridge penalty. Each held-out donor/section/type evaluation
stratum uses only its spatially disjoint same-section reference-pool mean. The frozen joint primary
endpoint names both scores: donor/type macro residual-coordinate R² and donor/section/type macro
residual-coordinate R². The first is equal over types within donor and then equal over donors; the
companion is equal over types within section, sections within donor, and donors. Each has a frozen
0.05 minimum and the decision rule requires both, so a large section cannot dominate a paired-section
donor.

Section and batch indicators are development-fold controls. Because locked sections and batches
are unseen categories, their one-hot columns do not fully adjust arbitrary held-out section or batch
effects; interpretation instead relies on measured stain/quality covariates, spatial controls,
section-balanced metrics, balance audits, and alternate reference splits.

The gate requires:

- macro residual-coordinate R² ≥ 0.05;
- section-balanced donor/section/type macro residual-coordinate R² ≥ 0.05;
- matched-minus-refitted-shuffle R² ≥ 0.03 with empirical p ≤ 0.01;
- exactly 333 unique training-set refits from each of three frozen permutation streams;
- at least 80% positive supported donor/type strata;
- positive donor consistency and no donor contributing over half the improvement;
- improvement over a coordinate/ROI-only ridge;
- at least 5% molecular error reduction over the independent reference mean; and
- an adequate low-rank ceiling.

Locked measurement reliability must be reported both by donor/type and by donor/section/type,
including the worst section and the fraction of every planned section/type stratum that passes the
frozen reliability threshold. H-MEAS and the H-CELL locked audit share one exact fail-closed contract:
8/12/8-µm absolute registration limits, 0.5/0.5 relative-geometry limits, and best/intermediate
registration-quality cutoffs of 0.25/0.6, together with the same segmentation, crop, and reliability
fields. G3 effects are additionally stratified as best, intermediate, and
near-threshold registration quality. Nucleus/cell contrasts and the full-versus-target-removed
intrinsic increment required for a mixed conclusion must be noninferior in the best-registration
subset within the frozen 0.01 delta-R2 margin; a near-threshold-only intrinsic effect is not credible.

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

H-MEAS may proceed only after the protocol names a hashed
`measurement_development_annotation_export` containing only development sections; the combined
GSE250346 export is not accepted because it would mix development rows with the historically
designated test rows. H-MEAS and confirmatory source construction both require
`--annotation-receipt`, `--annotation-predictions`,
`--training-label-provenance-receipt`, `--training-label-ontology-source`, and
`--annotation-validation-export`. Their bytes, row order, feature set, donor-training scope, and
cross-fitting folds must match the non-pending label-target contract. The ontology source and its
provenance receipt must establish that the training labels themselves were constructed without the
target genes; gene-disjoint classifier inputs alone are insufficient. The row-level prediction and
held-out validation exports must contain calibrated fine-type probability vectors and the frozen
abstention decision. Macro F1, multiclass ECE, per-class sensitivity, coverage, and per-validation-
donor metrics are recomputed from those rows rather than trusted from receipt summaries.
For same-cohort annotation, the cross-fitting receipt separately binds every development-donor LODO
validation fold and the model used for the prediction export: development predictions must come from
their held-out fold models, while disjoint test-donor predictions must come from one final model
trained on all and only the development donors.

Repository-generated receipts are necessary integrity bindings, but they are not independent
scientific provenance by themselves. Before H-MEAS can pass, the training labels and ontology must
trace to an immutable upstream assignment or an identified independent steward/curator attestation,
and the exact fitted annotation-model artifact and training-data identity must be archived and
hash-bound. Those external provenance materials do not currently exist.

H-MEAS first attempts one common reliable panel across all supported fine types, with at least six
genes to support the confirmatory rank-six candidate. A fallback to prespecified programs or type-specific panels is
permitted only as a new study version after a development-only H-MEAS failure, before any pristine
confirmatory outcomes are opened. Retrospective HEST outcomes cannot select that fallback.

Calibration finalization additionally requires a separate
`heir.confirmatory_stratum_topology.v1` JSON artifact created before H-CELL opening. It must list the
ordered donor/section/fine-type strata, the frozen minimum evaluation-cell count for each stratum,
the H-CELL analysis-plan hash, and `locked_outcomes_used=false`. A topology inferred after loading
locked molecular outcomes is not admissible.

Do not launch a long calibration from a notebook or service process. The calibration CLI marks a
dedicated child process, pins it to one logical CPU by default, checkpoints every trial, stops at a
16-GiB RSS ceiling, and applies a separate 64-GiB address-space ceiling so CUDA virtual mappings are
not confused with resident memory. No authorizing calibration receipt currently exists. Production
calibration must execute the exact gate under global-null, G2-boundary, nucleus-only, cell-only,
context-only, and mixed conditions and must preserve hash-addressed per-trial reports. The checked-in
implementation now routes synthetic row-level geometry, transcript halves, and reference-pool rows
through the same production measurement-audit and balance-report functions. Each actual gate report
binds both complete input artifacts and the trial/run identity, and calibration controls the union of
false hypothesis decisions per trial. This structural implementation and attestation are present,
but they are not authorizing until bound to the completed H-MEAS design and topology and executed
at the required scale. The literal
minimum of 1,000 trials for each of six conditions across ten stress families is at least 60,000
complete gate executions. With the full permutation and model-selection workload, that design is
computationally infeasible as currently specified and has not been executed. A sequential alternative
is admissible only if preregistered with simultaneous confidence bounds and non-opportunistic stopping
before any pristine confirmatory outcome is opened.

When such a run is eventually authorized, start the dedicated CLI under `tmux`, `screen`, or a
service manager so an SSH disconnect does not terminate the resumable worker. Aggregate pass counts
alone are not an authorizing calibration receipt; production evidence must preserve and
cryptographically bind the individual actual-gate reports from which those counts are recomputed.

Only after H-MEAS, independent-label, pristine-cohort topology, calibration, and manifest-opening
receipts exist, prepare confirmatory source rows from a genuinely unexposed cohort with the exact
receipt-bound inputs:

```bash
python scripts/prepare_morphology_ridge_artifacts.py \
  --study-manifest /external/pristine_cell_study.opened.json \
  --measurement-report /external/h_meas_report.json \
  --plan /external/frozen_plan.json \
  --source-observations /external/observations_and_frozen_features.npz \
  --development-output /external/ridge_development.npz \
  --locked-test-output /external/ridge_locked_test.npz
```

Run the primary probe:

```bash
python scripts/benchmark_morphology_state_gate.py \
  --study-manifest /external/pristine_cell_study.opened.json \
  --measurement-report /external/h_meas_report.json \
  --development-data /external/ridge_development.npz \
  --locked-test-data /external/ridge_locked_test.npz \
  --calibration-receipt /external/exact_gate_calibration_receipt.json \
  --report-output /external/pristine_uni2h_cell_gate.json
```

The five exposed HEST donors may be run only under a report explicitly labeled retrospective and
non-authorizing. If checkpoints become available, apply the frozen estimator with H-Optimus-1 and
H0-mini as sensitivities; same-cohort post-exposure runs are not independent replication. Use
`scripts/benchmark_reference_specificity.py` only after prospective morphology and independent
external confirmation pass.

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
