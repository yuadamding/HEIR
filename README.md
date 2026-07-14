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

No prospective biological gate capable of testing the current H-MEAS/H-CELL hypothesis has run.
Full HEIR development is not authorized, and the strict prospective primary hypothesis remains
**untested, not failed**.

Two distinct evidence tracks are now explicit. The strict prospective track next requires a
protected development export and independent labels, then development-only H-MEAS. H-MEAS cannot
select its target/type receipt from `final_CT` labels
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

The pragmatic track uses all 15 HEST donors in leave-one-biological-donor-out evaluation through
`scripts/benchmark_hest_retrospective.py`. It compares full 112-um context, cell-only, nucleus-only,
and target-cell-removed UNI2-h features against separate spatial, technical/stain, morphometry,
density, and combined controls. It refits both within-section/type and different-spatial-block nulls,
reports donor/type- and donor/section/type-balanced effects, and gives paired intrinsic crop
contrasts under strict-registration sensitivity. This can answer whether the available registered
images contain retrospective within-type molecular information; every report is irreversibly
labeled exposed and cannot authorize H-CELL, H-INTRINSIC, external validity, matched-reference
value, or full HEIR development.

The completed 100-permutation retrospective run did not support H-CELL or either intrinsic-crop
hypothesis in this fixed analysis. Full-context fine-type R² was -0.0605 by donor/type and -0.0699
by donor/section/type; its increment was -0.0159 versus the best non-image control and -0.00737
versus the combined control, with 0/15 donors positive. Cell-only and nucleus-only crops improved
only slightly over the target-removed crop (+0.00162 and +0.00169 R²), while their absolute R² and
increments over controls remained negative. These exposed retrospective results constrain the
current representation and ridge probe; they neither fail nor authorize the prospective hypothesis.
See the [explicit experiment report](reports/hest_retrospective.md) for the frozen design, complete
decision metrics, source QC, per-donor effects, null results, and artifact hashes.

The bounded H-optimus-1 qualification is implemented in
`scripts/benchmark_hest_scientific_reanalysis.py`. The H-optimus source matches the registered UNI2
source on all 217 non-encoder fields and uses full 1,536D features as the primary representation.
Its frozen visible-control gate failed: broad lineage, fine type, gray intensity, hematoxylin optical
density, and GLCM contrast passed, while natural-context nucleus area, perimeter, circularity, and
solidity all failed. Execution therefore stopped before H-optimus molecular fitting. The existing
UNI2 result was rerun only as an explicitly exposed, descriptive, non-authorizing baseline; UNI2
failed the same four geometry controls. A strong secondary nucleus-mask morphology score cannot
rescue the natural unmasked-image gate. The inherited registered-source QC also remains failed for
prospective registration/per-row crop criteria (the transcript-target audit passes); this limitation
is shared byte-for-byte across encoders and independently keeps the run non-authorizing. See the
[H-optimus qualification report](reports/hest_scientific_reanalysis.md).

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
- H-optimus-1 access is approved. Revision `3592cb220dec7a150c5d7813fb56e68bd57473b9`
  is downloaded outside Git, checksum-pinned, and is the **required encoder for every experiment
  started after 2026-07-13**. Existing UNI2-h artifacts remain the prespecified, separately scored
  encoder comparator; they are never substituted for H-optimus-1 inputs in a primary arm. H0-mini
  has not been materialized.
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

Current encoder: frozen `bioptimus/H-optimus-1`, 224 px at 0.5 µm/px, producing a
1,536-dimensional direct feature. Its `model.safetensors` checkpoint and `config.json` are pinned
outside Git by `manifests/encoders/hoptimus1.json`; loading is local-only and fine-tuning is
prohibited. A receipt-bound five-regime official-versus-local parity qualification passed the frozen
FP32 and mixed-precision cosine/error thresholds and verified the single-resampling 112-µm to
224-pixel input path.
[H-optimus-1 model card](https://huggingface.co/bioptimus/H-optimus-1)

Comparator encoder: frozen `MahmoodLab/UNI2-h`. It remains named in retrospective HEST reports so
those results preserve their true experiment identity, and it is retained as a fixed secondary
encoder sensitivity rather than being pooled with or substituted for the H-optimus-1 primary.
[UNI2-h model card](https://huggingface.co/MahmoodLab/UNI2-h)

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
retrospective signal is hypothesis-generating only; the completed exposed benchmark did not support
the three registered retrospective hypotheses in this fixed analysis. HESCAPE can provide only
development-stage regional sensitivity. Neither can, alone or together, validate prospective
H-CELL, personalized matched-snRNA value, or external-cohort generalization.

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
  --report-output /external/pristine_hoptimus1_cell_gate.json
```

The five exposed HEST donors may be run only under a report explicitly labeled retrospective and
non-authorizing. If checkpoints become available, apply the frozen estimator with H-Optimus-1 and
H0-mini as sensitivities; same-cohort post-exposure runs are not independent replication. Use
`scripts/benchmark_reference_specificity.py` only after prospective morphology and independent
external confirmation pass.

Reproduce the historical UNI2-h comparator source on the explicitly exposed cohort with four crop
arms, deterministic per-stratum sampling, CUDA extraction, and hard memory bounds (artifacts and
checkpoints remain outside Git):

```bash
mkdir -p /mnt/seagate/HEIR_runs/hest_retrospective
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  .venv/bin/python \
  scripts/build_hest_xenium_cell_source.py \
  --protocol configs/hest_lung_cell_protocol.json \
  --study-manifest manifests/studies/hest_lung_cell_association.draft.json \
  --encoder-manifest manifests/encoders/uni2h.json \
  --crop-manifest configs/crops/hest_crop_ladder.json \
  --data-root /mnt/seagate/HnE/HEST \
  --model-dir /mnt/seagate/HnE/pretrained/UNI2-h \
  --source-output /mnt/seagate/HEIR_runs/hest_retrospective/source.npz \
  --plan-output /mnt/seagate/HEIR_runs/hest_retrospective/plan.json \
  --qc-output /mnt/seagate/HEIR_runs/hest_retrospective/qc.json \
  --device cuda --batch-size 8 --retrospective
```

Then run the retrospective test with bounded BLAS threads:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  .venv/bin/python scripts/benchmark_hest_retrospective.py \
  --source /mnt/seagate/HEIR_runs/hest_retrospective/source.npz \
  --output /mnt/seagate/HEIR_runs/hest_retrospective/report.json \
  --permutations 100
```

Runs below 100 permutations are labeled smoke-only because they cannot resolve an empirical
one-sided p-value below `1 / (B + 1)` for each null family.

Reproduce the frozen H-optimus qualification with bounded CPU threads and deterministic CUDA:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  .venv/bin/python scripts/benchmark_hest_scientific_reanalysis.py \
  --source /mnt/seagate/HEIR_runs/hest_hoptimus1_qualification/source.npz \
  --output /mnt/seagate/HEIR_runs/hest_hoptimus1_qualification/report.json \
  --markdown-output /mnt/seagate/HEIR_runs/hest_hoptimus1_qualification/report.md \
  --phase full --representation-profile full --device cuda --inner-folds 3 \
  --seed 20260713 --torch-threads 2 --max-gpu-memory-gb 8 \
  --expected-source-sha256 f7e7d4e97727cc17e71a81a252ab35fd2ca1c0e70054cba3ed38c2f7b7f65636 \
  --expected-encoder bioptimus/H-optimus-1 \
  --comparison-source /mnt/seagate/HEIR_runs/hest_retrospective/source.npz \
  --comparison-report /mnt/seagate/HEIR_runs/hest_uni2h_same_runner_qualification/report.json
```

## Repository contents

```text
src/heir/data/                 strict registered-observation artifact
src/heir/evaluation/           oracle ridge and reference-specificity tests
scripts/                       preparation and core scientific benchmark entry points
manifests/                     local cohort readiness ledger
reports/                       no-run readiness decision
tests/                         deterministic scientific-contract tests
```

Raw data, extracted features, checkpoints, and run outputs must remain outside Git.

H-optimus-1 use is restricted to the accepted noncommercial research terms. Do not redistribute or
embed the checkpoint in a repository/container; each user must obtain access separately. Treat heads
or datasets derived from its outputs as subject to the model's derivative-use terms, and obtain
written permission before commercial or monetized use. HEIR and H-optimus-1 are research artifacts,
not certified medical devices, and require independent validation before healthcare use.
