# snPATHO-DeepBench-v1

`snPATHO-DeepBench-v1` applies the attached full benchmark plan to every
component supported by the frozen local artifacts. It is a **retrospective
diagnostic**, not a replacement for `snPATHO-Locked-v0.2`, an untouched
validation, or a compliant Track A result.

## Outcome

The requested primary endpoint is not yet testable: there are no redesigned
refined predictions, no FFPE-snPATHO-only R1 reference artifact, no
composition-adjustment inputs, and no frozen per-spot H&E tissue fraction. The
available historical round-0 diagnostic is negative.

| Specimen | Spots with >=3 nuclei | HEIR Spearman | Historical type mean | Final-record shuffle | Paired delta vs type mean | HEIR MSE | Type-mean MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4066 | 4,209 | -0.0221 | 0.1144 | 0.0045 | -0.0896 | 0.2066 | 0.1863 |
| 4399 | 4,050 | 0.0057 | 0.0000 | 0.0007 | 0.0057 | 0.0688 | 0.0726 |
| 4411 | 2,668 | -0.0035 | -0.0039 | 0.0008 | 0.0061 | 0.0829 | 0.0850 |
| **Equal-weight macro** | — | **-0.0066** | **0.0369** | **0.0020** | **-0.0259** | **0.1194** | **0.1146** |

The 10,000-resample paired specimen/gene abundance-stratified bootstrap gave a
95% interval of `[-0.0829, 0.0081]` for the historical HEIR-minus-type-mean
delta, with probability positive `0.2924`. This is not the plan's complete
hierarchical bootstrap because connected spatial-block definitions were not
frozen.

The historical diagnostic fails the available decision criteria: the macro
delta is negative, 4066 is below -0.01, and HEIR does not beat the final-record
shuffle in at least two specimens. HEIR improves median MSE over the historical
type mean in 4399 and 4411, but not in 4066.

## Reference and spot audit

The historical `reference500.npz` files were exported from the complete
integrated Seurat objects, not from FFPE snPATHO alone. The source
`processing_method` fields contain:

| Specimen | FFPE snPATHO | Frozen 3-prime | Frozen SNAP snPATHO/Flex | Total in historical reference |
|---|---:|---:|---:|---:|
| 4066 | 6,620 | 5,727 | 8,125 | 20,472 |
| 4399 | 4,471 | 9,163 | 9,446 | 23,080 |
| 4411 | 8,444 | 8,258 | 10,609 | 27,311 |

No separate scFFPE stratum is present in these downloaded integrated objects.
Consequently, the historical type-mean and pseudobulk controls are explicitly
named `historical_integrated_*` and cannot be interpreted as the plan's R1
FFPE-snPATHO-only controls.

The processed Visium RDS objects materialize the author-QC whitelist. The
DeepBench proxy additionally requires positive library size and at least three
assigned Space Ranger nuclei:

| Specimen | Author-QC spots | >=1 nucleus | >=3 nuclei | >=5 nuclei |
|---|---:|---:|---:|---:|
| 4066 | 4,769 | 4,659 | 4,209 | 3,466 |
| 4399 | 4,560 | 4,454 | 4,050 | 3,185 |
| 4411 | 2,812 | 2,758 | 2,668 | 2,496 |

The required `>=50%` H&E tissue-fraction field and explicit author-QC exclusion
flags/reasons are unavailable, so this remains a partial spot-QC proxy.

## What the executable scorer covers

- hash revalidation of the immutable locked plan, report, prediction, truth,
  reference, checkpoint, panel, and target-isolation contracts;
- RNA-mass and equal-cell spot aggregation performed in linear space;
- a hard predicted-cell-type mean constructed from pooled raw counts divided by
  pooled full-library mass;
- all-cell, selective, equal-cell, historical integrated type-mean,
  pseudobulk, and complete final-cell-record shuffle views;
- gene Spearman/Pearson/MSE/MAE/concordance, detection AUROC, top-10% hotspot
  Dice/Jaccard, location cosine/Spearman/MAE, and Moran's-I agreement;
- the required rule that a constant prediction receives correlation zero when
  observed expression varies, while constant observed genes are excluded with
  an explicit reason;
- per-gene TSV rows, equal-weight specimen macro summaries, paired deltas, and
  10,000 retrospective bootstrap replicates.

The final-cell-record shuffle is not the plan's shuffled-image-feature or
coordinate-shuffled-graph control. The v0.2 run also used target-H&E-derived OOD
calibration, making it noncompliant with the external freeze for both A1 and A2.
Space Ranger remains the default and common segmentation, but its historical
confidence was substituted as constant `1.0`; therefore the refinement
confidence gate is vacuous until a calibrated measurement is supplied.

## Still blocked

The executable readiness ledger records the exact status and reason for every
missing track. Major blockers are refined and five-seed predictions,
workflow-specific and wrong-donor references, generic-atlas and H&E-only runs,
image/graph shuffles, 384-µm features, program definitions, manual nucleus
labels, composition covariates, pathologist regions, segmentation sensitivity,
spatial blocks, and a native scANVI checkpoint.

Registered optional artifacts remain `registered_not_implemented` until a
schema-specific scorer consumes them; merely providing a path can never mark
the full plan complete.

## Reproduce

```bash
conda run -n hne python scripts/benchmark_snpatho_deepbench.py \
  --plan configs/experiments/snpatho_deepbench_v1.yaml \
  --output artifacts/snpatho/deepbench_v1/report.json \
  --tsv artifacts/snpatho/deepbench_v1/report.tsv \
  --markdown artifacts/snpatho/deepbench_v1/report.md
```

The final local run completed in 45.6 seconds with approximately 1.47 GiB peak RSS.
This command re-scores frozen predictions; it does not rerun historical CUDA
model inference. The compact committed result is
`reports/snpatho_deepbench_v1_summary.json`; the full JSON/TSV/Markdown outputs
remain in ignored `artifacts/` storage.

The locked report SHA-256 remains
`d002e601ffc0f1b69d141d507906f839815d305af9055780b38ff72bb4c12d65`.
