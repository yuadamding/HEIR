# snPATHO-DeepBench-v1

`snPATHO-DeepBench-v1` applies the attached full benchmark plan to every
component supported by the frozen local artifacts. It is a **retrospective
diagnostic**, not a replacement for `snPATHO-Locked-v0.2`, an untouched
validation, or a compliant Track A result.

## Outcome

The requested primary endpoint is not yet testable: there are no redesigned
refined predictions, independently reannotated/scANVI-encoded R1 reference, or
composition-adjustment inputs, and there is no frozen per-spot H&E tissue
fraction. FFPE-only counts now support explicitly labeled integrated-annotation
type-mean sensitivities, but not the requested refined primary contrast. The
available historical round-0 diagnostic is negative.

| Specimen | Spots with >=3 nuclei | HEIR Spearman | Historical hard type mean | Final-record shuffle draw 0 | `median_g(rho_HEIR,g - rho_hist-hard,g)` | HEIR MSE | Hard type-mean MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4066 | 4,209 | -0.0221 | 0.1144 | 0.0045 | -0.0896 | 0.2066 | 0.1863 |
| 4399 | 4,050 | 0.0057 | 0.0000 | 0.0007 | 0.0057 | 0.0688 | 0.0726 |
| 4411 | 2,668 | -0.0035 | -0.0039 | 0.0008 | 0.0061 | 0.0829 | 0.0850 |
| **Equal-weight macro** | — | **-0.0066** | **0.0369** | **0.0020** | **-0.0259** | **0.1194** | **0.1146** |

The paired statistic shown for specimen (d) is exactly
`median_g(rho_HEIR,dg - rho_historical-hard-type-mean,dg)`. It is not the
difference between the two marginal medians. The macro value is the
equal-weight mean of those three specimen statistics.

The 10,000-resample paired specimen/gene abundance-stratified bootstrap gave a
95% interval of `[-0.0829, 0.0081]` for the historical HEIR-minus-hard-type-mean
delta. The **bootstrap fraction with delta > 0** was `0.2924`; this is neither a
frequentist p-value nor a Bayesian probability of truth. This is not the plan's
complete hierarchical bootstrap because connected spatial-block definitions
were not frozen.

The strengthened type-mean ladder is:

| Specimen | Historical hard Spearman / MSE | Historical soft Spearman / MSE | FFPE-R1 hard Spearman / MSE | FFPE-R1 soft Spearman / MSE |
|---|---:|---:|---:|---:|
| 4066 | 0.1144 / 0.1863 | 0.1332 / 0.2536 | 0.1166 / 0.1846 | 0.1336 / 0.2616 |
| 4399 | 0.0000 / 0.0726 | -0.0442 / 0.0759 | 0.0000 / 0.0696 | -0.0426 / 0.0762 |
| 4411 | -0.0039 / 0.0850 | -0.0048 / 0.0852 | -0.0036 / 0.0876 | -0.0056 / 0.0840 |
| **Equal-weight macro** | **0.0369 / 0.1146** | **0.0280 / 0.1382** | **0.0377 / 0.1139** | **0.0284 / 0.1406** |

The FFPE-R1 columns use only `processing_method=FFPE_snPATHO`, but retain the
published integrated-workflow `major_annotation`; they are retrospective
annotation sensitivities, not clean reannotated/scANVI primary R1 baselines.

### Repeated final-cell-record shuffle null

Draw 0 is retained for exact backward comparison. The scorer now also moves
each assigned cell's complete expression-plus-library-size record in 100
independently seeded deterministic permutations per specimen:

| Specimen | HEIR Spearman | Null median | Null empirical 95% interval | HEIR percentile in null | Above upper bound? |
|---|---:|---:|---:|---:|---|
| 4066 | -0.0221 | 0.0011 | [-0.0016, 0.0043] | 0.00 | no |
| 4399 | 0.0057 | 0.0004 | [-0.0017, 0.0028] | 1.00 | yes |
| 4411 | -0.0035 | -0.0001 | [-0.0030, 0.0023] | 0.00 | no |

The equal-weight specimen-macro null median is `0.00035`, with empirical 95%
interval `[-0.00087, 0.00218]`. HEIR exceeds the per-specimen upper bound in
only one case, so the required at-least-two rule fails. This record shuffle does
not replace model reruns with shuffled image features or coordinates/graphs.

The historical diagnostic fails the available decision criteria: the macro
delta is negative, 4066 is below -0.01, and HEIR neither beats draw 0 nor exceeds
the repeated-null upper bound in at least two specimens. HEIR improves median MSE over the historical
type mean in 4399 and 4411, but not in 4066.

## Why Locked-v0.2 and DeepBench-v1 differ

The two reports use different estimands. Their numerical differences are
expected and do not change either negative conclusion.

| Feature | Locked-v0.2 | DeepBench-v1 |
|---|---|---|
| Minimum nuclei per spot | >=1 | >=3 |
| Cell aggregation | Equal-cell | Historical integrated-reference library-size weighting |
| Type profile | Historical locked implementation | Pooled raw counts divided by full-library mass |
| Constant prediction policy | Earlier metric implementation | Correlation fixed at zero when observed expression varies |
| Shuffle | Historical spatial shuffle | Complete final-cell-record shuffle draw 0 plus separate 100-permutation null |

The corresponding macro median-gene Spearman values are HEIR `-0.0054`, type
mean `0.0224`, and shuffle `0.0026` in Locked-v0.2, versus HEIR `-0.0066`, hard
type mean `0.0369`, and final-record shuffle `0.0020` in DeepBench-v1.

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

The source hashes, exact `processing_method` totals, and complete
workflow-by-`major_annotation` tables are committed in
[`snpatho_reference_workflow_audit.json`](../reports/snpatho_reference_workflow_audit.json).
No metadata rows were filtered for that audit.

All prediction type names are supported by both their historical integrated and
FFPE-only count references:

| Specimen | Prediction types | Historical supported | FFPE-R1 supported | Missing | Hard-fallback cells/fraction |
|---|---:|---:|---:|---|---:|
| 4066 | 12 | 12 | 12 | none | 0 / 0.0 |
| 4399 | 11 | 11 | 11 | none | 0 / 0.0 |
| 4411 | 9 | 9 | 9 | none | 0 / 0.0 |

The scorer now fails closed if a prediction type is absent; it never silently
substitutes a global reference profile. Thus the recorded global-fallback count
and soft fallback probability mass are both zero for these cases.

Constant predictions are explicitly counted rather than hidden. The historical
and FFPE-R1 hard baselines are constant for all 500 genes in 4399; the 4399 soft
baselines have 2 and 44 constant genes, respectively. FFPE-R1 hard/soft have
2/1 constant genes in 4066 and 7/3 in 4411. The spatially constant pseudobulk has
500 in every case. These receive correlation zero when observed expression
varies; there were no observed-constant exclusions.

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
- historical integrated-reference library-size weighting and equal-cell spot
  aggregation performed in linear space; the historical weights are not
  assay-corrected biological RNA-mass estimates;
- hard-argmax and probability-weighted soft predicted-cell-type means constructed
  from pooled raw counts divided by pooled full-library mass;
- all-cell, selective, equal-cell, historical integrated and FFPE-R1 hard/soft
  type-mean, pseudobulk, and complete final-cell-record shuffle views;
- gene Spearman/Pearson/MSE/MAE/concordance, expression-detection AUROC,
  top-10% hotspot Dice/Jaccard, location cosine/Spearman/MAE, and Moran's-I
  agreement;
- an exact-size top-10% hotspot rule with descending expression and ascending
  frozen spot-row index as the deterministic cutoff tie breaker;
- Moran's I on a directed, unweighted 6-nearest-neighbor graph that is neither
  symmetrized nor row-standardized; this is a historical sensitivity graph, not
  Visium hex-lattice adjacency;
- the required rule that a constant prediction receives correlation zero when
  observed expression varies, while constant observed genes are excluded with
  an explicit reason;
- per-gene TSV rows, explicit constant counts, equal-weight specimen macro
  summaries, paired deltas, 100-per-specimen shuffle-null summaries, and 10,000
  retrospective bootstrap replicates.

The repeated final-cell-record shuffle is not the plan's shuffled-image-feature or
coordinate-shuffled-graph control. The v0.2 run also used target-H&E-derived OOD
calibration, making it noncompliant with the external freeze for both A1 and A2.
Space Ranger remains the default and common segmentation, but its historical
confidence was substituted as constant `1.0`; therefore the refinement
confidence gate is vacuous until a calibrated measurement is supplied.

## Still blocked

The executable readiness ledger records the exact status and reason for every
missing track. Major blockers are refined and five-seed predictions,
alternative-workflow references/predictions, wrong-donor predictions,
generic-atlas and H&E-only runs,
image/graph shuffles, 384-µm features, program definitions, manual nucleus
labels, composition covariates, pathologist regions, segmentation sensitivity,
spatial blocks, and a native scANVI checkpoint.

The R1 manifest is consumed as a retrospective sensitivity. Other registered
optional artifacts remain `registered_not_implemented` until a schema-specific
scorer consumes them; merely providing a path can never mark the full plan
complete.

The FFPE-only counts and SVD fallback prototypes are hash-manifested in
[`snpatho_r1_reference_manifest.json`](../reports/snpatho_r1_reference_manifest.json).
The scorer consumes the counts for hard/soft type-mean sensitivities using
FFPE-only type-median library-size weights. Those labels came from the published
integrated-workflow annotation, so these are not clean primary R1 results. The
native-scANVI prototype-only/no-residual prediction and redesigned-refinement
contrast remain unavailable.

## Reproduce

```bash
conda run -n hne python scripts/benchmark_snpatho_deepbench.py \
  --plan configs/experiments/snpatho_deepbench_v1.yaml \
  --output artifacts/snpatho/deepbench_v1/report.json \
  --tsv artifacts/snpatho/deepbench_v1/report.tsv \
  --markdown artifacts/snpatho/deepbench_v1/report.md
```

The final 100-permutation local run completed in 143.94 seconds with
approximately 1.69 GiB peak RSS.
This command re-scores frozen predictions; it does not rerun historical CUDA
model inference. The compact committed result is
`reports/snpatho_deepbench_v1_summary.json`; the full JSON/TSV/Markdown outputs
remain in ignored `artifacts/` storage.

The locked report SHA-256 remains
`d002e601ffc0f1b69d141d507906f839815d305af9055780b38ff72bb4c12d65`.
The final DeepBench plan/JSON/TSV/Markdown SHA-256 values are recorded in
`reports/snpatho_deepbench_v1_summary.json`.
