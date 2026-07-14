# HEST retrospective experiment report

## Decision

The completed GSE250346/HEST experiment is
`retrospective_exposed_non_authorizing`. It provides no support for H-CELL,
H-INTRINSIC-cell, or H-INTRINSIC-nucleus in the frozen analysis. It does not fail the untouched
prospective hypotheses and cannot authorize full HEIR development.

All report-level authorization fields are false:

- `authorizes_h_cell=false`
- `authorizes_h_intrinsic=false`
- `authorizes_full_heir=false`

The strongest apparent intrinsic trend was nucleus-mask minus target-removed R2 = `+0.001688`,
positive in 14/15 donors. However, the nucleus-mask model had negative absolute R2, worsened the
best non-image control in every donor, and passed only one of two arm-level shuffle tests. The
shuffle p-values do not directly test the paired nucleus-minus-removed contrast. This is therefore
not evidence for H-INTRINSIC.

## Experiment identity

| Field | Frozen value |
|---|---|
| Run date | 2026-07-13 |
| Cohort | HEST/GSE250346 lung |
| HEST revision | `7e8d5a0b0aace41d8c8ec0f6ecea80e4ad2a61ec` |
| Biological donors / sections | 15 / 20 |
| Registered source / reference / evaluation cells | 36,121 / 18,059 / 18,062 |
| Broad lineages / supported fine types | 4 / 38 |
| Fine donor-type / donor-section-type strata | 448 / 570 |
| Image encoder | `MahmoodLab/UNI2-h` |
| Encoder revision | `d517a8dd47902dd7c308b3c36f63bce47e7b9a43` |
| Encoder checkpoint SHA-256 | `6e077eda234bebc595868d918d3458d9dd32a050199b0ff04443b2f46a0a3b1e` |
| Source schema | `heir.registered_observations_retrospective.v1` |
| Report schema | `heir.hest_retrospective_report.v2` |
| Source SHA-256 | `57b77c7be2e30026a2da9ba0f9d5b205cf630f5d138942db6366e15cae2ef7a3` |
| Plan SHA-256 | `412348218e3c8fa048ed0e60cd1803a2bd492af441590e4bea0bc6401f3d370b` |
| QC SHA-256 | `1a9ad33d48bbf163348bdef83d0c24869d6da78780b908b9f92e0034f13b0838` |
| Final report SHA-256 | `60bb7ec6a918b9e2e9382e0945b83ff6426afb9a86bb1e80346aff8ab576df79` |

Large artifacts remain outside Git:

| Artifact | Local path | Bytes |
|---|---|---:|
| Registered source | `/mnt/seagate/HEIR_runs/hest_retrospective/source.npz` | 1,915,144,238 |
| Preparation plan | `/mnt/seagate/HEIR_runs/hest_retrospective/plan.json` | 112,962 |
| Source QC | `/mnt/seagate/HEIR_runs/hest_retrospective/qc.json` | 89,449 |
| Full raw report | `/mnt/seagate/HEIR_runs/hest_retrospective/report.json` | 4,342,133 |
| Execution log | `/mnt/seagate/HEIR_runs/hest_retrospective/benchmark_v2.log` | 17,400 |

The raw JSON contains the complete per-donor, per-section, per-type, per-control, per-arm, and
permutation-level values. This Markdown records the decision-relevant results without copying that
4.34 MB artifact into the repository.

## Experiment 1: registered source construction

### Construction

Cells were joined by section-scoped native Xenium cell ID. Annotation-to-nucleus centroid distance
was retained as retrospective stress QC and was not used as the pairing key. Deterministic SHA-256
sampling retained at most 32 cells per section/fine-type/pool stratum and 36,121 cells overall.

Each cell has four 112-micrometre, 224-pixel crop arms at 0.5 micrometres per pixel:

1. `crop_112um`
2. `cell_mask_only`
3. `nucleus_mask_only`
4. `target_cell_removed_112um`

CUDA UNI2-h extraction produced a float32 tensor with shape `(36121, 4, 1536)`. The primary RNA
target contains 260 non-marker nucleus-overlapping log1p-CPM features. The builder excluded the
frozen broad- and fine-type marker proxies from the target.

### Source QC

| Check | Result | Interpretation |
|---|---:|---|
| Native ID join | One-to-one | Primary registration identity passed |
| Duplicate observation IDs | 0 | Passed |
| Target row invariants | 36,121/36,121 | Passed |
| Annotation-nucleus distance p50 / p95 | 5.876 / 29.953 micrometres | Prospective 8-micrometre stress threshold failed |
| Rows within all registration stress thresholds | 22.15% | Diagnostic only for this retrospective source |
| Nucleus centroid outside cell | 0.0388% | Below 1% |
| Crop padding cohort p95 | 0.0 for every crop | Passed the cohort-level 0.25 contract |
| Crop padding maximum | 0.4383 | Below the 0.5 hard exclusion limit |
| Rows above the stricter 0.25 padding threshold | 11/36,121 | All are in NCBI861 |
| Overall prospective-style QC flag | `false` | Correctly prevents authorizing use |

Among the 18,062 evaluation cells, 391 were in the registration-best stratum, 1,886 were best or
intermediate, 2,115 were near-threshold, and 14,061 were in the annotation-centroid stress-failed
stratum. A separate 3,873-cell subset passed the combined locked-measurement row flags. Because the
primary pairing is the native Xenium ID join, the stress-failed label must not be described as a
known cell-ID mismatch; it does limit cell- and nucleus-localization claims.

## Experiment 2: retrospective biological hypothesis benchmark

### Frozen analysis

| Component | Setting |
|---|---|
| Biological evaluation unit | Leave one true donor out; 15 folds |
| Molecular endpoint | Joint 260-feature nucleus-overlap residual state |
| Reference | Spatially separated same-donor/section/type reference-pool mean |
| Label resolutions | `final_lineage` and `final_CT` |
| Image representation | Fixed outcome-free 1,536-to-96 Rademacher projection |
| Model | Float64 ridge, alpha 100, train-fold-only standardization |
| Minimum evaluated stratum support | 5 cells |
| Non-image controls | Reference mean; technical 1D; spatial 14D; stain/QC 71D; morphometry+density 73D; deduplicated combined 158D |
| Null 1 | Within donor/section/type/role image derangement |
| Null 2 | Different-spatial-block reassignment within the same strata |
| Permutations | 100 per null, arm, and resolution; 1,600 refits total |
| Primary aggregation | Equal over types within donor, then equal over donors |
| Companion aggregation | Equal over types within section, sections within donor, then donors |

Within-stratum derangement changed all evaluation rows. The different-block null changed and crossed
blocks for all broad rows and 95.925% of fine rows (minimum 17,326 eligible). Frozen mapping-set
SHA-256s were:

| Resolution | Within-stratum mapping set | Different-block mapping set |
|---|---|---|
| Fine type | `c06a1c989868159d09cf767152681b35a52c3a6bdf755a4f958a59990db5e7c7` | `25f21eba5d242abd81ef5fc51814b1de1969d778db09701a18be95f7dfc347de` |
| Broad lineage | `cc126cf4ae4f1895001bd56bd09cc300626dfcd66c345311645fd458bc821566` | `a2293421760313f2839e15b382a0f4f193cec9d029a0aaeb0353d66ce4d671ed` |

### Non-image controls

The technical-only model was the observed best non-image control at both label resolutions.

| Control | Fine R2 | Fine section R2 | Fine reference error reduction | Broad R2 | Broad section R2 | Broad reference error reduction |
|---|---:|---:|---:|---:|---:|---:|
| Reference mean | -0.061075 | -0.070119 | 0.000000 | -0.010852 | -0.013123 | 0.000000 |
| Spatial | -0.062438 | -0.071480 | -0.001286 | -0.012523 | -0.014779 | -0.001654 |
| Technical | **-0.044578** | **-0.053579** | **0.015510** | **0.013344** | **0.011214** | **0.023922** |
| Stain/QC | -0.050274 | -0.059525 | 0.010134 | 0.007644 | 0.005378 | 0.018301 |
| Morphometry+density | -0.053321 | -0.062403 | 0.007307 | 0.003575 | 0.001275 | 0.014297 |
| Combined non-image | -0.053087 | -0.062363 | 0.007488 | 0.005372 | 0.003070 | 0.016078 |

### Fine-type image arms

`Delta combined` and `Delta best` are donor-paired R2 changes from adding the image representation
to the combined non-image model and from comparing it with the best non-image model, respectively.
The empirical p-values test whether correctly paired images have a larger increment than each
refitted shuffled-image null; they do not test whether absolute prediction is useful.

| Crop arm | R2 | Section R2 | Reference error reduction | Delta combined | Positive donors | Delta best | Within-stratum p | Different-block p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full 112 micrometres | -0.060454 | -0.069909 | 0.000518 | -0.007367 | 0/15 | -0.015876 | 0.128713 | 0.079208 |
| Cell mask only | -0.058823 | -0.068183 | 0.002078 | -0.005736 | 0/15 | -0.014245 | 0.227723 | 0.287129 |
| Nucleus mask only | -0.058752 | -0.068104 | 0.002146 | -0.005665 | 0/15 | -0.014175 | 0.029703 | 0.059406 |
| Target cell removed | -0.060440 | -0.069898 | 0.000532 | -0.007353 | 0/15 | -0.015863 | 0.366337 | 0.178218 |

No arm had positive absolute donor-balanced R2, no arm improved the combined or best control, and no
arm improved the combined model in any donor. The nucleus arm's within-stratum p-value did not
replicate under the different-block null and accompanies a negative model increment.

### Broad-lineage sensitivity

| Crop arm | R2 | Section R2 | Delta combined | Positive donors | Delta best | Within-stratum p | Different-block p |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full 112 micrometres | -0.000756 | -0.003226 | -0.006128 | 2/15 | -0.014100 | 0.009901 | 0.009901 |
| Cell mask only | 0.000351 | -0.001984 | -0.005021 | 0/15 | -0.012994 | 0.009901 | 0.009901 |
| Nucleus mask only | 0.000139 | -0.002181 | -0.005233 | 0/15 | -0.013205 | 0.009901 | 0.009901 |
| Target cell removed | -0.000753 | -0.003212 | -0.006125 | 2/15 | -0.014098 | 0.009901 | 0.009901 |

The small broad-lineage p-values mean matched images were less harmful than shuffled images; they
do not reverse the negative increments or show useful prediction. The technical control alone had
R2 = `0.013344`, above every broad image arm.

### Null distributions

The null statistic is the donor/type R2 increment over the combined non-image control. `Matched -
null` is positive when the correctly paired representation is less harmful or more helpful than the
mean shuffled representation.

| Resolution | Crop | Null | Observed increment | Null mean | Matched - null | Empirical p |
|---|---|---|---:|---:|---:|---:|
| Fine | Full | Within-stratum | -0.007367 | -0.007708 | 0.000341 | 0.128713 |
| Fine | Full | Different-block | -0.007367 | -0.007875 | 0.000508 | 0.079208 |
| Fine | Cell mask | Within-stratum | -0.005736 | -0.005832 | 0.000097 | 0.227723 |
| Fine | Cell mask | Different-block | -0.005736 | -0.005828 | 0.000093 | 0.287129 |
| Fine | Nucleus mask | Within-stratum | -0.005665 | -0.005869 | 0.000203 | 0.029703 |
| Fine | Nucleus mask | Different-block | -0.005665 | -0.005884 | 0.000219 | 0.059406 |
| Fine | Target removed | Within-stratum | -0.007353 | -0.007453 | 0.000100 | 0.366337 |
| Fine | Target removed | Different-block | -0.007353 | -0.007645 | 0.000292 | 0.178218 |
| Broad | Full | Within-stratum | -0.006128 | -0.007909 | 0.001781 | 0.009901 |
| Broad | Full | Different-block | -0.006128 | -0.008307 | 0.002179 | 0.009901 |
| Broad | Cell mask | Within-stratum | -0.005021 | -0.005442 | 0.000421 | 0.009901 |
| Broad | Cell mask | Different-block | -0.005021 | -0.005442 | 0.000420 | 0.009901 |
| Broad | Nucleus mask | Within-stratum | -0.005233 | -0.005796 | 0.000564 | 0.009901 |
| Broad | Nucleus mask | Different-block | -0.005233 | -0.005801 | 0.000569 | 0.009901 |
| Broad | Target removed | Within-stratum | -0.006125 | -0.007700 | 0.001575 | 0.009901 |
| Broad | Target removed | Different-block | -0.006125 | -0.008077 | 0.001952 | 0.009901 |

### Direct crop contrasts

These are paired differences between crop predictions. They were not assigned a dedicated
contrast-permutation distribution, so their signs and donor consistency are descriptive.

| Contrast | Fine R2 difference | Fine section difference | Positive donors | Strict-measurement difference | Broad R2 difference |
|---|---:|---:|---:|---:|---:|
| Full minus target removed | -0.000014 | -0.000011 | 6/15 | 0.000120 | -0.000003 |
| Cell mask minus target removed | 0.001617 | 0.001715 | 11/15 | 0.002412 | 0.001104 |
| Nucleus mask minus target removed | 0.001688 | 0.001794 | 14/15 | 0.004051 | 0.000893 |

The positive cell/nucleus differences are small relative to the negative absolute R2 and negative
image-over-control effects. The target-removed arm contains a white cell-shaped hole, while the mask
arms contain complementary white backgrounds; silhouette and fill artifacts therefore remain a
plausible explanation.

The strict-measurement fine-type R2 values were also negative: full `-0.182103`, cell mask
`-0.179811`, nucleus mask `-0.178172`, and target removed `-0.182223`. Positive strict-subset crop
differences therefore do not indicate useful absolute prediction.

### Full-context image increment by donor

Every donor's fine-type full-context increment was negative against both the combined control and
the observed best control.

| Donor | Delta combined | Delta best control |
|---|---:|---:|
| THD0008 | -0.005434 | -0.017376 |
| THD0011 | -0.006347 | -0.020401 |
| TILD117 | -0.006693 | -0.012060 |
| TILD175 | -0.005354 | -0.013580 |
| VUHD069 | -0.006625 | -0.022157 |
| VUHD116 | -0.009447 | -0.019144 |
| VUILD102 | -0.011392 | -0.019670 |
| VUILD105 | -0.005426 | -0.014007 |
| VUILD106 | -0.006409 | -0.013644 |
| VUILD107 | -0.006726 | -0.010008 |
| VUILD110 | -0.006900 | -0.014250 |
| VUILD115 | -0.008452 | -0.017525 |
| VUILD78 | -0.011052 | -0.017598 |
| VUILD91 | -0.005745 | -0.016164 |
| VUILD96 | -0.008504 | -0.010561 |

## Hypothesis decisions

| Hypothesis | Decisive observations | Reported status |
|---|---|---|
| H-CELL-retrospective | Full-context fine R2 and section R2 were negative; Delta combined and Delta best were negative in 15/15 donors; both p-values exceeded 0.05 | `not_supported_or_indeterminate_in_this_analysis` |
| H-INTRINSIC-cell-retrospective | Cell-minus-removed was positive, but absolute R2 and both control increments were negative; both arm-level null p-values exceeded 0.05 | `not_supported_or_indeterminate_in_this_analysis` |
| H-INTRINSIC-nucleus-retrospective | Nucleus-minus-removed was positive, but absolute R2 and both control increments were negative; only one of two arm-level nulls passed | `not_supported_or_indeterminate_in_this_analysis` |

The result is a negative result for this frozen retrospective representation/probe, not a general
biological null. It leaves the pristine prospective H-CELL and H-INTRINSIC hypotheses untested.

## Limitations and claim boundary

- All GSE250346 molecular outcomes were already exposed; this is internal retrospective evidence.
- `final_CT` and `final_lineage` are RNA-derived and may remain ontology-dependent despite marker
  proxy exclusion.
- The experiment tests one pooled multi-type linear ridge with fixed alpha and an outcome-free
  random 96-dimensional projection; it can miss nonlinear or type-specific morphology mappings.
- Joint unstandardized 260-feature SSE can be dominated by high-variance genes. The runner does not
  test the six stored programs, gene-wise endpoints, or reliability-weighted targets.
- Direct intrinsic crop contrasts lack their own permutation/sign test and multiplicity correction.
- White-fill mask interventions are not pure biological interventions.
- Only 3,873 evaluation rows pass the strict combined measurement flags, and only 391 are in the
  registration-best stratum.
- One hundred permutations give a minimum attainable p-value of `1/101`; p-values are not adjusted
  over resolutions, arms, or null families.
- The experiment does not test H-REGIONAL, external generalization, matched-reference utility,
  iterative refinement, or a prospective authorization gate.

## Reproduction and verification

The actual full command was:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 \
  .venv/bin/python scripts/benchmark_hest_retrospective.py \
  --source /storage/HE_GPT/.heir_runtime/hest_retrospective/source.npz \
  --output /storage/HE_GPT/.heir_runtime/hest_retrospective/report_v2.json \
  --alpha 100 --permutations 100 --seed 20260713 \
  --projection-dimension 96 --minimum-support 5
```

The run used the NumPy float64 backend, was capped at two CPU cores, and peaked at 1,727,524,864
bytes of resident service memory. Source construction used CUDA with batch size 8. The output was
written atomically, parsed successfully, and was checked for exact source, cohort, arm, null, and
authorization identities. Repository verification after the final report correction passed Ruff,
`git diff --check`, and 248 tests.
