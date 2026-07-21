# NatCommun next-validation scientific report

## Status and scientific decision

This report records validation of the frozen NatCommun model through the no-refit diagnostic audit,
the reciprocal adjacent-section matched-ST experiment, the fixed-effective-sample-size reference
sensitivity, the guarded same-section ST upper bound, and the leave-one-donor-out gene-panel
sensitivity. The added within-type state-diversity-matched control was attempted under a separately
audited fail-closed protocol, but no B1 type satisfied the complete
natural-state/joint-QC/exact-ESS eligibility contract. It therefore stopped without writing a
prediction artifact and before opening any score-target array, so it cannot produce a cohort-level
performance test.

The current evidence does **not** support H&E–molecular-reference synergy, general continuous-state
recovery, an ST measurement-floor ordering, an independent regional claim, or a cell-level claim.
The current fixed-bank sensitivity gives partial evidence consistent with matched-bank separation
after the implemented ESS/type/QC controls within this outcome-exposed cohort, but B1 does not
satisfy the stricter state-diversity eligibility contract and personalization remains unresolved.
The fold-local panel sensitivity preserves the M3-versus-M0 direction in all 13 donors, but again
fails M3 versus M1. This is not independent confirmation. The exposed development result remains
real but narrow: M3
improved over H&E-only M0 by 8.1096% in all 13 donors, while reference-only M1 remained better than
M3. In the two-donor adjacent-section diagnostic, same-assay ST references were substantially better
than suspension references, but adding the frozen H&E correction made the strong matched-ST
reference worse, not better. Correct image pairing carried information relative to shuffled H&E, yet
the aggregate correction audit shows that this information is over-applied.

No iterative refinement is authorized.

## Frozen scope and exact identities

The architecture, H-optimus-1 encoder, global 256-gene panel except for the explicitly prescribed
fold-local gene-selection sensitivity, 20-dimensional latent, loss, training schedule, fusion rule,
and decision thresholds were frozen. UNI2-h was not run. The no-refit audit performed no fit or
parameter update and changed no prediction rows or values after target access.

| Item | Exact identity |
| --- | --- |
| Image encoder | `bioptimus/H-optimus-1` |
| Encoder revision recorded by matched-ST report | `3592cb220dec7a150c5d7813fb56e68bd57473b9` |
| Frozen gene count / latent dimension | `256` / `20` |
| Frozen panel SHA-256 | `ce0b6b82440d7fccc69f24afccf0c68bb101b85f590e4a35a514309929fbb6ad` |
| Frozen model core SHA-256 | `55a63f1360e8cc76267e4b00ba8e2167f36259789e9bfdf2aa929c8cadd83b17` |
| Frozen baseline runner SHA-256 | `cf27504e25dfd8cd7e8bfe2894efc8b4a8f79306b47bc492d0e61406d20668ce` |
| Frozen development protocol SHA-256 | `2cb92b22b6870488a06e64b213e37ffbbdfe3044f1da8fc7442f506915e78197` |
| Frozen fit/predict manifest SHA-256 | `cb7ebdf9e22090a046937204993a7b2aa3ac1ba2d4c434883e43ed45d1e826ca` |

Machine-readable evidence used here:

| Evidence | External artifact | SHA-256 |
| --- | --- | --- |
| Exposed frozen development baseline | `/mnt/seagate/HEIR_runs/natcommun_generative_development/report.json` | `bf3144cf22405752488509dbb1a65b573967fe1b14110881020787187828cf29` |
| Frozen-prediction diagnostic audit | `/mnt/seagate/HEIR_runs/natcommun_frozen_validation_diagnostics/report.json` | `68232fa55b0b40711d3b649b60dc6ceb86953e0a7781cd5d17c4765116b8407e` |
| Reciprocal adjacent-section matched-ST diagnostic | `/mnt/seagate/HEIR_runs/natcommun_matched_st_validation/report.json` | `aff7b42ab45fa6cc69ae8085c8e095db12d8b90aec0ea403fb77453fe065c0ac` |
| Guarded same-section matched-ST upper bound | `/mnt/seagate/HEIR_runs/natcommun_matched_st_validation/same_section_upper_bound_report.json` | `d81bb54d18ae98e5f7cdfc2313fda43bf7cd868ad2d4c82525b0747418b65c5d` |
| Fixed-ESS/type-support reference sensitivity | `/mnt/seagate/HEIR_runs/natcommun_fixed_ess_reference_sensitivity/report.json` | `6514a90724dd0a65805d0f4ef8db0b1e36d9e00edf98db28a2d45b1f91b6e0db` |
| Fold-local training-only panel sensitivity | `/mnt/seagate/HEIR_runs/natcommun_fold_local_panel_sensitivity_v3/report.json` | `8812482d6f2a662007c4e603253b8866d42e590317a8c5ccbed7f0a4dee30568` |
| State-diversity v3 prepared manifest | `/mnt/seagate/HEIR_runs/natcommun_fixed_ess_reference_sensitivity_v3_6b73f418/prepared_manifest.json` | `bfc5216c505ce655212700af7a0e47c5b9a09df605a88ec0742fa8cf6d2cdd93` |
| State-diversity B1 ineligibility log | `/mnt/seagate/HEIR_runs/natcommun_fixed_ess_reference_sensitivity_v3_6b73f418/fit_predict.stderr.log` | `11ac6b50369b0f06157f41bd0299aa94fdcb87db1ba66343ea1e1c3228d318cd` |
| Validation plan | `/home/yding1995/.codex/attachments/68736369-ce2f-427c-89f7-c35d84bb50e5/pasted-text.txt` | `e184b815242c9e710054aed28a52cee155c73f819c9e8e6cb450b1f135678997` |

The matched-ST experiment additionally binds the following identities:

- validation runner: `31550aae6b7270d15a56d268cfab4e5ce3b4cc13ca7d10151f07a4ca292d7bee`;
- validation protocol: `e92338f608a5c78bf051a5daf072d1a8e648a5b48a1e655b3b1fd06fc3ed8eda`;
- prepared manifest: `7a91f37fcb79c021ac1ccdf2f1d22aab5eb49156dd91fe7ea1abaa67f4e9a345`;
- prepared identity: `460a68c9a52078de46add9fa24c85a70b2f7ef52f5055cca24d0f3243209c244`;
- score-target manifest: `242d3306b15d8279c88950c7262b40e4f50b5b005af05474442d5440d3a72cd1`;
- score-target manifest identity: `7bef2c8601f99a9bf87ab0c4d8adba4ce5f927b028ba4a05ffa1950f2c92fffd`;
- model manifest: `f6057aac16786166b2bc98b872d50078b11ce9d8a19a461ffde5537e142cc7fe`.

## Cohort and evidence boundaries

The current NatCommun evidence contains 15 Visium sections from 13 donors: three breast donors, six
DLBCL donors, and four lung donors. These outcomes influenced the global panel and architecture, so
all 13-donor analyses are outcome-exposed development diagnostics, not confirmation. The registered
suspension reference is annotated at cell/type level but is not independently verified as snRNA.

Only B1 and L1 have the reciprocal adjacent sections needed for the matched-ST primary diagnostic.
That analysis comprises four directions across distinct Visium slides from serial sections in the
same block. It is distributional rather than spot-paired and has only two donor inference units. It
therefore tests a mechanism but cannot establish donor-level replication or independent
confirmation. Its score-family predictions were globally preflighted before any selected target
manifest or target was opened; the prepared prediction manifest was physically target-free.

The current validation set still does not provide:

- an untouched independent regional cohort;
- a registered technical replicate or valid measurement floor;
- cell-resolved ST or individual-cell expression/state endpoints;
- authority to infer independent regional confirmation, cell-level performance, or independently
  confirmed personalization.

## Arms and endpoint

The endpoint is donor-balanced held-out negative-binomial deviance on the frozen global 256-gene
target, except that the prescribed fold-local sensitivity uses its donor-specific training-only
256-gene axis. Lower is better.

| Arm | Frozen input/role |
| --- | --- |
| M0 / S0 | H&E only |
| M1 | matched suspension reference only |
| M2 | H&E composition/type routing plus matched type means |
| M3 | H&E plus matched suspension reference |
| M4 | shuffled H&E plus matched suspension reference |
| M6 / M7 | H&E plus natural wrong-donor / pooled suspension reference |
| S1 | independent matched ST reference only |
| S3 | H&E plus independent matched ST reference |
| S4 | shuffled H&E plus matched ST reference |
| S6 / S7 | H&E plus natural wrong-donor / query-excluded pooled ST reference |

For contrast tables below, the reported effect is `loss(comparator) - loss(candidate)`. A positive
value favors the named candidate; a negative value means the candidate is worse.

## Frozen exposed-development baseline

These values establish context; they are not new confirmatory results.

| Arm | Donor-balanced NB deviance |
| --- | ---: |
| M0 | 2.528094 |
| M1 | 2.231896 |
| M2 | 2.357673 |
| M3 | 2.323076 |
| M4 | 2.526250 |
| M6, natural wrong donor | 2.504271 |
| M7, natural generic bank | 2.442669 |

| Development contrast | Mean effect | Positive donors | One-sided exact sign-flip p | Decision |
| --- | ---: | ---: | ---: | --- |
| M0 - M3 | +0.205018 (8.1096%) | 13/13 | 0.000122 | Development-only incremental reference signal |
| M1 - M3 | -0.091179 | 3/13 | 0.828003 | M3 does not beat reference-only M1 |
| M4 - M3 | +0.203175 | 13/13 | 0.000122 | Correct H&E pairing matters relative to shuffle |
| M2 - M3, identical supported rows | +0.113568 | 6/13 | 0.230591 | General continuous-state gate fails |
| M6 - M3, natural banks | +0.181195 | 12/13 | 0.000366 | Descriptive; implemented ESS/type/QC sensitivity is favorable |
| M7 - M3, natural banks | +0.119594 | 13/13 | 0.000122 | Descriptive; implemented ESS/type/QC sensitivity is favorable |

M8 was not a valid floor: on its common split-half target, M8 deviance was 1.495515 versus 1.427797
for M3, an effect of -0.067719 for `M3 - M8` with p = 0.75. A matched-ST reference is likewise an
explanatory assay-matched comparator, not a measurement floor.

## No-refit diagnostic results

### Composition versus continuous state

The exact supported comparison was M3 versus M2 on the same spots and identical supported
composition. The frozen rule requires a positive mean, at least 70% positive donors, no indication
reversal, adequate support, inferential support, and a same-row M3-to-M0 reliability-adjusted
variance ratio of at least 0.80.

| Stratum | M2 - M3 mean effect | Positive donors | One-sided p |
| --- | ---: | ---: | ---: |
| All donors | +0.113568 | 6/13 (46.15%) | 0.230591 |
| Breast | -0.084584 | 1/3 (33.33%) | 0.625000 |
| DLBCL | +0.322316 | 4/6 (66.67%) | 0.156250 |
| Lung | -0.050940 | 1/4 (25.00%) | 0.875000 |
| High reference coverage | +0.156393 | 6/13 (46.15%) | 0.184814 |
| Low reference coverage | -0.030165 | 3/13 (23.08%) | 0.586670 |

Every donor/section met the frozen minimal support definition, but inferential support failed. All
nine H&E-composition proxy type strata had Holm-adjusted p = 1.0; positive-donor fractions ranged
from 20.00% to 66.67%. The supported-row M3-to-M0 reliability-adjusted variance ratio was 0.514534,
and the global ratio was 0.559101, both below the frozen 0.80 requirement.

**Conclusion:** the positive overall mean is heterogeneous, reverses in breast and lung, appears
strongest but still sub-threshold in DLBCL, and does not preserve enough state variance. General
continuous molecular-state recovery is not supported.

### Image-correction alignment

For each prediction, the audit compared the frozen H&E correction `M3 - M1` with the residual
`observed ST - M1`; it did not fit a new predictor.

| Quantity | Correct H&E | Shuffled H&E |
| --- | ---: | ---: |
| Predictive-variance-weighted inner product | 0.522699 | 0.276052 |
| Predictive-variance-weighted cosine | 0.056336 | 0.025224 |
| Centered count correlation | 0.224769 | 0.123931 |
| Positive weighted-inner-product donors | 12/13 | 9/13 |
| Optimal correction scale | 0.130401 | 0.061487 |
| Correction RMS / predictive SD | 1.633344 | 1.601402 |

Correct-minus-shuffled weighted alignment was +0.246647 in 13/13 donors (one-sided p = 0.000122;
paired bootstrap interval 0.070602 to 0.553054). Alignment was positive within each indication, but
the optimal scale varied: breast 0.128530, DLBCL 0.169003, and lung 0.073901.

| Program | Dot per element | Positive donors | Holm p | Optimal scale |
| --- | ---: | ---: | ---: | ---: |
| Antigen presentation | -0.004111 | 7/13 | 1.000000 | 0.077667 |
| Cellular stress | +0.017014 | 8/13 | 0.496582 | 0.241827 |
| Epithelial injury | +0.065488 | 9/13 | 0.023926 | 0.563222 |
| Fibrosis/remodeling | -0.002356 | 6/13 | 1.000000 | -0.007267 |
| Hypoxia/glycolysis | +0.029937 | 6/13 | 0.424805 | 0.207861 |
| Inflammatory activation | +0.002164 | 8/13 | 0.852539 | 0.299565 |
| Interferon response | +0.012261 | 6/13 | 0.278320 | 0.136182 |
| Proliferation | +0.077067 | 12/13 | 0.004883 | 1.071375 |

**Conclusion:** correctly paired H&E contains residual-aligned information, particularly for the
epithelial-injury and proliferation programs, but the aggregate optimal scale is only 0.1304 while
the applied correction RMS is 1.63 times the predictive uncertainty. The frozen fusion therefore
points partly in the correct direction but over-applies the correction; this diagnostic does not
authorize recalibration or a new model.

### Molecular-distribution audit

`V` is predicted variance divided by reliable observed-ST variance. Covariance entries are relative
Frobenius errors, for which lower is better. PI columns are empirical 50%/80%/95% coverage. Gene
dynamic range is the predicted-to-held-out-ST ratio of within-section log1p-10,000-normalized
95th-minus-5th-percentile ranges.

| Arm | V | Calibration slope | Gene cov. error | Program cov. error | Rare-state recall | PI coverage | Gene dynamic-range ratio q10 / median / q90 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| M0 | 1.001682 | 0.176208 | 0.927293 | 0.767515 | 0.222154 | 0.5870 / 0.8366 / 0.9104 | 0.2302 / 0.5319 / 1.1999 |
| M1 | ~0 (`1.56e-16`) | not interpretable (0/13 donors) | 1.000000 | 1.000000 | 0.103480 | 0.5919 / 0.8470 / 0.9119 | 0 / 0 / 0 |
| M2 | 0.294346 | 0.310809 | 0.942228 | 0.881605 | 0.181314 | 0.5586 / 0.8242 / 0.9068 | 0.1081 / 0.2834 / 0.6991 |
| M3 | 0.560042 | 0.249716 | 0.884782 | 0.805972 | 0.188637 | 0.5759 / 0.8366 / 0.9130 | 0.1662 / 0.4067 / 0.9269 |

The observed pattern is `M1 < M2 < M3 < M0`. H&E restores variation relative to a constant
reference-only prediction and relative to composition routing, but M3 retains only about 56% of M0's
reliability-adjusted variance (about 51% on the identical supported rows). M3 also has lower median
dynamic range and rare-state recall than M0. Thus the mean-loss gain over M0 cannot be interpreted as
recovery of molecular heterogeneity. The machine decision is `state_variance_preserved: false`.

## Reciprocal adjacent-section matched-ST results

This primary matched-ST diagnostic used four reciprocal directions across B1 and L1. The table below
contains donor-balanced NB deviance.

| Arm | NB deviance |
| --- | ---: |
| S0 | 3.754059 |
| S1 | 2.341883 |
| S3 | 2.869157 |
| S4 | 3.141481 |
| S6, natural wrong donor | 3.310949 |
| S7, natural pooled bank | 3.695509 |
| M1 | 2.893109 |
| M3 | 3.138539 |

| Diagnostic contrast | Mean effect | Donors favoring candidate | Exact p | Interpretation |
| --- | ---: | ---: | ---: | --- |
| S1 - S3 | -0.527273 | 0/2 | 1.00 | H&E harms the strong matched-ST reference |
| S4 - S3 | +0.272325 | 2/2 | 0.25 | Exact image pairing matters relative to shuffle |
| S6 - S3 | +0.441792 | 2/2 | 0.25 | Natural-control separation only |
| S7 - S3 | +0.826353 | 2/2 | 0.25 | Natural-control separation only |
| S0 - S3 | +0.884903 | 2/2 | 0.25 | Matched ST plus H&E beats H&E alone |
| M3 - S3 | +0.269383 | 2/2 | 0.25 | Fusion cross-assay penalty favors ST reference |
| M1 - S1 | +0.551226 | 2/2 | 0.25 | Reference-only cross-assay penalty favors ST reference |

The same-assay advantage is consistent in both donors: S1 is substantially better than M1, and S3
is better than M3. However, it does not explain the synergy failure by itself, because the frozen H&E
fusion also degrades the stronger ST reference: S3 is worse than S1 in both donors. S3 beating S4
agrees with the no-refit alignment audit that correct morphology carries information, while the
S3-versus-S1 result shows that the current fusion magnitude/direction does not convert that
information into conditional improvement. S6 and S7 are natural banks without fixed effective
sample size or type support; depth, quality, and state diversity are uncontrolled, so they cannot
support a personalization claim.

## Answers to the unresolved scientific questions

| Question | Current answer |
| --- | --- |
| Does H&E add conditional information to a strong reference? | **No performance evidence under the frozen fusion.** M3 is worse than M1, adjacent-section S3 is worse than S1, and same-section S3 is worse than S1 at both physical guards. Correct-pairing and residual-alignment diagnostics show image information exists, but it is over-applied. |
| Is the suspension-to-ST assay gap limiting? | **Yes, descriptively, but it is not the sole failure.** S1 beats M1 by 0.551226 and S3 beats M3 by 0.269383 in both adjacent-section donors; S3 still fails to beat S1. |
| Is matched-bank value donor-specific rather than bank-driven? | **Unresolved.** The advantage persists after exact cell-weight ESS, common coarse-type support, and available QC-stratum control. The stricter within-type control is not evaluable: no B1 type satisfies the complete frozen natural-state/joint-QC/exact-ESS contract, so no cohort-level performance contrast exists. This is not independent confirmation. |
| Is continuous molecular state recovered beyond composition/reference means? | **No.** Only 6/13 donors favor M3 over M2, indications reverse, inferential support fails, and variance preservation fails. |
| Is fusion above a valid ST measurement floor? | **Blocked.** M8 did not establish a floor, and an independent registered replicate is absent. |
| Does any regional result extend to individual cells? | **Blocked.** Visium regional spots cannot answer a cell-level endpoint. |

## Additional frozen sensitivities

### Fixed-reference validation — ESS/type/QC stage complete; state-diversity stage ineligible

Expected machine report:
`/mnt/seagate/HEIR_runs/natcommun_fixed_ess_reference_sensitivity/report.json`.

The frozen natural training distribution was unchanged. At prediction time, matched, wrong, and
generic banks used common Level-1 type support, uniform type mass, a common distribution over joint
binary UMI/n-feature/mitochondrial-QC strata, three soft components per type, exact cell-weight
effective sample size, and donor-equal generic pooling. Weighting was used rather than hard
subsampling. This completed stage does not yet distribution-match within-type state diversity, so
it is a partial implementation of Validation 2.2 rather than the final result.

| Arm | Donor-balanced NB deviance |
| --- | ---: |
| Fixed-support matched M3 | 2.334897 |
| Equal-weight mean of per-wrong-donor M6 losses | 2.504436 |
| Fixed generic M7 | 2.452652 |

| Contrast | Mean improvement | Median | Positive donors | 95% paired interval | Exact one-sided p | Holm p |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| M3 vs equal-weight mean of per-wrong-donor M6 losses | +0.169539 | +0.137416 | 12/13 (92.31%) | [0.101968, 0.237111] | 0.000244 | 0.000244 |
| M3 vs fixed generic M7 | +0.117755 | +0.107697 | 13/13 (100%) | [0.068054, 0.167457] | 0.000122 | 0.000244 |

Both effects were positive in breast, DLBCL, and lung. The wrong-donor effect was negative only for
B1 (`-0.000845`), while the generic-bank effect was positive for every donor. The fixed-support
results closely track the natural-bank effects (`+0.181195` and `+0.119594`), so the prior matched
advantage is not explained solely by available cell count or coarse type coverage.

**Interim decision:** the exposed NatCommun cohort gives partial evidence consistent with
matched-bank separation after the implemented ESS/type/QC controls. This does not yet exclude
within-type state-diversity as an alternative bank explanation and is not full Validation 2.2. It
remains a regional, outcome-exposed sensitivity rather than independent confirmation. Training
remains naturally imbalanced; DV200 and block age are donor-constant, percent-ribosomal metadata are
uninformative, and the registered reference is annotated as cell but is not independently verified
snRNA.

The first clean attempt stopped before target access on one D3 diagnostic receipt: exact float64
support mass `0.8999999836705683` had been serialized as float32 `0.8999999761581421`, whose display
rounded to the frozen `0.90` threshold. The validation wrapper now reconstructs and promotes only
that diagnostic receipt to float64, asserts the original ineligible decision, and leaves all rates,
eligibility values, targets, and endpoints unchanged. All 13 donors were then refit cleanly. The
preserved failed resource log and manifest have SHA-256 identities
`a25871a54328e71c9bc6d81107ecb95869d446b120f9301c1d0799febee521cd` and
`812b1df1072582d53b1796db3f92ec3d035a2bf562eb6a667fcd7f2a786ca0be`.

The stricter state-diversity-matched protocol used query-excluded training anchors, natural hard
assignments, common joint type-by-state-by-QC support, componentwise ESS, donor-equal generic
pooling, and frozen minimum support/coverage/concentration gates. Its runner and protocol SHA-256
identities are `6b73f418be4261e6c783481fae56d73138d44514eec15c709ea4d1524570233c`
and `097803da2f84e2f1e93aaaf3f2bc6c95e5be3de9844929c5524964b4c904e2f5`.
Preparation completed for all 13 donors in 2:24 with 1,466,348 KiB peak RAM and zero swaps; score
targets remained sealed. During B1 fit/predict, no type satisfied the complete frozen
natural-state/joint-QC/exact-ESS contract. The runner stopped with
`fold has no evaluable naturally supported state type` after 5:52, 2,578,272 KiB peak RAM, zero
swaps, and exit status 1. It wrote zero prediction artifacts, zero fit receipts, zero score reports,
and no aggregate report. The fit resource log SHA-256 is
`7631ee9ea3fcf70b605a81e45304d80644cc641e9faa55ac4f1b8b2d9d774eb6`.
The runner computes per-type K=3/K=2 failure reasons in memory but does not serialize them before the
all-ineligible exception, so the evidence cannot identify a specific biological, joint-QC, ESS, or
weighting subgate as the cause.

**State-diversity decision:** full Validation 2.2 is not estimable in the registered cohort under
the prespecified natural-state/joint-QC/exact-ESS rules. Skipping B1, reducing support thresholds,
or forcing equal state occupancy after observing this failure would change the frozen
cohort/protocol. The correct conclusion is therefore not that personalization passed or failed,
but that the available cohort cannot support identification under the prespecified full
state-diversity eligibility contract. The earlier ESS/type/QC result remains partial evidence only.

### Fold-local gene-panel sensitivity — complete

Machine report:
`/mnt/seagate/HEIR_runs/natcommun_fold_local_panel_sensitivity_v3/report.json`.

Each held-out donor used a 256-gene panel selected from the other 12 donors only. The 13 panels
covered a 358-gene union, with 119 genes shared by every fold. The architecture, H-optimus-1
encoder, 20-dimensional latent, loss, schedule, and fusion rule were unchanged. All 13 donors were
freshly prepared and refit exactly once; prediction artifacts were globally preflighted before the
first score target opened.

| Fold-local arm | Donor-balanced NB deviance |
| --- | ---: |
| M0 | 2.080895 |
| M1 | **1.807645** |
| M2 | 1.902579 |
| M3 | 1.904063 |

| Fold-local contrast | Mean effect | Positive donors | 95% paired bootstrap interval | Exact one-sided p |
| --- | ---: | ---: | --- | ---: |
| M0 - M3 | +0.176833 | 13/13 | [0.110297, 0.270174] | 0.000122 |
| M1 - M3 | -0.096418 | 3/13 | [-0.195059, 0.008936] | 0.959717 |

Within-donor M3-versus-M0 relative reduction averaged 7.8898% (median 7.2791%); all 13 donors were
positive, with a paired bootstrap interval of 5.5756% to 10.8295%. Compared descriptively with the
prespecified original 8.1096% donor-balanced aggregate gain, 7.8898% is 97.29%; these are not
identical estimands. The fold-local donor-balanced aggregate ratio is 8.4979% (104.79% of the
original), but aggregate absolute losses are descriptive because fold gene axes differ.
M3-versus-M1 again points against fusion.

The clean runner SHA-256 is
`35f5e270b544a196210eb2e8bc65c255b777c97122dc91a21bd2c40c623e8726`;
the fit manifest SHA-256 is
`7a9ba91d90d9da157a579fba238942e9b37de1b150413a05ffce10cdf541eeaf`.
Preparation took 2:14 with 3,759,060 KiB peak RAM; all 13 refits took 1:40:42 with 6,778,072 KiB
peak RAM; scoring took 7:27 with 1,547,844 KiB peak RAM. Every stage recorded zero swaps and exit
status 0.

One D1 fold-local panel contains zero proliferation genes. The frozen trainer therefore emitted a
NaN threshold for that inactive diagnostic program while its generic artifact validator required
all thresholds finite. An independently audited wrapper stores a zero sentinel only for inactive,
zero-gene programs; scoring indexes active programs only and the primary M0-M3 rates are unchanged.
Only D1 has one inactive program; the other 12 folds have none. The failed `_v2` run is preserved as
`/mnt/seagate/HEIR_runs/natcommun_fold_local_panel_sensitivity_v2_failed_D1_inactive_program_nan`;
no old preparation or prediction was reused.

**Fold-local decision:** the incremental reference-over-H&E direction is robust to replacing the
global panel with training-only fold-local panels, but the central H&E-over-reference synergy
remains absent. This outcome-exposed sensitivity is not confirmation.

### Guarded same-section ST upper bound — complete

Expected aggregate location after scoring:
`/mnt/seagate/HEIR_runs/natcommun_matched_st_validation/same_section_upper_bound_report.json`.

Each guard has 15 section directions from 13 donors. B1 and L1 each contribute two sections, which
are averaged within donor before donor-level inference. Reference and query tails are disjoint and
separated by the frozen physical guard.

| Guard | S0 | S1 | S3 | S4 | S6 | S7 | M1 | M3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 mm | 2.465333 | **1.831164** | 2.020663 | 2.172835 | 2.508603 | 2.418833 | 2.182793 | 2.262358 |
| 2 mm | 2.417500 | **1.830794** | 1.998668 | 2.177939 | 2.472105 | 2.370127 | 2.158569 | 2.225248 |

| Contrast, positive favors S3 or ST reference | 1 mm effect / donors / p | 2 mm effect / donors / p |
| --- | --- | --- |
| S1 - S3, H&E conditional value | -0.189500; 3/13; 0.991455 | -0.167874; 3/13; 0.987061 |
| S4 - S3, exact image pairing | +0.152172; 13/13; 0.000122 | +0.179271; 13/13; 0.000122 |
| S0 - S3, incremental ST-reference value | +0.444670; 13/13; 0.000122 | +0.418832; 13/13; 0.000122 |
| S6 - S3, natural wrong-donor control | +0.487940; 13/13; 0.000122 | +0.473437; 13/13; 0.000122 |
| S7 - S3, natural pooled control | +0.398170; 13/13; 0.000122 | +0.371460; 13/13; 0.000122 |
| M1 - S1, reference-only cross-assay penalty | +0.351629; 13/13; 0.000122 | +0.327776; 13/13; 0.000122 |
| M3 - S3, fusion cross-assay penalty | +0.241695; 13/13; 0.000122 | +0.226581; 13/13; 0.000122 |

**Decision:** even this optimistic same-section, same-assay, same-batch upper bound does not show H&E
value conditional on a strong ST reference. Correct image pairing is consistently informative, but
the frozen correction worsens S1 at both guards. Same-assay references outperform suspension
references in every donor. S6 and S7 remain natural unmatched-support controls, so their favorable
results cannot establish personalization. This is neither independent confirmation, a technical
replicate, a measurement floor, nor cell-level evidence.

## Frozen stopping rules and present status

| Frozen outcome | Required scientific decision | Present status |
| --- | --- | --- |
| M3 fails to beat M0 independently | Stop: reference-fusion effect did not replicate | Independent test not available |
| M3 beats M0 but not M1 | Reference-driven imputation, not H&E–reference synergy | Matches exposed development pattern |
| M3 beats M1 but not M4 | Fusion does not depend on correct pairing | Not reached; M3 fails M1 |
| M3 beats M0/M1/M4 but not M2 | Composition/type information, not continuous state | Not reached; M1 and state requirements fail |
| S3 beats S1 while M3 fails M1 | Suspension-to-ST gap likely limiting | Not observed; S3 also fails S1 |
| Neither S3 beats S1 nor M3 beats M1 | No evidence that H&E adds beyond a strong reference in the tested design | Observed in adjacent ST and both guarded same-section upper bounds; still non-confirmatory |
| Matched fails fixed-support wrong/generic controls | No personalized reference evidence | Not observed under partial ESS/type/QC controls; full state-diversity control is not estimable because no B1 type satisfies the complete natural-state/joint-QC/exact-ESS contract |
| Mean loss improves while variance/calibration collapses | Denoised conditional mean only; no state-reconstruction claim | Observed variance compression; state claim stopped |
| Regional passes but cell-level fails | Restrict claim to regional inference | Neither independent regional nor cell test exists |
| Measurement benchmark is not below fusion | Do not claim the left side of the hierarchy | Current M8 fails; valid floor remains blocked |
| All independent regional and cell gates pass | Scientific hypothesis supported; only then reconsider later methodology | Not reached |

Iterative refinement remains excluded. Its entry condition is an independent one-step result with
`L(M3) < L(M0), L(M1), L(M2), L(M4)` and acceptable molecular-state preservation. The present
evidence fails that condition, so no iteration result should be generated or interpreted.
