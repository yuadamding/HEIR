# NatCommun generative reference-fusion development results

## Executive decision

The revised 13-donor experiment is complete. It supports one narrow, outcome-exposed development
result:

\[
L(M3_{\mathrm{H\&E+matched\ reference}}) < L(M0_{\mathrm{H\&E}}).
\]

M3 reduced donor-balanced held-out negative-binomial (NB) deviance from `2.528094` to `2.323076`,
an `8.1096%` relative improvement. The effect favored M3 in `13/13` donors, its paired-donor
bootstrap interval was `[0.100829, 0.333460]` NB-deviance units, and the exact one-sided sign-flip
test gave `p=0.000122`.

The proposed full hierarchy is **not supported**. Reference-only M1 was better than fusion M3
(`2.231896` versus `2.323076`), conditional continuous-state fusion did not significantly beat
matched type routing, the reliability-adjusted variance guard failed, and the empirical molecular
oracle M8 did not beat M3 when both were scored against the same split-half target. Therefore this
run does not establish H&E--reference synergy, continuous within-type state inference, a molecular
floor, donor personalization, regional authorization, cell-level biology, or independent
confirmation.

The correct status is:

- Gate 1 incremental-reference development signal: **passed**.
- Full model chain and quality-preserving development candidate: **failed**.
- Candidate for external preregistration without further model revision: **no**.
- Scientific confirmation or cell-level HEIR authorization: **no**.

## Experiment identity and scope

| Item | Frozen value |
| --- | --- |
| Analysis scope | Exposed NatCommun development; non-confirmatory |
| Biological donors | 13 (`B1`, `B3`, `B4`, `D1`--`D6`, `L1`--`L4`) |
| Indications | Breast, DLBCL, lung |
| Eligible Visium spots | 38,945 |
| Primary scored spots | 38,906; 39 zero-depth spots excluded identically from all arms |
| Scored sections | 15 |
| Reference profiles | 86,306 registered suspension profiles |
| Reference assay qualification | Source metadata say `cell`; not verified snRNA |
| Image representation | Frozen 112-µm `bioptimus/H-optimus-1` embeddings |
| UNI2-h | Forbidden and not run |
| Molecular endpoint | Frozen 256-gene raw-count panel |
| Outer validation | Leave one biological donor out; every section from a donor stays together |
| Primary endpoint | Donor-balanced held-out NB deviance; lower is better |
| Frozen fit | seed 1729, 80 epochs, batch 256, latent dimension 20, CUDA `cuda:0` |

NatCommun outcomes had already influenced the panel and architecture, and the archived v1 result
informed v2. Donor holdout prevents within-run target leakage but cannot restore cohort independence.

## Primary model ladder

The following values use full held-out targets and equal donor weighting. M8 is intentionally absent
because it has a different split-half target and is reported separately.

| Arm | Scientific role | NB deviance ↓ | Plug-in NB log likelihood ↑ |
| --- | --- | ---: | ---: |
| M0 | H&E-only generative baseline | 2.528094 | -2.217843 |
| M1 | Matched reference only | **2.231896** | **-2.069745** |
| M2 | H&E composition plus matched type means; mixed support | 2.357673 | -2.132633 |
| M3 | H&E composition/state plus matched multistate reference | 2.323076 | -2.115334 |
| M4 | Composition-stratified, within-section deranged H&E plus reference | 2.526250 | -2.216922 |
| M5 blank | Blank image control plus reference | 4.754346 | -3.330970 |
| M5 coordinates | Coordinate-only control plus reference | 3.308558 | -2.608075 |
| M6 | Natural same-indication wrong-donor reference | 2.504271 | -2.205932 |
| M7 | Query-excluded generic same-indication reference | 2.442669 | -2.175131 |
| BLEEP-style | Compact contrastive retrieval comparator | 2.546020 | -2.226806 |

M1 is the best full-target arm. M3 beating M0 therefore cannot be interpreted as demonstrated
multimodal synergy: the matched reference supplies useful predictive information, but the current
fusion does not use H&E well enough to beat the reference alone.

The indication-equal M0 and M3 deviances were `2.635544` and `2.400428`, respectively. Gate-1
relative gains remained positive in every indication: breast `10.3492%`, DLBCL `4.1756%`, and lung
`11.6566%`; hence there was no prespecified severe indication reversal.

## Ordered gates

All effects below are `control deviance - M3 deviance`, so positive values favor M3. Donor is the
inference unit.

| Gate/comparison | Mean effect | Donors favoring proposed arm | 95% paired bootstrap interval | Exact one-sided p | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Gate 1: M3 vs M0 | 0.205018 | 13/13 | [0.100829, 0.333460] | 0.000122 | **Passed**; 8.1096% relative gain |
| Gate 2a: M3 vs M1 | -0.091179 | 3/13 | [-0.255664, 0.095666] | 0.828003 | **Failed** |
| Gate 2b: M3 vs M4 | 0.203175 | 13/13 | [0.103118, 0.319240] | 0.000122 | Passed individually; Holm p=0.000244 |
| Gate 3: M3-supported vs M2-supported | 0.113568 | 6/13 | [-0.125923, 0.407671] | 0.230591 | Evaluable, not reached, and would fail |
| Gate 4a: M3 vs natural M6 | 0.181195 | 12/13 | [0.109534, 0.253514] | 0.000366 | Not reached; attribution restricted |
| Gate 4b: M3 vs generic M7 | 0.119594 | 13/13 | [0.073599, 0.175680] | 0.000122 | Not reached; attribution restricted |
| Descriptive: M3 vs BLEEP-style | 0.222944 | 8/13 | [-0.038684, 0.528258] | 0.091187 | Did not meet 70% consistency |
| Gate 5: M8 vs same-target M3 | -0.067719 | 8/13 | [-0.233640, 0.060586] | 0.750000 | Failed; M8 did not establish headroom |

Gate 2 is a two-comparison Holm family. It fails because M3 does not beat M1, even though exact H&E
pairing is informative relative to M4. Sequential testing stops scientific attribution at Gate 2.
Gate 3 and the M8 diagnostic were nevertheless computed as prespecified diagnostics; neither is
positive. Gate 4's nominal natural-bank separation cannot be called donor personalization because
the fixed-effective-sample-size/type-support wrong-bank sensitivity was not run.

## Donor-level primary results

| Donor | Indication | Spots | M0 | M1 | M3 | M3 relative gain over M0 | M1 − M3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B1 | Breast | 4,381 | 4.133983 | 3.407834 | 3.681536 | 10.945% | -0.273701 |
| B3 | Breast | 2,604 | 3.664767 | 2.847372 | 3.246904 | 11.402% | -0.399532 |
| B4 | Breast | 2,110 | 2.302433 | 1.951721 | 2.127348 | 7.604% | -0.175627 |
| D1 | DLBCL | 4,957 | 3.355698 | 2.574791 | 3.145696 | 6.258% | -0.570905 |
| D2 | DLBCL | 4,713 | 1.727564 | 1.627783 | 1.723299 | 0.247% | -0.095516 |
| D3 | DLBCL | 4,966 | 2.568413 | 2.390916 | 2.437495 | 5.097% | -0.046579 |
| D4 | DLBCL | 1,778 | 2.272115 | 2.841345 | 2.176512 | 4.208% | 0.664834 |
| D5 | DLBCL | 2,340 | 1.600336 | 1.506630 | 1.520935 | 4.962% | -0.014305 |
| D6 | DLBCL | 1,998 | 2.293140 | 2.680972 | 2.236377 | 2.475% | 0.444595 |
| L1 | Lung | 2,150 | 3.374136 | 2.378384 | 2.595543 | 23.075% | -0.217159 |
| L2 | Lung | 996 | 2.176332 | 1.729043 | 1.954131 | 10.210% | -0.225089 |
| L3 | Lung | 2,566 | 1.885294 | 1.500461 | 1.858535 | 1.419% | -0.358074 |
| L4 | Lung | 3,347 | 1.511010 | 1.577400 | 1.495672 | 1.015% | 0.081728 |

All M3-versus-M0 effects are positive, but only D4, D6, and L4 favor M3 over M1. That donor pattern
is the clearest evidence that the current gain is reference-dominated rather than a robust H&E and
reference interaction.

## Conditional continuous-state test

The Gate-3 comparison used identical H&E compositions in M2-supported and M3-supported, renormalized
only over reference types having at least two components. Spots required at least 0.90 supported
composition mass and every section required at least three eligible spots.

- All 13 donors and all sections were evaluable.
- 21,401 of 38,906 scored spots were eligible (`55.01%` pooled; `55.64%` donor-balanced).
- Donor eligibility ranged from `21.84%` to `84.68%`.
- Donor-balanced NB deviance was `2.387788` for M2-supported and `2.274220` for M3-supported.
- The average effect was positive, but only `6/13` donors favored M3-supported; the interval crossed
  zero and `p=0.230591`.
- Indication-mean effects were heterogeneous: breast `-0.084584`, DLBCL `0.322316`, and lung
  `-0.050940`.
- The non-gating, mixed-support full-target M3-versus-M2 contrast was also inconclusive: effect
  `0.034597`, `6/13` donors, interval `[-0.114028, 0.202074]`, and `p=0.348511`.

Thus continuous reference-state fusion beyond type/composition routing is not established. The
training-only donor×type alignment diagnostic used a composition-weighted proxy, not observed spot
cell-type truth, so it cannot support a cell-level statement even though its engineering criterion
passed in every fold.

## Empirical molecular-oracle diagnostic

M8 and M3 were compared against exactly the same held-out split-half target. Ninety-three zero-depth
target halves were excluded only from this diagnostic.

| Model on common half target | Donor-balanced NB deviance ↓ | NB log likelihood ↑ |
| --- | ---: | ---: |
| M3, depth-scaled to the M8 target | **1.427797** | **-1.413068** |
| M8, training-only NB-compatible cross-half predictor | 1.495515 | -1.446927 |

M8 was worse by `0.067719` deviance units on average; its paired interval crossed zero and the
one-sided sign-flip test gave `p=0.75`. Therefore the left side of
`L_molecular-oracle < L_H+R < L_H` is not observed. M8 is an empirical benchmark rather than a
mathematical ST floor, and this source has no registered independent full-depth replicate from which
to estimate full-depth measurement risk.

## Quality and uncertainty diagnostics

| Check | M0 or target | M3 | Frozen margin | Result |
| --- | ---: | ---: | ---: | --- |
| Reliability-adjusted variance | 1.001682 | 0.560042 | M3/M0 ≥ 0.80 | **Failed**; ratio 0.559101 |
| Program covariance relative error | 0.767515 | 0.805972 | M3/M0 ≤ 1.10 | Passed; ratio 1.050105 |
| Rare-program extreme recall | 0.222154 | 0.188637 | M0−M3 ≤ 0.05 | Passed; difference 0.033517 |
| 20D latent MSE | 1.657479 | 1.441901 | Descriptive | Improved |
| Gene correlation | 0.089143 | 0.094869 | Descriptive | Slightly improved |
| Program correlation | 0.135830 | 0.136071 | Descriptive | Essentially unchanged |
| Calibration slope | 0.176208 | 0.249716 | Descriptive; ideal 1 | Under-dispersed |
| Approximate posterior-predictive NB log score | -1.961486 | -1.942522 | M3 > M0 | Passed, nonblocking approximation |
| 50/80/95% approximate interval coverage | — | 0.5759 / 0.8366 / 0.9130 | Frozen tolerances | Passed, exploratory |

The overall quality guard fails because M3 preserves only `55.91%` of M0's reliability-adjusted
variance, below the frozen `80%` margin. The result therefore retains the earlier scientific concern:
improved mean loss accompanies excessive molecular compression.

The matched-reference labels were all recognized, but donor-balanced model-vocabulary type support
was `0.6838`, H&E-composition-weighted state coverage was `0.7960`, and the exploratory out-of-support
flag rate was `0.5642`. The latter two miss their uncalibrated 0.90/0.20 development thresholds but
are nonblocking diagnostics. The flag is not operational abstention: no prediction or scoring row
was withheld.

## Cross-assay alignment and retrieval controls

The revised scale-normalized alignment was applied on every molecular minibatch and passed its
training-only criterion in all 13 folds:

| Diagnostic, donor-balanced mean | Aligned model | Same-seed λ=0 comparator |
| --- | ---: | ---: |
| Global matched/mismatched ratio after training | 0.038821 | 0.330555 |
| Global ratio before aligned training | 0.195605 | — |
| Donor×type composition-proxy matched/mismatched ratio | 0.465755 | 0.499536 |

This establishes that v2 corrected the v1 optimization defect; it does not by itself establish a
biological shared latent. The donor×type metric is based on training-only composition proxies.

BLEEP hard negatives used the exact same-indication and dominant-composition stratum for `97.48%`
of training queries on average, a same-indication fallback for `2.52%`, and the global-emergency
fallback for `0%` in every fold. M3's average advantage over the BLEEP-style comparator was not
statistically decisive (`p=0.0912`, 8/13 donors).

## Leakage and artifact audit

The execution was physically staged:

1. `prepare` wrote separate public fit/predict bundles and score-only targets.
2. `fit-predict` produced all 13 predictions without opening held-out ST; every fold receipt records
   `heldout_ST_opened: false`.
3. Before any target was opened, `score` globally revalidated all prediction schemas, semantic
   hashes, frozen identities, and the absence of target fields. It then revalidated each donor
   immediately before opening that donor's score target.

All 13 predictions use schema `heir.natcommun_generative_predictions.v2`; every alignment and
donor×type proxy support check passed; every BLEEP global-emergency fraction was zero. The aggregate
report uses schema `heir.natcommun_generative_development_report.v2`.

| Artifact | SHA-256 |
| --- | --- |
| Registered source | `ec37d5717a9b737dfac226ae9267258fb728ee024496a7655bb69a913aa3cf20` |
| Projected 256-gene source | `71479f891b5945762e20ec5b91d85bac097230b12ed9192aeacd965be119607f` |
| Gene-panel configuration | `ce0b6b82440d7fccc69f24afccf0c68bb101b85f590e4a35a514309929fbb6ad` |
| Protocol configuration | `2cb92b22b6870488a06e64b213e37ffbbdfe3044f1da8fc7442f506915e78197` |
| Benchmark runner | `cf27504e25dfd8cd7e8bfe2894efc8b4a8f79306b47bc492d0e61406d20668ce` |
| Generative core | `55a63f1360e8cc76267e4b00ba8e2167f36259789e9bfdf2aa929c8cadd83b17` |
| Fit/predict manifest | `cb7ebdf9e22090a046937204993a7b2aa3ac1ba2d4c434883e43ed45d1e826ca` |
| Final machine-readable report | `bf3144cf22405752488509dbb1a65b573967fe1b14110881020787187828cf29` |

The complete machine-readable report is outside Git at
`/mnt/seagate/HEIR_runs/natcommun_generative_development/report.json`. The scientifically invalid
under-aligned v1 run remains separate under `revisions/underaligned_v1` and was not pooled with v2.

## Resource record

- Panel projection: peak RSS 2.17 GiB, below its frozen 6-GiB preparation ceiling.
- Final target-separated preparation: 1 minute 43 seconds; no swap.
- Full serial 13-fold fit/predict: 1 hour 21 minutes 49 seconds; peak RSS 6,709,516 KiB (6.40 GiB),
  average CPU occupancy 327%, no swap, one fold at a time, CUDA allocation capped at 60% of the
  10-GiB device.
- Score-only stage: 46.68 seconds; peak RSS 1,564,968 KiB (1.49 GiB), no swap.

The protocol's 6-GiB RSS ceiling applies specifically to panel preparation, not neural fitting.
Training exceeded 6 GiB transiently but left substantial system memory available and did not swap.

## Scientific interpretation and next boundary

The experiment shows that matched-reference information can improve an H&E-only regional predictor,
and that exact image pairing contains signal relative to deranged images. It does **not** show that
the current fusion is better than the reference alone, that it recovers continuous within-type
state, or that it preserves enough reliable molecular variation. The parsimonious interpretation is
reference-driven prediction with a smaller exact-image contribution, not validated multimodal
synergy.

No result here can authorize cell annotation because the outcome is regional Visium and the
reference is not verified snRNA. No result is confirmatory because the cohort, panel, and architecture
are outcome-exposed. Before a claim progresses, the model needs revision and a new frozen evaluation
that at minimum passes M3 versus M0, M1, M4, and supported M2 together with the variance guard.
Independent confirmation then requires a pristine matched H&E + spatial-count + independently
qualified sc/snRNA cohort.

The following prespecified sensitivities remain unrun and cannot be silently inferred from this
central experiment:

- registered 55-µm H-optimus-1 localization, unavailable in the source;
- fold-local LODO-panel model reruns;
- soft composition-weighted reference sensitivity;
- fixed-ESS/type-support wrong-reference sensitivity; and
- a full-depth measurement-noise estimate from registered molecular replicates.
