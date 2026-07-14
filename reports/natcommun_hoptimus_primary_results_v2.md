# NatCommun H-optimus-1 regional-fusion experiment report v2

## Executive decision

**The registered H-optimus-1 regional matched-reference hypothesis was not supported.** The run
completed successfully, both crop arms were evaluable, and the result is a scientific
`not_supported` decision rather than an execution block. Neither `target_55um` nor
`context_112um` passed the complete registered conjunction of global-Holm comparisons, effect
thresholds, and molecular-quality guardrails.

This result does not justify a regional HEIR research implementation. It cannot authorize
cell-level HEIR development or production/clinical use. The experiment has Visium-spot truth, not
matched cell-resolved spatial truth, so the cell-level molecular-state and spatial-cell-annotation
hypotheses remain untested rather than failed.

UNI2-h was excluded by user instruction. No UNI2-h supplement, preflight, benchmark, result
pooling, rescue analysis, or override was run for this decision.

## Scope and frozen design

The primary observation is a 55-um Visium spot. Frozen H-optimus-1 H&E features are the base
predictor, matched Chromium FLEX snRNA is the molecular reference, and held-out spatial expression
is truth. Evaluation is leave-one-biological-donor-out across 13 primary donors from breast, lung,
and DLBCL. The source contains 41,748 spots across 16 registered sections; B2 is sensitivity-only
and is excluded from the primary folds.

The primary reference has eight deterministic molecular-state prototypes per training donor/type.
The reference latent may use only the applicable fold-training-donor ST calibration; outer
held-out and inner-validation-donor ST outcomes are excluded. One donor/type centroid is a
non-authorizing diagnostic. The two image arms are:

| Arm | H&E field | Permitted interpretation |
|---|---|---|
| `target_55um` | Registered 112-um canvas whitened outside the centered 55-um spot | Target-matched regional signal |
| `context_112um` | Natural registered 112-um field | Spot plus immediate architecture |

Each arm contains four primary experiments: program and PCA endpoints crossed with natural and
composition-equalized reference banks. Every primary experiment must pass all eight paired M3
comparisons after one global Holm correction over 64 tests, at least 5% M3-versus-M0 relative MSE
gain, at least 70% positive donors, at least two positive indications without a severe reversal,
and every registered molecular-quality guard. M8 is a secondary, nonblocking measurement-floor
diagnostic.

## Execution and artifact receipts

The amended run started from a clean worktree at commit
`69c1865daeb3b1eaec8d05f5b3ecad646a889e3b`, used deterministic CUDA on an NVIDIA GeForce RTX 3080,
disabled CPU fallback, and bounded CPU libraries to four threads. It exited with code 0 on
2026-07-14 and wrote all 10 atomic checkpoints, `report.json`, and `report.md`.

| Artifact | SHA-256 |
|---|---|
| Registered source `source.npz` | `ec37d5717a9b737dfac226ae9267258fb728ee024496a7655bb69a913aa3cf20` |
| Registration review | `93def2b69a809374df6e8c84d75c58639bc7676f3e417386c93b9bb8182eca3a` |
| 55-um H-optimus supplement | `4c5a71cb5504dded8283e8f8be541454ef4526e0eafa2624808b65ca7426941e` |
| Amended H-optimus preflight | `756f1d8e1d549f3359bf8ef8b903353c8a93c0603c41042c914e9316a31dad96` |
| Frozen v2 protocol JSON | `17ff9ee61f73507ffb903a179a75b30fbd2cf230502ad21929b044ffd6af02c9` |
| Frozen v2 runner | `64d23f9f86054d628e225450e5b01055713671c3f1a717bbd4648152715988e2` |
| Frozen v2 reference implementation | `894ad3de42cc3d253aee400cb17cd7ff2257c2aca016601d7eb0f2bc2136bf3a` |
| Final `report.json` | `81f8c0e703e54fa1010c096ae4d001e01c9623af5ffa5f0e4e5668acabcb08b4` |
| Final generated `report.md` | `8047fbada530bc2c1106f6327f3c8ccf1325a7f45d75ca3176fa4b5091942b9e` |

The exact H-optimus-1 revision was
`3592cb220dec7a150c5d7813fb56e68bd57473b9`; its checkpoint SHA-256 was
`c4f1e5b457ddf00679626053b0bf2899be6a19c3a04ad191c87ad1cdfd1abfe1`. The final external result
tree is
`/mnt/seagate/HEIR_runs/natcommun_regional_v2/hoptimus_primary_amended` and remains outside Git.

The earlier clean-head attempt stopped before any experiment checkpoint or report because the
original 100-iteration molecular k-means safeguard was too short for exact convergence. Its log is
preserved with SHA-256
`e9e53c9de2a7faaee2fa36b4641b5ca4ce56da4ef2f7e6d1bdb2c83a6f09a18e`. The pre-result amendment
changed only the universal cap to 1,000 and added fail-closed assignment-cycle detection; the
registered model arms, endpoints, effects, multiplicity, and decision rules were unchanged.

## Preflight result

The preflight was recomputed inside the benchmark and passed independently for both crop arms. The
donor-held-out visible indication control used 1,920 outcome-free sampled rows per crop:

| Crop | H&E accuracy | Outer-training-majority baseline | Increment | Gate |
|---|---:|---:|---:|---|
| `target_55um` | 0.8056 | 0.3333 | +0.4722 | pass |
| `context_112um` | 1.0000 | 0.3333 | +0.6667 | pass |

Thus, the negative primary decision is not an encoder-loading, feature-degeneracy, registration,
or visible-control execution failure.

## Primary experiment results

M0 is H&E alone and M3 is H&E plus the matched molecular reference. `Controls` is the number of
registered paired comparisons with positive direction and global Holm-adjusted p-value at most
0.05. `Effect` is the complete gain/donor/indication gate; `Quality` is the complete molecular
variance/coverage/abstention/rare-state gate.

| Experiment | M0 loss | M3 loss | Relative gain | Positive donors | Positive indications | Controls | Effect | Quality |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 55-um program, natural | 0.764190 | 0.752174 | 1.572% | 76.9% | 3/3 | 6/8 | fail | fail |
| 55-um program, equalized | 0.764190 | 0.758267 | 0.775% | 84.6% | 3/3 | 5/8 | fail | fail |
| 55-um PCA, natural | 4.319368 | 4.148594 | 3.954% | 92.3% | 3/3 | 4/8 | fail | fail |
| 55-um PCA, equalized | 4.319368 | 4.208072 | 2.577% | 92.3% | 3/3 | 4/8 | fail | fail |
| 112-um program, natural | 0.711771 | 0.702953 | 1.239% | 76.9% | 3/3 | 5/8 | fail | fail |
| 112-um program, equalized | 0.711771 | 0.706818 | 0.696% | 84.6% | 3/3 | 6/8 | fail | fail |
| 112-um PCA, natural | 3.941747 | 3.727880 | 5.426% | 92.3% | 3/3 | 8/8 | pass | fail |
| 112-um PCA, equalized | 3.941747 | 3.805876 | 3.447% | 92.3% | 3/3 | 7/8 | fail | fail |

All eight point estimates favored M3 over M0, and all eight had positive mean direction in every
indication without a severe reversal. Nevertheless, seven of eight gains were below the frozen 5%
threshold. Only natural-bank 112-um PCA passed both the complete effect gate and all eight paired
comparisons; it still failed the preregistered molecular-variance guard.

## Global-Holm comparison failures

All 64 registered comparison effects had the favorable direction. After one global correction,
26/32 comparisons passed for the 112-um arm and 19/32 passed for the 55-um arm. The following were
the adjusted-p failures; every omitted comparison passed:

| Crop/endpoint/bank | Failed registered comparisons, global Holm-adjusted p |
|---|---|
| 112-um PCA, equalized | M6 matched specificity, 0.3142 |
| 112-um PCA, natural | none |
| 112-um program, equalized | M1 image beyond reference, 0.3867; M2 continuous state beyond type routing, 0.6315 |
| 112-um program, natural | M0 incremental reference, 0.1721; M1, 0.6315; M2, 0.6315 |
| 55-um PCA, equalized | M1, 0.1143; M2, 0.1426; M6, 0.3867; M7 generic specificity, 0.3142 |
| 55-um PCA, natural | M1, 0.1965; M2, 0.1965; M6, 0.1143; M7, 0.09277 |
| 55-um program, equalized | M0, 0.1965; M1, 0.6315; M2, 0.6315 |
| 55-um program, natural | M1, 0.6315; M2, 0.6315 |

This pattern does not support the required conjunction that the matched reference adds reliable
incremental value, H&E adds information beyond reference-only/type routing, pairing matters, and
matched-donor specificity survives all registered endpoints and bank conditions.

Exact pairing and both image-content controls passed in all 8/8 primary experiments. In contrast,
image beyond reference and continuous state beyond type routing each passed only 2/8, making these
the least consistently supported components of the registered hypothesis.

## Molecular-quality guardrails

Thresholds are variance ratio at least 0.5, median type coverage at least 0.5, and abstention at
most 0.5. Program rare-state values are median M0-minus-M3 recall drops and must be at most 0.2;
PCA has no registered rare-state endpoint.

| Experiment | Variance ratio | Type coverage | Abstention | Rare-state drop | Quality decision |
|---|---:|---:|---:|---:|---:|
| 55-um program, natural | 0.218 fail | 0.499 fail | 0.047 pass | 0.0025 pass | fail |
| 55-um program, equalized | 0.237 fail | 0.188 fail | 0.036 pass | 0.0000 pass | fail |
| 55-um PCA, natural | 0.265 fail | 0.508 pass | 0.049 pass | not applicable | fail |
| 55-um PCA, equalized | 0.293 fail | 0.202 fail | 0.032 pass | not applicable | fail |
| 112-um program, natural | 0.243 fail | 0.503 pass | 0.055 pass | 0.0039 pass | fail |
| 112-um program, equalized | 0.251 fail | 0.182 fail | 0.058 pass | 0.0002 pass | fail |
| 112-um PCA, natural | 0.451 fail | 0.514 pass | 0.035 pass | not applicable | fail |
| 112-um PCA, equalized | 0.479 fail | 0.207 fail | 0.041 pass | not applicable | fail |

Every primary experiment failed the molecular-variance preservation rule. All
composition-equalized banks also failed type coverage. Abstention and the applicable rare-state
collapse checks passed throughout. The closest experiment, natural-bank 112-um PCA, preserved
0.451 of within-section variance rather than the required 0.5.

## Secondary diagnostics

### Approximate ST measurement floor

M8 was significantly below M3 in all eight endpoint/bank/crop comparisons, with within-floor-family
Holm-adjusted p = 0.000488 for each. The M3-minus-M8 loss differences ranged from 0.603 to 0.658 for
program endpoints and 3.544 to 4.024 for PCA endpoints. M3 closed only 0.8% to 5.7% of the
approximate M0-to-M8 gap. This indicates substantial residual predictability in the spatial target;
it does not rescue M3 and was never allowed to block or authorize the primary decision.

### One-centroid reference diagnostic

| Crop | Eight-state M3 gain | One-centroid M3 gain | Centroid minus state-aware M3 loss |
|---|---:|---:|---:|
| `target_55um` program, natural | 1.572% | 2.166% | -0.004535 |
| `context_112um` program, natural | 1.239% | 1.976% | -0.005245 |

The simpler centroid was descriptively better in both registered diagnostics. These two experiments
were excluded from the 64-test family and cannot authorize or rescue the primary result, but they
provide no descriptive evidence that eight within-type molecular prototypes improved this
endpoint.

## Independent integrity audit

The final report contains exactly eight primary experiments and two centroid diagnostics. All 64
registered p-values were present; no test was filled as p=1. An independent reconstruction of the
global Holm step-down correction matched all 64 stored adjusted p-values exactly, with maximum
absolute difference 0.0. Recomputing crop support from the stored comparison, effect, and quality
subdecisions reproduced both `not_supported` crop decisions and the overall encoder/regional
decision. The report preserves the 1,000-iteration amendment, exact consecutive repaired-label
convergence, fail-closed nonconsecutive repeats, training-only calibration boundary, nonblocking M8
role, and cell-level prohibition.

All 10 checkpoint file hashes, canonical identity hashes, canonical result hashes, and embedded
report payloads matched. The generated `report.md` was byte-for-byte the expected rendering of
`report.json`. This is an artifact-integrity and decision-logic audit, not an independent numerical
rerun of model inference or endpoint fitting.

## Scientific interpretation and next decision

The experiment provides evidence for weak, broadly positive regional increments, especially in
natural 112-um PCA, but not for the registered matched-reference hypothesis. The gains are small for
reliability-qualified gene programs, matched-reference specificity is inconsistent, and fusion
compresses too much within-section molecular variance. The fact that the one-centroid diagnostic
outperformed the state-aware bank argues against adding more reference-state machinery on the
current evidence.

Accordingly:

- do not advance regional HEIR implementation on this result;
- do not tune thresholds or select a favorable endpoint/crop after seeing this report;
- do not run UNI2-h as a rescue or comparator under the current user-directed scope;
- preserve this result as the completed H-optimus primary decision;
- keep the strict cell-level hypothesis separate and blocked until an independent, matched,
  cell-resolved cohort can test it.
