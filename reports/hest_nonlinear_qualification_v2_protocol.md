# HEST nonlinear estimator qualification v2

**Status:** registered, retrospective, exposed, nonauthorizing, and not yet executed.

This protocol tests one narrow SCHAF-inspired proposition: whether a bounded nonlinear,
type-conditioned model on frozen H-optimus-1 features exposes donor-generalizing molecular
residual information that the full 1,536-dimensional ridge arm B2 does not expose. It does not
implement transcriptome expansion, alignment, optimal transport, a GAN, or encoder fine-tuning.

## Immutable predecessor

V1 remains permanently blocked. V2 inherits its source, B0-N7 ladder, leave-one-donor-out outer
and inner splits, training grid, rank grid `[2, 4, 6]`, molecular target, support-20 hierarchical
R2 endpoints, diagnostics, thresholds, and two complete-refit null families from protocol SHA-256
`656419fcb75919fb938ba5a351014b7b72ba54b077da4c557103796f269d044f`.

Only the missing blank input, sparse registration estimand, algorithm-level controls,
multiplicity, D1/D2 separation, and pre-outcome calibration are changed.

## Outcome-blind inputs

The compact blank supplement contains one real white-patch H-optimus embedding and an explicit
broadcast rule for 36,121 rows. Independent CUDA recomputation reproduced feature SHA-256
`d81926fd76fef69e9e59b8777e068a008716eedbf939b0b85ef1e4018b57df6f`; batch sizes 1 and 32
were exactly equal. The supplement SHA-256 is
`a7495d795c6a73e9a8bfc172448d0d161a99402b0bc63cd1ae6966e1c3968219` and its receipt SHA-256
is `41917b39c0e50f27e252d76a4507e0a7599591e00c09b25074f49b9413bbd962`.

The topology receipt SHA-256 is
`8f619537486e7e5814797551c5e8f22681118f4cfef49a98bf343f6aea170742`. It opened only the
seven allowlisted identity/registration members and no molecular, image, prediction, or report
payload.

| Band | Formal rows | Formal donors | Result |
| --- | ---: | ---: | --- |
| Q1: best | 331 | 6 | Fails donor count, total rows, and 20% donor cap |
| Q2: best + intermediate | 1,874 | 13 | Fails the 20% donor cap (21.72%) |
| Q3: best + intermediate + near-threshold | 3,991 | 14 | Passes; largest donor 19.22% |

Q3 is frozen. VUILD110 is excluded from formal Q3 sensitivity because it has 10 Q3 evaluation
rows, below the prespecified 20-row included-donor minimum. No registration cutoff was changed.

## Algorithms and controls

Inner donor validation selects the primary algorithm `A*` from N1, N2, N6, and N7. A separate
single-view algorithm selects between N1 and N2, then reuses that architecture and all
hyperparameters without retuning for N3, N4, N5, and the single-view blank control.

`C_blank` uses the selected architecture with the real blank embedding in every image branch.
For N7, its real nuisance branch remains present. `C_target_removed` is N5 under the frozen
single-view algorithm. A blank model may learn intercepts and type effects; that is the intended
no-image comparator.

## Registration estimand

For each held-out cell and focal/comparator pair:

```text
g_i = (||z_i - comparator_i||^2 - ||z_i - focal_i||^2) / coordinate_rank
```

Cells are averaged within donor/type, represented types are averaged within donor, and donors
are averaged equally. A donor/type needs five cells for this paired mean-loss sensitivity only;
the primary R2 support remains 20. The all-row estimate reuses exactly the donor/type strata
supported in Q3, preventing composition changes from driving noninferiority.

Q3 minus all rows must be at least `-0.01`, and its one-sided 95% equal-donor bootstrap lower
bound (20,000 resamples; seed 20260721) must also be at least `-0.01`. Best-registration results
remain separately descriptive. Models are never retrained on a registration subset.

## Multiplicity and decisions

An exact donor sign-flip max statistic controls the direct family `A* - B2`, `A* - N0`, and
`A* - C_blank`. A separate family controls N1, N3, and N4 versus N5. The complete-refit image
derangement and different-spatial-block nulls each use 20 permutations for smoke and 100 for the
final run, with one synchronous image mapping across multiview branches.

D1 supports design of a prospective local-context estimator only when every frozen absolute,
incremental, variance, coverage, rare-state, null, multiplicity, and registration criterion
passes. D2 is separate: at least N3 or N4 must beat N5 by 0.01 R2, pass its adjusted test and
registration gate, have positive absolute R2, beat nuisance and blank controls, and D1 must pass.

Neither decision authorizes a biological HEIR hypothesis.

## Calibration and stopping

Before opening `nucleus_molecular_targets`, the exact scoring, multiplicity, registration, and
D1/D2 code must run on synthetic held-out predictions for 200 trials in each of ten frozen
conditions on the real donor/section/type/block/registration topology. The separate implementation
smoke covers nested fitting, checkpoint replay, and complete refits. Exact one-sided
Clopper-Pearson bounds use Bonferroni control across all 20 D1/D2 expectations. False-positive
upper bounds must be below 0.05 and true-positive lower bounds above 0.80.

Calibration failure blocks HEST molecular execution. A revision is allowed only while no HEST
molecular member has been opened under v2. After a passing calibration receipt, a clean committed
runner may open the target once, run all observed arms, registration sensitivity, and both final
100-refit null families, then write one atomic JSON report and one Markdown report.

Even a complete pass can only motivate a new unexposed prospective protocol.
