# Development refinement redesign

The committed snPATHO v0.2 result is a one-pass personalized benchmark. It did
not run refinement, its primary spatial correlation endpoint was negative, and
it remains the immutable historical result.

The current development refiner addresses the code-level audit findings:

- UOT cost is evaluated on the same latent mean decoded into expression and
  uses image plus prototype diagonal covariance.
- New v3 checkpoints may use a learned local query to form candidate prototype
  probabilities, but the UOT cost is recomputed on the final decoded latent.
  Legacy v1/v2 query and unrestricted-residual behaviors remain loadable only for
  exact checkpoint compatibility.
- Known prototype probabilities are normalized conditional on not-unknown, so
  unknown confidence cannot shrink the molecular latent toward zero.
- Sample dustbin mass is prespecified and fixed by default (0.05, to be
  calibrated only on development data) or derived from explicit
  `unknown_targets`. Detached model-estimated mass is available only as an explicit
  sensitivity because it otherwise feeds the unknown head back into its own UOT
  target.
- UOT uses a numerical convergence tolerance with a bounded iteration budget;
  a nonconverged molecular E-step is rejected rather than silently reused.
- The EMA teacher's UOT plan is detached and normalized by complete row mass,
  including the dustbin. The resulting known-state subprobabilities supervise
  molecular routing, cell type, latent mean, marker, program, and frozen-teacher
  objectives in the M-step while preserving unassigned mass.
- Trusted fine or parent anchors mask both local prototype routing and UOT.
- Anchors are provisional for one round, trusted after two agreeing rounds,
  challenged by contradictory evidence, relabelled after two contradictory
  rounds, and revoked by technical/model rejection.
- Round 0 is evaluated and snapshotted before the first candidate. A first or
  later candidate beyond the configured validation-loss tolerance restores the
  best student, EMA teacher, prototype prior,
  training batches, and anchor lifecycle immediately; the failed student never
  updates the EMA.
- The default schedule uses two parent-gated rounds (enough to establish trusted
  parent anchors) and then two fine rounds (enough to establish trusted fine
  anchors). Broad rounds train only the parent head and cannot update
  fine-prototype priors or terminate refinement
  before the fine phase.
- Measured prototype priors are fixed by default. Lower measured-prior weights
  are explicitly sensitivity analyses.
- Biological aggregate losses are weighted by detached transported known-state
  row mass. Spot aggregation accepts external RNA-mass weights.
- New checkpoints replace the algebraically cancelling unrestricted residual
  with a zero-initialized type-conditioned low-rank correction. A smooth
  unit-ball projection and sigmoid gate hard-bound deterministic and sampled
  latent corrections.
- Inference intervals sample prototype assignment, prototype covariance, and
  coefficient-space residual uncertainty. They are explicitly conditional on a
  measured known state and are suppressed for abstained cells; finite means are
  retained only for aggregate internal scoring.
- scVI panel output preserves the full-library 10,000 normalization and applies
  only `log1p`; corrected exports carry an explicit v2 normalization contract.

Run development experiments from a prespecified plan:

```bash
conda run -n hne python scripts/run_refinement_development.py \
  --plan configs/refinement_development_plan.example.json
```

The orchestrator accepts only `development` or `development_validation` roles
and enforces this order:

```text
train
→ predict_round_0
→ refine_round_1 → predict_round_1
→ ...
→ development_spatial_evaluation
```

Before an `--execute` run, replace every example command and output with the
cohort-specific arguments and freeze the plan in version control.

## Remaining requirements

This refactor does not supply missing biological evidence. A publishable result
still requires a disjoint generic H&E-to-molecular initializer, donor-held-out
scVI/scANVI validation, at least three development seeds, cell-resolution or
reliably registered spatial truth, a matched-refined versus matched-one-pass
contrast, and a new untouched final cohort. The broad E-step still gates a fine
prototype bank rather than transporting against moment-matched parent
Gaussians. The simple mean-aggregation graph, calibrated segmentation/OOD
semantics, biological RNA-mass calibration, and parameter-ensemble uncertainty
also remain development targets.
