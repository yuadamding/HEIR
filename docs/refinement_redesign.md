# Development refinement redesign

The committed snPATHO v0.2 result is a one-pass personalized benchmark. It did
not run refinement, its primary spatial correlation endpoint was negative, and
it remains the immutable historical result.

The current development refiner addresses the code-level audit findings:

- UOT cost is evaluated on the same latent mean decoded into expression and
  uses image plus prototype diagonal covariance.
- The legacy independent prototype-query module remains loadable for checkpoint
  compatibility but is not used by new routing unless explicitly requested.
- Known prototype probabilities are normalized conditional on not-unknown, so
  unknown confidence cannot shrink the molecular latent toward zero.
- Sample dustbin mass uses calibrated unknown targets or detached predictions
  with Beta-style prior shrinkage instead of an unconditional fixed 5% target.
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
- A degraded round restores the best student, EMA teacher, prototype prior, and
  anchor lifecycle immediately; the failed student never updates the EMA.
- Broad rounds apply parent-derived constraints to molecular transport before
  fine-state rounds.
- Biological aggregate losses and spot aggregation are weighted by known-state
  probability. Spot aggregation accepts external RNA-mass weights.
- Inference intervals sample prototype assignment, prototype covariance, and
  residual uncertainty rather than the residual Gaussian alone.
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
contrast, and a new untouched final cohort. The simple mean-aggregation graph,
free residual width, technical OOD semantics, and learned RNA-mass calibration
also remain development targets.
