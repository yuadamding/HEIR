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
  with a zero-initialized type-conditioned low-rank correction. Primary runs
  freeze deterministic within-type RNA PCA directions and calibrate a separate
  bound for each type from measured state separation/covariance. A smooth
  unit-ball projection and sigmoid gate hard-bound deterministic and sampled
  latent corrections. The coefficient log-variance head starts at `-6`.
- Inference intervals sample prototype assignment, prototype covariance, and
  coefficient-space residual uncertainty. They are explicitly conditional on a
  measured known state and are suppressed for abstained cells. The v8 artifact
  has an explicit mean-availability mask: public cell means fail closed while
  finite internal means remain available only through the aggregate accessor.
- scVI panel output preserves the full-library 10,000 normalization and applies
  only `log1p`; corrected exports carry an explicit v2 normalization contract.

Run development experiments from a prespecified plan:

```bash
conda run -n hne python scripts/run_refinement_development.py \
  --plan configs/refinement_development_plan.example.json
```

Build and freeze the RNA residual geometry before primary training:

```bash
conda run -n hne heir fit-residual-geometry \
  --reference reference_with_latent.npz \
  --prototypes prototypes.npz \
  --rank 4 \
  --output residual_geometry.npz
```

The full snPATHO development matrix is resumable and CUDA-first:

```bash
conda run -n hne python scripts/prepare_snpatho_refinement_inputs.py --execute

conda run -n hne python scripts/run_snpatho_refinement_benchmark.py \
  --sample all --execute --controls \
  --manifest-output reports/snpatho_refinement_v1_five_seed_manifest.json
```

The preparation step consumes the hash-bound native-scANVI provenance, frozen
histology splits, and histology-only calibrated OOD artifacts. It creates the
rare-complete prototypes, RNA residual geometry, and train/validation batches
through the public HEIR CLI. Per-stage receipts resume interrupted work and a
compact final manifest binds every input and output hash. Pre-existing outputs
without receipts fail closed; `--adopt-existing` accepts them only after the
frozen CLI recipe reproduces byte-identical artifacts.

The runner freezes five endpoint seeds (17, 41, 89, 131, and 197), round 0 and
round 4, and three control seeds (17, 41, and 89) for prototype-only,
image-record-shuffle, degree-preserving graph-node-shuffle, and no-graph
controls. Wrong-donor coverage requires both alternative donors for every
specimen: six directed target/source pairings at each control seed, or 18 cases
in total. Under `heir predict --wrong-donor-control`, source prototypes are
deterministically intersected with the target checkpoint ontology in memory,
with at least two retained types/prototypes required. The PredictionBundle
remains hash-bound to the full source bank, while telemetry records the exact
retained/omitted types and counts. Matched predictions do not permit this
filtering.

Score the complete development matrix, then derive the compact public summary:

```bash
PYTHONPATH=src conda run -n hne python \
  scripts/benchmark_snpatho_refinement_matrix.py \
  --run-manifest reports/snpatho_refinement_v1_five_seed_manifest.json \
  --evidence-manifest reports/snpatho_refinement_matrix_evidence_v1.json \
  --json-output artifacts/snpatho/refinement_matrix_v1/report.json \
  --tsv-output artifacts/snpatho/refinement_matrix_v1/report.tsv \
  --markdown-output artifacts/snpatho/refinement_matrix_v1/report.md

PYTHONPATH=src conda run -n hne python \
  scripts/summarize_snpatho_benchmarks.py \
  --full-json artifacts/snpatho/refinement_matrix_v1/report.json \
  --full-tsv artifacts/snpatho/refinement_matrix_v1/report.tsv \
  --full-markdown artifacts/snpatho/refinement_matrix_v1/report.md \
  --output reports/snpatho_refinement_matrix_v1_summary.json
```

The current scorer consumed 93 of 93 requested artifacts. Strict ordering
failed with 49 passing and 59 failing checks, so the public summary status is
`blocked_evidence`. The remaining gaps are a clean independent reannotation,
generic-atlas control, label permutation, state omission, reference
downsampling, and an untouched external cohort. The fixed unknown-mass sweep is
blocked because 14 of 15 cases lack checkpoint-serialized mass provenance; a
post-hoc command/output manifest does not prove the settings used at execution.
The v2 run manifest binds every input/output hash and canonical command across
138 stages. Its final pass conservatively records all outputs as adopted, so it
records `execution_provenance_verified=false`: current validation proves
recipe/output consistency, source-bound CLI identity, and every shuffle-map
hash, but not the original executing source revision for adopted outputs.
Full-primary gating fails closed on that distinction.
The full per-gene JSON, TSV, and Markdown outputs stay in ignored `artifacts/`;
the compact hash-bound summary in `reports/` is tracked.

Run the prespecified seed-17 unknown-mass sensitivity separately at fixed masses
0, 0.01, 0.05, 0.10, and 0.20, then evaluate its hash-validated outputs:

```bash
PYTHONPATH=src conda run -n hne python \
  scripts/run_snpatho_refinement_benchmark.py \
  --sample all --unknown-mass-sensitivity --execute \
  --manifest-output artifacts/snpatho/unknown_mass_sensitivity_v1/run_manifest.json

PYTHONPATH=src conda run -n hne python \
  scripts/benchmark_snpatho_unknown_mass.py \
  --json-output artifacts/snpatho/unknown_mass_sensitivity_v1/report.json \
  --tsv-output artifacts/snpatho/unknown_mass_sensitivity_v1/report.tsv \
  --markdown-output artifacts/snpatho/unknown_mass_sensitivity_v1/report.md
```

The evaluator currently validates only the 4411/mass-0.20 pair. The other 14
cases predate checkpoint serialization of `uot_unknown_mass` and
`uot_unknown_mass_mode`, so the full sweep and its cross-mass conclusion remain
**blocked**. See [the evidence report](snpatho_unknown_mass_sensitivity.md).

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

Five development seeds, three-seed controls, and the matched-refined versus
matched-one-pass comparisons are now scored, but their strict ordering failed.
A publishable result still requires the missing evidence listed above,
donor-held-out scVI/scANVI validation, cell-resolution or reliably registered
spatial truth, and a new untouched final cohort. The broad E-step still gates a
fine prototype bank rather than transporting against moment-matched parent
Gaussians. The simple mean-aggregation graph, calibrated segmentation/OOD
semantics, biological RNA-mass calibration, and parameter-ensemble uncertainty
also remain development targets.
