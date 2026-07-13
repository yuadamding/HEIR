# Development refinement redesign

The committed snPATHO v0.2 result is a one-pass personalized benchmark. It did
not run refinement, its primary spatial correlation endpoint was negative, and
it remains the immutable historical result.

The current development refiner addresses the code-level audit findings:

- UOT cost is evaluated on the same latent mean decoded into expression and
  uses image plus prototype diagonal covariance.
- New v4 checkpoints may use a learned local query to form candidate prototype
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
- Primary refinement consumes a hash-bound E-step artifact produced by an
  independently frozen morphology/cross-modal teacher. The live student type,
  query, and unknown heads never create their own M-step targets, and the direct
  live UOT optimizer term is disabled. The historical self-E-step is retained
  only as an excluded negative control.
- In the strict path, repeated reads of one fixed artifact cannot mint
  pseudo-anchors. In the excluded live-E-step sensitivity, trusted fine or parent
  pseudo-anchors contribute confidence-weighted classification losses but do not
  hard-mask prototype routing or UOT.
- Anchors are provisional for one round, trusted after two agreeing rounds,
  challenged by contradictory evidence, relabelled after two contradictory
  rounds, and revoked by technical/model rejection.
- Round 0 is evaluated and snapshotted before the first candidate. A first or
  later candidate beyond the configured validation-loss tolerance restores the
  best student, round teacher, prototype prior,
  training batches, and anchor lifecycle immediately; the failed student never
  updates the round teacher. Every candidate is compared with the immutable
  round-0 safety ceiling, and output selection keeps the lowest-loss safe state
  with automatic round-0 fallback.
- The default strict schedule uses two fixed-target parent-head rounds and then
  two fixed-target fine rounds. Broad rounds train only the parent head and cannot
  update fine-prototype priors or terminate refinement before the fine phase.
  Parent-gated transport and longitudinal anchor lifecycle behavior apply only to
  the excluded live-E-step sensitivity.
- Measured prototype priors are fixed by default. Lower measured-prior weights
  are explicitly sensitivity analyses.
- Biological aggregate losses are weighted by detached transported known-state
  row mass. Spot aggregation accepts external RNA-mass weights.
- New checkpoints replace the algebraically cancelling unrestricted residual
  with a zero-initialized type-conditioned low-rank correction. Primary runs
  freeze deterministic within-type RNA PCA directions. RNA cells use
  regularized diagonal-Gaussian assignment to same-type prototypes; residual,
  covariance, and nearest-neighbor state-separation scales are calibrated in
  the learned residual subspace. Single-state types fall back to projected
  covariance or empirical residual evidence. A detached maximum-type basis and
  continuous detached posterior-concentration gate replace the non-orthonormal
  weighted basis. This is a numerical safeguard, not the separate independent-label
  broad-type development gate, which must pass before residual-on results are
  interpreted. Then
  a smooth unit-ball projection hard-bounds deterministic and sampled latent
  corrections. The coefficient log-variance head starts at `-6`. Legacy v1
  geometry and early v3 mixed-basis checkpoints require explicit migration.
- The graph encoder and graph loss are off by default. The current
  distance-weighted graph is an explicit ablation with a zero-initialized
  learnable context gate; directional edge conditioning remains future work.
- Pathology feature OOD remains separate from biological unknown-state
  supervision. Target score quantiles are descriptive telemetry only and never
  replace the development-calibrated OOD threshold.
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
conda run -n hne python scripts/prepare_snpatho_refinement_inputs.py \
  --molecular-generation r2 --execute

conda run -n hne python scripts/run_snpatho_refinement_benchmark.py \
  --molecular-generation r2 --sample all --execute --controls \
  --artifact-root /mnt/seagate/HEIR_runs/snpatho_refinement_nested_v2 \
  --prohibit-adoption \
  --manifest-output artifacts/snpatho/refinement_matrix_v2/run_manifest.json
```

R2 is the preparation default and requires the specimen-preserving molecular
run under `artifacts/snpatho/r2_scanvi`; `--molecular-generation r1` is reserved
for reproducing the historical specimen-batch sensitivity. The preparation
step consumes the hash-bound native-scANVI provenance, frozen
histology splits, and histology-only calibrated OOD artifacts. It creates the
rare-complete prototypes, RNA residual geometry, and train/validation batches
through the public HEIR CLI. Per-stage receipts resume interrupted work and a
compact final manifest binds every input and output hash. Pre-existing outputs
without receipts fail closed; `--adopt-existing` accepts them only after the
frozen CLI recipe reproduces byte-identical artifacts.

At execution start, the runner freezes the complete refinement source-tree and
source-bound CLI identities. It revalidates both immediately before and after
every subprocess, around every existing-output adoption, and before writing the
final manifest. The manifest stores the initial execution identity separately
from the final validation identity; a final-tree hash alone cannot set
`original_execution_source_verified`. The long-running molecular trainer uses
the same fail-closed pattern for every H5AD, `reference500.npz`, and gene-panel
input: all are hashed before any read and rehashed before output families and
the provenance manifest.

The prospective molecular leave-one-donor-out path uses one completely separate
R2 family per target. For each of `4066`, `4399`, and `4411`, run
`train_snpatho_scanvi.py --held-out-sample TARGET` with distinct native-model,
decoder, latent-root, and provenance outputs, then run
`prepare_snpatho_refinement_inputs.py` against that fold's explicit scanVI root
and provenance. Supply all three resulting preparation manifests to the runner:

```bash
conda run -n hne python scripts/run_snpatho_refinement_benchmark.py \
  --molecular-generation r2 --sample all --execute --controls \
  --molecular-fold-preparation-manifest 4066=/path/fold_4066/preparation_manifest.json \
  --molecular-fold-preparation-manifest 4399=/path/fold_4399/preparation_manifest.json \
  --molecular-fold-preparation-manifest 4411=/path/fold_4411/preparation_manifest.json \
  --artifact-root /mnt/seagate/HEIR_runs/snpatho_true_loo_v1 \
  --prohibit-adoption \
  --manifest-output artifacts/snpatho/true_loo_v1/run_manifest.json
```

The runner refuses a shared latent space or decoder across folds. Held-out
target labels are removed before query registration and replaced by predictions
from the frozen training-donor scANVI classifier. Those classifier labels still
originate from published annotations on the two training donors, not an
independently reviewed ontology. No real three-fold CUDA artifacts were produced
by this code change, and the current HEIR runner remains an explicitly excluded
uninitialized/live-E-step engineering control until the independent initializer
and frozen E-step exist.

The independently fitted folds do not share a latent space, so a prototype bank
from one fold cannot be evaluated through another fold's checkpoint. The
true-LOO runner therefore omits cross-fold wrong-prototype-bank stages and marks
their coverage unavailable rather than claiming them as completed. Score this
run with the same three fold declarations; the scorer validates each target's
native manifest, decoder, and latent identity and preserves the negative-control
claim scope:

```bash
PYTHONPATH=src conda run -n hne python \
  scripts/benchmark_snpatho_refinement_matrix.py \
  --molecular-generation r2 \
  --artifact-root /mnt/seagate/HEIR_runs/snpatho_true_loo_v1 \
  --run-manifest artifacts/snpatho/true_loo_v1/run_manifest.json \
  --molecular-fold-preparation-manifest 4066=/path/fold_4066/preparation_manifest.json \
  --molecular-fold-preparation-manifest 4399=/path/fold_4399/preparation_manifest.json \
  --molecular-fold-preparation-manifest 4411=/path/fold_4411/preparation_manifest.json \
  --json-output artifacts/snpatho/true_loo_v1/report.json \
  --tsv-output artifacts/snpatho/true_loo_v1/report.tsv \
  --markdown-output artifacts/snpatho/true_loo_v1/report.md
```

Because the default score request includes `wrong_prototype_bank`, the report
above is intentionally `matrix_status: blocked` with
`requested_control_unavailable_cross_latent_space`, even when every available
same-latent artifact scores successfully. This is an incomplete requested
matrix, not a generic complete result. A deliberately reduced diagnostic matrix
can list only available controls with repeated `--control` arguments, but it
does not restore wrong-bank coverage.

The runner freezes five endpoint seeds (17, 41, 89, 131, and 197), round 0 and
round 4, and three control seeds (17, 41, and 89) for the fully nested
round0/refined × residual-on/off design, image-record shuffle,
degree-preserving graph-node shuffle, and no graph. The nested cases use the
same `--prototype-only` inference ablation at each checkpoint, allowing separate
round-zero residual, refined residual, routing-refinement, and total-refinement
effects. In the shared-latent development matrix, wrong-prototype-bank coverage
requires both alternative sources for
every specimen: six directed target/source pairings at each control seed, or 18
cases in total. The legacy CLI flag remains `heir predict --wrong-donor-control`;
source prototypes are
deterministically intersected with the target checkpoint ontology in memory,
with at least two retained types/prototypes required. The PredictionBundle
remains hash-bound to the full source bank, while telemetry records the exact
retained/omitted types and counts. Matched predictions do not permit this
filtering.

Score the complete development matrix, then derive the compact public summary:

```bash
PYTHONPATH=src conda run -n hne python \
  scripts/benchmark_snpatho_refinement_matrix.py \
  --molecular-generation r2 \
  --artifact-root /mnt/seagate/HEIR_runs/snpatho_refinement_nested_v2 \
  --run-manifest artifacts/snpatho/refinement_matrix_v2/run_manifest.json \
  --evidence-manifest reports/snpatho_refinement_matrix_evidence_v1.json \
  --practical-delta-threshold 0.002 \
  --json-output artifacts/snpatho/refinement_matrix_v2/report.json \
  --tsv-output artifacts/snpatho/refinement_matrix_v2/report.tsv \
  --markdown-output artifacts/snpatho/refinement_matrix_v2/report.md

PYTHONPATH=src conda run -n hne python \
  scripts/summarize_snpatho_benchmarks.py \
  --full-json artifacts/snpatho/refinement_matrix_v2/report.json \
  --full-tsv artifacts/snpatho/refinement_matrix_v2/report.tsv \
  --full-markdown artifacts/snpatho/refinement_matrix_v2/report.md \
  --output artifacts/snpatho/refinement_matrix_v2/summary.json
```

The committed 93-artifact report is a historical pre-nested diagnostic. The
hardened plan requests 102 scored predictions across 147 stages, and classifies
paired deltas using a prespecified practical margin of 0.002 as pass, tie, or
fail while retaining raw sign separately. A new clean-root run is required
before the nested decomposition can be interpreted. The remaining gaps are a clean independent reannotation,
generic-atlas control, label permutation, state omission, reference
downsampling, and an untouched external cohort. The historical fixed-mass sweep
was provenance-blocked; the clean replacement completed all 75 source-bound
CUDA stages and 15 cases but is practically unstable and selects no mass.
The historical v2 run manifest binds every input/output hash and canonical command across
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
  --molecular-generation r1 \
  --sample all --unknown-mass-sensitivity --execute \
  --artifact-root /mnt/seagate/HEIR_runs/snpatho_unknown_mass_clean_v2_run4 \
  --prohibit-adoption \
  --manifest-output artifacts/snpatho/unknown_mass_sensitivity_v2/run_manifest.json

PYTHONPATH=src conda run -n hne python \
  scripts/benchmark_snpatho_unknown_mass.py \
  --molecular-generation r1 \
  --artifact-root /mnt/seagate/HEIR_runs/snpatho_unknown_mass_clean_v2_run4 \
  --run-manifest artifacts/snpatho/unknown_mass_sensitivity_v2/run_manifest.json \
  --practical-delta-threshold 0.002 \
  --json-output artifacts/snpatho/unknown_mass_sensitivity_v2/report.json \
  --tsv-output artifacts/snpatho/unknown_mass_sensitivity_v2/report.tsv \
  --markdown-output artifacts/snpatho/unknown_mass_sensitivity_v2/report.md
```

The runner now validates checkpoint-bound fixed-mode mass metadata before it can
classify a stage as `skipped_valid`. Legacy cases fail immediately with guidance
to use a clean output root. `--prohibit-adoption` additionally rejects any
pre-existing planned endpoint before starting the canonical 75-stage grid (3
specimens × 5 masses × 5 stages). The completed result is **unstable**: refined
prediction does not beat round zero and both matched type-mean baselines by the
0.002 practical margin at every mass. No mass is selected. See [the clean v2
report](snpatho_unknown_mass_sensitivity_v2.md) and the [historical provenance
failure](snpatho_unknown_mass_sensitivity.md).
The clean plan reads immutable molecular inputs from the canonical preparation
root, including `residual_geometry_rare_complete_v2.npz`, while writing every
model and prediction endpoint only under the supplied `--artifact-root`.

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
