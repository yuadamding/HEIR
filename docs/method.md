# Method contract

For sample (s), HEIR receives H&E nuclei (\{(x_i,r_i,m_i)\}_{i=1}^{N_s}) and an unregistered matched snRNA reference (\{(y_j,c_j)\}_{j=1}^{M_s}). There is no assumed correspondence (i\leftrightarrow j).

The image graph predicts hierarchical type probabilities, a conditional
known-state prototype distribution (q_{ik}), a transport-unassigned probability
(u_i) that is explicitly distinct from biological-unknown annotations, and a
restricted residual posterior. For current restricted checkpoints, the decoded
molecular mean is

\[
z_i^{\mathrm{proto}}=\sum_k q_{ik}\widetilde\mu_{sk},
\qquad
\Delta z_i=
g_i\delta_{c_i^*}\sigma(h_i)
\frac{B_{c_i^*}a_i}{\sqrt{1+\lVert B_{c_i^*}a_i\rVert_2^2}},
\qquad
z_i=z_i^{\mathrm{proto}}+\Delta z_i,
\]

where (c_i^*=\arg\max_c\operatorname{stopgrad}p_i(c)),
(g_i=\operatorname{sigmoid}((\max_c\operatorname{stopgrad}p_i(c)-\tau)/T)), and each orthonormal (B_c) contains
frozen within-type RNA directions with (\operatorname{rank}(B_c)\le r).
The continuous concentration gate smoothly attenuates molecular residual prediction
when broad type is uncertain, and detachment prevents residual losses from sharpening the type
posterior merely to unlock a correction. Each (\delta_c) is calibrated in the
corresponding RNA residual subspace. The coefficient-mean head is initialized
exactly at zero and every deterministic or sampled residual satisfies
(\lVert\Delta z_i\rVert_2<\delta_{c_i^*}); a fresh deterministic model therefore
equals its routed prototype baseline. Residual activity, effective-zero fraction,
and residual norms are emitted overall and by predicted type. Checkpoint
v1/v2 models retain their
historical unrestricted behavior. Early v3 restricted checkpoints that used a
probability-weighted sum of nonaligned bases are rejected unless the caller
explicitly requests legacy mixed-basis migration.

RNA cells are assigned to same-type prototypes with a regularized diagonal
Gaussian negative log likelihood when prototype variances are available. State
bounds use a low quantile of projected nearest-neighbor prototype separation,
not median all-pairs distance. Covariance and empirical residual scales are also
computed after projection; a type with one prototype therefore falls back to
its projected covariance or projected within-type residual evidence.

Unknown probability therefore changes confidence and loss participation, not the
biological latent by scaling it toward zero.

Prototype means are shrunken toward a training atlas by (n/(n+\kappa)), with (\kappa=50) by default. A frozen RNA decoder jointly maps (z_i) to the selected expression panel. Prototype banks and the decoder carry the same immutable latent-space identifier; matching width alone is not accepted. An optional second image head is aligned to frozen, donor-audited scGPT type prototypes and within-type variances. When morphology is weak, the residual is regularized toward zero, uncertainty rises, and the cell may be left unknown.

## Population alignment

An independently frozen E-step aligns an image-to-latent bridge to prototype
Gaussians with covariance-aware entropic unbalanced transport. A dustbin column absorbs image
objects or states that should remain unassigned. Unlike balanced transport, this
does not force the H&E section to reproduce biased snRNA capture fractions
exactly. The primary path uses a prespecified fixed dustbin mass (default 0.05,
to be calibrated only on development data). Explicit biological
`unknown_targets` are not mapped onto this transport-unassigned variable.
Estimation from the model's own
unknown head is retained only as an explicit sensitivity because it can create
self-reinforcing dustbin assignments. The UOT plan is normalized by
each row's complete transported mass,
including the dustbin, and detached to form known-state subprobabilities. Their
row sum preserves molecular known-state mass; conditional known responsibilities
directly supervise prototype routing, cell type, latent mean, marker, program,
and frozen-teacher objectives in the M-step. Detached transported known-state
row mass, rather than the live unknown-head probability, weights the weak
biological objectives. In the strict M-step the frozen responsibilities are
hash-bound to ordered nucleus/prototype identities, direct live-student UOT is
disabled, and neither the live type, latent, nor unknown head can change its own
target. Live-student transport remains only as the explicitly tagged
`live_student_negative_control` mode.

Pathology feature OOD and biological unknown state are distinct variables. An
`ood_mask` can gate or abstain on out-of-distribution morphology, but it is never
converted into `unknown_targets`; those targets must come from an explicit,
independent development artifact.

The optimized objective combines:

- UOT latent/prototype alignment;
- a low-concentration Dirichlet composition prior (rather than equality to biased snRNA fractions);
- normalized pseudobulk agreement;
- per-cell marker ranking and cell-type-conditioned program consistency;
- frozen scGPT prototype, contrastive, and within-type moment alignment when supplied;
- prototype-covariance residual regularization;
- cycle consistency through a frozen RNA encoder when available;
- boundary-aware graph regularization;
- sparse trusted/pseudo anchor loss;
- hierarchy and unknown calibration.

Every term is implemented independently under `src/heir/losses/` and can be ablated. The graph path and graph loss are off by default until held-out controls support them; `distance_only` is an explicit experimental mode with a learnable context gate initialized at zero. When a specimen is split into graph patches, HEIR forwards each graph independently and aggregates outputs before applying specimen-level composition, pseudobulk, marker, and program losses; it does not force every local patch to reproduce the whole-specimen mixture. Live sample-level UOT is limited to the negative-control mode, while strict runs consume already frozen bag-aligned responsibilities. Generic spatial pretraining merges shared spot IDs across disjoint graph patches, verifies repeated spot targets, and weights its pseudobulk by assigned spot mass. Empty spots and overlapping-nucleus spatial patches are rejected rather than silently creating irreducible or double-counted targets.

Each batch carries a hash-derived `weak_target_scope_id`. Strict personalized
training rejects any scope shared between training and checkpoint-selection
validation, even when the H&E nuclei are spatially disjoint; the historical
same-specimen split is available only in explicitly excluded negative controls.
Checkpoint `training_donors` is transitive exposure provenance: it includes
direct fitting donors, validation/model-selection donors, and every upstream
initializer, decoder, prototype/teacher, and residual-geometry donor. Direct
training and validation scopes are also recorded separately. Exact batch bytes
are hashed before loading and checked again before checkpoint publication.

## Training stages

1. Fit/adapt the RNA teacher on real snRNA counts only.
2. Optionally pretrain the image mapping on decontaminated public H&E–ST samples.
3. Personalize on H&E plus matched snRNA; target ST is prohibited. Personalized
   training requires a passing `heir.validated_initialization.v1` receipt and a
   `heir.molecular_e_step` v3 artifact for every train/validation bag. Random
   morphology initialization is available only as an excluded negative control.

The strict producer sequence is executable and fail-closed:

```bash
conda run -n hne python scripts/validate_initialization_checkpoint.py \
  --plan initialization_validation_plan.json \
  --output initialization_evidence.json --device cpu
conda run -n hne python scripts/create_initialization_receipt.py \
  --checkpoint generic_heir.pt --evidence-report initialization_evidence.json \
  --output initialization_receipt.json
conda run -n hne python scripts/create_molecular_e_step.py \
  --teacher-checkpoint generic_heir.pt \
  --initialization-receipt initialization_receipt.json \
  --histology histology.npz --prototypes prototypes.npz \
  --rna-reference reference.npz --output molecular_e_step.npz --device cuda
```

The validation plan binds the checkpoint and held-out evidence hashes, donor
scope, three or more shuffle seeds, and fail-closed broad-type/image-latent
thresholds. The E-step producer accepts no student checkpoint and constructs
its image latent without passing an RNA prototype bank into the teacher.
4. Before refinement, evaluate and snapshot round 0 (student, round teacher,
   batches, measured priors, and empty anchor state). In the strict path, refine
   against the same immutable E-step for at most five rounds (default four), using
   two parent-head rounds followed by two fine-head rounds. Re-reading that one
   artifact cannot constitute longitudinal confirmation, so strict rounds mint no
   pseudo-anchors. Parent-gated live transport and revocable pseudo-anchors exist
   only in the explicitly excluded live-student-E-step sensitivity. Measured priors
   are fixed by default; updating them requires a new round-specific E-step artifact.
   A first or later candidate whose validation loss
   exceeds the immutable round-0 loss plus `maximum_validation_loss_degradation` restores
   the best complete snapshot immediately. Development cohorts can select a
   round with a frozen spatial scorer; target cohorts use a development-locked
   fixed round count and retain the lowest-loss safe snapshot, automatically
   falling back to round 0 when no candidate improves it. Same-checkpoint scale
   views are diagnostics by default, not independent anchor evidence.
   The frozen-spatial-scorer path is currently an injected programmatic API;
   the production CLI does not yet construct that scorer, so a CLI fixed-round
   run must not be described as spatially selected.
5. Calibrate on development donors.
6. Open locked spatial measurements once the checkpoint and thresholds are frozen.

The distilled H&E-only student is a separate model and evaluation setting.

## Oracle ladder

`scripts/benchmark_oracle_ladder.py` scores decoder reconstruction, oracle
type mean, oracle type/prototype, predicted type with oracle state, oracle type
with predicted state from an explicit same-checkpoint oracle-type-conditioned
forward pass, an explicit same-checkpoint residual-disabled HEIR forward pass,
and full HEIR on
one truth-defined gene mask. Every rung reports cell-level, frozen
RNA-mass-weighted spot, and pseudobulk metrics. This is a required diagnostic
before architecture changes: it separates decoder, type-mapping, within-type
state, residual, aggregation, and end-to-end error ceilings.

## Identifiability limitation

Composition, pseudobulk, and distribution losses are invariant to permutations among visually similar nuclei. They establish distributional plausibility, not cell-level truth. Spatial validity must come from morphology anchors, generic cross-modal pretraining, uncertainty/abstention, and independent registered or aggregated spatial evaluation.

The corrected refiner is development code until it beats the one-pass model and
matched type-mean baseline on a separate development cohort, then succeeds on a
new untouched cohort. It does not retroactively change the negative locked
snPATHO v0.2 result.

The current model output named `unknown_probability` is explicitly the
prototype-transport unassigned probability. Pathology feature OOD is a separate
detector output. Strict training rejects legacy biological `unknown_targets`
rather than fitting those labels to the transport head; a future biological
unknown output requires its own independently supervised head and artifact
contract.

The strict broad phase is **fixed-target parent-head fitting**: it aggregates the
immutable fine-prototype responsibilities to parent targets and trains only the
parent head. It does not rerun or parent-gate transport. The excluded
live-student-E-step sensitivity can apply a binary parent-support mask to its
fine-prototype transport, but that is neither the primary path nor a true
parent-Gaussian transport contract.

Restricted residual bases are fitted from within-type RNA latent residuals with
`heir fit-residual-geometry`, then loaded and frozen with
`heir train --residual-geometry`. The same artifact records per-type PCA rank,
state/covariance/residual scale evidence, and calibrated maximum displacement;
the scalar `residual_max_norm` is only a backward-compatible fallback.
