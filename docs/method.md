# Method contract

For sample (s), HEIR receives H&E nuclei (\{(x_i,r_i,m_i)\}_{i=1}^{N_s}) and an unregistered matched snRNA reference (\{(y_j,c_j)\}_{j=1}^{M_s}). There is no assumed correspondence (i\leftrightarrow j).

The image graph predicts hierarchical type probabilities, a conditional
known-state prototype distribution (q_{ik}), a separate biological-unknown
probability (u_i), and a restricted residual posterior. For new v3 checkpoints,
the decoded molecular mean is

\[
z_i^{\mathrm{proto}}=\sum_k q_{ik}\widetilde\mu_{sk},
\qquad
\Delta z_i=
\alpha_i\frac{B_i a_i}{\sqrt{1+\lVert B_i a_i\rVert_2^2}},
\qquad
z_i=z_i^{\mathrm{proto}}+\Delta z_i,
\]

where (B_i=\sum_c p_i(c)B_c) mixes type-conditioned bases,
(\operatorname{rank}(B_c)\le r), and
(0<\alpha_i<\alpha_{\max}). The coefficient-mean head is initialized exactly at
zero and every deterministic or sampled residual has
(\lVert\Delta z_i\rVert_2<\alpha_{\max}); a fresh deterministic model therefore
equals its routed prototype baseline. Checkpoint v1/v2 models retain their
historical unrestricted residual behavior when loaded and are never silently
upgraded.

Unknown probability therefore changes confidence and loss participation, not the
biological latent by scaling it toward zero.

Prototype means are shrunken toward a training atlas by (n/(n+\kappa)), with (\kappa=50) by default. A frozen RNA decoder jointly maps (z_i) to the selected expression panel. Prototype banks and the decoder carry the same immutable latent-space identifier; matching width alone is not accepted. An optional second image head is aligned to frozen, donor-audited scGPT type prototypes and within-type variances. When morphology is weak, the residual is regularized toward zero, uncertainty rises, and the cell may be left unknown.

## Population alignment

HEIR aligns the final decoded latent mean to prototype Gaussians with
covariance-aware entropic unbalanced transport. A dustbin column absorbs image
objects or states that should remain unassigned. Unlike balanced transport, this
does not force the H&E section to reproduce biased snRNA capture fractions
exactly. The primary path uses a prespecified fixed dustbin mass (default 0.05,
to be calibrated only on development data), or explicit `unknown_targets` when
supplied. Estimation from the model's own
unknown head is retained only as an explicit sensitivity because it can create
self-reinforcing dustbin assignments. The UOT plan is normalized by
each row's complete transported mass,
including the dustbin, and detached to form known-state subprobabilities. Their
row sum preserves molecular known-state mass; conditional known responsibilities
directly supervise prototype routing, cell type, latent mean, marker, program,
and frozen-teacher objectives in the M-step. Detached transported known-state
row mass, rather than the live unknown-head probability, weights the weak
biological objectives. Live image type predictions are not the responsibility
target.

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

Every term is implemented independently under `src/heir/losses/` and can be ablated. When a specimen is split into graph patches, HEIR forwards each graph independently and aggregates outputs before applying sample-level UOT, composition, pseudobulk, marker, and program losses; it does not force every local patch to reproduce the whole-specimen mixture. Generic spatial pretraining merges shared spot IDs across disjoint graph patches, verifies repeated spot targets, and weights its pseudobulk by assigned spot mass. Empty spots and overlapping-nucleus spatial patches are rejected rather than silently creating irreducible or double-counted targets.

## Training stages

1. Fit/adapt the RNA teacher on real snRNA counts only.
2. Optionally pretrain the image mapping on decontaminated public H&E–ST samples.
3. Personalize on H&E plus matched snRNA; target ST is prohibited.
4. Before refinement, evaluate and snapshot round 0 (student, EMA teacher,
   batches, measured priors, and empty anchor state). Refine for at most five
   rounds (default four), using two parent-gated rounds followed by two fine
   rounds, confidence/entropy/OOD/segmentation/view gates, and two-round
   revocable anchors. Measured priors are fixed by default; updating them is an
   explicit sensitivity. A first or later candidate whose validation loss
   exceeds the best loss plus `objective_stability_tolerance` restores the best
   complete snapshot immediately.
5. Calibrate on development donors.
6. Open locked spatial measurements once the checkpoint and thresholds are frozen.

The distilled H&E-only student is a separate model and evaluation setting.

## Identifiability limitation

Composition, pseudobulk, and distribution losses are invariant to permutations among visually similar nuclei. They establish distributional plausibility, not cell-level truth. Spatial validity must come from morphology anchors, generic cross-modal pretraining, uncertainty/abstention, and independent registered or aggregated spatial evaluation.

The corrected refiner is development code until it beats the one-pass model and
matched type-mean baseline on a separate development cohort, then succeeds on a
new untouched cohort. It does not retroactively change the negative locked
snPATHO v0.2 result.

The current broad phase is deliberately described as **parent-gated
fine-prototype transport**: parent probabilities restrict compatible fine
states, and only the parent head is trainable, but the UOT bank is still fine
grained. A genuine parent-Gaussian transport contract remains future work.
