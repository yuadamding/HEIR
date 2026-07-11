# Method contract

For sample (s), HEIR receives H&E nuclei (\{(x_i,r_i,m_i)\}_{i=1}^{N_s}) and an unregistered matched snRNA reference (\{(y_j,c_j)\}_{j=1}^{M_s}). There is no assumed correspondence (i\leftrightarrow j).

The image graph predicts hierarchical type probabilities and a residual RNA posterior. The model routes each image nucleus over sample-supported prototypes and writes

\[
z_i=\sum_k p_{ik}\,\widetilde\mu_{sk}+\Delta z_i.
\]

Prototype means are shrunken toward a training atlas by (n/(n+\kappa)), with (\kappa=50) by default. A frozen RNA decoder jointly maps (z_i) to the selected expression panel. Prototype banks and the decoder carry the same immutable latent-space identifier; matching width alone is not accepted. An optional second image head is aligned to frozen, donor-audited scGPT type prototypes and within-type variances. When morphology is weak, the residual is regularized toward zero, uncertainty rises, and the cell may be left unknown.

## Population alignment

HEIR aligns image cells to prototypes with entropic unbalanced transport. A dustbin column absorbs image objects or states that should remain unassigned. Unlike balanced transport, this does not force the H&E section to reproduce biased snRNA capture fractions exactly.

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
4. Refine for at most five rounds (default three) with an EMA teacher, confidence/entropy/OOD/segmentation/view-agreement gates, sample-level prior updates, per-class caps, and rollback to the best weak-validation checkpoint.
5. Calibrate on development donors.
6. Open locked spatial measurements once the checkpoint and thresholds are frozen.

The distilled H&E-only student is a separate model and evaluation setting.

## Identifiability limitation

Composition, pseudobulk, and distribution losses are invariant to permutations among visually similar nuclei. They establish distributional plausibility, not cell-level truth. Spatial validity must come from morphology anchors, generic cross-modal pretraining, uncertainty/abstention, and independent registered or aggregated spatial evaluation.
