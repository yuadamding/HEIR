# Validation and negative controls

The donor is the biological unit. Sections, cells, spots, and patches from one donor never cross an outer fold.

## Primary comparisons

1. H&E-only morphology/graph model.
2. Generic tissue atlas.
3. Matched snRNA, no refinement.
4. Matched snRNA with constrained refinement.
5. Matched cell-type mean expression.
6. Wrong-donor snRNA.

The expression claim survives only if HEIR beats the matched type-mean baseline on within-type residual programs.

## Mandatory negative controls

- donor-permuted RNA;
- shuffled H&E features;
- shuffled graph coordinates;
- composition-only prediction;
- label-shuffled prototypes;
- no-refinement and no-unknown variants.

Correct matched RNA should outperform a generic atlas, which should outperform permuted/no RNA. Similar performance for correct and wrong donors falsifies the sample-personalization claim.

## Metrics

- Cell types: macro-F1, balanced accuracy, per-class F1/AUPRC, ECE, Brier, risk–coverage.
- Composition: per-type Pearson/Spearman, JS divergence, Aitchison distance, RMSE.
- Expression: every prespecified gene, donor-wise Spearman/Pearson, log-MSE, spatial pattern, fraction beating baseline.
- Programs: cell-type-conditioned score correlation and co-expression preservation.
- Spatial organization: Moran's I, domain agreement, boundaries, adjacency/neighborhood enrichment.
- Registration: target-registration error, match/ambiguity fractions, performance versus error.

Locked snPATHO Visium expression/counts may be opened only after architecture,
gene panel, calibration, and stopping rules are fixed. Target H&E and
non-expression spot metadata may be used transductively when declared. A later
model change creates a new version and requires a new untouched test set.

`heir prepare-spatial-truth` enforces this boundary at artifact creation: the
selected manifest row must have a locked target role, and an RDS-derived count
matrix must include a hash-verified conversion sidecar. The resulting
`heir.spatial_truth` NPZ stores the target role and all source hashes but is not
accepted by the personalized training/refinement interfaces. Barcode matching
uses one collision-free policy for the complete sample, and nuclei outside the
physical Visium spot disks remain explicitly unassigned.

The one-shot `scripts/benchmark_snpatho.py` runner additionally binds every
prediction to one frozen checkpoint and gene-panel hash before opening truth.
It rejects overlap between locked spatial hashes and prediction/reference
inputs, requires all three donors by default, and emits donor-bootstrap macro
confidence intervals. Constant matched-snRNA pseudobulk, spatially shuffled
HEIR predictions, and matched type means are generated without consulting the
target Visium values.

Methodology tests additionally exercise identifiable and non-identifiable
transport, missing-reference-state dustbin routing, revocable wrong anchors,
parent-to-fine constraints, scVI full-library scale equivalence, final-latent UOT
consistency, and prototype-mixture uncertainty. These are necessary behavioral
checks, not substitutes for independent cell-resolution biological validation.
