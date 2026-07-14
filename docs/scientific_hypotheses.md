# HEIR scientific hypotheses

These identifiers are permanent. Every locked study and benchmark report must cite at least one
identifier. A passing component is evidence only for the stated claim; it does not inherit a broader
HEIR claim.

## H-MEAS — measurement validity

Question: Are registration, segmentation, transcript assignment, and molecular measurement reliable
enough for a morphology experiment?

Null: Apparent cell-level variation is dominated by registration error, low counts, segmentation
artifacts, or measurement noise.

Primary evidence: unique cell/transcript identities, row-level registration geometry, transcript
detection and zero fractions, split-half gene/program reliability, target-basis ceiling, crop padding,
and reference/evaluation separation.

Execution prerequisite: H-MEAS may proceed only from a hashed annotation export containing the ten
development donors and no historically designated test rows. The combined 20-section annotation
source is not an admissible analyst input. Fine types must come from a receipt-bound independent
annotation whose inputs and training-label ontology are disjoint from the full candidate target
panel; development `final_CT`/`final_lineage` labels cannot select the H-MEAS panel.

Reference and evaluation cells may share a section and source file, but they must have disjoint
observation, cell, and spatial-block identities with a positive physical guard. Development donors
select the common reliable target panel and supported fine types. The panel must contain at least six
genes to support the confirmatory rank-six candidate. If that common panel fails, a type-specific-panel or
prespecified-program fallback requires a new development-only study version; no confirmatory result
may trigger the switch. The identical frozen measurement thresholds are then audited on a fresh
pristine confirmatory cohort without changing that selection; every failed or unsupported planned
donor/section/type stratum remains in the intention-to-analyze coverage denominator. Reliability is
reported by donor/type and donor/section/type, including the worst section.

Authorization: a pass permits morphology experiments to run. It does not support a morphology claim.

## H-REGIONAL — regional H&E–expression association

Question: At 55-µm HESCAPE pseudo-spots, do frozen pathology features predict molecular residuals
beyond a donor/niche reference mean, coordinates, stain, density/composition, and registration nulls?

Primary unit: biological donor. Primary crop: target-matched approximately 55 µm. The 109-µm crop
and context-only annulus are sensitivities.

Authorization: development-stage engineering confidence and exploratory GSE250346 regional
tissue-context evidence only. HESCAPE uses development-donor cross-validation and cannot authorize
a validated regional association.

## H-CELL — registered cell-level morphology–state association

Question: Given an independently derived fine RNA type and a spatially independent
donor/section/type RNA reference mean, do frozen H&E features predict residual molecular state in
held-out donors?

For cell `i`, donor `d`, section `s`, and fine type `t`:

```text
r_i = y_i - mean_reference[d,s,t] - technical_i @ Gamma_t
z_i = B_t.T @ r_i
z_hat_i = f_t(image_i)
```

The frozen joint primary endpoint explicitly contains donor/type macro residual-coordinate R² and
donor/section/type macro residual-coordinate R². The first averages supported fine types within donor
and then donors; the second averages fine types within section, sections within donor, and donors.
Each must be at least 0.05 and the frozen intersection rule requires both. Technical correction,
weighted basis, and model selection use development donors only.

The fine-type label must be backed by an exact external or development-donor-cross-fitted annotation
receipt whose ordered RNA features are disjoint from the frozen target panel. Its training-label
ontology must independently establish zero target dependence; reproducing target-derived labels with
gene-disjoint classifier inputs is insufficient. Marker-list exclusion or an independence boolean
alone cannot open H-CELL.

Authorization: a pass on a genuinely unexposed, preregistered internal cohort is an internal go/no-go
result supporting progression to external confirmation. It is not population-level validation,
external generalization, or H&E-only deployment.

The five historically designated HEST test donors (`THD0008`, `THD0011`, `TILD117`, `VUILD78`, and
`VUILD96`) are not eligible for that authorization. Their molecular and image outcomes were
materialized and the test artifact was reloaded at commit `28c6fff`. No endpoint report or tuning
evidence was found, so the hypothesis remains untested rather than failed; these donors may support
only explicitly retrospective internal/exploratory analysis.

Section and batch indicators are fitted only as development-fold controls. New locked-section and
locked-batch categories do not have estimable development coefficients, so their inclusion cannot
be described as fully adjusting away arbitrary section or batch effects.

## H-INTRINSIC — cell- or nucleus-intrinsic morphology

Question: Does signal survive a nucleus or cell mask and exceed a context-only view with the target
cell removed?

Required arms: nucleus mask, cell mask, 32/64/112-µm crops, context rings, target-cell-removed crop,
blank patch, stain, coordinates, and handcrafted nucleus/cell morphometrics.

All 18 prespecified direct contrasts are tested together with an exact donor sign-flip max-statistic
family. The allowed conclusions include nucleus dominant, cell dominant, context dominant, mixed
intrinsic and contextual information, multiple sources without incremental combination, and no
morphology-specific information. A strongest-comparator diagnostic cannot authorize a claim.

Each G3 effect is also reported in best, intermediate, and near-threshold registration-quality
strata. Nucleus- or cell-local evidence must clear the frozen effect threshold in the best-registration
subset and be noninferior to the all-row and fully supported near-threshold estimates within the
prespecified 0.01 delta-R2 margin. The same rule applies to the full-context-versus-target-removed
incremental intrinsic contrasts required for a mixed conclusion. Context-only effects remain
stratified diagnostics because their hypothesized source is outside the registered target cell. A
signal appearing only near the registration limit cannot support an intrinsic claim.

## H-REF — matched-reference utility

Question: Holding image queries, image model, target genes, decoder, type probabilities, bank size,
assay, and quality fixed, does a matched-donor bank improve locked molecular prediction relative to
every hard wrong-donor, generic-atlas, and leave-query-donor-out population bank?

Primary endpoint: donor-paired comparator loss minus matched-bank loss against unchanged locked RNA.

Authorization: a pass after cell-level and external morphology evidence permits a personalized
reference claim. Molecular-bank proximity alone is diagnostic and cannot pass H-REF.

## H-END2END — oracle-free inference

Question: Can H&E-derived type routing plus H&E molecular state beat the corresponding predicted-type
reference mean at identical evaluation coverage?

Required comparisons: oracle type, hard predicted type, soft probabilities, type-agnostic model, and
reference mean only. Type metrics include donor-level broad/fine macro F1, calibration, occupancy,
confusion matrices, and fixed abstention coverage.

Authorization: a pass after external confirmation permits an H&E-only inference claim.

## H-COMP — HEIR component value

Question: After the simple ridge and matched-reference premises pass, does each reference prototype,
transport/UOT, graph, unknown/abstention, or refinement component improve its immediate predecessor?

Authorization: only components with a donor-paired, minimum-sized improvement and no material
coverage/calibration/external degradation may be included in validated HEIR.

## H-EXT — independent external generalization

Question: Does the frozen primary effect replicate in a different study, donor population, acquisition
batch, and preferably institution?

HEST and HESCAPE share GSE250346 and cannot jointly test H-EXT. The previously designated five HEST
test donors are also prospectively ineligible because their outcomes were already materialized.
Cohort selection is based on declared
suitability before image–RNA outcomes are inspected. No external retuning or gene reselection is
permitted.

Authorization: a pass permits a general morphology–state claim within the locked external scope.

## Frozen encoder hierarchy

For experiments begun after the 2026-07-13 access decision, the H-CELL primary encoder is frozen
H-optimus-1 revision `3592cb220dec7a150c5d7813fb56e68bd57473b9`. UNI2-h is the fixed secondary
encoder comparator because the already-opened negative HEST evidence used UNI2-h; it is never
silently substituted into a new primary arm. H0-mini remains a gated second replication when
available. These roles were frozen before any H-optimus-1 molecular outcome was opened. The existing
HEST outcomes remain retrospective, so a same-cohort H-optimus-1 run is a bounded encoder
qualification rather than pristine confirmation.
