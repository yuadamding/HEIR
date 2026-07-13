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

Reference and evaluation cells may share a section and source file, but they must have disjoint
observation, cell, and spatial-block identities with a positive physical guard. Development donors
select the target panel and supported fine types. The identical frozen measurement thresholds are
then audited on locked rows without changing that selection; every failed or unsupported locked
donor/section/type stratum remains in the intention-to-analyze coverage denominator.

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

The primary endpoint is residual-coordinate R² macro-averaged first over supported fine types within
donor and then equally across donors. Technical correction, weighted basis, and model selection use
development donors only.

The fine-type label must be backed by an exact external or development-donor-cross-fitted annotation
receipt whose ordered RNA features are disjoint from the frozen target panel. Marker-list exclusion
or an independence boolean alone cannot open H-CELL.

Authorization: a pass across the five locked donors is an internal go/no-go result supporting
progression to external confirmation. It is not population-level validation, external
generalization, or H&E-only deployment.

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

HEST and HESCAPE share GSE250346 and cannot jointly test H-EXT. Cohort selection is based on declared
suitability before image–RNA outcomes are inspected. No external retuning or gene reselection is
permitted.

Authorization: a pass permits a general morphology–state claim within the locked external scope.

## Frozen encoder hierarchy

The H-CELL primary encoder is UNI2-h. H-Optimus-1 is replication 1 when its gated checkpoint is
available, and H0-mini is replication 2 when available. These labels are preregistered and may not
be reassigned after any locked molecular outcome is observed.
