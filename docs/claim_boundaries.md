# Claim boundaries and mandatory failure actions

## Dataset boundaries

- HESCAPE targets are sum-pooled 55-µm Xenium pseudo-spots. HESCAPE is restricted to development
  donors and can provide exploratory H-REGIONAL evidence only; it cannot authorize validation.
- HEST is cell-resolved, but its 20 lung sections are the same GSE250346 source material as HESCAPE.
  It can test internal donor generalization, H-MEAS, H-CELL, and H-INTRINSIC, but not H-EXT.
- RNA-derived broad or fine type is an oracle. An oracle-type pass does not test H-END2END.
- H-CELL cannot lock from marker exclusion alone: its exact label-annotation features, training
  donors, cross-fit receipt, and zero overlap with the frozen target panel must be verified.
- A 112-µm unmasked crop measures local tissue context. It is not nucleus-intrinsic evidence.
- CellViT is a crop/segmentation sensitivity and never replaces the native Xenium RNA join.
- The five locked HEST donors provide an internal go/no-go falsification gate, not population-level
  validation or external generalization.
- Section and batch indicators are development-fold controls, not proof that unseen locked-section
  or locked-batch effects have been fully adjusted away.
- Encoder roles are frozen as UNI2-h primary, H-Optimus-1 replication 1, and H0-mini replication 2.
- Synthetic calibration can become authorizing only after H-MEAS and after a quantitative truth
  matrix is frozen for the global null, G2 boundary, nucleus-only, cell-only, context-only, and
  mixed boundaries. Power is assessed only where a decision is true; false-pass control includes
  every global or partial null. The checked-in shared-latent runner remains preliminary and cannot
  issue an authorizing receipt. The authorizing runner must also reproduce the exact H-MEAS
  donor/section/type support topology and retain verifiable per-trial gate evidence.
- A matched molecular bank is useful only if changing the bank changes image-conditioned prediction
  error against locked RNA; molecular proximity is exploratory.

## Interpretation rules

| Result | Permitted conclusion/action |
| --- | --- |
| HESCAPE passes, HEST fails | Pivot to regional context prediction; abandon cell/nucleus claim |
| Full context passes, masks fail | Local microenvironment association only |
| Coordinates or stain equal image | Treat as spatial/batch confounding |
| Fine-type conditioning removes signal | Subtype recognition, not within-type state |
| Only one encoder passes | Representation-specific association |
| Internal HEST passes, external fails | Cohort-specific association; no generalization |
| Matched bank fails against any required bank | Remove personalization and use a generic reference |
| Oracle type passes, predicted type fails | Mechanistic oracle result; no H&E-only deployment |
| Ridge passes, later HEIR component fails | Retain the ridge and remove the unsupported component |
| Refinement degrades round zero | Immutable round zero remains final |
| Measurement ceiling is inadequate | Redefine/improve measurement before model development |

Negative and failed gates remain in the final evidence bundle. They are never omitted or relabeled as
successful development evidence.
