# Claim boundaries and mandatory failure actions

## Dataset boundaries

- HESCAPE targets are sum-pooled 55-µm Xenium pseudo-spots. HESCAPE can test H-REGIONAL only.
- HEST is cell-resolved, but its 20 lung sections are the same GSE250346 source material as HESCAPE.
  It can test internal donor generalization, H-MEAS, H-CELL, and H-INTRINSIC, but not H-EXT.
- RNA-derived broad or fine type is an oracle. An oracle-type pass does not test H-END2END.
- A 112-µm unmasked crop measures local tissue context. It is not nucleus-intrinsic evidence.
- CellViT is a crop/segmentation sensitivity and never replaces the native Xenium RNA join.
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

