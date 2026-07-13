# Claim boundaries and mandatory failure actions

## Dataset boundaries

- HESCAPE targets are sum-pooled 55-µm Xenium pseudo-spots. HESCAPE is restricted to development
  donors and can provide exploratory H-REGIONAL evidence only; it cannot authorize validation.
- HEST is cell-resolved, but its 20 lung sections are the same GSE250346 source material as HESCAPE.
  It can support H-MEAS development and retrospective H-CELL/H-INTRINSIC exploration, but not a
  prospective confirmatory claim or H-EXT.
- The five historically designated HEST test donors had molecular and image outcomes materialized
  and their test artifact reloaded in the run
  `/mnt/seagate/HEIR_runs/mr_hescape_uni2h_28c6fff_1783921182` at commit `28c6fff`. No endpoint
  report or metric-guided tuning evidence was found. Their prospective eligibility is nevertheless
  permanently false; the scientific hypothesis is untested, not failed, and a fresh pristine cohort
  is required.
- RNA-derived broad or fine type is an oracle. An oracle-type pass does not test H-END2END.
- H-CELL cannot lock from marker exclusion alone: its exact label-annotation features, training
  donors, cross-fit receipt, and zero overlap with the frozen target panel must be verified. The
  training-label ontology must also have target-independent provenance; gene-disjoint classifier
  inputs do not cure target-derived training labels.
- H-MEAS cannot use unresolved `final_CT`/`final_lineage` labels to choose the common panel or
  supported types. Its development annotations must already be disjoint from the full candidate
  target panel and carry target-independent training-label provenance.
- A locally authored receipt cannot establish that provenance on its own. The label ontology and
  source assignments require an immutable upstream identity or independent steward/curator
  attestation, plus hashes for the exact training data and fitted annotation model. These materials
  are currently missing.
- A 112-µm unmasked crop measures local tissue context. It is not nucleus-intrinsic evidence.
- CellViT is a crop/segmentation sensitivity and never replaces the native Xenium RNA join.
- The exposed five-donor HEST subset is retrospective internal/exploratory evidence only. It cannot
  issue an internal prospective go/no-go decision, population-level validation, or external
  generalization.
- Section and batch indicators are development-fold controls, not proof that unseen locked-section
  or locked-batch effects have been fully adjusted away.
- G2's joint primary endpoint freezes both the donor/type macro endpoint and its
  donor/section/type-balanced companion at R² >= 0.05; both must pass.
  Locked reliability must be reported by donor/type and donor/section/type, with worst-section and
  planned-stratum coverage summaries.
- Confirmatory registration uses the same exact H-MEAS/H-CELL audit object: 8/12/8-µm absolute
  criteria, 0.5/0.5 relative-geometry limits, and 0.25/0.6 best/intermediate quality cutoffs, plus
  the shared segmentation, crop, and reliability fields. Looser source-ingestion tolerances cannot
  authorize a biological result.
- The primary H-MEAS target is one common reliable panel with at least six genes. A type-specific or
  program fallback requires a new development-only study version and calibration before any pristine
  confirmatory opening; no confirmatory outcome may choose the fallback.
- G3 effects must be stratified by best, intermediate, and near-threshold registration quality.
  Nucleus/cell contrasts and the full-versus-target-removed intrinsic increment needed for a mixed
  conclusion must remain noninferior in the best-registration subset within the frozen 0.01 delta-R2
  margin. An apparent intrinsic effect confined to near-threshold rows cannot be promoted to strong
  nucleus- or cell-local evidence.
- Encoder roles are frozen as UNI2-h primary, H-Optimus-1 replication 1, and H0-mini replication 2.
- Synthetic calibration can become authorizing only after H-MEAS and after a quantitative truth
  matrix is frozen for the global null, G2 boundary, nucleus-only, cell-only, context-only, and
  mixed boundaries. Power is assessed only where a decision is true; false-pass control includes
  every global or partial null. The checked-in implementation structurally separates the exact
  six-condition production DGP from a non-authorizing smoke mode and hash-attests each actual-gate
  trial report. That structure is not an authorization: it must be bound to the completed H-MEAS
  design and donor/section/type topology, then executed at the required scale. No authorizing receipt
  exists.
- The literal calibration minimum is 1,000 complete gate trials for six conditions across ten stress
  families: at least 60,000 full gate executions. It is computationally infeasible as currently
  specified and remains unexecuted. A sequential alternative is valid only if preregistered with
  simultaneous confidence bounds and a non-opportunistic stopping rule before confirmatory opening.
- A matched molecular bank is useful only if changing the bank changes image-conditioned prediction
  error against locked RNA; molecular proximity is exploratory.

## Interpretation rules

| Result | Permitted conclusion/action |
| --- | --- |
| Development HESCAPE signal, no prospective cell cohort | Regional hypothesis generation only; no cell/nucleus claim |
| Full context passes, masks fail | Local microenvironment association only |
| Coordinates or stain equal image | Treat as spatial/batch confounding |
| Fine-type conditioning removes signal | Subtype recognition, not within-type state |
| Only one encoder passes | Representation-specific association |
| Retrospective HEST signal, prospective external fails | Historical cohort association only; no confirmation or generalization |
| Matched bank fails against any required bank | Remove personalization and use a generic reference |
| Oracle type passes, predicted type fails | Mechanistic oracle result; no H&E-only deployment |
| Ridge passes, later HEIR component fails | Retain the ridge and remove the unsupported component |
| Refinement degrades round zero | Immutable round zero remains final |
| Measurement ceiling is inadequate | Redefine/improve measurement before model development |

Negative and failed gates remain in the final evidence bundle. They are never omitted or relabeled as
successful development evidence.
