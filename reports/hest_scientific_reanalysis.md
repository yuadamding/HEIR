# HEST H-optimus-1 qualification

## Decision

**STOP before molecular fitting.** The frozen H-optimus-1 visible positive-control gate failed.
Broad lineage, supported fine type, and all three natural-image appearance controls passed, but all
four required natural-context geometry controls failed. The runner therefore produced no
H-optimus-1 molecular models, scientific summary, comparator preflight, or paired encoder claim.

This is a retrospective, outcome-exposed encoder qualification on HEST. It is not the central
H&E-plus-matched-scRNA hypothesis test: HEST has no independent matched sc/snRNA aliquot.

## Frozen experiment identity

| Item | Value |
|---|---|
| Cohort | HEST/GSE250346 human lung Xenium |
| Registered cells | 36,121 |
| Biological donors | 15 |
| Sections | 20 |
| Primary crop | natural unmasked 112-um field, 224 px at 0.5 um/px |
| Primary encoder | frozen `bioptimus/H-optimus-1` |
| Encoder revision | `3592cb220dec7a150c5d7813fb56e68bd57473b9` |
| Features | full 1,536D primary; training-only PCA-256/PCA-512 planned sensitivities |
| Evaluation | leave one biological donor out; three inner training-donor folds |
| Comparator | fixed, separately scored `MahmoodLab/UNI2-h` baseline |
| Outcome exposure | retrospective; permanently non-authorizing |

The H-optimus source and registered UNI2 source match on all 217 non-encoder fields. Model bytes,
official-versus-local parity, normalization, crop scale, and the single Pillow bicubic resize were
independently verified before opening the H-optimus gate.

The inherited registered-source QC remains `pass=false`: the prospective registration audit and
per-row crop QC do not pass, while the molecular transcript-target audit does pass. Those exact QC
fields and source rows are shared with the UNI2 comparator, so this is not an encoder-specific
explanation for their descriptive difference. It is an independent reason the qualification remains
retrospective and non-authorizing even if its visible gate had passed.

## Positive controls

The morphology metric is donor/type-macro error reduction over an outer-training-only reference.
Positive values pass; zero or negative values fail.

| Natural 112-um endpoint | H-optimus-1 | UNI2-h | H-optimus pass |
|---|---:|---:|:---:|
| Broad-lineage balanced accuracy | 0.4888 | 0.4640 | yes |
| Broad-lineage training-majority baseline | 0.2500 | 0.2500 | -- |
| Fine-type balanced accuracy | 0.1619 | 0.1437 | yes |
| Fine-type training-majority baseline | 0.0355 | 0.0355 | -- |
| Nucleus area | -0.0400 | -0.0716 | **no** |
| Nucleus perimeter | -0.0740 | -0.1068 | **no** |
| Nucleus circularity | -0.0905 | -0.0967 | **no** |
| Nucleus solidity | -0.0602 | -0.0663 | **no** |
| Gray intensity | 0.1305 | 0.1332 | yes |
| Hematoxylin optical density | 0.1401 | 0.1411 | yes |
| GLCM contrast | 0.0399 | 0.0223 | yes |

H-optimus-1 improved broad/fine classification and several geometry/texture values descriptively,
but every required natural-context geometry endpoint remained below zero. The nucleus-mask-only arm
had high morphology R2 (about 0.806), but it is a secondary segmentation-sensitive attribution arm
and cannot rescue the natural-image gate.

Gate result:

```text
execution_status = blocked_positive_control_gate_failed
positive_control_gate.passed = false
molecular_interpretation_allowed = false
```

## UNI2 historical comparator

UNI2-h also failed the same current natural-context gate on exactly the four geometry endpoints. A
default-off amendment allowed its already-exposed molecular analysis to finish only as a descriptive,
non-authorizing baseline. Its PCA-512 mean delta-R2 / positive-donor fractions were:

| Program | Mean delta R2 | Positive donors |
|---|---:|---:|
| Fibrotic mesenchymal | -0.0450 | 0.067 |
| Macrophage inflammation | -0.0140 | 0.133 |
| Epithelial injury | -0.0175 | 0.067 |
| Stress/hypoxia | -0.0815 | 0.000 |
| Interferon/chemokine | -0.0202 | 0.133 |
| Proliferation | 0.0242 | 0.533 |

These values are descriptive only. No p-value, decision, evidence, or encoder-comparison claim is
authorized from the gate-failed UNI2 baseline.

## Artifact receipts

| Artifact | SHA-256 |
|---|---|
| H-optimus source | `f7e7d4e97727cc17e71a81a252ab35fd2ca1c0e70054cba3ed38c2f7b7f65636` |
| H-optimus plan | `5866f5a626fc4521f9d349ab7cc6638c98e7773f0bd7e597fc26ca4a5348c547` |
| H-optimus QC | `31d532c289e5fb24881d5ae90cafb3da79c3c1dae1f9ffc2da9c9ca41d5f0e7c` |
| H-optimus blocked report | `2685efc9574a1b6c9b2ff8f5a08cf372b038a1eaadd271f91ff24228b6060f1f` |
| H-optimus gate | `54f4a2b6fdb5bb47705073468d31cc51c52cc63b6462b0691a883b720b1e4cae` |
| H-optimus parity receipt | `a67ca37feae12a3ca444399f12dc983de01283b05f14ffe16adfcdae80a4d761` |
| UNI2 registered source | `57b77c7be2e30026a2da9ba0f9d5b205cf630f5d138942db6366e15cae2ef7a3` |
| UNI2 descriptive report | `25c57263a28fb9733a7ff31d76988540386d778315299570d4ee22a3a761ac44` |
| Frozen HEST runner | `8fb8ac6c1d2c9f2f0cfa230690e59d360cf0b6507c51f5dde38c2f92462fd0b9` |

Large sources, embeddings, checkpoints, and raw reports remain outside Git.

## Scientific boundary

- H-optimus molecular-state qualification: blocked by the visible gate.
- Prospective HEST source eligibility: independently blocked by inherited registration/crop QC and
  prior outcome exposure; the transcript-target audit itself passed.
- Paired H-optimus-versus-UNI2 molecular comparison: not run.
- Regional NatCommun M0-M8 interpretation: blocked because its frozen runner requires a completed,
  passing H-optimus HEST gate.
- Central cell-level H&E-plus-matched-scRNA hypothesis: blocked because no downloaded cohort has
  registered cell-resolved ST and an independent matched sc/snRNA reference.
- Independent replication: unavailable.
- Full HEIR development, refinement, fine-tuning, and biological claims: not authorized.

The gate may be redesigned only as a new prospective protocol. It cannot be relaxed after observing
this result and then used to continue the frozen experiment.
