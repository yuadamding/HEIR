# HEST nonlinear estimator qualification v1

## Registered decision scope

This protocol is `retrospective_exposed_non_authorizing`. It asks one bounded engineering
question:

> Can a small nonlinear, type-conditioned, multiview translator recover independently
> reference-centered within-type molecular residuals that the historical ridge model class did
> not expose?

All 15 HEST/GSE250346 donor outcomes have already been exposed. A complete pass may set only
`supports_new_prospective_estimator_protocol=true`. It cannot authorize H-CELL, H-INTRINSIC,
H-REF, or full HEIR. Those four authorization fields remain false under every result.

This document registers the scientific protocol; it is not a biological-results report. The
WP-2/WP-3 core implementation and WP-4 preflight/synthetic smoke are complete. The smoke used no
HEST molecular rows and produced no engineering or biological decision. Its separate report is at
`/mnt/seagate/HEIR_runs/hest_nonlinear_v1/report.md`.

Execution is currently blocked. The registered source has no receipt-bound blank-patch embedding,
and its 391 best-registration rows yield zero donor/type strata at the primary support of 20. V1
fails closed: it may neither fabricate a blank vector nor replace the registration band/support
after outcomes. A frozen input supplement and a new protocol identity are required before running.

## Relationship to frozen evidence

The earlier negative results retain their original meaning. This new protocol does not relax the
failed H-optimus-1 natural-context geometry gate, reinterpret the frozen UNI2-h ridge result, or
promote the unsupported regional reference-fusion result.

| Frozen artifact | SHA-256 |
|---|---|
| `scripts/benchmark_hest_retrospective.py` | `853a2801e94e3c727631d4e3eb8e0a0ba766bd3768b3302b4219ce14a61c9878` |
| `scripts/benchmark_morphology_state_gate.py` | `5975f2cc0ff586a89f39f78bbf989f1ecd1e03347ea9fba9b74673b316385919` |
| `src/heir/evaluation/morphology_gate.py` | `4c92b3c853203cf9e580db311536cf428283e41fd473436b8a6f3d4be80b3cad` |
| `src/heir/evaluation/reference_fusion_v2.py` | `894ad3de42cc3d253aee400cb17cd7ff2257c2aca016601d7eb0f2bc2136bf3a` |
| `reports/hest_retrospective.md` | `1e833867cef32504f1f0f2d4d52471e9ff17ee992cfd8c4e914e347495a95027` |
| `reports/hest_scientific_reanalysis.md` | `2c571cc3e611466dd0c0f957cb738aa9ffa319a93562c4519bdfe02b412c733d` |
| `reports/natcommun_hoptimus_primary_results_v2.md` | `b4771b6d10dfbbe22cc4ecf873435cf370fb117a608ab9e6982634bde7374af7` |

The registered source is the frozen H-optimus-1 HEST source at
`/mnt/seagate/HEIR_runs/hest_hoptimus1_qualification/source.npz`, SHA-256
`f7e7d4e97727cc17e71a81a252ab35fd2ca1c0e70054cba3ed38c2f7b7f65636`. H-optimus-1
revision `3592cb220dec7a150c5d7813fb56e68bd57473b9` remains frozen. Encoder fine-tuning
and UNI2-h execution are prohibited in this protocol; UNI2-h remains historical evidence only.

## Frozen estimator matrix

| ID | Representation | Estimator | Registered question |
|---|---|---|---|
| B0 | None | Best existing non-image ridge | Strongest nuisance baseline |
| B1 | Historical-form 96D projection | Per-type ridge | Contemporaneous H-optimus linear baseline |
| B2 | Full 1,536D feature | Per-type ridge | Did dimensionality reduction remove signal? |
| B3 | Full 1,536D feature | Shared linear model + type adapter | Does cross-type sharing help? |
| N0 | Combined nuisance features | Small MLP | Architecture-matched nonlinear nuisance baseline |
| N1 | Natural 112-µm crop | Small MLP | Nonlinear global image signal |
| N2 | Natural 112-µm crop | MLP + fine-type adapter | Low-capacity type conditioning |
| N3 | Nucleus-mask crop | Same selected MLP family | Nucleus-local signal |
| N4 | Cell-mask crop | Same selected MLP family | Cell-local signal |
| N5 | Target-cell-removed crop | Same selected MLP family | Context counterfactual |
| N6 | Full + nucleus + cell views | Late-fusion MLP | Prespecified multiview increment |
| N7 | Combined nuisance + N6 views | Late-fusion MLP | Image value beyond nonlinear nuisance |

B1 reruns the historical outcome-independent 1,536-to-96 Rademacher *form* on the registered
H-optimus-1 features. It is not the exact archived baseline: that result used UNI2-h, a pooled
multi-type ridge, and direct 260-gene residual prediction. The archived result remains a frozen,
non-comparable anchor and UNI2-h is not rerun. Focal and comparator arms must use the same donor
folds and capacity/search rules where applicable.

## Molecular target and leakage boundary

The primary target remains the existing HEIR residual construction, in this order:

1. Subtract the spatially independent same-donor/section/type reference-pool mean.
2. Fit technical correction using outer-training donors only.
3. Fit the type-specific low-rank molecular basis using outer-training donors only.
4. Predict residual coordinates.
5. Reconstruct and score the measured target genes.

Raw whole-transcriptome prediction is not a primary target. The outer split is leave-one-biological-
donor-out across the 15 exposed donors. Inner model selection is leave-one-training-donor-out;
local spatial blocks remain intact. No held-out donor may affect standardization, target fitting,
early stopping, architecture choice, or epoch choice.

The molecular rank grid is `[2, 4, 6]`, and every inner-development-fold basis ceiling must be at
least R² = 0.30. Each inner fold refits feature normalization, technical correction, target basis,
and donor/type-weighted coordinate standardization from its inner-training donors only. Coordinates
are inverse-transformed before scoring. After selection, the outer model refits every transform on
all outer-training donors only.

## Frozen architecture and training settings

- `shared_linear_type_adapter`: shared linear map plus a rank-8 type-specific linear correction.
- `mlp_tiny`: LayerNorm(input width) → 64 → GELU → dropout 0 → rank.
- `mlp_small`: LayerNorm(input width) → 256 → GELU → dropout 0.2 → 64 → GELU → rank.
  Input width is the registered arm width: 1,536D for an image arm, 158D for N0, or the exact
  concatenated width for a combined single-view comparator.
- Fine-type conditioning: width-16 embedding, rank-8 FiLM scale/shift on the first hidden layer.
- Late fusion: each registered-width view projects to 64D; concatenate, project to 64D, then to
  rank. N6 uses three 1,536D full, nucleus-mask, and cell-mask views. N7 adds a separate 158D
  nuisance branch before fusion;
  the nuisance vector is not padded or treated as a 1,536D image. Target-removed remains a
  comparator, never a focal input.
- No BatchNorm and no encoder fine-tuning.
- AdamW, learning rate `1e-3`, weight decay in `{1e-4, 1e-2}`, batch size 256, maximum 100 epochs,
  patience 10, gradient clipping 1.0, and fixed seeds 17/29/41.
- Selection uses donor-balanced validation R². Reject any configuration with a failed inner donor,
  any inner-fold basis ceiling below 0.30, or median variance ratio below 0.5; then maximize
  donor/section/type R², donor/type R², prefer fewer parameters, larger weight decay, and the
  lexicographically smaller model ID.
- Refit each outer model for the median selected inner-fold epoch and average the three seed
  predictions while reporting each seed separately.

N0/N7 use the frozen deduplicated 158D `all_controls` matrix. The stored 31D
`full_nuisance_covariates` matrix is not a substitute. The exact paired families are
`neural_reference_mean_only`, `neural_combined_nuisance_only`, `neural_image_only`,
`neural_combined_nuisance_plus_image`, `neural_blank_patch`, and `neural_target_removed`. A
blank-patch arm remains required, but the registered source lacks its embedding; execution stays
`blocked_missing_feature_supplement` until a receipt-bound supplement is frozen.

## Engineering support rule

Every row below must pass. The 0.01 improvement over B2 is the nonlinear complexity tax. The
intrinsic localization diagnostic is the minimum of the paired N1, N3, and N4 donor/type macro
increments over N5; it remains non-authorizing even when positive.

| Requirement | Frozen threshold |
|---|---:|
| Donor/type macro residual-coordinate R² | ≥ 0.05 |
| Donor/section/type macro residual-coordinate R² | ≥ 0.05 |
| Improvement over B2 | ≥ 0.01 R² |
| Improvement over N0 | ≥ 0.03 R² |
| Donor/type macro relative-RMSE reduction over reference mean | ≥ 5% |
| Positive supported donor/type strata | ≥ 80% |
| Positive donors versus N0 | ≥ 80% |
| Maximum contribution from one donor | < 50% of gain |
| Molecular variance ratio | ≥ 0.50 |
| Median type coverage | ≥ 0.50 |
| Abstention | ≤ 0.50 |
| Rare-state recall drop | ≤ 0.20 |
| Within-section/type refitted-null empirical p | ≤ 0.01 |
| Different-block refitted-null empirical p | ≤ 0.01 |
| Prespecified focal arm versus target-removed increment | ≥ 0.01 R² |
| Best-registration R² minus all-row R² | ≥ -0.01 |

The best-registration criterion is presently non-evaluable at primary support 20 and therefore
fails closed. It cannot be dropped, imputed, or replaced after model results.

Predictions are dense: every registered row must receive a finite prediction, so v1 has no
post-hoc OOD abstention. Variance is the median gene-wise predicted/truth standard-deviation ratio
within supported donor/type strata, macro-averaged over types and donors. Rare states are the
outer-training-only lower and upper coordinate deciles, and rare-state recall drop is measured
against B2. The maximum donor gain fraction is the largest positive donor gain divided by the sum
of positive donor gains.

The final decision uses 100 complete refitted permutations for each of two null families:
within-section/type image derangement and different-spatial-block reassignment. Twenty permutations
per family are permitted only for schema and execution smoke testing. Each null repeats
preprocessing, target fitting, hyperparameter selection, training, checkpoint selection,
prediction, and scoring. A post-fit feature shuffle is not valid.

## Ordered interpretation

| Observed result | Permitted action |
|---|---|
| B2 beats B1; neural arms do not beat B2 | Retain full-feature ridge; historical projection was harmful |
| Neural image arm beats B2 but not N0 | Nonlinearity helps nonspecifically; stop claim progression |
| Focal and target-removed arms are similar | Attribute to context/technical structure, not target-cell information |
| Intrinsic increment appears only in poor registration | Do not pursue an intrinsic claim |
| N6 or N7 passes every support criterion | Register a new prospective estimator protocol before pristine outcomes |
| All neural arms fail | Stop neural expansion; prioritize measurement, crop representation, or a new encoder protocol |
| R² improves but variance preservation fails | Consider a separate RNA-target qualification; do not proceed to reference fusion |

Reference expansion, RNA autoencoding, encoder fine-tuning, adversarial alignment, optimal
transport, graphs, and iterative refinement are outside v1. A Stage B reference-expanded result can
never rescue a failed directly measured Stage A target.

## Immutable authorization block

Every generated report must contain:

```json
{
  "analysis_status": "retrospective_exposed_non_authorizing",
  "supports_new_prospective_estimator_protocol": false,
  "authorizes_h_cell": false,
  "authorizes_h_intrinsic": false,
  "authorizes_h_ref": false,
  "authorizes_full_heir": false
}
```

Only `supports_new_prospective_estimator_protocol` may become true after a complete pass. The four
biological authorization fields may not change.
