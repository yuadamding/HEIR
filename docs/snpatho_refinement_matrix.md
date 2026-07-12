# Native snPATHO refinement matrix

Status: **blocked_evidence**; matrix: **complete**; strict ordering: **fail**; full primary evidence: **blocked**.

Execution provenance verified: **false**; control transform hashes verified: **true**.

Scored 93 of 93 requested artifacts.

## Blockers

- `posthoc_adoption_not_original_execution_proof`: The exact-plan manifest validates current artifacts but adopted one or more pre-existing stages, so it cannot prove their original execution source revision.
- `missing_evidence_generic_atlas`: The generic-atlas RNA control requested by the benchmark plan is unavailable.
- `missing_evidence_label_permutation`: The label-permutation negative control requested by the benchmark plan is unavailable.
- `missing_evidence_state_omission`: The state-omission sensitivity requested by the benchmark plan is unavailable.
- `missing_evidence_reference_downsampling`: The 1,000/2,500/5,000/all-cell reference-downsampling sensitivity is unavailable.
- `invalid_evidence_unknown_mass_sweep`: 14 of 15 fixed-mass cases lack checkpoint-serialized unknown-mass metadata; post-hoc recipe adoption cannot prove the mass used to create those endpoints
- `missing_evidence_clean_independent_reannotation`: Native scANVI still uses published integrated annotations rather than an independent clean reannotation.
- `missing_evidence_untouched_external_cohort`: No untouched external cohort is available for confirmatory validation.

## Wrong-donor coverage and aggregates

Coverage: **18/18** directed target/source/seed cases; complete: **true**.

| Scope | Evaluable / expected | Mean paired delta | Worst paired delta | All positive |
|---|---:|---:|---:|---|
| all_directed | 18 / 18 | -0.004044 | -0.023129 | false |
| site_matched | 6 / 6 | -0.000026 | -0.002286 | false |

## HEIR median gene Spearman

| Sample | Seed | Variant | Value |
|---|---:|---|---:|
| 4066 | 17 | round0 | -0.089120 |
| 4066 | 17 | refined | -0.087674 |
| 4066 | 41 | round0 | 0.068757 |
| 4066 | 41 | refined | 0.066522 |
| 4066 | 89 | round0 | 0.059716 |
| 4066 | 89 | refined | 0.055250 |
| 4066 | 131 | round0 | 0.023925 |
| 4066 | 131 | refined | 0.020699 |
| 4066 | 197 | round0 | -0.074367 |
| 4066 | 197 | refined | -0.074603 |
| 4066 | 17 | round1 | -0.089319 |
| 4066 | 17 | round2 | -0.089450 |
| 4066 | 17 | round3 | -0.087734 |
| 4066 | 17 | prototype_only | -0.088167 |
| 4066 | 17 | image_shuffle | 0.003127 |
| 4066 | 17 | graph_shuffle | -0.050375 |
| 4066 | 17 | no_graph | -0.080415 |
| 4066 | 17 | wrong_donor_4399 | -0.068033 |
| 4066 | 17 | wrong_donor_4411 | -0.082280 |
| 4066 | 41 | prototype_only | 0.067313 |
| 4066 | 41 | image_shuffle | -0.003829 |
| 4066 | 41 | graph_shuffle | 0.044785 |
| 4066 | 41 | no_graph | 0.053626 |
| 4066 | 41 | wrong_donor_4399 | 0.101294 |
| 4066 | 41 | wrong_donor_4411 | 0.109377 |
| 4066 | 89 | prototype_only | 0.059154 |
| 4066 | 89 | image_shuffle | 0.002017 |
| 4066 | 89 | graph_shuffle | 0.062864 |
| 4066 | 89 | no_graph | 0.056472 |
| 4066 | 89 | wrong_donor_4399 | 0.052073 |
| 4066 | 89 | wrong_donor_4411 | 0.061645 |
| 4399 | 17 | round0 | -0.035372 |
| 4399 | 17 | refined | -0.035161 |
| 4399 | 41 | round0 | -0.006846 |
| 4399 | 41 | refined | -0.006589 |
| 4399 | 89 | round0 | -0.043400 |
| 4399 | 89 | refined | -0.043641 |
| 4399 | 131 | round0 | -0.019889 |
| 4399 | 131 | refined | -0.018038 |
| 4399 | 197 | round0 | -0.038167 |
| 4399 | 197 | refined | -0.037069 |
| 4399 | 17 | round1 | -0.035336 |
| 4399 | 17 | round2 | -0.035953 |
| 4399 | 17 | round3 | -0.035105 |
| 4399 | 17 | prototype_only | -0.035257 |
| 4399 | 17 | image_shuffle | 0.002850 |
| 4399 | 17 | graph_shuffle | -0.029424 |
| 4399 | 17 | no_graph | -0.033972 |
| 4399 | 17 | wrong_donor_4066 | -0.034254 |
| 4399 | 17 | wrong_donor_4411 | -0.034014 |
| 4399 | 41 | prototype_only | -0.007506 |
| 4399 | 41 | image_shuffle | 0.004464 |
| 4399 | 41 | graph_shuffle | -0.018829 |
| 4399 | 41 | no_graph | -0.012334 |
| 4399 | 41 | wrong_donor_4066 | -0.028946 |
| 4399 | 41 | wrong_donor_4411 | -0.017221 |
| 4399 | 89 | prototype_only | -0.043623 |
| 4399 | 89 | image_shuffle | -0.001219 |
| 4399 | 89 | graph_shuffle | -0.029932 |
| 4399 | 89 | no_graph | -0.045184 |
| 4399 | 89 | wrong_donor_4066 | -0.037719 |
| 4399 | 89 | wrong_donor_4411 | -0.040162 |
| 4411 | 17 | round0 | 0.000379 |
| 4411 | 17 | refined | 0.000001 |
| 4411 | 41 | round0 | 0.006617 |
| 4411 | 41 | refined | 0.006351 |
| 4411 | 89 | round0 | 0.007142 |
| 4411 | 89 | refined | 0.006157 |
| 4411 | 131 | round0 | 0.002507 |
| 4411 | 131 | refined | 0.001393 |
| 4411 | 197 | round0 | 0.001638 |
| 4411 | 197 | refined | 0.000070 |
| 4411 | 17 | round1 | -0.000366 |
| 4411 | 17 | round2 | -0.000279 |
| 4411 | 17 | round3 | -0.000962 |
| 4411 | 17 | prototype_only | 0.000323 |
| 4411 | 17 | image_shuffle | -0.001755 |
| 4411 | 17 | graph_shuffle | -0.012101 |
| 4411 | 17 | no_graph | -0.005124 |
| 4411 | 17 | wrong_donor_4066 | 0.006555 |
| 4411 | 17 | wrong_donor_4399 | -0.001599 |
| 4411 | 41 | prototype_only | 0.006192 |
| 4411 | 41 | image_shuffle | -0.001149 |
| 4411 | 41 | graph_shuffle | 0.014180 |
| 4411 | 41 | no_graph | 0.010760 |
| 4411 | 41 | wrong_donor_4066 | 0.007969 |
| 4411 | 41 | wrong_donor_4399 | 0.005641 |
| 4411 | 89 | prototype_only | 0.006031 |
| 4411 | 89 | image_shuffle | 0.001440 |
| 4411 | 89 | graph_shuffle | 0.010178 |
| 4411 | 89 | no_graph | 0.008873 |
| 4411 | 89 | wrong_donor_4066 | 0.006277 |
| 4411 | 89 | wrong_donor_4399 | 0.002886 |

## Strict ordering checks

Pass/fail and delta both use the paired median across per-gene Spearman differences.

| Check | Sample | Seed | Status | Paired delta |
|---|---|---:|---|---:|
| refined_gt_round0 | 4066 | 17 | pass | 0.000544 |
| refined_gt_hard_baseline | 4066 | 17 | pass | 0.004618 |
| refined_gt_soft_baseline | 4066 | 17 | pass | 0.004590 |
| refined_gt_round0 | 4066 | 41 | fail | -0.002379 |
| refined_gt_hard_baseline | 4066 | 41 | fail | -0.013814 |
| refined_gt_soft_baseline | 4066 | 41 | fail | -0.016614 |
| refined_gt_round0 | 4066 | 89 | fail | -0.002321 |
| refined_gt_hard_baseline | 4066 | 89 | fail | -0.011588 |
| refined_gt_soft_baseline | 4066 | 89 | fail | -0.011289 |
| refined_gt_round0 | 4066 | 131 | fail | -0.001166 |
| refined_gt_hard_baseline | 4066 | 131 | fail | -0.026783 |
| refined_gt_soft_baseline | 4066 | 131 | fail | -0.024514 |
| refined_gt_round0 | 4066 | 197 | pass | 0.000625 |
| refined_gt_hard_baseline | 4066 | 197 | pass | 0.012268 |
| refined_gt_soft_baseline | 4066 | 197 | pass | 0.012175 |
| round0_gt_prototype_only | 4066 | 17 | pass | 0.000291 |
| refined_gt_prototype_only | 4066 | 17 | pass | 0.000455 |
| refined_gt_image_shuffle | 4066 | 17 | fail | -0.093369 |
| refined_gt_graph_shuffle | 4066 | 17 | fail | -0.013780 |
| refined_gt_no_graph | 4066 | 17 | fail | -0.001818 |
| refined_gt_wrong_donor | 4066 | 17 | fail | -0.004225 |
| refined_gt_wrong_donor | 4066 | 17 | fail | -0.004176 |
| round0_gt_prototype_only | 4066 | 41 | pass | 0.001259 |
| refined_gt_prototype_only | 4066 | 41 | fail | -0.000864 |
| refined_gt_image_shuffle | 4066 | 41 | pass | 0.066307 |
| refined_gt_graph_shuffle | 4066 | 41 | pass | 0.020086 |
| refined_gt_no_graph | 4066 | 41 | pass | 0.012297 |
| refined_gt_wrong_donor | 4066 | 41 | fail | -0.017693 |
| refined_gt_wrong_donor | 4066 | 41 | fail | -0.023129 |
| round0_gt_prototype_only | 4066 | 89 | pass | 0.000853 |
| refined_gt_prototype_only | 4066 | 89 | fail | -0.001165 |
| refined_gt_image_shuffle | 4066 | 89 | pass | 0.051224 |
| refined_gt_graph_shuffle | 4066 | 89 | fail | -0.002974 |
| refined_gt_no_graph | 4066 | 89 | pass | 0.001742 |
| refined_gt_wrong_donor | 4066 | 89 | fail | -0.000542 |
| refined_gt_wrong_donor | 4066 | 89 | fail | -0.001604 |
| refined_gt_round0 | 4399 | 17 | pass | 0.000472 |
| refined_gt_hard_baseline | 4399 | 17 | fail | -0.004029 |
| refined_gt_soft_baseline | 4399 | 17 | fail | -0.005433 |
| refined_gt_round0 | 4399 | 41 | pass | 0.000044 |
| refined_gt_hard_baseline | 4399 | 41 | fail | -0.000259 |
| refined_gt_soft_baseline | 4399 | 41 | pass | 0.005132 |
| refined_gt_round0 | 4399 | 89 | pass | 0.000350 |
| refined_gt_hard_baseline | 4399 | 89 | fail | -0.004165 |
| refined_gt_soft_baseline | 4399 | 89 | fail | -0.001986 |
| refined_gt_round0 | 4399 | 131 | pass | 0.001046 |
| refined_gt_hard_baseline | 4399 | 131 | fail | -0.004063 |
| refined_gt_soft_baseline | 4399 | 131 | fail | -0.006825 |
| refined_gt_round0 | 4399 | 197 | pass | 0.000787 |
| refined_gt_hard_baseline | 4399 | 197 | fail | -0.001439 |
| refined_gt_soft_baseline | 4399 | 197 | fail | -0.000599 |
| round0_gt_prototype_only | 4399 | 17 | fail | -0.000529 |
| refined_gt_prototype_only | 4399 | 17 | pass | 0.000030 |
| refined_gt_image_shuffle | 4399 | 17 | fail | -0.038364 |
| refined_gt_graph_shuffle | 4399 | 17 | fail | -0.009596 |
| refined_gt_no_graph | 4399 | 17 | fail | -0.007172 |
| refined_gt_wrong_donor | 4399 | 17 | fail | -0.006016 |
| refined_gt_wrong_donor | 4399 | 17 | fail | -0.002286 |
| round0_gt_prototype_only | 4399 | 41 | pass | 0.000247 |
| refined_gt_prototype_only | 4399 | 41 | pass | 0.000062 |
| refined_gt_image_shuffle | 4399 | 41 | fail | -0.009140 |
| refined_gt_graph_shuffle | 4399 | 41 | pass | 0.004550 |
| refined_gt_no_graph | 4399 | 41 | pass | 0.003990 |
| refined_gt_wrong_donor | 4399 | 41 | pass | 0.007513 |
| refined_gt_wrong_donor | 4399 | 41 | pass | 0.003077 |
| round0_gt_prototype_only | 4399 | 89 | fail | -0.000304 |
| refined_gt_prototype_only | 4399 | 89 | pass | 0.000042 |
| refined_gt_image_shuffle | 4399 | 89 | fail | -0.044362 |
| refined_gt_graph_shuffle | 4399 | 89 | fail | -0.013458 |
| refined_gt_no_graph | 4399 | 89 | fail | -0.004187 |
| refined_gt_wrong_donor | 4399 | 89 | fail | -0.008490 |
| refined_gt_wrong_donor | 4399 | 89 | fail | -0.001412 |
| refined_gt_round0 | 4411 | 17 | fail | -0.000772 |
| refined_gt_hard_baseline | 4411 | 17 | pass | 0.001255 |
| refined_gt_soft_baseline | 4411 | 17 | pass | 0.002334 |
| refined_gt_round0 | 4411 | 41 | fail | -0.001003 |
| refined_gt_hard_baseline | 4411 | 41 | pass | 0.001095 |
| refined_gt_soft_baseline | 4411 | 41 | fail | -0.000287 |
| refined_gt_round0 | 4411 | 89 | fail | -0.000373 |
| refined_gt_hard_baseline | 4411 | 89 | pass | 0.001393 |
| refined_gt_soft_baseline | 4411 | 89 | pass | 0.001500 |
| refined_gt_round0 | 4411 | 131 | fail | -0.000173 |
| refined_gt_hard_baseline | 4411 | 131 | pass | 0.002527 |
| refined_gt_soft_baseline | 4411 | 131 | pass | 0.003150 |
| refined_gt_round0 | 4411 | 197 | fail | -0.001185 |
| refined_gt_hard_baseline | 4411 | 197 | pass | 0.001151 |
| refined_gt_soft_baseline | 4411 | 197 | fail | -0.000133 |
| round0_gt_prototype_only | 4411 | 17 | pass | 0.000793 |
| refined_gt_prototype_only | 4411 | 17 | pass | 0.000011 |
| refined_gt_image_shuffle | 4411 | 17 | fail | -0.001286 |
| refined_gt_graph_shuffle | 4411 | 17 | pass | 0.004195 |
| refined_gt_no_graph | 4411 | 17 | pass | 0.001559 |
| refined_gt_wrong_donor | 4411 | 17 | fail | -0.006894 |
| refined_gt_wrong_donor | 4411 | 17 | fail | -0.000920 |
| round0_gt_prototype_only | 4411 | 41 | pass | 0.000871 |
| refined_gt_prototype_only | 4411 | 41 | pass | 0.000043 |
| refined_gt_image_shuffle | 4411 | 41 | pass | 0.004371 |
| refined_gt_graph_shuffle | 4411 | 41 | fail | -0.005748 |
| refined_gt_no_graph | 4411 | 41 | fail | -0.000428 |
| refined_gt_wrong_donor | 4411 | 41 | fail | -0.006684 |
| refined_gt_wrong_donor | 4411 | 41 | pass | 0.000268 |
| round0_gt_prototype_only | 4411 | 89 | pass | 0.000302 |
| refined_gt_prototype_only | 4411 | 89 | pass | 0.000017 |
| refined_gt_image_shuffle | 4411 | 89 | pass | 0.005546 |
| refined_gt_graph_shuffle | 4411 | 89 | fail | -0.003485 |
| refined_gt_no_graph | 4411 | 89 | fail | -0.001109 |
| refined_gt_wrong_donor | 4411 | 89 | fail | -0.000692 |
| refined_gt_wrong_donor | 4411 | 89 | pass | 0.001117 |
