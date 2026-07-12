# snPATHO-DeepBench-v1 retrospective result

This report does not replace or reinterpret `snPATHO-Locked-v0.2`.

## Scope and benchmark contracts

The available v0.2 artifacts form a retrospective capture-aware diagnostic. Although target H&E is an allowed input, v0.2 derived its OOD threshold from each target slide's 95th percentile. That target-specific calibration violates the external freeze, so these results establish neither Track A1 nor Track A2 compliance.

The required primary R1 reference is the matched **FFPE snPATHO-seq-only** reference. The historical v0.2 reference instead pools FFPE snPATHO-seq, frozen Flex, and frozen 3-prime nuclei; its results and type-mean baseline are therefore retrospective diagnostics, not the primary R1 comparison.

Hash-bound FFPE-only count references and prototype banks are now consumed. The type-mean ladder separates hard-assigned mass, shared soft mass, expected soft mass, and equal-cell hard/soft estimands. It retains the published integrated-workflow annotation. A native-scANVI prototype-only control is scored in the development matrix, but the annotation and execution-provenance gates remain incomplete, so this is not the requested clean primary R1 comparison.

Cell-to-spot weights in this diagnostic are **historical integrated-reference library-size weights**, not assay-corrected biological RNA-mass estimates. Both hard-argmax and probability-weighted soft historical type-mean baselines are reported under each explicit mass estimand. Missing prediction types fail closed; no global profile is substituted.

The available null is a complete shuffle of final cell records across assigned nuclei. It does not substitute for the separately required shuffled-image-feature and coordinate-shuffled-graph controls.

Space Ranger supplies the common segmentation but no calibrated segmentation confidence. Historical v0.2 substituted a constant confidence of 1.0, making the >=0.50 anchor gate vacuous; refinement benchmarking remains blocked until that measurement is available.

Expression-detection AUROC uses observed expression > 0 as its label; top-10% Dice/Jaccard are the hotspot metrics. Exact top-decile sets break cutoff ties by ascending frozen spot-row index. Moran's I uses a directed, unweighted 6-NN graph that is not symmetrized or row-standardized.

## Executability

| Component | Status | Reason |
|---|---|---|
| locked_round0_predictions | ready | Hash-frozen v0.2 predictions for all three specimens |
| historical_integrated_hard_type_mean | ready | Hard argmax profile with shared soft expected RNA-mass weights derived from the v0.2 pooled multi-workflow reference |
| historical_integrated_soft_type_mean | ready | Probability-weighted baseline derived from the v0.2 pooled multi-workflow reference |
| historical_integrated_hard_assigned_mass_type_mean | ready | Hard argmax profile with hard-assigned type-median RNA-mass weights |
| historical_integrated_equal_cell_type_means | ready | Hard and soft type-profile baselines with equal-cell aggregation |
| historical_integrated_pseudobulk | ready | Derived from the v0.2 pooled integrated multi-workflow reference |
| historical_final_cell_record_shuffle | ready | 100 deterministic independently seeded final-cell-record permutations are summarized compactly; draw 0 remains the single-method backward comparison. These historical permutations do not substitute for image-feature or graph shuffle controls; those controls are consumed separately by the native-scANVI refinement matrix |
| historical_integrated_reference_library_size_weighting | ready | Type-median library sizes from the historical pooled multi-workflow reference |
| primary_ffpe_snpatho_reference | partial_materialized_not_benchmark_ready | FFPE-snPATHO-only native scANVI references and rare-complete prototype banks are hash-validated, and the scored refinement matrix consumes the native prototype-only control. The labels still come from the published integrated-workflow annotation; an independent clean reannotation is absent, so this remains a sensitivity analysis. |
| primary_spot_qc | partial | processed RDS spots are author-QC-whitelisted and positive-library with >=3 nuclei; the required >=50% per-spot H&E tissue fraction plus explicit exclusion flags and reasons are absent |
| hierarchical_spatial_bootstrap | partial | paired donor/gene bootstrap is available; the historical run lacks frozen connected spatial-block definitions |
| alternative_rna_raw_inputs | partial | downloaded integrated objects expose FFPE snPATHO, frozen SNAP snPATHO/Flex, and frozen 3-prime strata, but workflow-specific frozen references/predictions have not been prepared and no scFFPE stratum is present in those objects |
| externally_frozen_ood_rule | blocked_noncompliant_historical_input | v0.2 recalibrated the OOD threshold from each target H&E slide's 95th percentile; the capture-aware historical run therefore does not establish Track A/A2 compliance |
| segmentation_confidence_anchor_gate | blocked_nonfunctional_gate | Space Ranger exports no calibrated segmentation confidence and the refinement runs substitute 1.0, so the scored development matrix does not satisfy the primary benchmark's calibrated anchor-confidence requirement |
| graph_sensitivity_and_rewiring | partial_consumed_via_refinement_matrix | The provenance-validated matrix consumes scored graph-shuffle and no-graph controls. The requested 8-NN, radius, multiscale, and degree-preserving rewiring sensitivities remain absent. |
| refinement_trajectory_and_ablations | partial_consumed_via_refinement_matrix | The provenance-validated matrix consumes round 0/final predictions across five seeds, the complete round 1-4 score trajectory at the prespecified trajectory seed, and prototype-only/image-shuffle/graph-shuffle/no-graph/wrong-donor controls. E-step, prior-update, refinement-gate, and anchor/map stability analyses remain unscored. |
| complete_negative_control_matrix | partial_consumed_via_refinement_matrix | Prototype-only, image-feature-shuffle, graph-shuffle, no-graph, and wrong-donor controls are consumed by the provenance-validated matrix. Label and prototype permutations, generic-atlas RNA, state omission, reference downsampling, block shuffles, toroidal shifts, and coordinate perturbations remain absent. |
| seed_ensemble_stability | partial_consumed_performance_matrix_only | Five-seed prediction-level performance is consumed, but map, anchor, assignment-overlap, and between-model stability have not been scored; the matrix must not be interpreted as ensemble-stability evidence. |
| track_a1_external_personalization | blocked_not_implemented_or_missing_artifact | No externally frozen H&E-plus-snRNA-only predictions exist |
| track_b_leave_one_specimen_out | blocked_not_implemented_or_missing_artifact | No nested leave-one-specimen-out configurations or predictions are frozen |
| independent_snpatho_reannotation | blocked_not_implemented_or_missing_artifact | Historical labels came from integrated workflow objects; no snPATHO-only frozen ontology and marker-review artifact exists |
| reference_size_and_per_type_caps | blocked_not_implemented_or_missing_artifact | The requested five draws at 1k/2.5k/5k/all and 100/250/500/1k per-type caps were not generated |
| hierarchical_ontology_scoring | blocked_not_implemented_or_missing_artifact | No frozen compartment/major-type/supported-subtype mapping and evaluation output is available |
| manual_segmentation_roi_audit | blocked_not_implemented_or_missing_artifact | The 24 stratified ROIs per specimen and independent detection annotations do not exist |
| image_multiscale_and_morphology_ablations | blocked_not_implemented_or_missing_artifact | The 32/128/384-um and explicit-morphology ablation predictions were not generated |
| composition_controlled_residuals | blocked_not_implemented_or_missing_artifact | Five spatial-block folds, independent composition covariates, library covariates, and pathologist regions are absent |
| manual_cell_type_benchmark | blocked_not_implemented_or_missing_artifact | No two-reviewer consensus nucleus labels or evaluation-only confidence scores exist |
| spot_composition_consensus | blocked_not_implemented_or_missing_artifact | No frozen RCTD/cell2location/DestVI consensus artifact is available |
| uncertainty_calibration_and_risk_coverage | blocked_not_implemented_or_missing_artifact | Historical artifacts do not contain full posterior ensembles or fixed-coverage evaluation outputs |
| unknown_state_omission_stress_test | blocked_not_implemented_or_missing_artifact | Per-major-type reference-omission predictions were not generated |
| core_model_ablation_matrix | blocked_not_implemented_or_missing_artifact | No-UOT, balanced-OT, query/final-latent UOT, no/low-rank residual, no covariance, fixed/updated prior, and initializer ablations are absent |
| expanded_spatial_structure_metrics | blocked_not_implemented_or_missing_artifact | Geary C, semivariogram, spatial EMD, and boundary-localization scorers are not implemented in this executable subset |
| per_gene_block_permutation_fdr | blocked_not_implemented_or_missing_artifact | Frozen connected blocks and block-permutation nulls needed for BH-FDR are absent |
| biological_case_study_endpoints | blocked_not_implemented_or_missing_artifact | Prespecified HER2/DCIS/calcium, tumor-liver, and liver-resident program definitions and region labels are absent |
| complete_computational_benchmark | blocked_not_implemented_or_missing_artifact | Historical inference telemetry exists, but segmentation, feature extraction, training, refinement, CPU memory, checkpoint/cache size, and energy are incomplete |
| primary_ffpe_snpatho_reference_manifest | partial_consumed_retrospective_sensitivity | Hash-validated FFPE-only counts are consumed for the matched type-mean estimand ladder. Native scANVI references and rare-complete prototype banks are separately hash-validated, but the published integrated annotations are not an independent clean R1 reannotation. |
| refined_predictions | consumed_via_provenance_validated_refinement_matrix | Round-0 and final refined predictions for every prespecified specimen and five-seed case are scored in the hash-bound matrix; strict ordering is fail |
| five_seed_predictions | ready_provenance_validated_five_seed_matrix | Every prespecified specimen/seed PredictionBundle is hash-validated and bound to the native scANVI latent/expression identities. This establishes scored performance coverage, not map or anchor ensemble stability |
| refinement_matrix_summary | consumed_provenance_validated_matrix_strict_ordering_failed | The compact summary is plan-hash-bound, covers every requested specimen, seed, round 0-4 trajectory artifact, prototype-only/image-shuffle/graph-shuffle/no-graph/wrong-donor control, and strict comparison, and reports strict ordering fail |
| alternative_workflow_references | blocked_missing_artifact | No scFFPE/Flex/3-prime reference artifacts are frozen |
| wrong_donor_predictions | consumed_via_provenance_validated_refinement_matrix | The wrong-donor prototype control is scored for every prespecified control case in the hash-bound refinement matrix; strict ordering is fail |
| generic_atlas_predictions | blocked_missing_artifact | Generic-atlas HEIR predictions have not been generated |
| h_and_e_only_predictions | blocked_missing_artifact | No RNA-free H&E prediction artifact is supplied |
| image_shuffle_predictions | consumed_via_provenance_validated_refinement_matrix | The shuffled-image-feature control is scored for every prespecified control case in the hash-bound refinement matrix; strict ordering is fail |
| graph_shuffle_predictions | consumed_via_provenance_validated_refinement_matrix | The shuffled-graph control is scored for every prespecified control case in the hash-bound refinement matrix; strict ordering is fail |
| no_geometry_predictions | partial_no_graph_consumed_via_refinement_matrix | The hash-bound matrix consumes the prespecified no-graph control, but a dedicated no-geometry prediction that removes every spatial input remains absent; strict ordering is fail |
| manual_nucleus_labels | blocked_missing_artifact | No evaluation-only consensus nucleus annotations are available |
| spot_composition_covariates | blocked_missing_artifact | No frozen independent spot-composition covariates exist |
| pathologist_regions | blocked_missing_artifact | No frozen pathologist-region artifact exists |
| published_program_definitions | blocked_missing_artifact | The 15 published program definitions are not frozen locally |
| author_qc_tissue_fraction | blocked_missing_artifact | The processed RDS materializes the author-QC spot whitelist, but explicit exclusion flags/reasons and the required >=50% per-spot H&E tissue fraction are absent |
| segmentation_sensitivity_predictions | blocked_missing_artifact | Only the common Space Ranger nucleus set is frozen |
| regional_384um_features | blocked_missing_artifact | Historical OmiCLIP features contain only 32 and 128 um scales |
| native_scanvi_checkpoint | ready_recursively_hash_validated_native_scanvi | The external native model directory, decoder gene order and expression-normalization contract, per-specimen latent references, rare-complete prototype banks, and RNA residual geometries (source reference, latent identity, type order, rank, and bounds) were parsed and recursively hash-validated; published integrated annotations remain a declared sensitivity |

## Locked-v0.2 versus DeepBench-v1 reconciliation

The two reports use different estimands, so their values need not match.

| Feature | Locked-v0.2 | DeepBench-v1 |
|---|---|---|
| minimum_nuclei_per_spot | >=1 | >=3 |
| cell_aggregation | equal-cell | historical_integrated_reference_library_size_weighted |
| type_profile | historical locked implementation | pooled raw counts divided by full-library mass |
| constant_prediction_policy | earlier metric implementation | correlation fixed at zero when observed expression varies |
| shuffle | historical spatial shuffle | complete final-cell-record shuffle draw 0; repeated null reported separately |

## Type-mean baseline estimands

The legacy hard method IDs remain available, but their shared-soft-mass estimand is now explicit.

| Method | Reference | Cell profile | Cell RNA mass |
|---|---|---|---|
| historical_integrated_hard_type_mean_hard_assigned_type_mass | historical_integrated_multi_workflow_reference | hard_argmax_type_profile | hard_assigned_type_median_library_size |
| historical_integrated_hard_type_mean | historical_integrated_multi_workflow_reference | hard_argmax_type_profile | shared_soft_expected_type_median_library_size |
| historical_integrated_soft_type_mean | historical_integrated_multi_workflow_reference | probability_weighted_soft_type_profile | expected_soft_type_median_library_size |
| historical_integrated_hard_type_mean_equal_cell | historical_integrated_multi_workflow_reference | hard_argmax_type_profile | equal_cell |
| historical_integrated_soft_type_mean_equal_cell | historical_integrated_multi_workflow_reference | probability_weighted_soft_type_profile | equal_cell |
| r1_ffpe_snpatho_integrated_annotation_sensitivity_hard_type_mean_hard_assigned_type_mass | matched_ffpe_snpatho_count_reference_integrated_annotation_sensitivity | hard_argmax_type_profile | hard_assigned_type_median_library_size |
| r1_ffpe_snpatho_integrated_annotation_sensitivity_hard_type_mean | matched_ffpe_snpatho_count_reference_integrated_annotation_sensitivity | hard_argmax_type_profile | shared_soft_expected_type_median_library_size |
| r1_ffpe_snpatho_integrated_annotation_sensitivity_soft_type_mean | matched_ffpe_snpatho_count_reference_integrated_annotation_sensitivity | probability_weighted_soft_type_profile | expected_soft_type_median_library_size |
| r1_ffpe_snpatho_integrated_annotation_sensitivity_hard_type_mean_equal_cell | matched_ffpe_snpatho_count_reference_integrated_annotation_sensitivity | hard_argmax_type_profile | equal_cell |
| r1_ffpe_snpatho_integrated_annotation_sensitivity_soft_type_mean_equal_cell | matched_ffpe_snpatho_count_reference_integrated_annotation_sensitivity | probability_weighted_soft_type_profile | equal_cell |

## Reference type support

| Specimen | Prediction types | Supported | Missing | Hard fallback cells |
|---|---:|---:|---|---:|
| 4066 | 12 | 12 | none | 0 (0.0000) |
| 4399 | 11 | 11 | none | 0 (0.0000) |
| 4411 | 9 | 9 | none | 0 (0.0000) |

FFPE-only R1 count-reference support (integrated-annotation sensitivity):

| Specimen | Prediction types | Supported | Missing | Hard fallback cells |
|---|---:|---:|---|---:|
| 4066 | 12 | 12 | none | 0 (0.0000) |
| 4399 | 11 | 11 | none | 0 (0.0000) |
| 4411 | 9 | 9 | none | 0 (0.0000) |

Count-reference support and prototype-bank support are distinct:

| Specimen | Count-reference-supported types | Prototype-supported types | Prototype-omitted types |
|---|---|---|---|
| 4066 | B_cells, CAF, DC, Endothelial, Epithelial_basal, Epithelial_cancer, Epithelial_luminal, MAST, Macrophage, Myoepithelial, PVL, T_cells | B_cells, CAF, DC, Endothelial, Epithelial_basal, Epithelial_cancer, Epithelial_luminal, MAST, Macrophage, Myoepithelial, PVL, T_cells | none |
| 4399 | CAF, Cholangiocyte, Endothelial, Epithelial_cancer, Hepatocyte, LSEC, Lymphatic_endothelial, Macrophage, Mixed_lymphocytes, PVL, RBCs | CAF, Cholangiocyte, Endothelial, Epithelial_cancer, Hepatocyte, LSEC, Lymphatic_endothelial, Macrophage, Mixed_lymphocytes, PVL, RBCs | none |
| 4411 | CAF, Cholangiocyte, Endothelial, Epithelial_cancer, Hepatocyte, LSEC, Macrophage, Mixed_lymphocytes, RBCs | CAF, Cholangiocyte, Endothelial, Epithelial_cancer, Hepatocyte, LSEC, Macrophage, Mixed_lymphocytes, RBCs | none |

Native rare-complete support is reported separately from the legacy SVD sensitivity bank. Refined-run fairness uses the native bank:

| Specimen | Fairness source | Native supported / omitted | Legacy supported / omitted |
|---|---|---|---|
| 4066 | native_scanvi_rare_complete_prototype_bank | B_cells, CAF, DC, Endothelial, Epithelial_basal, Epithelial_cancer, Epithelial_luminal, MAST, Macrophage, Myoepithelial, PVL, T_cells / none | B_cells, CAF, DC, Endothelial, Epithelial_basal, Epithelial_cancer, Macrophage, Myoepithelial, PVL, T_cells / Epithelial_luminal, MAST |
| 4399 | native_scanvi_rare_complete_prototype_bank | CAF, Cholangiocyte, Endothelial, Epithelial_cancer, Hepatocyte, LSEC, Lymphatic_endothelial, Macrophage, Mixed_lymphocytes, PVL, RBCs / none | CAF, Endothelial, Epithelial_cancer, Hepatocyte, LSEC, Macrophage, PVL / Cholangiocyte, Lymphatic_endothelial, Mixed_lymphocytes, RBCs |
| 4411 | native_scanvi_rare_complete_prototype_bank | CAF, Cholangiocyte, Endothelial, Epithelial_cancer, Hepatocyte, LSEC, Macrophage, Mixed_lymphocytes, RBCs / none | CAF, Endothelial, Epithelial_cancer, Hepatocyte, LSEC, Macrophage, Mixed_lymphocytes, RBCs / Cholangiocyte |

## Type-probability map audit

Hard occupancy and hard/soft spot-mixture variation are computed over assigned nuclei in the primary evaluated spots.

| Specimen | Occupied hard types | Hard assignments | Mean normalized entropy | Hard-mixture constant types | Soft-mixture constant types |
|---|---:|---|---:|---|---|
| 4066 | 2 | B_cells=0, CAF=10091, DC=0, Endothelial=0, Epithelial_basal=0, Epithelial_cancer=28511, Epithelial_luminal=0, MAST=0, Macrophage=0, Myoepithelial=0, PVL=0, T_cells=0 | 0.656925 | B_cells, DC, Endothelial, Epithelial_basal, Epithelial_luminal, MAST, Macrophage, Myoepithelial, PVL, T_cells | none |
| 4399 | 1 | CAF=0, Cholangiocyte=0, Endothelial=0, Epithelial_cancer=28180, Hepatocyte=0, LSEC=0, Lymphatic_endothelial=0, Macrophage=0, Mixed_lymphocytes=0, PVL=0, RBCs=0 | 0.382165 | CAF, Cholangiocyte, Endothelial, Epithelial_cancer, Hepatocyte, LSEC, Lymphatic_endothelial, Macrophage, Mixed_lymphocytes, PVL, RBCs | none |
| 4411 | 2 | CAF=0, Cholangiocyte=0, Endothelial=0, Epithelial_cancer=31768, Hepatocyte=1857, LSEC=0, Macrophage=0, Mixed_lymphocytes=0, RBCs=0 | 0.510953 | CAF, Cholangiocyte, Endothelial, LSEC, Macrophage, Mixed_lymphocytes, RBCs | none |

## Historical round-0 diagnostic

The paired statistic is `median_g(rho_HEIR,g - rho_historical-integrated-hard-type-mean,g)`; it is not the difference between the two marginal medians.

| Specimen | Median paired per-gene delta | MSE improvement vs hard type mean |
|---|---:|---:|
| 4066 | -0.089628 | -0.020298 |
| 4399 | 0.005702 | 0.003778 |
| 4411 | 0.006076 | 0.002119 |

## Repeated final-cell-record shuffle null

The preserved draw-0 method is one member of a 100-per-specimen null. Expression and its library-size weight move together. This retrospective record shuffle does not replace image-feature or coordinate/graph reruns.

| Specimen | HEIR median-gene Spearman | Null median | Null empirical 95% interval | HEIR percentile in null | Above null upper? |
|---|---:|---:|---:|---:|---|
| 4066 | -0.022134 | 0.001058 | [-0.001573, 0.004288] | 0.000 | no |
| 4399 | 0.005702 | 0.000396 | [-0.001688, 0.002768] | 1.000 | yes |
| 4411 | -0.003504 | -0.000088 | [-0.002973, 0.002320] | 0.000 | no |

The equal-weight specimen-macro null median was **0.000347**, with empirical 95% interval **[-0.000867, 0.002183]**. HEIR exceeded the specimen null upper bound in one of three cases, so the prespecified at-least-two rule failed.

## Equal-weight specimen macro summaries

| Method | Median-gene Spearman | Median-gene MSE | Spot coverage |
|---|---:|---:|---:|
| heir_round0_historical_integrated_reference_library_size_weighted | -0.006645 | 0.119435 | 1.000000 |
| heir_round0_historical_integrated_reference_library_size_weighted_nonabstained | -0.002905 | 0.112608 | 0.936638 |
| heir_round0_equal_cell | -0.006968 | 0.119504 | 1.000000 |
| historical_integrated_hard_type_mean_hard_assigned_type_mass | 0.036932 | 0.114915 | 1.000000 |
| historical_integrated_hard_type_mean | 0.036853 | 0.114635 | 1.000000 |
| historical_integrated_soft_type_mean | 0.028037 | 0.138230 | 1.000000 |
| historical_integrated_hard_type_mean_equal_cell | 0.036932 | 0.114536 | 1.000000 |
| historical_integrated_soft_type_mean_equal_cell | 0.028335 | 0.138615 | 1.000000 |
| historical_integrated_snrna_pseudobulk | 0.000000 | 0.118924 | 1.000000 |
| heir_final_cell_record_shuffle_historical_integrated_reference_library_size_weighted | 0.001978 | 0.119865 | 1.000000 |
| r1_ffpe_snpatho_integrated_annotation_sensitivity_hard_type_mean_hard_assigned_type_mass | -0.033400 | 0.173240 | 1.000000 |
| r1_ffpe_snpatho_integrated_annotation_sensitivity_hard_type_mean | -0.032600 | 0.176071 | 1.000000 |
| r1_ffpe_snpatho_integrated_annotation_sensitivity_soft_type_mean | -0.038313 | 0.174913 | 1.000000 |
| r1_ffpe_snpatho_integrated_annotation_sensitivity_hard_type_mean_equal_cell | -0.029407 | 0.201001 | 1.000000 |
| r1_ffpe_snpatho_integrated_annotation_sensitivity_soft_type_mean_equal_cell | -0.035291 | 0.199777 | 1.000000 |
| refined_heir_matched_ffpe_r1_reference_library_size_weighted | -0.040944 | 0.165701 | 1.000000 |

## Constant-prediction audit

Only nonzero counts are listed. A constant prediction receives correlation zero when observed expression varies; an observed-constant gene is excluded.

| Specimen | Method | Prediction-constant scored zero | Observed-constant excluded |
|---|---|---:|---:|
| 4066 | historical_integrated_snrna_pseudobulk | 500 | 0 |
| 4066 | r1_ffpe_snpatho_integrated_annotation_sensitivity_hard_type_mean_hard_assigned_type_mass | 1 | 0 |
| 4066 | r1_ffpe_snpatho_integrated_annotation_sensitivity_hard_type_mean | 1 | 0 |
| 4066 | r1_ffpe_snpatho_integrated_annotation_sensitivity_soft_type_mean | 1 | 0 |
| 4066 | r1_ffpe_snpatho_integrated_annotation_sensitivity_hard_type_mean_equal_cell | 1 | 0 |
| 4066 | r1_ffpe_snpatho_integrated_annotation_sensitivity_soft_type_mean_equal_cell | 1 | 0 |
| 4399 | historical_integrated_hard_type_mean_hard_assigned_type_mass | 500 | 0 |
| 4399 | historical_integrated_hard_type_mean | 500 | 0 |
| 4399 | historical_integrated_soft_type_mean | 2 | 0 |
| 4399 | historical_integrated_hard_type_mean_equal_cell | 500 | 0 |
| 4399 | historical_integrated_soft_type_mean_equal_cell | 2 | 0 |
| 4399 | historical_integrated_snrna_pseudobulk | 500 | 0 |
| 4399 | r1_ffpe_snpatho_integrated_annotation_sensitivity_hard_type_mean_hard_assigned_type_mass | 56 | 0 |
| 4399 | r1_ffpe_snpatho_integrated_annotation_sensitivity_hard_type_mean | 56 | 0 |
| 4399 | r1_ffpe_snpatho_integrated_annotation_sensitivity_soft_type_mean | 44 | 0 |
| 4399 | r1_ffpe_snpatho_integrated_annotation_sensitivity_hard_type_mean_equal_cell | 56 | 0 |
| 4399 | r1_ffpe_snpatho_integrated_annotation_sensitivity_soft_type_mean_equal_cell | 44 | 0 |
| 4411 | historical_integrated_snrna_pseudobulk | 500 | 0 |
| 4411 | r1_ffpe_snpatho_integrated_annotation_sensitivity_hard_type_mean_hard_assigned_type_mass | 3 | 0 |
| 4411 | r1_ffpe_snpatho_integrated_annotation_sensitivity_hard_type_mean | 3 | 0 |
| 4411 | r1_ffpe_snpatho_integrated_annotation_sensitivity_soft_type_mean | 3 | 0 |
| 4411 | r1_ffpe_snpatho_integrated_annotation_sensitivity_hard_type_mean_equal_cell | 3 | 0 |
| 4411 | r1_ffpe_snpatho_integrated_annotation_sensitivity_soft_type_mean_equal_cell | 3 | 0 |

Macro mean of specimen median paired per-gene Spearman deltas: **-0.025950**.

Bootstrap fraction with delta > 0: **0.2924** (this is neither a p-value nor a posterior probability).

Requested refined-versus-type-mean endpoint: **developmental_joint_contrast_only_not_primary**.

Developmental seed-17 joint matched-R1 contrast: **passes_developmental_joint_contrast**. This one-seed contrast is reported separately and cannot unlock a full-primary claim.

Full-primary refinement matrix: completeness **complete**; strict ordering **fail**.

Spot QC is a partial proxy. Inclusion in the processed RDS materializes the author-QC whitelist, but explicit per-spot exclusion flags/reasons and the required >=50% H&E tissue-fraction field are not present in the historical truth contract.
