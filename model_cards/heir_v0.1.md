# HEIR v0.1 model card

**Status:** runnable research implementation; no pretrained checkpoint is bundled.
The locked snPATHO v0.2 run was one-pass, not refined, and its primary expression
endpoint was negative. The covariance-aware UOT/refinement redesign is
development code and has not been revalidated on an untouched cohort.

**Intended use:** spatialize broad/intermediate cell types and selected nuclear-compatible transcriptional programs on H&E using sample-matched snRNA-seq.

**Not intended for:** diagnosis, treatment selection, full-transcriptome replacement, exact cell pairing, or forced fine immune-state calls.

**Inputs:** validated nucleus segmentation and cached H&E features; sample-matched annotated snRNA reference/prototypes in personalized mode.

**Outputs:** nucleus-indexed type/prototype posteriors, selected expression/program means and intervals, OOD score, unknown/abstain decision, and provenance.

**Known limitations:** distributional supervision remains non-identifiable when
morphology does not separate states; nuclear RNA differs from whole-cell spatial
assays; segmentation/stain/section mismatch can dominate error; uncertainty
requires development-donor calibration; the portable scVI path distills
full-library-normalized decoder means and is not a replacement for scVI's
complete count likelihood. The current graph is still a mean-aggregation GNN,
and biological unknown-state detection needs leave-state-out validation.

**Required reporting:** mode (personalized or distilled), modalities used by every comparator, donor-level results, matched type-mean residual comparison, wrong-donor control, calibration/risk–coverage, and matching tier.
