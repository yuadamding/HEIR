# NatCommun generative reference-fusion development protocol

## Decision and scope

This experiment is an **outcome-exposed development analysis** of the regional HEIR hypothesis. It
cannot confirm the hypothesis, authorize a cell-level claim, or justify iterative refinement. Its
purpose is to determine whether a compact count-generative composition/state model is worth freezing
for a genuinely independent matched H&E--spatial-transcriptomics--reference cohort. The registered
reference consists of suspension expression profiles with donor and type labels; its source metadata
say `cell`, and this protocol does not claim that it is a verified single-nucleus assay.

The principal development contrast is

\[
L(M3_{\mathrm{H\&E+matched\ reference}}) < L(M0_{\mathrm{H\&E}}),
\]

where both arms use the same H&E branch, molecular backbone, spatial count decoder, optimization
budget, and evaluation observations. Reference fusion is the only difference. NatCommun has already
influenced the architecture and endpoint, so every result is descriptive development evidence even
when donor-held-out.

## Development revision and current run state

The first complete 13-fold run is retained at
`/mnt/seagate/HEIR_runs/natcommun_generative_development/revisions/underaligned_v1`. It is an
outcome-exposed, under-aligned development baseline; it is not valid evidence that the reference and
ST encoders learned a shared biological latent space. The audit found that its alignment penalty was
applied once per epoch while the negative-binomial reconstruction objective was accumulated over
hundreds of molecular minibatches and 257 modeled features (256 panel genes plus the registered
other-transcript count bin). Raw matched-donor pseudobulk MSE therefore increased from 0.01445 before
training to 0.26185 after training, with every fold worsening by 15.77- to 23.11-fold on that scale.
The apparent pretraining advantage was also scale-confounded, so raw matched MSE alone is no longer
an alignment acceptance criterion.

The archived v1 score remains useful only as an architecture-informing result. M3 nominally improved
donor-balanced full-target NB deviance over M0 from 2.46641 to 2.29611 (6.90%; 11 of 13 donors;
one-sided sign-flip `p=0.000610`; paired donor-bootstrap effect interval 0.08584--0.26954), but Gate 2
failed because reference-only M1 was better than M3 (2.23860 versus 2.29611). The M3-versus-M2 state
increment was not significant, the reliability-adjusted M3/M0 variance ratio was 0.580 and failed its
0.80 quality margin, and the same-half M8 diagnostic did not beat same-half M3. Consequently, v1 did
not pass claim progression even before considering the alignment failure. These exposed outcomes
were used to revise the architecture; they must not be pooled with, or presented as independent
support for, the revised experiment.

The registered revision is
`v2_scale_normalized_every_minibatch_cross_assay_alignment`. For each outer fold, the molecular model
now applies the alignment term at every molecular minibatch:

\[
L_{\mathrm{align}}=
\frac{\operatorname{MSE}(\text{matched donor ST/reference pseudobulks})}
{\operatorname{stopgrad}\!\left[\operatorname{MSE}(\text{off-diagonal mismatched donor
ST/reference pseudobulks})\right]}.
\]

The aligned model uses `lambda=1.0`. A second training-only model uses the same seed, data, schedule,
capacity, and optimization budget with `lambda=0`; it is an alignment diagnostic only and never sees
held-out-donor ST. A fold is accepted only when the aligned post-training matched-to-mismatched ratio
is below 1, below its own pretraining ratio, and below the post-training ratio of the same-seed
`lambda=0` comparator. The alignment term must also have been applied on every molecular minibatch.
Failure of any condition fails the fold closed rather than permitting score interpretation.

The revised full 13-fold run completed on 2026-07-15 with all predictions and score artifacts bound
to the frozen v2 identities below. The explicit result is reported in
`reports/natcommun_generative_development_results.md`. Gate 1 passed: M3 improved donor-balanced NB
deviance over M0 from 2.528094 to 2.323076 (8.1096%; 13 of 13 donors; paired bootstrap interval
0.100829--0.333460; exact one-sided sign-flip `p=0.000122`). The full hierarchy failed because M1
outperformed M3, conditional Gate 3 was not significant, reliability-adjusted variance failed its
margin, and M8 did not beat M3 on the common split-half target. Because v2 was selected after
examining v1, this result remains outcome-exposed development and cannot restore independence.

## Frozen source and representation

- Source: `/mnt/seagate/HEIR_runs/natcommun_regional_source/source.npz`
- Source SHA-256: `ec37d5717a9b737dfac226ae9267258fb728ee024496a7655bb69a913aa3cf20`
- Primary observations: 38,945 eligible Visium spots from 13 biological donors; every section from a
  donor remains in one leave-one-donor-out fold.
- Matched reference: 86,306 primary-eligible Chromium expression profiles with donor and Level-1
  type labels. The source metadata call these profiles `cell`; the experiment does not relabel the
  suspension as unequivocally single-nucleus.
- Primary image: natural registered 112-micrometre H&E context represented by frozen H-optimus-1
  revision `3592cb220dec7a150c5d7813fb56e68bd57473b9`.
- Encoder manifest: `manifests/encoders/hoptimus1.json` (SHA-256
  `f6852288e1ae146a4865bf19e38ce994c0be9ce1c2bfa09bdf77747043ac8fd9`); checkpoint SHA-256
  `c4f1e5b457ddf00679626053b0bf2899be6a19c3a04ad191c87ad1cdfd1abfe1`.
- Secondary localization sensitivity: unavailable and not run. The registered source contains no
  55-micrometre H-optimus-1 embeddings, so this sensitivity cannot rescue or modify the primary
  112-micrometre result.
- UNI2-h is operationally forbidden: it is not loaded, qualified, run, pooled, or used for rescue.
- Primary molecular endpoint: raw counts for one deterministic 256-gene development panel selected
  from the 17,612-gene assay intersection. The panel artifact records the exact source, row scope,
  ranking rules, gene order, and content hash.
- Secondary molecular endpoint: a fixed 20-dimensional training-only molecular representation.

The 256-gene endpoint excludes mitochondrial, ribosomal, and haemoglobin-dominated technical genes;
requires adequate detection in both assays; incorporates split-half reliability and within-coarse-
type reference variation; and preserves eligible genes from the eight prespecified biological
programs. A panel frozen with all NatCommun outcomes is suitable only for a later independent cohort.
Any nominal NatCommun donor-held-out diagnostic using that panel remains outcome-exposed; a fold-local
panel sensitivity is required to quantify selection leakage.

The external-development panel was materialized on 2026-07-14 before model fitting. It requires ST
split-half correlation of at least 0.05, retained 37 program genes that passed that requirement, and
selected the remainder from 1,659 eligible genes. The
receipt is `configs/natcommun_generative_gene_panel.json` (SHA-256
`ce0b6b82440d7fccc69f24afccf0c68bb101b85f590e4a35a514309929fbb6ad`; semantic identity
`12c075a0b7639b64a6a0c521e855cd10f44852212865924279b05d40b38083cf`). A 132-MiB projected
count/image artifact was written outside Git at
`/mnt/seagate/HEIR_runs/natcommun_generative_development/panel_256_projected_counts.npz`
(SHA-256 `71479f891b5945762e20ec5b91d85bac097230b12ed9192aeacd965be119607f`). It contains
38,945 eligible spots, 86,306 reference profiles, full and split count matrices and library totals,
the frozen eight-program membership matrix, the 112-micrometre H-optimus-1 features, the blank
vector, and coordinate controls. The sequential projection stayed below the 6-GiB ceiling (observed
peak RSS was 2.17 GiB during the final materialization).

The real-fold integration pilot and 556-test suite froze the completed v2 rerun to these exact
identities:

| Frozen artifact | Schema or SHA-256 |
| --- | --- |
| Registered source archive | `ec37d5717a9b737dfac226ae9267258fb728ee024496a7655bb69a913aa3cf20` |
| Projected 256-gene source | `71479f891b5945762e20ec5b91d85bac097230b12ed9192aeacd965be119607f` |
| `configs/natcommun_generative_gene_panel.json` | `ce0b6b82440d7fccc69f24afccf0c68bb101b85f590e4a35a514309929fbb6ad` |
| `configs/natcommun_generative_development_protocol.json` | schema `heir.natcommun_generative_development_protocol.v2`; `2cb92b22b6870488a06e64b213e37ffbbdfe3044f1da8fc7442f506915e78197` |
| `scripts/benchmark_natcommun_generative_development.py` | `cf27504e25dfd8cd7e8bfe2894efc8b4a8f79306b47bc492d0e61406d20668ce` |
| `src/heir/evaluation/generative_fusion.py` | `55a63f1360e8cc76267e4b00ba8e2167f36259789e9bfdf2aa929c8cadd83b17` |

Prepare, prediction, and score receipts record these identities together with the fold, seed, and
parent hashes. A change to any bound artifact creates a different experiment and requires fresh
sealed artifacts; it cannot reuse this v2 identity.

All 13 fold-local training-only panel membership receipts were also materialized outside Git in
`gene_panel_external_and_lodo_receipts.json` (file SHA-256
`7dcd5bbd6fe6f1a18625afd1fddd36bbd1a9ccad284612cab5248cd21a09f982`; semantic receipt
`3faa42ff97906e2f3040de6a3b74e0a0942e85ac8bf98fec151a6ae5398406ff`). Each shared 222--251
of 256 genes with the all-development panel (mean 240.46); Jaccard similarity ranged from 0.7655 to
0.9617 (mean 0.8872). This high but imperfect stability is why the all-development-panel result is
explicitly exposed and the fold-local sensitivity remains scientifically necessary. No model was
fitted or scored on these fold-local panels; the LODO-panel model sensitivity is not run.

Thirty-nine primary spots with zero full-library depth are excluded from every primary arm before
fitting or scoring. A split-half M8 comparison additionally excludes observations with a zero target
half. These masks are applied identically across compared models and reported by donor and section.

## Compact model

The molecular backbone is a 20-dimensional diagonal-Gaussian count model fitted with training-donor
Visium and matched registered-suspension counts. It receives the 256 panel counts plus one
other-transcript bin, full-library exposure, and a two-level assay-modality indicator. It uses one
shared encoder and separate reference and ST negative-binomial decoders and dispersion parameters.
The primary endpoint dispersion is fitted from training-donor ST only.

CountVAE has **no donor, type, or indication covariate**. Within CountVAE, donor IDs are used only to
construct the training-pseudobulk alignment loss and its receipts. Downstream, donor IDs group
reference banks; reference type labels define mixture components, type anchors, and the H&E
composition/state vocabulary; indication labels select control banks and BLEEP hard-negative strata.
None of those labels is an input to the molecular encoder or decoder. The v2 model therefore must
not be described as explicitly adjusting CountVAE for donor, type, or indication.

The separate reference and ST decoders use the scale-normalized matched-versus-mismatched
training-donor cross-assay alignment term defined above, with weight 1.0 at every molecular
minibatch. A same-seed, same-schedule `lambda=0` model provides the fold-local negative control; the
aligned model must satisfy the frozen support criterion relative to both its pretraining state and
that comparator. Per-type state variance is calibrated from training residual likelihood and then
frozen.

Training is staged to reduce composition/state non-identifiability:

1. Fit and freeze the molecular backbone on training donors.
2. Derive a training-ST-only coarse-composition proxy from frozen reference type signatures, fit the
   H&E composition branch, and freeze it. The proxy is not observed spot-composition truth.
3. Fit the H&E per-type state posterior with composition and type signatures frozen.

For donor `d` and type `t`, the held-out matched reference is represented by a naturally weighted
multi-component diagonal-Gaussian state distribution, not one centroid. M3 analytically combines
the H&E state evidence with that distribution using a product of experts. Each component is decoded
separately using 32 deterministic Sobol samples before exact mixture moments are assembled; a single
Gaussian moment match is not passed through the nonlinear decoder. The common spatial decoder
mixes per-type expected expression with H&E-predicted composition and scores raw Visium counts under
one training-only gene-dispersion vector shared by every arm.

The normalized-Mahalanobis rule is a diagnostic, not an inference-time rejection rule. A
donor/type reference state is marked supported when its minimum normalized Mahalanobis distance from
the H&E state is at most 4.0. The reported coverage is the H&E-composition-weighted mass of those
supported states. Coverage below 0.90 sets a per-spot out-of-support flag; its mean is serialized as
`abstention_rate` for compatibility. No prediction is withheld, no primary scoring row is removed,
and M3 is still produced. Both the distance threshold and flag rate are fixed-development,
exploratory, nonblocking diagnostics that were not calibrated by an inner training-donor search.
Missing full-target reference types fall back to the H&E-only state rather than silently borrowing
another type.

The implementation is deliberately shallow and PyTorch-only. It does not add scvi-tools, graph
learning, optimal transport, image fine-tuning, diffusion, or iterative refinement.

## Frozen execution defaults

These are fixed pre-score engineering defaults. They were not selected by inner donor folds:

- base seed 1729, with each fold seed deterministically derived as the first four bytes of
  `SHA256("1729:<donor>")` interpreted as little-endian;
- neural-module initialization seed 17; the fold seed controls row ordering and stochastic draws, so
  the aligned and `lambda=0` CountVAEs start from the same deterministic parameters;
- exactly 80 runner epochs, batch size 256, latent dimension 20, and Adam learning rate `1e-3` for
  the molecular, H&E state/composition, coordinate-control, and retrieval fits;
- a 257-input CountVAE (256 panel genes plus the other-transcript bin), hidden width 256, aligned
  weight 1.0 at every molecular minibatch, and a same-seed/same-schedule `lambda=0` training-only
  comparator;
- H&E and coordinate composition/state models with hidden width 80; 40 composition epochs, 80 state
  epochs, type-anchor penalty 0.10, and 100 training-residual variance-calibration steps;
- three deterministic soft components per donor/type, 25 mixture iterations, mixture temperature
  1.0, variance floor `1e-4`, and 32 Sobol decoder samples per component;
- BLEEP projection dimension 20, temperature 0.07, hard-negative weight 0.5, and the fallback rule
  and receipts described below;
- normalized-Mahalanobis diagnostic threshold 4.0 and diagnostic coverage-mass flag threshold 0.90;
- Gate 3 minimum supported H&E-composition mass 0.90 and at least three eligible spots in every
  section; and
- M8 half fraction 0.5, ridge 10.0, and log-count normalization scale 10,000.

Any later sensitivity changes the model identity and must be reported as a separate analysis.

## Required arms

| Arm | Frozen difference from the common model | Scientific question |
| --- | --- | --- |
| M0 | H&E composition and state only | H&E baseline |
| M1 | Natural matched-reference distribution only; spatially constant within donor | Is reference alone sufficient? |
| M2 | Full-target H&E composition plus matched type means, with H&E-state fallback for missing types | Descriptive type/composition-routing arm |
| M2_supported | Gate-3-only H&E composition renormalized over matched types with at least two components, plus matched type means | Conditional routing comparator |
| M3 | H&E composition/state plus matched multi-state reference distribution | Central model |
| M3_supported | The same Gate-3 support mask and renormalized H&E composition as M2_supported, plus matched-reference PoE state | Conditional continuous-state arm |
| M4 | Deterministically deranged H&E within section and coarse composition stratum | Does exact image pairing matter? |
| M5a | Frozen white-blank H&E vector | Does image content matter? |
| M5b | Same-budget coordinate branch | Is spatial location sufficient? |
| M6 | Each eligible natural same-indication wrong-donor bank | Natural-bank separation diagnostic |
| M7 | Query-excluded same-indication generic bank with donor-equal weights | Is a generic reference sufficient? |
| M8 | Training-only negative-binomial-compatible cross-half molecular predictor | Empirical molecular-oracle headroom diagnostic |

M6 reports every naturally available wrong donor, the registered equal-wrong-donor mean, and a
conservative best-wrong diagnostic. Bank size and state quality are not fixed across donors, and the
fixed-effective-sample-size wrong-bank sensitivity has not been run. M6 can therefore describe
natural wrong-bank separation but cannot by itself attribute an effect specifically to donor
personalization.

M8 is compared with M3 scored against exactly the same half-depth targets; it is not compared with
full-depth M3 through a post-hoc scaling constant. It is an empirical molecular oracle, not a
mathematical lower bound or hard `L_ST floor`. Full-depth measurement-noise risk cannot be estimated
from this source because no registered independent full-depth replicate is available.

A BLEEP-style baseline learns a contrastive H&E--training-ST query space and retrieves from the
held-out donor's reference-state mixture. It is explicitly a compact prototype-retrieval comparator,
not an exact reproduction of BLEEP. M3 must outperform it before extra model complexity is argued to
have value. Its frozen temperature is 0.07 and hard-negative weight is 0.5. For each training query,
hard negatives first require a different donor in the same indication and the same dominant
training-only composition-proxy stratum. If that exact stratum is empty, selection falls back first
to a different donor in the same indication and then, only as an emergency fallback, to any different
donor. Within the first nonempty candidate stratum, the deterministic choice maximizes separation
along one fixed training-latent projection. Each fold saves the count and fraction assigned through
the exact, same-indication fallback, and global fallback routes; the three counts must sum to the
number of training spots and their fractions to one.

## Hidden-outcome and leakage boundary

The execution is physically divided into fit, predict, and score artifacts.

- Fit receives training-donor ST, training/query H&E, and training/query reference counts, but no
  query-donor ST.
- Predict receives only the frozen fold model, query H&E, query reference, coordinates/metadata
  needed by prespecified controls, and hash-bound masks.
- Score alone opens query-donor ST and cannot update predictions or hyperparameters.

The held-out donor must be absent from panel selection for the fold-local sensitivity, dispersion,
molecular-backbone fitting, image fitting, calibration, donor/type proxy adequacy, and the generic
reference bank. The Mahalanobis coverage and out-of-support thresholds are fixed diagnostics, not
fitted abstention mechanisms. The retrieval temperature and other listed execution defaults are
fixed before score opening rather than selected with held-out outcomes. Its matched reference may be
encoded by the frozen molecular encoder because this is the intended inference-time input. Held-out
ST library depth may be used as an observation exposure only inside scoring and is never an image or
reference-model feature.

Every artifact records source, protocol, panel, code, fold, training donors, query donor, encoder,
model-arm, seed, and parent-prediction hashes. A resumed fold is accepted only if all identities
match. Writes are atomic.

## Endpoints and aggregation

The primary loss is gene-mean negative-binomial deviance, averaged spots within section, sections
within donor, then donors equally. An indication-equal aggregate is reported separately. The same
training-only gene-dispersion vector is used to score all arms within a fold.

Required secondary outputs for M0--M3 are:

- plug-in held-out NB log likelihood;
- standardized 20-dimensional molecular MSE;
- donor/section-balanced gene correlation and calibration slope;
- reliability-adjusted variance using positive, training-qualified split-half covariance strata;
- covariance error over the eight fixed programs;
- rare-state recall under training-defined quantiles;
- moment-normal 50%, 80%, and 95% predictive-interval coverage;
- H&E-composition-weighted Mahalanobis reference-state coverage, reported as
  `H_composition_weighted_state_coverage` (the fold-level compatibility field remains
  `reference_state_coverage`);
- out-of-support flag rate, serialized as `abstention_rate` for compatibility even though
  predictions are not withheld;
- normalized retrieval entropy for reference-query models.

Raw predicted-to-observed variance is descriptive, not a standalone failure rule. Predictive
interval and state-diversity diagnostics distinguish a calibrated shrunken posterior mean from
molecular collapse. The eight-point Gauss--Hermite, moment-matched-lognormal posterior-predictive NB
log score and the moment-normal interval coverages are exploratory approximations. They are reported
but nonblocking; neither can replace the primary held-out NB deviance or determine a gate.

## Ordered development gates

All effects are defined as `control loss - M3 loss`, so positive favors M3. Donor is the inference
unit. Confidence intervals use a deterministic donor bootstrap, and exact one-sided donor sign-flip
tests determine the prespecified significance conditions.

1. **Central incremental value:** M3 must beat M0 by at least 5% in the donor-balanced aggregate,
   favor M3 in at least 70% of donors, have a 95% paired-effect interval above zero, have an exact
   one-sided donor sign-flip `p<=0.05`, and show no prespecified severe indication reversal. Failure
   stops scientific attribution.
2. **H&E necessity:** M3 must beat M1 and M4; Holm adjustment is applied only across these two tests.
3. **Continuous state on supported composition:** this gate does not use the mixed-support,
   full-target M3-versus-M2 contrast. For each evaluable type, the training-only proxy compares
   composition-proxy-weighted ST donor latent means with registered-suspension donor-by-type means;
   it requires at least two paired training donors and weights evaluable types equally. For every
   outer fold, its matched-to-off-diagonal-mismatched ratio must be below 1 and below the same-seed
   `lambda=0` ratio. M2_supported and M3_supported then use exactly the same H&E composition
   renormalized over types with at least two matched-reference components. A spot is eligible only
   when those types carry at least 0.90 of its original H&E-predicted composition, and every section
   must contain at least three eligible spots. Gate 3 is evaluable only if these requirements hold
   for every outer donor; when evaluable, M3_supported must beat M2_supported by a positive
   donor-balanced effect with one-sided sign-flip `p<=0.05`. The composition-weighted proxy is not
   observed type truth and cannot support a cell-level alignment claim.
4. **Matched-reference separation:** M3 must beat natural-bank M6 and generic-bank M7; Holm adjustment
   is applied only across these two tests. Without the unrun fixed-ESS M6 sensitivity, this gate does
   not isolate donor personalization from bank size or state-quality differences.
5. **Empirical molecular headroom:** M8 must beat M3 in the same split-target space. This is secondary,
   is not a hard ST floor, and cannot rescue or block Gate 1.

M5 and contrastive retrieval are reported controls and are not silently added to either Holm family.
Quality preservation is a separate conjunct after Gates 1--3. Because this run is development-only,
even a complete nominal pass means only `candidate_for_external_preregistration`; it never means
`confirmed`, `cell_level_supported`, or `iterative_refinement_authorized`.

## Protocol completion boundary

The central 112-micrometre donor-held-out experiment using the all-development panel can complete and
test Gate 1 as an exposed development analysis. That is distinct from full implementation of every
sensitivity in this protocol:

| Required sensitivity | Status |
| --- | --- |
| Registered 55-micrometre H-optimus-1 localization | Unavailable; source has no such embeddings; not run |
| Fold-local LODO-panel model reruns | Panel receipts exist; model sensitivity not run |
| Soft-weighted reference-composition sensitivity | Not run |
| Fixed-ESS wrong-reference-bank sensitivity | Not run |
| Full-depth measurement-noise estimate | Not estimable from this registered source |

Accordingly, a finished central development run may be labeled
`central_development_experiment_complete`, while `full_protocol_implementation_complete` remains
false. No aggregate report may equate metric-suite completion with full scientific-protocol
completion.

At this revision, `under_aligned_v1_development_run_complete` and
`scale_normalized_alignment_v2_central_development_run_complete` are true. Full protocol
implementation remains false because the listed sensitivities are not run. Neither completed
development run is independent confirmation.

## Resource contract

- CUDA is required for the primary neural fits; CPU fallback is for tests only and cannot be silently
  substituted into a primary run.
- Exactly one GPU is visible. At most four CPU threads, zero to two data-loader workers, one outer
  fold at a time, and no concurrent encoder execution are allowed. The aligned and `lambda=0`
  molecular fits are evaluated within that serial fold contract.
- CUDA allocation is capped at 60% of the 10-GiB device; mixed precision is permitted for matrix
  operations, while negative-binomial likelihood and `lgamma` calculations remain FP32.
- Dense 256-gene projection is materialized once outside Git. Broad CSR arrays are read and released
  sequentially, with a 6-GiB panel-preparation resident-memory ceiling and a 1-GiB projected-count
  artifact ceiling.
- Checkpointing occurs per stage and fold. Fixed maximum steps and training-only stopping criteria
  prevent outcome-dependent runtime.
- An out-of-memory condition fails closed. Batch size cannot change under the current frozen
  identity; any reduced-batch run requires a new protocol and checkpoint identity. Folds are not run
  concurrently, and the model or endpoint is not changed to rescue a run.

## Independent confirmation status

No currently downloaded cohort contains a pristine, donor-safe, same-specimen H&E + spatial-count +
independent matched sc/snRNA system. The local spatialDLPFC support files identify a promising
10-donor study but contain no high-resolution images or molecular payload and do not yet freeze
same-block routing. Therefore this implementation can produce only an explicit NatCommun development
report. Independent confirmation remains blocked by cohort acquisition, not by a favorable or
unfavorable NatCommun result.
