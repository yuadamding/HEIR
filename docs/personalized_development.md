# Personalized HEIR workflow

This revision applies the user-supplied **HEIR software development plan** to the existing
repository. The development target is a specimen-specific probabilistic RNA-state assignment at
H&E nuclei, constrained by measured proteins and geometry, followed by relative-expression
prediction. Spatial transcriptomics (ST) supervises and evaluates that same pathway during
development. It is not an ordinary prediction input.

The package is an engineering prototype. Synthetic contract tests can verify calculations and
input separation; they cannot qualify an HTAPP specimen, establish spatial resolution, or support
a positive biological claim. The historical HEST, NatCommun, and GSE315411 results and protocol
identities remain unchanged.

The pre-cleanup repository passed **913 tests**. That broader implementation, its original
review, and its scientific records are preserved in the [source archive](../archive/README.md).
The initial compact prototype at `d955c01` passed **53 focused tests**. Version 0.3 adds regression
tests for the assessment's numerical and identity failures; that earlier pass did not exclude them.
Its synthetic H5AD/Parquet workflow covers preparation, fitting, bundle reload, target-free
prediction, and scoring on CPU. These checks establish engineering behavior only; they are not
real-data, GPU, or biological qualification.

Retained frozen encoder adapters live in `heir.features`. Historical study-specific validation,
prototype, and evaluation code is preserved in the archive rather than maintained as an active
runtime dependency. Its exact-block matching assumptions must not be substituted for HTAPP's
patient, episode, lesion, represented-region, core, and section evidence.

## Prediction and target contract

| Information | Training | Held-out personalized prediction |
| --- | --- | --- |
| H&E nuclei, physical coordinates, cached image features | Input | Input |
| Documented specimen-matched sc/snRNA | Input | Input |
| Measured proteins, availability masks, marker interpretation, registration | Input when the arm uses proteins | Input when the arm uses proteins |
| Predefined LR resources | Input in LR comparisons | Input in LR comparisons |
| Measured spatial RNA | Supervision | Evaluation after prediction |
| Spatial-RNA-derived labels, `RCTD`, `OT`, transferred annotations | Only explicitly declared training/evaluation use | Excluded from ordinary prediction inputs |

Matched RNA may contain the genes being predicted: the personalized task is to place its molecular
information correctly in space. The prediction API must never read target ST to construct the
assignment, select a reference, define a protein feature, or determine an evaluation compartment.
The H&E-only arm must be independent of RNA and protein values.

For each H&E nucleus the output contract includes its ID and coordinates; state probabilities with
an explicit unsupported-reference component; conditional known-state probabilities;
reference-transferred expression; bounded learned
log correction; final relative expression; assignment ambiguity and reference support; registration
sensitivity when measured; and gene-level supervision/evaluation flags. Missing registration
sensitivity must remain missing, not a false claim of zero sensitivity. Model variability is a
separate repeated-fit quantity.

In personalized arms, genes without spatial training supervision retain reference transfer with
zero learned correction. The H&E-only arm supports its spatially supervised genes only.
Reference support is diagnostic only. The unsupported head is zero-initialized and frozen until a
qualified support objective exists. Transfer, message passing, refinement, protein discrepancies,
and specimen abundance all use conditional known-state probabilities, computed directly by a
known-state softmax to remain stable even when joint support numerically reaches zero. All cells
therefore retain expression contributions; support cannot suppress an inconvenient cell's RNA.
Exports label this provisional policy `conditional_reference_unvalidated` and set
`reference_support_calibrated=false`. This conditional reference extrapolation is not a validated
prediction for truly unsupported cells. Absolute molecule counts and calibrated expression
intervals require additional measurement and validation models.

### Fixed normalization and spatial endpoints

Model configuration separates `normalization_gene_ids` from `supervised_gene_ids`; the evaluation
command's `--genes` (alias `--evaluation-genes`) selects outcomes only. The normalization panel is
saved in the bundle and is reused from prediction provenance when scoring. The contract is:

1. Aggregate nonnegative linear cell expression into assay regions.
2. Divide by the sum over the fixed normalization panel, scale to 10,000, and apply `log1p`.
3. Select supervised/evaluation genes and apply their outcome-availability mask.

Every normalization gene must be measured in every retained region, and both prediction and target
denominators must be positive. Incomplete coverage is rejected; users must qualify a common complete
panel/coverage before running a comparison. Missing outcomes outside that panel do not change its
denominator. Changing the outcome subset therefore cannot redefine an existing gene's scale.

A panel needs at least two distinct genes. One supervised gene is allowed with a separate
normalization basis; training rejects an effectively one-gene denominator with no measured and
initially predicted companion mass in any common training region. Individual sparse regions remain
valid. H&E-only models must supervise every normalization gene because they have no RNA transfer
for untrained genes. Exposure-based/count normalization is not implemented.

`score()` computes gene-wise spatial correlations within one specimen. `score_patient()` first
computes each specimen's median gene-wise correlation, then gives eligible specimens equal weight
in `mean_specimen_median_gene_spearman`. Between-specimen mean differences cannot create this
spatial endpoint. Reports retain per-specimen gene exclusions; a constant prediction for a
target-evaluable gene invalidates the headline rather than selecting a favorable subset.

## Package boundaries

The active inference modules live in `src/heir/personalized/`; optional frozen encoder adapters
remain in `src/heir/features/`. Historical study implementations are retained in the source archive.

| Module | Responsibility |
| --- | --- |
| `schema.py` | Validate and separate `PredictionInputs`, `RNAReference`, protein evidence, graph geometry, `SpatialTargets`, and `Prediction` |
| `data.py` | Read explicitly described local inputs and keep target loading separate |
| `geometry.py` | Directed coordinate transforms, physical units, regional protein evidence, and inspection overlays |
| `reference.py` | Construct specimen RNA prototypes and means on a nonnegative linear scale |
| `graph.py` | Sparse directed neighborhoods, generic messages, and complex-aware LR compatibility |
| `model.py` | Shared assignment, optional bounded refinement, reference transfer, and expression correction |
| `observation.py` | Aggregate cell expression into measured regions on the linear scale |
| `train.py` | Shared supervised optimization and model persistence |
| `evaluate.py` | Join completed predictions to separate targets and report explicit metric eligibility |
| `serialization.py` | Validate prediction axes and values, preserve supplied provenance, and write/read pickle-free NPZ exports |
| `cli.py` | Preparation, training, prediction, and evaluation entry points |

### Small-workflow commands

Install the core and the optional HTAPP file readers (`--extra encoders` adds frozen image adapters):

```bash
uv sync --frozen --extra dev --extra htapp
```

Start from the [specimen manifest](../configs/personalized/htapp.example.json) and
[model configuration](../configs/personalized/model.example.json), replacing their placeholders and
dimensions with the qualified specimen inputs. The
[comparison plan](../configs/personalized/comparisons.json) records the seven variants, decisive
contrasts, and shared requirements; it is a comparison index rather than a model JSON accepted by
`--config`.

The CLI keeps prediction separate from target loading:

```bash
heir prepare --manifest specimen.json --output prepared/
heir train --manifests train-specimen-1.json train-specimen-2.json \
  --folds folds.csv --outer-fold 0 --config model.json --output bundle/
heir predict --manifest held-out-specimen.json --bundle bundle/ --output predictions.npz
heir evaluate --prediction predictions.npz --manifest held-out-specimen.json \
  --genes evaluation-genes.json --output scores.json
```

All four context variants (`generic_fixed`, `lr_fixed`, `generic_refined`, `lr_refined`) require
`heir train ... --baseline protein-bundle/`, pointing to the same fitted `he_rna_protein` baseline
for a controlled comparison. Their model JSON must supply the same explicit `lr_interactions`
programs for generic and LR arms. The base mapper and its image-feature transform are copied and
frozen; refined variants learn their additional bounded assignment updates.

Use `heir predict ... --transfer-only --output transfer-only.npz` to export a fitted personalized
model's transfer-only diagnostic. Training a fixed-context transfer-only model is rejected because
its frozen mapper leaves no decoder to optimize. H&E-only prediction has no transfer diagnostic.
Training and prediction accept `--device cpu` (the default) or `--device cuda` for a locally
available CUDA device. This revision's checks ran on CPU; CUDA execution remains unqualified.

Paths in these commands are examples and require actual qualified local input files. The manifest
identifies the sparse cell-to-observation NPZ through `paths.observation`, including explicit cell
and observation identifiers. `folds.csv` supplies patient roles `train`, `validation`, and `test`
for the selected outer fold. Training reads targets only for the training specimens. Model
selection and a complete outer-fold study remain distinct from executing one training command.
The training CLI performs the requested fit; it does not perform hyperparameter selection or
training resumption. Bundles retain the training configuration, manifests, and optimizer state for
future reproducibility/resumption work, while inference loads numeric weights and transforms.
`prepare` creates an input summary/eligibility record and geometry overlay without opening target
ST. That target-free diagram is not the complete multimodal acceptance overlay.

Version 0.3 requires specimen manifest, bundle, and prediction schema v2. Old model weights are
not silently loaded under the changed numerical semantics: update the normalization panel and
protein/registration identities, replace protein `radius_um` with declared bin geometry, and
prepare/refit/export again. The archived v1 evidence remains unchanged.

For a synthetic integration fixture, use:

```bash
heir demo --output /tmp/heir-personalized-demo
```

The demo is a software check, not an HTAPP result. Prediction exports currently materialize dense
expression arrays and are intended for small development specimens. The bundle preserves fitted
model state and configuration; reproducing a scientific prediction also requires retaining the
matched-reference inputs, preparation settings, mapping metadata, and cached-feature identities.

Segmentation and image encoding remain preprocessing tasks. The existing frozen encoder adapters
are reusable; the new core consumes cached, nucleus-ID-keyed features. A new CellViT adapter,
automated two-scale physical crops, and complete image-cache provenance still need specimen-level
qualification. Existing studies retain their recorded H-optimus-1/UNI2-h roles and checkpoint
identities. The attached plan's optional UNI2-h suggestion does not rewrite those studies.

## HTAPP qualification

The attached plan identifies the [SCP2702 portal](https://singlecell.broadinstitute.org/single_cell/study/SCP2702)
and a previous 13-participant intersection as the candidate source. Those are planning inputs,
not evidence that the corresponding payloads or their metadata were verified in this revision.
The final eligibility table must derive from actual local files and exclude incomplete or
unresolved specimens with explicit reasons.

Preparation consumes the documented canonical local format. Portal bundles with different AnnData
keys or metadata columns need explicit adaptation before loading; the importer does not infer
scientific identities or registration from arbitrary downloaded files.

The durable cohort contract should retain:

| Artifact | Required content |
| --- | --- |
| `specimens.parquet` | Patient, biopsy, episode, lesion, represented region, core, section, assay, source paths, and molecular matching evidence |
| `cells.parquet` | H&E nucleus IDs, coordinates in µm, geometry, segmentation quality, and tissue components |
| `image_features.zarr` or bounded ID-keyed cache | Frozen cell-centered and neighborhood features, crop dimensions, encoder and source identities |
| `rna.h5ad` | Raw counts, stable gene IDs, RNA-cell IDs, specimen identity, sc/snRNA flag, broad-type provenance |
| `proteins.h5ad` | Intensities, marker identities and interpretation, coordinates, observed/quality masks |
| `registration.json` | Coordinate frames, transform direction, units, correspondence type, uncertainty, and overlap |
| `spatial_targets.h5ad` | Measured RNA, observation IDs and geometry, assay, coordinates, available-gene masks |
| `folds.csv` | Patient IDs and fixed train/validation/test roles for each outer fold |

The implemented nucleus table requires `cell_id`, `x_um`, `y_um`, `segmentation_quality`, and
`tissue_component_id`. An NPZ feature cache contains `cell_ids` and `image_features`; its rows are
joined by ID. The observation NPZ contains `cell_ids`, `observation_ids`, integer `rows` and
`columns`, and nonnegative `weights`. These arrays describe a sparse observation-by-cell matrix;
each retained observation must have positive cell coverage.

RNA `.obs` requires patient, sampling episode, lesion, and represented-region identities matching
the manifest's molecular reference. Default names are `patient_id`, `sampling_episode_id`,
`lesion_id`, and `represented_region_id`; use `reference.patient_id_column` and
`reference.matching_columns` for explicit source-name mappings. Target `.obs` requires matching
`patient_id` and `specimen_id`, with optional `targets.patient_id_column` and
`targets.specimen_id_column` mappings. Target coordinates are `obsm["spatial"]` in micrometres and
availability is Boolean `layers["observed"]`.

For proteins, `matching.protein` names patient, sampling episode, lesion, represented region,
specimen, core, section, and assay. The seven identity columns must also be present in the actual
protein `.obs`; measurement IDs must be unique. `matching.query.specimen_id` must equal the
manifest's H&E specimen. Registration `source_identity` must match the protein source and
`target_identity` the H&E source on all seven fields. Unscoped identity aliases are rejected.
Serial sections share the documented core with distinct section IDs; verified same-cell or
same-section evidence requires matching core/section. Region-level evidence may span documented
cores while preserving the shared biological match. These checks verify consistency, not the
truth of independently unverified metadata.

For tissue with internal gaps, supply `paths.tissue_mask`: an NPZ containing Boolean `mask[y, x]`,
`origin_um` in `[x, y]` order, and `pixel_size_um` in `[x, y]` order. Graph edges exclude paths
crossing empty tissue-mask pixels, and protein bins are split by connected mask islands.
Component IDs alone prevent crossing
between disconnected labeled components; they do not establish that every path within a component
stays in tissue.

The importer must inspect the actual RNA count store and gene axis. In the supplied plan's portal
example, the processed matrix has 5,000 variables while additional counts and gene metadata are in
`obsm["counts"]` and `uns["counts_var"]`. The adapter must align those counts with their own gene
identifiers, not assume `var_names` describes that wider matrix or silently call processed `X` raw
counts. Transcriptome-wide coverage is a property of the loaded gene axis, not a filename.

Document equality of patient, sampling episode, lesion, and represented region for the molecular
reference. Record separate cores and serial sections explicitly. Broad labels used for reference
construction must come from the allowed dissociated RNA workflow; transferred spatial annotations
do not become prediction evidence merely because they reside in the same AnnData file.

The adapter checks RNA observation metadata against the declared biological identities, including
mapped columns when source column names differ. Its initial reference embedding is a fixed ordered
gene view of `log1p` state means; it does not fit a shared PCA or other learned RNA representation.
The initial protein proxy supports identity and total-receptor markers using explicit fixed RNA
and intensity half-saturation constants. Those constants expose a calibration assumption; their
presence does not qualify RNA as a quantitative measurement of protein abundance. Phosphorylation
and localization constraints need separate qualified implementations.

Begin with approximately 100 × 100 µm common evaluation regions, conditional on actual overlap and
registration quality. Serial-section protein evidence represents local mixtures or marker
distributions. A nearest CODEX cell is not automatically the same cell as an H&E nucleus.
Same-cell constraints require independently justified correspondence. A complete first-specimen
overlay must show H&E tissue, nuclei, protein coverage, ST coverage, and the common eligible
evaluation regions. A synthetic coordinate diagram does not fulfill that milestone.

Protein supervision has its own `ProteinObservationOperator`, independent of communication edges.
For serial/soft/region evidence, `proteins.bin_size_um` and `origin_um` define common physical bins
and tissue components. Measured intensities and predicted state-protein mixtures both use uniform
means within those bins. A region-marker is available only with complete positive-confidence
source coverage; incomplete markers are masked as a whole instead of changing the averaging
kernel. Mean quality is a separate absolute confidence weight. The loss divides by observed-marker
count, not quality sum, so lowering confidence weakens its influence. Excluded regions and markers
are reported in `protein_coverage` during preparation.

Correct assignments have zero discrepancy in a noiseless matched example. If both cells lie in
one coarse bin, correct assignments and a 50/50 mixture can have the same regional mean; the data
cannot identify their individual placements. Finer independently supported regions can resolve
that difference. Altering the communication radius or degree cap does not change protein geometry.

## Model and comparison contract

RNA means are computed on a defined nonnegative linear scale. Prototype counts are bounded and
rare broad types are preserved; 64–256 states per specimen is the plan's starting range, subject
to validation. Reference abundance is soft sampling information. Any abundance penalty applies
across the specimen, not independently to each small training tile.

| Configuration | Inputs/context | Assignment behavior |
| --- | --- | --- |
| `he` | H&E features | No RNA-state mapping claim |
| `he_rna` | H&E and matched RNA | Base mapper |
| `he_rna_protein` | H&E, matched RNA, measured proteins | Protein-guided base mapper |
| `generic_fixed` | Same inputs and generic molecular neighborhood context | Fixed base assignment during context decoding |
| `lr_fixed` | Same inputs with curated LR structure | Fixed base assignment during context decoding |
| `generic_refined` | Generic context | Bounded assignment updates |
| `lr_refined` | LR context | Bounded assignment updates |

Every personalized arm also needs a transfer-only diagnostic. Reference transfer and correction
must remain separately inspectable. For genes with supervision, use a small low-rank bounded
multiplicative correction; for other genes, retain transfer unchanged.

The decisive contrasts are `he_rna_protein` versus `he_rna`, `lr_fixed` versus `generic_fixed`,
`lr_refined` versus `generic_refined`, and `lr_refined` versus `lr_fixed`. The generic and LR pairs
must share spatial edges, receptive field, molecular features, comparable capacity, and optimization
budget. The fixed-assignment stage should start from the same trained base mapper; merely naming
two independently trained models "fixed" does not establish a controlled assignment comparison.

LR programs retain ligand/receptor complexes, sender/receiver direction, physical distances,
contact/diffusible distinctions, tissue boundaries, and bounded neighborhood contributions.
Measured functional proteins gate programs only when those measurements exist and their marker
interpretation supports that use. Candidate compatibility is not measured signaling activity.
One or two constrained refinement updates use identical computation during training and inference.
Optimizing compatibility alone cannot establish biologically correct placement.
The generic/LR comparison tests the interaction representation (incoming ligand and receiver
receptor versus their product). Shuffled-LR controls are still required to attribute a gain to
curated biological pairing.

The observation operator applies `Y = A X` to linear cell expression, normalizes using the fixed
complete panel, applies `log1p`, then selects outcomes and computes a masked loss.
Regional correspondence across serial sections does not establish exact shared physical cells.
Count likelihoods remain deferred until RNA content and assay exposure are specified.

## Validation design

If all 13 candidate participants qualify, use five outer patient folds of 3, 3, 3, 2, and 2, with a
fixed two-patient validation subset inside each training partition. Every core, section, and region
from a patient stays in the same role. Use metadata such as RNA assay, metastatic site, and modality
coverage to define partitions before outcomes inform model selection. Fewer eligible patients
require an explicitly revised design, not silent reuse of a 13-patient claim.

The principal endpoint is the equal-specimen mean of within-specimen median gene-wise spatial
Spearman correlations on a fixed eligible gene/region set, accompanied by normalized-expression
error. With one specimen per patient this reduces to that specimen's median. Undefined correlations and
excluded genes are reported explicitly. Comparisons must use common coverage; each method cannot
choose its own favorable subset. Seeds describe numerical variability and do not add biological
replicates. Repeated model redesign using outer-fold results makes the analysis developmental
cross-validation rather than independent confirmation.

Within-type spatial variation is a separate mechanistic endpoint. Define types or regions
independently of the model's predictions, and account for composition when only regional evidence
exists. Add expression-prevalence/structure-preserving LR shuffles, external RNA references,
reference-state deletion, functional-marker removal where measured, registration perturbations,
and readouts outside the protein panel. Choose a practical effect threshold from pilot variability
and assay reliability before the principal comparison.

The ST-informed comparator needs its own input interface. Use either one spatial assay to predict
an independently measured assay on aligned coverage, or spatial anchor genes to predict explicitly
withheld genes. All methods receive the same evaluation outcome. Measured ST scored against itself
is not informative, and an empirical ST-informed result is not a universal lower bound.

External MaxFuse, CytoBulk, and SCHAF wrappers remain planned. They should consume the same prepared
specimens, preserve modality differences, emit the same prediction schema, and keep incompatible
dependencies in optional environments.

## Acceptance and remaining work

Version 0.3 passes 127 CPU tests, Ruff lint/format checks, frozen-lock validation, and wheel/source
distribution builds. Regression cases cover fixed-panel missingness and outcome subsetting,
single-gene gradients and degenerate exposure, support-only logit changes, matched protein
aggregation and confidence, protein/registration identity mismatches, and specimen-first scoring.

The user confirmed on October 26, 2026 that no local HTAPP specimen inputs are available. The
real-specimen milestone is blocked on acquiring those files and their matching metadata; no
real-data result or memory qualification is claimed. Once available, prepare one complete specimen,
inspect its actual count axis and full H&E/protein/ST overlap, compare H&E-only, RNA-transfer, and
protein-guided outputs, and score them using the fixed normalization contract before expanding
to patient-held-out experiments.

| Milestone | Software scope in this revision | Scientific acceptance still required |
| --- | --- | --- |
| Data qualification | Local input contracts, HTAPP-oriented adapter, geometry helpers | Actual file-derived eligibility, matching provenance, count-axis inspection, and a complete first-specimen overlay |
| Baseline path | Shared mapping/decoder, linear aggregation, training and scoring interfaces | One real specimen from qualified images and RNA through scored common-region predictions |
| Personalized baseline | H&E-only and RNA comparison configuration | Patient-held-out added value with transfer and correction reported separately |
| Protein contribution | Shared regional observation operator, confidence weights, and source identity checks | Validated marker interpretation, registration uncertainty, and incremental expression/localization evidence |
| Neighborhood/communication | Generic/LR fixed/refined configuration family | Controlled staged comparison with identical base mapper, budgets, and independent readouts |
| Robustness/ST reference | Explicit output and evaluation boundaries | Full perturbation/control suite, within-type endpoint, calibrated uncertainty, separate ST-informed comparator |
| Packaging and scale | Small cached-feature workflow and persistence contract | Full inference provenance, restart qualification, block halos, chunked/factorized exports, external comparators, and measured real-specimen resources |

Large-scale execution is not qualified by a small synthetic pass. A 100,000-cell × 20,000-gene
float32 matrix is approximately 8 GB; transfer, correction, and final-expression matrices multiply
that footprint. Production exports should store assignment and reference matrices plus residual
factors and requested gene maps, with optional chunked reconstruction. Full-specimen dense outputs
are for bounded development fixtures until that path is implemented and measured.

Required checks include target removal/replacement invariance, physical transform direction and
units, analytical aggregation, gene/state reorder invariance, masked-protein value invariance,
reference-state deletion behavior, model save/reload identity, and small-graph versus tiled
inference with adequate halos. Unsupported mass needs reference-dropout training and validation;
the presence of an extra softmax state is not calibrated out-of-reference detection. Tiling tests
and calibrated gene-error intervals remain required before those capabilities can be claimed.

Progress follows demonstrated outcomes. If RNA adds little, investigate matching, coverage,
initialization, and aggregation. If proteins improve only broad identity, report that narrower
result. If LR matches generic context, retain the simpler model. If LR helps decoding but not
placement, claim only the expression improvement. Robust independent within-type localization
would justify further joint refinement and external validation. An inconclusive result remains a
valid outcome for a small cohort.
