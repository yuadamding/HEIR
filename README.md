# HEIR

HEIR maps the molecular states measured by sample-matched single-nucleus RNA-seq back onto H&E nuclei. It is implemented as weakly supervised spatialization—not as fictitious nucleus-to-RNA paired regression.

Two modes are kept distinct:

- **HEIR-Personalized:** H&E plus matched snRNA-seq are inputs. This is the primary method.
- **HEIR-Distilled:** an H&E-only student learns from frozen personalized teachers. Its results must be reported separately.

Target-sample spatial expression and derived labels are blocked from personalized
training and refinement by manifest and runtime checks. Personalized HEIR is
transductive: target H&E is an input, and Visium spot positions/scalefactors may
be used for capture-area filtering. The target count matrix is opened only after
the selected predictions are frozen.

## What is implemented

The repository contains a runnable HEIR core and a development-only refinement redesign:

- immutable donor/block/section manifests with leakage checks;
- exact local NatCommun/MOSAIC and snPATHO cohort mappings;
- backed, sparse H5AD selection and versioned NPZ artifact contracts;
- calibrated PIL/OpenSlide access, coordinate transforms, nucleus tables, image QC, sparse cellular graphs, Visium assignment, and conservative registration;
- a transferable RNA VAE fallback, scVI-decoder export, and frozen scGPT teacher objective;
- shrunken sample/type RNA prototypes;
- a graph-aware, hierarchical prototype model with a zero-initialized,
  RNA-PCA-informed type-conditioned low-rank residual whose latent L2 norm is
  bounded relative to measured type-specific molecular geometry;
- differentiable unbalanced optimal transport with prespecified fixed unknown
  mass, or explicit unknown targets, in the primary path;
- composition, pseudobulk, marker/program, residual, cycle, hierarchy, graph, anchor, and uncertainty objectives;
- covariance-aware UOT on the decoded molecular latent, detached transport responsibilities, and direct molecular/type M-step supervision;
- two parent-gated rounds followed by fine refinement, with an EMA teacher,
  revocable anchors, scale-held-out-view gates, a validated round-0 rollback target,
  and fixed measured priors by default;
- H&E-only distillation, calibration, OOD detection, abstention, v8
  known-state-conditional expression availability and intervals (public cell
  means are fail-closed on abstention while finite internal aggregate values
  and legacy migrations remain explicit), biological baselines, and
  donor-aware metrics;
- pull-request and main-branch CI for formatting, lint, and the unit/synthetic suite;
- synthetic, unit, and local-data smoke paths.

Large pretrained components are deliberately external assets. HEIR contains the
source, manifests, hashes, loaders, tests, and orchestration scripts, but not
multi-gigabyte weights. The default external root is
`../HEIR_assets/pretrained`, or set `HEIR_PRETRAINED_DIR`. Verify every weight
against the committed manifest before use:

```bash
export HEIR_PRETRAINED_DIR=/storage/HE_GPT/HEIR_assets/pretrained
conda run -n hne python scripts/manage_pretrained_assets.py list
conda run -n hne python scripts/manage_pretrained_assets.py verify
```

Generated run artifacts remain under the ignored `artifacts/` path because their
contracts bind canonical source paths as well as hashes. They are never tracked
by Git; do not relocate a completed locked run behind a symlink.

## Quick start

Use the existing H&E environment:

```bash
conda run -n hne python -m pip install -e . --no-deps
conda run -n hne heir doctor --require-files
conda run -n hne heir demo --output outputs/demo --epochs 3
```

Prepare the audited NatCommun B1 reference. The bundled panel is a small smoke panel; replace it with a training-only, frozen ~500-gene panel for a real experiment:

```bash
conda run -n hne heir prepare-reference \
  --manifest manifests/natcommun.tsv \
  --section-id B1_4 \
  --cell-type-key Level1 \
  --genes manifests/gene_panel_example.tsv \
  --output artifacts/B1/reference.npz

conda run -n hne heir build-prototypes \
  --reference artifacts/B1/reference.npz \
  --reference-with-latent artifacts/B1/reference_latent.npz \
  --fit-latent-transform artifacts/B1/shared_svd.npz \
  --manifest manifests/natcommun.tsv \
  --section-id B1_4 \
  --output artifacts/B1/prototypes.npz
```

The SVD is fitted once on an allowed development reference and reused with `--latent-transform` for every other specimen; HEIR rejects independently refitted or merely same-width latent spaces. This B1 command is a local smoke path, not the paper-grade molecular model. For the latter, use donor-validated scVI/scANVI and `SCVIAdapter.export_transferable_decoder_checkpoint`, which writes the gene order, training donors, latent ID, and decoder-only flag required by `heir train`.

For snPATHO, the native FFPE-only CUDA stage is reproducible but remains in its
own heavy environment. Its native checkpoint and distilled decoder are written
to `../HEIR_assets/pretrained`, while only hash-bound latent/reference artifacts
return to the ignored run directory:

```bash
PYTHONPATH=src conda run -n scdiffeq python scripts/train_snpatho_scanvi.py
```

The R2 default does **not** use `section_id` as a removable batch and does not
marginalize decoder expression over specimens. It runs a donor-rotated
decoder-distillation audit, writes those audit checkpoints beside the external
decoder, and uses one hash-bound, 32-sample posterior-mean expression target for
every rotation and the deployable decoder. `--molecular-design
technical_batch_only` is accepted only when every declared technical level is
observed across specimens and every specimen contains multiple levels. Output
families are immutable: reruns use fresh versioned paths rather than deleting a
completed checkpoint. The historical specimen-corrected model remains available as the explicitly named
`specimen_batch_sensitivity`; it is not the default.

The current input labels are the published integrated-workflow annotations;
the script records them as an annotation sensitivity and does not call them an
independent clean reannotation.

When `--reference-with-latent` is used, downstream `assemble-batch` must consume
that emitted reference. The prototype artifact is hash-bound to the enriched
reference so latent identity and source provenance remain simultaneously
verifiable.

## Artifact-to-result CLI

Heavy pathology/scGPT/scVI models remain isolated producers of cached artifacts. HEIR then provides the checked path between them:

For molecular-refinement development, first run the independent reviewer-label
[broad-type gate](docs/broad_type_supervised_gate.md) on frozen features with
both graph and molecular residual disabled. That gate is currently
`blocked_evidence` because reviewed labels are absent; the commands below remain
an engineering path and must not be interpreted as a validated biological path
until the gate passes.

```text
prepare-histology + prepare-reference
              ↓
build-prototypes + fit-ood
              ↓
         assemble-batch
              ↓
 independent broad-type gate [no graph, no residual]
              ↓ pass
 train [optional generic H&E–ST pretraining checkpoint]
              ↓
 refine → predict → evaluate-spatial
```

Use `heir prepare-histology` to join a canonical nucleus table with a pickle-free cached feature NPZ, calibrate pixels to micrometers, and construct a weighted graph. The command requires donor/sample/block and source-H&E provenance (directly or from a manifest) plus a stable `--feature-space-id` identifying the pathology encoder checkpoint and preprocessing recipe. `assemble-batch` accepts optional scGPT, program, OOD, domain, calibration, and non-target spatial-pretraining artifacts. `train --initial-heir-checkpoint` starts personalization from a generic H&E–ST model. `refine` requires independently identified view predictions by default and emits both a refined checkpoint and updated per-sample prototype banks. `predict` records source/model hashes, OOD provenance, RNG seed, Monte Carlo count, and decision thresholds; `evaluate-spatial` requires exact nucleus, gene, and type identities.

Run `heir COMMAND --help` for each versioned contract. Training intentionally caps the total cells retained in one sample-level autograd graph (16,384 by default); use a prespecified donor-balanced tissue-region sample for fitting, then patchwise inference over the complete WSI.

For snPATHO RDS files, convert the RNA object in the existing R environment and selectively extract the Visium/H&E archive:

```bash
conda run -n r_env Rscript scripts/export_seurat.R \
  /mnt/seagate/HnE/snPATHO_seq/dryad/files/4399_integrated_seuarat_object.rds \
  artifacts/4399/reference.h5ad RNA

python3 scripts/extract_snpatho.py \
  --archive /mnt/seagate/HnE/snPATHO_seq/GEO/GSE268427/suppl/GSE268427_RAW.tar \
  --sample 4399 --output artifacts/4399/visium

conda run -n hne heir prepare-reference \
  --manifest manifests/snpatho.tsv --section-id 4399 \
  --input artifacts/4399/reference.h5ad \
  --output artifacts/4399/reference.npz
```

The frozen v0.2 three-sample run is automated by a fail-closed dry-run-first
orchestrator. Space Ranger is the default segmentation method; OmiCLIP,
training, and prediction use CUDA mixed precision. No locked Visium expression
is opened until all selected predictions validate. This historical locked run is
`train → predict`, not a refinement experiment, and remains unchanged:

```bash
conda run -n hne python scripts/run_snpatho_pipeline.py --sample all
conda run -n hne python scripts/run_snpatho_pipeline.py --sample all --execute
```

See [the snPATHO pipeline guide](docs/snpatho_pipeline.md) for resume behavior,
stage outputs, telemetry, and the target-isolation boundary.

The completed locked v0.2 result is reported in
[the benchmark report](docs/benchmark_report.md). Its primary endpoint is
negative: macro median-gene Spearman is -0.0054 and does not beat spatial
shuffle or the matched type-mean baseline. The repository reports that result
as-is; it is evidence for further development, not a successful performance
claim.

The attached comprehensive plan is implemented as the separate retrospective
[`snPATHO-DeepBench-v1`](docs/snpatho_deepbench.md) scorer. It revalidates the
immutable locked inputs, applies historical integrated-reference library-size
weighting and the expanded metric panel, consumes hash-bound FFPE-only counts for
explicitly labeled hard/soft integrated-annotation sensitivities, writes per-gene
and equal-weight specimen summaries, and records every
unavailable track without substituting weaker evidence. The available
historical round-0 diagnostic is also negative (equal-weight mean across
specimens of `median_g(rho_HEIR,dg - rho_hard-type-mean,dg)`: -0.0259).

The native-scANVI refined endpoint is now testable as an opened-cohort
developmental sensitivity, not as the requested clean-R1 primary endpoint. The
native model uses FFPE-snPATHO counts with the published integrated-workflow
annotations; those labels are not an independent clean reannotation. All five
prespecified endpoint seeds (17, 41, 89, 131, and 197) and the three control
seeds (17, 41, and 89) are complete. The refinement matrix scored all 93 of 93
requested artifacts, but strict ordering failed: 49 of 108 checks passed and 59
failed. All 18 directed wrong-prototype-bank cases are present; their all-directed mean
refined-minus-control paired median gene-Spearman delta is -0.0040. Its overall
status is `blocked_evidence`, with missing clean independent
reannotation, generic-atlas, label-permutation, state-omission,
reference-downsampling, and untouched-external-cohort evidence. The historical
unknown-mass attempt was provenance-blocked, but its clean replacement is now
complete: 75 of 75 CUDA stages and 30 endpoints produced all 15 fixed-mass
comparisons. The result is unstable and does not select a mass or rescue
refinement. See the hash-bound
[compact refinement-matrix summary](reports/snpatho_refinement_matrix_v1_summary.json).
The [v2 unknown-mass summary](reports/snpatho_unknown_mass_sensitivity_v2_summary.json)
records the clean result; the v1 report remains the historical provenance failure.
The v2 refinement run manifest recursively validates all 138 canonical stage
recipes and current output hashes. Its final validation pass conservatively
records all 138 outputs as adopted, verifies the source-bound CLI identity and
all shuffle transformation hashes, and honestly records
`execution_provenance_verified=false`; the matrix remains development evidence
and cannot unlock a full-primary claim even if its numerical ordering passed.
Full per-gene JSON, TSV, and Markdown reports remain under the ignored
`artifacts/` tree; only compact summaries and manifests under `reports/` are
tracked.

For refinement development, use only a development cohort with spatial truth.
The independent broad-type gate must pass first; the internal residual
posterior-concentration threshold is a separate numerical safeguard and is not
a substitute for reviewer-label evidence.
The separate orchestrator enforces the prespecified
`train → predict-0 → refine/predict rounds → development evaluation` order and
rejects locked roles:

```bash
conda run -n hne python scripts/run_refinement_development.py \
  --plan configs/refinement_development_plan.example.json
```

Replace the example commands/outputs with cohort-specific, prespecified CLI
arguments before adding `--execute`. See
[the refinement redesign](docs/refinement_redesign.md) for the implemented
algorithm and remaining validation requirements.

After the model, gene panel, and thresholds are frozen, create the separate
locked Visium truth contract. The command accepts a QC H5AD, a 10x HDF5 file,
or a filtered Matrix Market directory. An RDS-derived H5AD must carry the JSON
lineage sidecar emitted by `scripts/export_seurat.R`:

```bash
conda run -n hne heir prepare-spatial-truth \
  --manifest manifests/snpatho.tsv --section-id 4399 \
  --counts artifacts/4399/visium.h5ad \
  --conversion-provenance artifacts/4399/visium.h5ad.provenance.json \
  --positions artifacts/4399/visium/tissue_positions.csv \
  --scalefactors artifacts/4399/visium/scalefactors_json.json \
  --nuclei artifacts/4399/nuclei.csv \
  --genes artifacts/snpatho/gene_panel.tsv \
  --output artifacts/4399/spatial_truth.locked.npz
```

The adapter preserves the canonical panel order, normalizes panel counts with
the full-transcriptome library size, deterministically resolves Seurat `_N`
and 10x `-N` barcode suffixes, and assigns nuclei only inside the full-resolution
Visium spot disk. Its versioned NPZ is pickle-free and records hashes for the
counts, coordinates, scale factors, segmentation, panel, manifest, and any
RDS conversion lineage. `evaluate-spatial` consumes it directly; training and
refinement commands do not.

Generate each already-frozen prediction with measured inference telemetry:

```bash
conda run -n hne heir predict \
  --checkpoint artifacts/snpatho/frozen/heir.pt \
  --histology artifacts/4399/histology.npz \
  --prototypes artifacts/4399/prototypes.npz \
  --genes artifacts/snpatho/gene_panel.tsv \
  --donor-id 4399 --device cuda \
  --output artifacts/4399/predictions.npz \
  --telemetry-output artifacts/4399/prediction.telemetry.json
```

Copy `configs/snpatho_benchmark_plan.example.json`, insert the SHA-256 of the
frozen checkpoint, and keep all three cases. If personalization emits one
checkpoint per donor, put `checkpoint_sha256` inside each case instead of at
the plan root. Then open and score the locked targets once:

```bash
conda run -n hne python scripts/benchmark_snpatho.py \
  --plan configs/snpatho_benchmark_plan.json \
  --output artifacts/snpatho/benchmark.json \
  --tsv artifacts/snpatho/benchmark.tsv \
  --iterations 10000 --seed 17
```

The report contains donor-level and macro bootstrap results for HEIR, constant
matched-snRNA pseudobulk, a deterministic spatial shuffle, and the matched
type-mean baseline, plus explicit per-gene correlation and error rows for every
panel gene (undefined constant-baseline correlations remain `null`). It also reports abstention/coverage, inference time,
throughput, CUDA device/precision, and peak allocated CUDA memory. The runner
requires all three snPATHO donors by default and rejects checkpoint, reference,
or panel provenance that overlaps or disagrees with locked target artifacts.

For general manifest-driven `prepare-histology`, an explicitly supplied TIFF is
checked against the manifest source. The snPATHO orchestrator instead uses the
already audited local full-resolution TIFF and a frozen SHA-256 in the v0.2
config, avoiding repeated decompression of multi-gigabyte archive members while
retaining byte-level provenance. These rows remain locked validation inputs
regardless of conversion or extraction.

## Local cohort decision

The downloaded DLPFC directory contains imagery and alignment files but not the
expression objects needed for the blueprint's primary DLPFC experiment.
NatCommun/MOSAIC is directly usable for personalization, matched-versus-wrong
molecular falsification, and pipeline smoke tests: it contains 16 H&E sections
from 14 donors and 86,356 annotated nuclei (15 sections from 13 donors after the
prespecified B2 exclusion). Its current local derivatives do not provide the
registered spatial truth required to validate redesigned refinement.

The three snPATHO samples (4066, 4399, 4411) have a completed historical locked
round-0 Visium benchmark and a completed native-scANVI refinement development
matrix. The clean-independent-annotation/full-primary plan remains incomplete,
and these already opened cases cannot serve as a new untouched confirmation.
Their spatial measurements must not enter personalized optimization. No
downloaded cohort currently supports the full proposed refined validation claim.

See [data.md](docs/data.md), [method.md](docs/method.md),
[validation.md](docs/validation.md), and
[snpatho_pipeline.md](docs/snpatho_pipeline.md), and
[snpatho_deepbench.md](docs/snpatho_deepbench.md), plus
[refinement_redesign.md](docs/refinement_redesign.md), for the scientific and
engineering contracts and retrospective development status.

## Repository map

```text
src/heir/data/          manifests, H5AD streaming, artifact contracts
src/heir/image/         calibrated slide/coordinate/nucleus/graph operations
src/heir/image_features cached multiscale extraction
src/heir/prior/         RNA models, programs, state prototypes
src/heir/models/        personalized and distilled models
src/heir/losses/        UOT and biological constraints
src/heir/refinement/    constrained generalized EM and EMA teacher
src/heir/uncertainty/   calibration, OOD, abstention
src/heir/evaluation/    donor-aware and spatial metrics
manifests/              reviewed public-cohort ledgers
workflows/              reproducible stage orchestration
tests/                  unit and synthetic safeguards
```

HEIR does not claim full-transcriptome recovery or morphology-resolved fine immune states. The initial target is broad/intermediate types, 20–50 programs, and a prespecified nuclear-compatible gene panel.
