# Local data ledger

## Benchmark-readiness decision

A **full SIGHT/HEIR benchmark** requires all of the following for a specimen:

1. native-resolution H&E with a reviewed pixel-size calibration;
2. stable, instance-level nucleus IDs (Space Ranger is the default segmenter);
3. frozen pathology features extracted from the H&E, rather than geometry alone;
4. an annotated, specimen-matched snRNA-seq reference;
5. held-out spatial expression aligned to the H&E section or an adjacent section;
6. a frozen gene panel and a nucleus-to-spatial-observation assignment;
7. donor-level splits with target spatial expression and derived labels excluded
   from training, refinement, feature selection, and threshold selection; any
   transductive use of target H&E or non-expression spatial metadata is declared.

“Runnable” below means that the current local payload can support the complete
evaluation without obtaining a missing reference or assay payload; it does not
mean that every derived artifact has already been built.

| Downloaded data | Full proposed benchmark locally complete? | Valid role now | Decisive limitation |
|---|---|---|---|
| snPATHO-seq breast cancer (3 specimens) | **No; historical round-0 benchmark only** | Truth-scored retrospective development/diagnostic | No redesigned refined predictions or clean reannotated/scANVI R1; all three cases have been opened and cannot be a new untouched confirmation |
| NatCommun/MOSAIC (15 usable sections, 13 donors) | **Not currently** | Personalized inference, matched-vs-generic/wrong-reference ablations, runtime and spatial-plausibility checks | Raw Visium FASTQs and slide metadata are present, but no processed registered matrix or exact local reference is available for reproducible reprocessing |
| spatialDLPFC snapshot | **No** | Coordinate/format smoke tests only | Full-resolution H&E and spatial-expression objects are absent |
| HEST prostate ST (35 samples) | **No, not nucleus-resolved** | Spot-level expression baselines and transfer pretraining | Only approximately 1000-pixel embedded images; no matched snRNA reference |
| Generic prostate single-cell atlas | **Not a benchmark cohort** | Unmatched/generic-reference baseline | No paired H&E or spatial truth |
| TRACERx421 WES | **Out of scope** | None for SIGHT/HEIR | DNA sequencing resource; no required H&E/snRNA/spatial triplet |

Consequently, **no downloaded cohort currently completes the proposed refined,
clean-R1, untouched validation**. snPATHO is the only downloaded cohort with a
completed truth-scored historical run and can support retrospective development
diagnostics. NatCommun is still scientifically important: it tests whether a
sample-matched molecular prior improves over generic, wrong-donor, and permuted
priors. Its existing local derivatives cannot establish per-cell or spot-level
spatial accuracy; a future reproducible Visium reprocessing could change that
assessment.

## NatCommun/MOSAIC development cohort

Root: `/mnt/seagate/HnE/NatCommun_2025_s41467_025_59005_9`

- 16 high-resolution H&E sections from 14 donors: B1–B4, L1–L4, D1–D6; B1 and L1 have two sections.
- Three Cellxgene H5AD references totaling 86,356 nuclei by 18,063 genes;
  86,306 nuclei remain across the 13 usable donors after excluding B2.
- Audited mappings are encoded in `manifests/natcommun.tsv` and enforced in code.
- B2 is excluded because its curated reference has only 50 nuclei.
- Breast/lung TIFFs are non-pyramidal and lack trustworthy physical calibration. The PIL backend therefore requires an explicit MPP override. DLBCL Aperio slides report 0.2528 µm/pixel.
- All sections from a donor remain in one outer fold.

The smallest practical case is B1:

- H&E: `.../processed_data/B1_4.tif` (14,784 × 9,280)
- quick CytAssist image: `.../processed_data/cytassist_V6_B1_4_OPHI.tiff` (3,000 × 3,000)
- RNA filter: `donor_id == "7"` and `sample_id == "6"` (3,937 nuclei)

These data are donor matched, not cell registered.

## snPATHO locked validation

Root: `/mnt/seagate/HnE/snPATHO_seq`

| Sample | Biology | annotated nuclei | Visium spots |
|---|---|---:|---:|
| 4066 | primary ER+/HER2+ breast cancer | 20,472 | 4,769 |
| 4399 | metastatic TNBC in liver | 23,080 | 4,560 |
| 4411 | metastatic ER+/HER2− breast cancer in liver | 27,311 | 2,812 |

The RDS references, Visium objects, and exact GEO archive members are recorded in `manifests/snpatho.tsv`. All three rows have `analysis_role=locked_validation`. Runtime manifest validation rejects spatial data in a training role.

The GEO archive contains the full-resolution H&E, filtered/raw Visium matrices,
tissue positions, scale factors, CytAssist images, and alignment metadata for all
three specimens. The spatial RDS barcodes add a synthetic `_1` suffix; stripping
`_[0-9]+$` is collision-free and maps every retained barcode to a Visium position.
The local Visium metadata do not contain meaningful spot cell-type labels, so the
locked benchmark can score gene-expression reconstruction but must not describe
composition as observed ground truth unless an independent annotation is added.

Current derived-artifact status (2026-07-10): all three specimens have audited
full-resolution H&E, converted single-nucleus references, Space Ranger nucleus
segmentations, CUDA OmiCLIP pathology features, hash-bound training/inference
artifacts, and frozen Visium truth. The completed run evaluated 40,739, 31,499,
and 34,103 segmented nuclei for 4066, 4399, and 4411, respectively. Results are
reported in `docs/benchmark_report.md` and the lightweight machine-readable
summary `reports/snpatho_v0_2_summary.json`.

FFPE-snPATHO-only count references have also been isolated for all three cases
with exact workflow filters and hashes in
`reports/snpatho_r1_reference_manifest.json`. Their `major_annotation` labels
come from the published integrated-workflow objects, and their latent/prototype
adapter is an SVD development fallback rather than primary scANVI. They support
explicit hard/soft type-mean sensitivity analyses, not the requested clean R1
refined primary endpoint.

## DLPFC limitation

The local `/mnt/seagate/HnE/spatialDLPFC` snapshot has low-resolution images, alignments, and nucleus metrics, but no usable expression objects or complete high-resolution H&E payload. It can test coordinate code, not the HEIR transcriptomic experiment. The blueprint's DLPFC study requires a separate complete download and a reviewed matching ledger.

The snapshot contains 34 morphology tables (1,713,224 nucleus rows), small
raw/segmentation thumbnails, 33 alignment JSON files, and 12 low-resolution tissue
PNGs. These are not substitutes for the omitted whole-slide TIFFs and Visium
expression matrices.

## HEST prostate and prostate single-cell reference

`/storage/HE_GPT/reference/hest_raw/st` contains 35 Visium H5AD files with 108,217
spots in total. Each file embeds only an approximately 1000-pixel image, where a
Visium spot spans roughly 6--7 pixels. This is adequate for spot-level baselines,
but not for Space Ranger nucleus segmentation or nucleus morphology. The derived
`reference_hest_prostate.h5ad` is a concatenated spatial matrix, not an snRNA-seq
reference.

`/storage/HE_GPT/single_cell/merged_all_lineages.rds` contains 175,203 cells from
54 samples, 32,524 genes, and 64 annotated cell types. It is a useful unmatched
prostate atlas comparator, but it is not specimen-matched to the HEST slides and
therefore cannot turn HEST into a full personalized benchmark.

## Artifact contracts

- `HistologyBag`: nucleus IDs, cached image/morphology features, micron coordinates, confidence, artifact probability, graph edges, donor/sample/block identity, image/segmentation/feature hashes, and pathology feature-space identity.
- `RNAReference`: sparse counts, frozen gene order, cell IDs, donor/sample/block IDs, labels, optional latent/program scores, source-count hash, and a latent-space identifier.
- `PrototypeSet`: sample/type prototype means, diagonal variances, weights, cell counts, donor/block/reference provenance, and the same latent-space identifier used by the decoder.
- `HEIRTrainingBatch`: one coherent image graph patch plus sample-level weak targets, donor/block/bag and feature-space identity, molecular-teacher training donors, spatial spot IDs, and role-tagged source hashes.
- `PredictionBundle`: cell/type/prototype posteriors, latent uncertainty,
  known-state-conditional expression intervals with an explicit availability
  mask, OOD/abstention outputs, source hashes, and the complete
  stochastic/decision policy used for inference.

NPZ files carry a contract/version tag and never enable pickle.
