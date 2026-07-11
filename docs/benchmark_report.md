# HEIR/SIGHT benchmark on downloaded cohorts (NatCommun/MOSAIC + snPATHO)

Historical locked run: 2026-07-10. DeepBench refresh: 2026-07-11.
Environment: `hne` conda env, RTX 3080 (10 GB), Space Ranger 4.1.0.
Scope: (1) audit the method ("check HEIR"); (2) benchmark it on the local cohorts.

## TL;DR

- **Locked snPATHO validation is now complete** for all three downloaded cases,
  using 500 frozen genes, Space Ranger nuclei, CUDA OmiCLIP features, matched
  snRNA prototypes, and target Visium expression opened only after all predictions
  were hash-frozen.
- The primary endpoint **did not succeed**: macro median-gene Spearman was
  **-0.0054**, versus **0.0026** for a spatial shuffle and **0.0224** for the
  matched type-mean baseline. The current model therefore does not establish
  general histology-to-expression spatial prediction on snPATHO.
- HEIR slightly reduced macro median-gene MSE versus spatial shuffle
  (**0.1225 vs 0.1234**) and matched pseudobulk (**0.1465**), but not versus the
  matched type mean (**0.1092**). Location-level cosine was also below both
  molecular baselines.
- Historical v0.2 round-0 inference was computationally efficient: mean CUDA inference was **18.7 s**
  per slide, **1,907 nuclei/s**, and **2.33 GiB** peak allocation. Mean cell and
  spot coverage were **83.1%** and **92.2%**.
- The earlier 13-specimen molecular falsification still supports the narrower
  premise that matched donor RNA is more informative than a wrong donor or generic
  atlas. It does not rescue the failed image-to-spatial-expression endpoint.

## Retrospective snPATHO-DeepBench-v1

The full attached benchmark plan is now represented by a separate, fail-closed
retrospective scorer. It preserves `snPATHO-Locked-v0.2`, applies the available
expanded metrics and historical integrated-reference library-size weighting to
all three cases, and reports
unavailable tracks explicitly. The historical round-0 statistic is the
equal-weight specimen mean of
`median_g(rho_HEIR,dg - rho_historical-hard-type-mean,dg)`: **-0.0259** (10,000
resamples: 95% interval **-0.0829 to 0.0081**), so the retrospective diagnostic
is also negative.

This is not the plan's requested primary result: the historical references pool
FFPE snPATHO, frozen SNAP snPATHO/Flex, and frozen 3-prime nuclei. Hash-bound
FFPE-only counts now provide hard/soft integrated-annotation type-mean
sensitivities, but clean reannotation, primary scANVI, prototype-only, refined and
five-seed predictions, composition-adjustment inputs, per-spot H&E tissue
fraction, and the required image/graph shuffles are absent. Target-H&E-derived
OOD calibration also prevents a Track A1/A2 compliance claim. See
[snPATHO-DeepBench-v1](snpatho_deepbench.md) for specimen-level results,
reference composition, spot QC, the readiness matrix, and the reproduction
command.

## 0. Locked snPATHO v0.2 benchmark

This was a single prespecified seed-17 proof-of-concept run. It used the
development-only B1 latent transform and RNA decoder, a three-layer 256-wide graph,
20 Monte Carlo latent samples, and a target-H&E-only OOD threshold calibration.
It ran one-pass personalized training followed by prediction; it did **not** call
the iterative refiner and therefore provides no empirical refinement contrast.
The latter copied the B1 Mahalanobis mean/precision unchanged and calibrated only
the scalar 95th-percentile cutoff; every provenance artifact records
`target_expression_accessed: false`. The run is transductive: target H&E and
Visium positions/scalefactors were used before training for morphology and
capture-area filtering, but target Visium expression/counts remained locked.

| Case | Evaluable spots | Nuclei | Cell coverage | Spot coverage | Median gene Spearman | Pearson | MSE | Location cosine | Inference s | Peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4066 | 4,659 | 40,739 | 0.595 | 0.809 | -0.0186 | -0.0308 | 0.2141 | 0.6620 | 22.70 | 2.68 |
| 4399 | 4,454 | 31,499 | 0.949 | 0.993 | 0.0070 | 0.0165 | 0.0698 | 0.6702 | 17.28 | 2.07 |
| 4411 | 2,758 | 34,103 | 0.950 | 0.965 | -0.0045 | 0.0030 | 0.0837 | 0.6803 | 16.22 | 2.24 |
| **Macro** | — | — | **0.831** | **0.922** | **-0.0054** | **-0.0038** | **0.1225** | **0.6708** | **18.73** | **2.33** |

| Method, macro over donors | Median gene Spearman | Pearson | MSE ↓ | Location cosine |
|---|---:|---:|---:|---:|
| **HEIR** | -0.0054 | -0.0038 | 0.1225 | 0.6708 |
| HEIR spatial shuffle | 0.0026 | 0.0011 | 0.1234 | 0.6722 |
| Matched snRNA pseudobulk | undefined | undefined | 0.1465 | 0.8084 |
| Matched type mean | 0.0224 | 0.0207 | **0.1092** | 0.7813 |

Against spatial shuffle, the donor-bootstrap mean Spearman difference was
-0.0080 (95% CI -0.0233 to 0.0052; bootstrap fraction with delta > 0 of 0.14).
This fraction is descriptive, not a probability of truth or a p-value. HEIR's MSE
improvement was 0.00087 (95% CI 0.00007 to 0.00235), so it captured a small
amplitude advantage without robust gene-wise spatial rank correlation. A subset
of genes did show signal—34/500 genes exceeded Spearman 0.2 in 4066, 22/500 in
4399, and 2/500 in 4411. Examples include ERBB2/KRT8/KRT19 in 4066 and
APOE/APOC1/CST3 in 4399.

The locked primary result is negative and must remain so. Further architecture,
loss, panel, or threshold tuning should be performed on development cohorts and
then assessed on a new untouched spatial cohort, not tuned and re-reported on
these same three snPATHO targets.

The complete local, hash-bound outputs are
`artifacts/snpatho/orchestration_v0_2/benchmark.all.{json,tsv}`; the frozen plan
SHA-256 is `471dbf8f2b5632918772219d78722af36ccb4949b4d840e1c8ef227bf8f4c6b2`.

Sections 1–6 below retain the earlier development benchmarks and audit record.
They are useful provenance, but Section 0 and the current test suite supersede
their then-current status statements.

## 1. Data verified

All 14 audited donor/sample filters in `manifests/natcommun.tsv` reproduce the exact
curated nucleus counts (B1=3937, L3=17804, D1=8936, …); B2 excluded (50 nuclei).
H5AD `X` is raw counts; all 70 panel genes map uniquely via `feature_name`.
The 16 manifest H&E TIFFs total ~29 GB (breast/lung 0.2–1.3 GB, DLBCL 1–6 GB).

## 2. Molecular falsification benchmark

`scripts/benchmark_molecular.py`. HEIR's own validation ladder (docs/validation.md)
at the level that needs no H&E: predict each held-out cell's 70-gene panel
expression (frozen log1p-CPM-10k space) from a reference's cell-type means (oracle
type), and predict specimen composition, under four reference conditions. Scored
with HEIR's shipped `expression_metrics` / `composition_metrics`. Donor-aware
(matched = same specimen held-out; generic/wrong = other specimens), 3 seeds,
13 specimens (breast B1/B3/B4, lung L1–L4, DLBCL D1–D6).

| Condition | Recon MSE ↓ | median-gene Spearman ↑ | Composition JS ↓ | matched wins |
|---|---:|---:|---:|:--:|
| **matched donor**   | **0.451** | **0.227** | **3e-5** | — |
| generic atlas       | 0.629 | 0.201 | 0.117 | 13/13 |
| wrong donor         | 0.696 | 0.191 | 0.157 | 13/13 |
| permuted labels     | 0.643 | −0.020 | — | 13/13 |

Reading: matched < generic/wrong/permuted on MSE for **every** specimen. The key
falsification test — "similar performance for matched vs wrong donor falsifies
personalization" (validation.md) — is **passed**: matched clearly beats wrong
donor. The permuted-label control collapses Spearman to ~0, as it should.

Caveat: the matched advantage reflects recoverable *donor-specific* signal, which
mixes true biological state with donor batch/technical effects. Disentangling the
within-type biological residual requires morphology-driven routing (the image arm),
which is untested here. This is a **necessary, not sufficient**, condition for the
full cell-level claim.

## 3. Image→RNA pipeline on real H&E (B1_4)

Per the user's direction, nucleus segmentation was done with **`spaceranger segment`**
(GPU StarDist), not HistoPLUS/CellViT. `scripts/segment_slide.sh` (H&E TIFF → tiled
BigTIFF via pyvips → segment) then `scripts/geojson_to_heir.py` bridges the geojson
to HEIR (centroids + 10 geometric morphology descriptors).

| Stage | Result |
|---|---|
| `spaceranger segment` B1_4 | **53,197 nuclei** (geojson + instance mask), ~4 s GPU |
| bridge → nucleus table + features | 53,197 rows, feature space `geom-morph-v1` |
| `prepare-reference` (Level1, panel) | 3,937 cells, 6 types, 70 genes |
| `build-prototypes` (fit SVD) | 44 prototypes, latent `sha256:c000ed37…` |
| `prepare-histology` (mpp=0.5) | 53,197 nuclei, **720,704** graph edges |
| `train` (region 15,001 nuclei, 40 ep) | best epoch 16, val loss −1.509 |
| `predict` (full WSI 53,197) | **100% abstain** (unknown prob ≈ 0.999; OOD=0) |

The universal abstention is correct behavior: geometry-only features + an untrained
(random) decoder are uninformative, so HEIR routes mass to the unknown/dustbin
rather than hallucinating states. The conditional (forced) composition still leans
to the dominant B1 types — Tumor_Breast 37.7%, T/NK 32.6%, Stroma 28.4%, B 1.3% —
and predicted P(Tumor) is spatially coherent: **Moran's I = 0.240** on the real
graph vs **0.002** permuted. To get confident, non-abstaining predictions requires
(a) a pathology foundation-model feature space (UNI/CONCH/HistoPLUS embeddings) and
(b) a donor-held-out scVI/scANVI decoder — both documented next steps.

MPP for B1_4 is an assumption (0.5 µm/px): the TIFF's resolution tag is a 300-DPI
placeholder, so absolute graph radius is uncalibrated (affects neighborhood size,
not the molecular comparison).

## 4. Bugs found and fixed by running the real pipeline

All three are string-dtype truncation via the `np.full(n, value, dtype="U"/np.str_)`
footgun (zero-width dtype → `<U1`), which silently truncated every multi-character
donor/specimen ID ("B1"→"B") and broke `assemble-batch`'s provenance/latent checks.
Only triggers on real multi-char IDs, so the synthetic tests missed it.

1. `src/heir/cli.py` `command_prepare_reference` — donor_ids/sample_ids.
2. `src/heir/prior/prototypes.py` `build_sample_prototypes` — sample_ids.
3. `src/heir/cli.py` refine — refined prototype sample_ids.

Fixed to `dtype="U%d" % max(1, len(value))`. Regression coverage for the
completed locked pipeline and development-only refinement redesign now runs in
the committed GitHub Actions quality workflow as well as locally.

## 5. Method audit highlights (26-agent adversarial review)

Sound & verified: versioned prototype plus bounded, type-conditioned low-rank
residual `z=Σ q·μ+Δz`, where `q` is conditional on a known state;
`α=n/(n+κ)` prototype shrinkage;
SHA-256-bound immutable latent space (rejects refitted same-width SVD); genuine
log-domain unbalanced OT with a dustbin (does not force snRNA capture fractions);
per-term ablatable losses; separate distilled student; four independent
target-ST blocks; honest, code-backed non-identifiability disclaimer.

At audit time, the type-mean baseline, donor bootstrap, and decisive expression
criterion were not yet wired into an end-to-end target benchmark. Section 0
supersedes that finding: the completed locked runner executes the type-mean
baseline and donor bootstrap and reports the prespecified expression endpoint.
The historical locked implementation had sticky refinement anchors and did not
run refinement. The current development branch adds revocable anchors,
transport-derived responsibilities, a validated round-0 rollback target, fixed
measured priors by default, and a restricted zero-initialized residual, but those changes
remain unvalidated biologically and do not alter the locked result. Broader
cautions remain around generic initialization, graph boundaries, residual
identifiability, and independent cell-level truth.

## 6. Runnable next steps (infrastructure in place)

- Segment the remaining 14 sections: `scripts/batch_segment_natcommun.sh` (resumable).
- Add pathology foundation-model features per nucleus (replace `geom-morph-v1`).
- Train a donor-held-out scVI/scANVI decoder (removes `--allow-random-decoder`).
- The locked snPATHO spot-level spatial validation described in the original plan
  is complete; see Section 0. Do not tune the reported architecture, panel, or
  thresholds on these three cases and present a re-run as independent validation.
