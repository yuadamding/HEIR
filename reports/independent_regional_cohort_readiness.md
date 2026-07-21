# Independent regional confirmation and spatialDLPFC pilot readiness

## Executive decision

**No downloaded cohort is eligible for the sealed definitive confirmation.** The 18--24-donor
confirmation target therefore remains unselected, its spatial-transcriptomic outcomes remain
unopened, and no Gate A--F validation has run.

spatialDLPFC is a **conditional, prospectively registered external-pilot candidate**, not a
definitive cohort. Paper and local metadata identify 19 claimed same-block H&E--Visium--snRNA rows
across 10 donors, but **zero rows are admitted exact-specimen matches**. High-resolution H&E and raw
or minimally processed Space Ranger payloads are absent, reference aliquot provenance is unresolved,
and reference/gene coverage is unqualified. The two Br2720/Br2770 filename discrepancies have been
resolved as metadata-level filename typos, but this does not verify the missing payload relationships.

Opaque copies of the official processed Visium and snRNA objects have been acquired and
checksum-bound. Neither expression object has been loaded, extracted, filtered, summarized, or used
for model fitting or scoring. Because the exFAT location cannot enforce role-separated ACL access,
the Visium object is not a genuinely blinded target seal and cannot authorize execution.

UNI2-h remains prohibited. H-optimus-1 revision
`3592cb220dec7a150c5d7813fb56e68bd57473b9` is the only registered H&E encoder. Iterative
refinement remains zero.

## Two separate frozen contracts

### Definitive high-confidence confirmation

The definitive protocol remains byte-for-byte unchanged at
`configs/independent_regional_confirmation_protocol.json`, SHA-256
`6033a2d7db6cb5095d014f984c6f6519a4a7c073e41b627379a47525236b668a`. Its 18--24 independent
donors, single-tissue design, model identity, 256-gene panel, 20-dimensional secondary latent,
M0--M7/F-ST/S1/S3/S4 arms, Gate A--F thresholds, resource limits, stopping rules, and registered
decision are not amended by this report or the pilot.

The sealed definitive decision still requires A + B + C + stringent F because that is what was
prospectively registered. Existing tests now fail if any byte of this contract changes.

### External-replication pilot

The separate pilot is frozen at `configs/independent_regional_pilot_protocol.json`, SHA-256
`f483c0d40e8e29746cb7e4694ca8a3666e2d7196acc7ab2c02e3cc6c0c9b20e5`. It binds the latest
scientific review (SHA-256
`c8daafb697c40d49d10ac7df2059bbf9fa3553fdaf73e916e98129ebb701b378`) and the definitive
protocol hash without amending it.

The pilot requires exactly 8--12 independent donors from one tissue, at least eight payload-
qualified exact same-block/specimen H&E--ST--sc/snRNA triples, at least one registered query
section per donor, sections nested within donor, and untouched held-out ST outcomes. Its primary
external-replication evidence is limited to:

1. `L(M3 H&E + matched snRNA) < L(M0 H&E)`; and
2. `L(M3 paired H&E + reference) < L(M4 shuffled/offset H&E + reference)`.

Even a complete pilot pass is not definitive population confirmation, does not establish the full
left-hand ST-floor inequality, and cannot be pooled into the definitive analysis.

No complete learned HEIR checkpoint exists that could make every pilot ST count score-only. The
registered executable identity is therefore the unchanged frozen leave-one-donor-out procedure:
training-donor ST may fit the frozen decoder, while each held-out donor ST remains score-only after
its fold seal. “Run once” means one registered execution containing all prespecified serial folds,
not one literal fit.

### Strict specimen-level admission addendum

The new instruction is frozen separately at
`configs/specimen_level_matching_addendum.json`, SHA-256
`f0a252602b8d0927a3ca259fb24385c5afae4bca7952261e96d0f47b8ddca97f`. It binds review SHA-256
`845ada0952c7a76efb4ee84e92538e33fed144faa8982257f135de21c4dea42b` and both parent protocol
hashes. It prospectively narrows parent-primary cohort admission and changes no parent model,
primary arm, estimand, gate, threshold, or frozen decision. Separately, it registers a new T2
same-donor/different-region-or-block diagnostic with its own non-gating thresholds and run order; T2
does not amend or rescue a parent primary decision.

For primary M3, “matched” now means the same donor and canonical physical block/specimen, with an
independent sc/snRNA aliquot or an immediately adjacent curl whose chain of custody returns to that
same specimen. A paper-level claim, same donor, same lesion, same tissue, or same disease is not
sufficient without a row-level query/reference assignment and payload proof.

| Match tier | Relationship | Role |
|---|---|---|
| T0 | Same physical block/specimen with independent aliquot, or adjacent curl from that specimen | Only tier eligible for primary M3 |
| T1 | Same lesion/procedure/time, separate physical core | Secondary close-match transport only |
| T2 | Same donor, different anatomical block (and, in spatialDLPFC, usually a different ant/mid/post region) | Weaker-reference control, never primary |
| T3 | Different donor, matched tissue/disease/assay/support | Existing wrong-donor control M6 |
| T4 | Query-excluded generic tissue reference | Existing generic control M7 |
| X | Same tissue or “integrated” modalities without specimen proof | Excluded |

Exactly one payload-qualified T0 block will be selected per spatialDLPFC donor using target-free
constrained assignment: maximize eligible donors, balance anterior/middle/posterior positions, then
use frozen SHA-256 tie-breaks. Selection is sealed before ST count access. A later QC failure makes
that donor non-evaluable; another block may not replace it. Multiple blocks never increase the
biological donor count.

## Layered scientific interpretation

The unchanged definitive pass rule and the biological interpretation are reported separately so a
stronger-tier failure cannot erase valid lower-tier evidence.

| Evidence tier | Required contrast or evidence | Permitted conclusion |
|---|---|---|
| Core regional inequality | `L_ST < L(M3) < L(M0)` and paired `M3 <` shuffled/offset `M4` | Exact regional hypothesis supported |
| Pilot primary replication | `M3 < M0` and paired `M3 < M4` | Independent right-hand/reference-augmentation and spatial-attribution evidence only |
| Strong multimodal synergy | `M3 < M1` reference-only | H&E and reference are complementary in average loss |
| Continuous-state inference | `M3_supported < M2_supported` | State signal beyond composition/type routing |
| Same-block reference advantage | same-block M3 beats same-donor/different-region-or-block T2 control | The exact reference is better than the available weaker same-donor reference; pure block specificity remains unresolved when anatomy changes too |
| Personalized matching | matched M3 beats fixed-support wrong M6 and generic M7 | Donor/sample personalization |
| High-fidelity reconstruction | reliability-adjusted variance, calibration, covariance, dynamic range and rare-state recovery | Molecular heterogeneity/state reconstruction rather than only conditional-mean prediction |

If no valid ST floor exists, the left-hand inequality is **unresolved, not failed**. If M3 beats M0
and M4 but stringent F fails, those lower-tier contrasts remain reportable; the biological wording
is restricted to conditional-mean regional prediction rather than molecular heterogeneity
reconstruction. The definitive protocol's stricter pre-existing A+B+C+F decision remains unchanged.
No new numerical “catastrophe” thresholds were invented after development; qualitative collapse
checks are claim-limiting diagnostics in the pilot.

Cell-resolved validation remains prohibited until a regional result passes, and would require a
separate registered cell-level cohort and protocol.

## Donor-level power and precision audit

Power was calculated on the sealed primary absolute donor contrast
`D_i = L_i(M0) - L_i(M3)`, with sections averaged within donor. The exposed NatCommun development
artifact has 13 primary donors, donor-balanced M0 loss `2.5280939211`, mean absolute contrast
`0.2050183904`, aggregate reduction `8.1096%`, and paired SD `0.2236080`. The one-sided 90% upper
confidence bound for that SD is `0.3085154` and is the conservative sensitivity value.

The planning alternative is a 5% reduction, absolute contrast `0.1264047`. This is 38.3% below the
exposed 8.11% estimate and is the smallest effect that can satisfy the sealed 5% materiality rule.
Paired-normal working-model power is:

| Analyzable donors | One-sided alpha=.05, observed SD | Positive two-sided 95% interval, observed SD | One-sided alpha=.05, conservative SD | Positive two-sided 95% interval, conservative SD |
|---:|---:|---:|---:|---:|
| 8 | 42.0% | 28.2% | 27.5% | 17.0% |
| 10 | 50.3% | 35.9% | 32.8% | 21.2% |
| 12 | 57.6% | 43.1% | 37.7% | 25.4% |
| 18 | 74.4% | 61.9% | 51.0% | 37.5% |
| 21 | 80.4% | 69.3% | 56.7% | 43.2% |
| 24 | 85.1% | 75.6% | 61.9% | 48.5% |

These values are not full Gate A power. At a true effect exactly equal to 5%, the observed estimate
clears the 5% materiality threshold only about half the time. The joint probability of clearing the
materiality threshold and obtaining a positive 95% interval is at most 48.5%, 49.7%, and 49.9% at
18, 21, and 24 donors with the observed SD, and 36.3%, 40.9%, and 44.6% with the conservative SD.
The 70% favorable-donor, exact sign-flip, and severe-stratum criteria can only reduce full-gate
power further.

Therefore, the frozen 18--24 range has a nominal directional rationale but is **not conservatively
80%-powered for the complete Gate A contract**. It stays unchanged because the review explicitly
forbids a post-development amendment; 24 analyzable donors are preferred within that range. The
8--12 pilot has directional external-replication value only, and 12 donors are preferred when
available.

The calculation is limited by its outcome-exposed 13-donor, pooled-indication source; L1 influence
and skew; uncertain transport of the baseline loss scale and variance to DLPFC; normal-theory
assumptions; and the fact that a mean and SD do not identify exact sign-flip or full joint-gate
power. It is a transparent planning sensitivity, not fresh biological evidence.

## Downloaded-cohort audit

| Cohort | Local state and matched structure | Eligibility and permitted role |
|---|---|---|
| NatCommun MOSAIC | `/mnt/seagate/HnE/NatCommun_2025_s41467_025_59005_9`; its consecutive sections and FLEX nuclei support an exact-block design, but the registered source has only 13 primary donors across three indications and ST was repeatedly opened in development. | T0 design evidence, endpoint definition, calibration, power and failure-mode analysis only; never independent confirmation or pilot replication. |
| snPATHO | `/mnt/seagate/HnE/snPATHO_seq`; all six expected Visium/snRNA processed objects and all 123 download destinations are present. Same-block/close matching is not row-level verified, there are only three heterogeneous specimens, and all ST truths are opened. | Runtime and matched/wrong-reference development only; underpowered and confounded for independent validation. |
| HESCAPE lung | `/mnt/seagate/HnE/HESCAPE/hescape-pyarrow/human-lung-healthy-panel`; 5 of 195 shards are present, one is truncated, leaving about 95.1 GB missing. Full study is 20 sections from 15 true donors, H&E plus Xenium-derived pseudospots, no independently matched sc/snRNA or valid floor, and only 54/256 corrected-panel genes overlap. | Incomplete retrospective regional exploration only; cannot test matched-reference value or the complete inequality. |
| HEST GSE250346 lung | `/mnt/seagate/HnE/HEST/hest-lung-xenium`; 20 WSIs, transcript parquet, native and CellViT segmentations, and metadata are local for 20 sections/15 donors. It is the same source study as HESCAPE; no independent matched dissociated reference or valid replicate structure, only 54/256 corrected-panel genes, and outcomes were materialized. | Retrospective measurement/morphology development only; neither external replication nor personalized-reference validation. |
| spatialDLPFC | Support checkout plus metadata are local. Processed Visium/snRNA objects are checksum-bound but not ACL-separated. Nineteen metadata candidates span 10 donors; zero row-level T0 assignments are admitted. | Priority-1 conditional 8--10-donor regional pilot candidate. Definitive confirmation remains impossible because there are only 10 donors and no validated ST floor. |
| [HTAN metastatic breast cancer](https://www.nature.com/articles/s41591-024-03215-z) | No local payload or sample-level manifest. The study has spatial data for 15 biopsies from serial sections of a second core from the same lesion/procedure as the sc/snRNA core. | T1 secondary close-match transport candidate only; never primary exact-specimen M3. The unique evaluable donor count among the 15 biopsies remains unverified. |
| [GSE243280 breast cancer](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE243280) | GEO family metadata are local and checksum-bound; expression/spatial payloads are not. GSM7782698 scFFPE uses two 25-um curls and GSM7782699 Visium uses a Sample 1 serial section; Xenium replicates are GSM7780153/154. HEST's `NCBI776` preservation label conflicts with the FFPE Visium description. | Claimed T0 adjacent-curl regional and later cell-level mechanism candidate; payload identity and adjacency are unverified, and approximately one deeply profiled block precludes population inference. Do not route HEST assets until its preservation/identity annotation is resolved. |

HEST GSE315411 (`NCBI885`--`NCBI888`) is metadata-only locally and has no identified matched
independent sc/snRNA bank. The local HEST prostate H5AD is spot expression without its paired local
H&E/reference system. `/mnt/seagate/TRACERx421_WES` is a WES resource, not an H&E--ST--sc/snRNA
cohort. None changes the eligibility decision.

## Query-specific routing and acquisition receipts

The original metadata-frozen candidate inventory is
`manifests/spatialdlpfc_pilot_routing.tsv`, SHA-256
`47e7830e1aa2263ddfecf0dff8bb10a5bd399638ae0ed3753be72fa87e1ca2a8`. It contains 19 candidate
same-block rows across donors `Br2720`, `Br2743`, `Br3942`, `Br6423`, `Br6432`, `Br6471`,
`Br6522`, `Br8325`, `Br8492`, and `Br8667`. Anterior, middle, and posterior blocks remain distinct;
they are neither interchangeable references nor replicate measurements.

The strict query-specific assignment manifest is
`manifests/spatialdlpfc_specimen_reference_assignments.tsv`, SHA-256
`724759379a9442cc16bb1e5e4c10b447fa3ff85e5cb4585774cbfc9eef36fe1f`. Every row has an explicit
query donor/block/capture, proposed reference donor/block, claimed and verified tier, payload and
registration status, eligibility flag, target state and frozen tie-break. All 19 verified tiers are
`unresolved`; all eligibility and selection flags are false. The `DLPFC_<donor>_<region>` specimen
and `snRNA_<donor>_<region>` sample strings are provisional routing keys constructed from metadata,
not payload-verified canonical specimen or aliquot identifiers; their equality cannot admit a row.

The Br2720 metadata identity receipt is
`manifests/spatialdlpfc_metadata_identity_receipt.json`, SHA-256
`22301d7a201a661ef0e7c4ba1b19961325e3ac7dec112b78e8346239fc0ded89`. Both alignment JSONs
embed serial `V10U24-091` and areas B1/C1, while imaging/RNA/subject metadata consistently map those
captures to `Br2720`/`NDAR_INVGP756DP1`. This resolves the `Br2770` token as a filename typo only;
payload-level T0 identity and independent snRNA aliquot provenance remain unverified.

The post-registration acquisition receipt is
`manifests/spatialdlpfc_pilot_acquisition_receipt.json`, SHA-256
`5988dff71dab6bcaf8dd0297ad8352e10ed1108ed60d05c6a753254008f0e698`.

| Payload | State on 2026-07-15 | Integrity/boundary |
|---|---|---|
| Processed snRNA archive | 4,035,795,545 bytes; SHA-256 `15176538edd4d632fb19376229fd83b9446b90cfbc7cf0de7fbc599443a49c75` | ZIP test passed for `se.rds` and `assays.h5`; not extracted or loaded |
| Processed Visium SpatialExperiment | 1,458,409,716 bytes; SHA-256 `24c98ddcbc47083591a68957f8b8dd3934160e0dde9649d2349c5908e42fe77e` | Opaque and not loaded; processed object does not satisfy the raw/minimally processed requirement |
| High-resolution H&E and raw Space Ranger outputs | Not acquired | Controlled Globus/NDA transfer requires authentication and a configured destination |

The official spatialDLPFC project is documented at
[spatialDLPFC](https://research.libd.org/spatialDLPFC/). The official spatialLIBD downloader and
[source implementation](https://github.com/LieberInstitute/spatialLIBD/blob/devel/R/fetch_data.R)
identify the processed payloads. Raw spatial material is associated with Globus collection UUID
`6cd81564-ed47-11ec-8358-cd84b862b754`; the snRNA collection UUID is
`6f9322c4-5eaf-11ed-b0b5-bfe7e7197080`. This host has no authenticated Globus destination, so the
controlled raw transfer cannot proceed autonomously.

The current historical reference matcher selects “matched” banks at donor level and lacks a
query-specific canonical parent-block/independent-aliquot identity. It is explicitly prohibited for
primary M3 assignment. A separate fail-closed query-specific validator and synthetic positive and
negative tests now exist, but execution still requires a versioned payload-qualification receipt,
selected-assignment receipt, validator wiring for that typed receipt, and frozen query-specific T2,
T3/M6 and T4/M7 assignments. The new T2 comparison is separately non-gating; it is not M6 and cannot
rescue C1 or C2. Because spatialDLPFC alternates change anterior/middle/posterior anatomy together
with block identity, T2 can estimate a same-donor different-region-or-block penalty, not pure
physical-block specificity.

GSE243280 metadata were acquired without expression targets at
`/mnt/seagate/HnE/GSE243280/metadata`. The receipt
`manifests/gse243280_metadata_receipt.json` (SHA-256
`81ca9c108ac495f373f7117e52b823f061710204d3352f6dfab08239b8b1020f`) freezes the GSM/BioSample/SRA
mapping and keeps all regional/cell-level execution unauthorized.

## Target-safe qualification and execution order

The protocol resolves the review's target-access tension with two stages:

1. Before any ST count access: metadata-only query/reference routing, high-resolution H&E
   registration, reference aliquot and support qualification, and freezing exactly one T0 block per
   donor in a versioned assignment receipt.
2. After the full protocol and subset are sealed: QC-only ST access for depth, detection,
   split-half/program reliability and a restricted receipt. These quantities may not choose donors,
   blocks, genes, states, thresholds, or model identity. Held-out outcomes are then score-only.

If fewer than eight unique donors survive exact-specimen payload qualification, the pilot stops without
opening the target. If 8--12 survive, execution is serial and fail-closed:

1. run frozen M0 and M3 for all prespecified LODO folds;
2. issue the registered phase-1 decision;
3. run M4 and physical offsets 0/55/110/220/440 micrometres;
4. report the ST floor only if a valid technical replicate, target-blind registered serial-section
   mapping, or validated count-split oracle exists;
5. report M1, M2, M6, M7 and matched-ST diagnostics in their separate evidence tiers; and
6. after the phase-1 decision, report T2 same-donor/different-region-or-block controls separately where at
   least one alternate qualified block exists, averaging multiple controls within donor.

A serial section does not contain identical target observations. It may support a floor only through
a target-blind, prespecified common-region mapping frozen before expression access; expression
similarity may never define that mapping.

Execution is limited to four CPU threads, one visible GPU, 60% GPU memory, serial outer folds and no
swap. An out-of-memory event stops the run without changing model identity or thresholds.

## Current status

| Action | Status | Next admissible step |
|---|---|---|
| Preserve definitive protocol | **Complete** | Keep its exact SHA regression test passing |
| Register separate pilot | **Complete** | Keep its exact SHA regression test passing |
| Register strict specimen addendum | **Complete** | Keep parent/addendum/assignment hashes passing; matched means T0 only |
| Donor-level power audit | **Complete** | Prefer 24 definitive and 12 pilot analyzable donors; do not overstate full-gate power |
| Query-specific specimen routing | **Candidate inventory frozen; zero admitted T0 rows** | Cross-check raw payloads, independent snRNA aliquots and registration; freeze one selected block per donor |
| Processed snRNA acquisition | **Opaque integrity verified** | Extract only for target-free reference qualification after raw routing inputs are available |
| Processed Visium acquisition | **Checksum-bound, not access-separated** | Keep unopened; create a role-separated ACL-capable or encrypted target store before scoring |
| Raw H&E/Space Ranger acquisition | **Blocked externally** | Authenticate a Globus endpoint/destination and transfer spatial collection `6cd81564-ed47-11ec-8358-cd84b862b754` |
| Pilot eligibility | **Blocked** | Demonstrate at least eight unique donors with one payload-qualified T0 block, frozen-panel/reference support and a true target seal |
| HTAN secondary route | **Blocked at metadata** | Obtain its 15-biopsy core-A/core-B sample/file manifest; never pool T1 with T0 |
| GSE243280 mechanism route | **Metadata-only claimed-T0 receipt complete** | Verify file-level parent-block chain of custody and curl adjacency before promotion; resolve HEST preservation identity; population claims remain prohibited |
| Biological hypothesis validation | **Not run** | Authorized only after pilot eligibility and target seal |
| Definitive confirmation | **Blocked** | Acquire a separate one-tissue 18--24-donor tri-modal cohort with a valid floor |
| Cell-level validation | **Prohibited** | Register separately only after regional success |

The correct scientific conclusion remains: the NatCommun findings are development evidence; the
external regional hypothesis is **untested**, not supported or refuted. The next in-scope work is
controlled raw spatialDLPFC acquisition and target-blind payload qualification, not model tuning.
