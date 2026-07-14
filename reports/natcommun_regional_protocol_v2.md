# NatCommun regional-fusion validation v2

## Current status

**IMPLEMENTED, NOT YET EXECUTED.** The scientific protocol and analysis path are ready, but the
registered source cannot be built yet. Space Ranger has completed 12 of 16 sections; `B3_2`,
`L4_2`, `D1`, and `D4` remain incomplete. No regional molecular result or biological conclusion has
been produced.

This document is a protocol/readiness report, not an experiment report.

## Why this is a new protocol

The completed HEST H-optimus-1 result remains unchanged and failed its frozen gate. Its precise
conclusion is:

> The natural unmasked 112-µm H-optimus-1 embedding did not linearly resolve the geometry of the
> single centered nucleus beyond a donor-balanced fine-type reference.

That is useful evidence about target-cell localization. It is not evidence that H-optimus is
misloaded, generally uninformative, or unable to query a matched molecular reference for a regional
Visium spot. HEST also lacks the independent matched sc/snRNA aliquot required for the central
tri-modal comparison.

The frozen HEST report and v1 NatCommun runner were therefore not edited. Protocol v2 creates a new,
hash-bound regional path in which the HEST result is recorded as a non-gating architecture
diagnostic.

## Registered scientific test

The observation is a 55-µm Visium spot. H&E is the base predictor, matched Chromium FLEX snRNA is
the reference, and held-out spatial expression is hidden truth. All fit and model selection is
leave-one-donor-out, with grouped training-donor inner selection.
Hyperparameter objectives first average donors within indication and then weight breast, lung, and
DLBCL equally, so the six DLBCL donors cannot dominate selection.

The primary reference representation is deterministic molecular k-means within each donor/type,
with eight prototypes. One donor/type centroid is a secondary diagnostic. Identity-hash averages
are prohibited. Cross-assay calibration is diagonal, identity-regularized, indication-aware, and
fit using training donors only; a global diagonal transform is used when an inner fold has fewer
than two training donors for an indication. Its ridge penalty is selected by leave-one-training-
donor-out evaluation of that same indication/fallback mapping, rather than a different global map.
The global fallback itself weights indications equally and donors equally within indication.

The one-step fusion grid is `0, 0.1, 0.25, 0.5, 0.75, 1.0`. H&E centrality is tested empirically; it
is not forced by capping the reference contribution at 0.5. Iterative refinement remains
prohibited.

The decisive family is:

```text
M3 < M0  H&E plus matched reference beats H&E alone
M3 < M1  H&E adds information beyond reference-only imputation
M3 < M2  continuous H&E adds information beyond type routing
M3 < M4  correct H&E pairing matters
M3 < M6  the matched donor reference matters
M3 < M7  donor matching adds value beyond a same-indication generic bank
```

Blank-image and coordinate M5 comparisons are registered image-content controls. Holm correction
is applied once across both crop arms, both endpoints, both bank conditions, and all eight M3
control comparisons: exactly 64 tests at familywise alpha 0.05. A registered test that is absent
because its endpoint is blocked enters this family conservatively as `p=1`; the family may not
shrink.

Initial support additionally requires at least 5% donor-equal relative MSE reduction, positive
direction in at least 70% of donors, positive direction in at least two of three indications, and no
indication with a relative MSE reversal of 5% or worse. Reports contain donor-equal,
indication-equal, breast, lung, DLBCL, and descriptive fixed-effect meta summaries. Every primary
experiment must also preserve at least 50% of the observed within-section molecular variance, have
at least 50% median reference type coverage, abstain to H&E on no more than 50% of spots, and avoid
the registered median and single-program rare-state recall collapses. For the program endpoint,
variance and rare-state metrics use only the programs qualified in that held-out donor's
outer-training reliability gate; all-candidate metrics cannot pass or fail this decision. These are
decision gates, not post-hoc descriptive metrics.

M8 remains reported as a split-half, factor-of-four-corrected approximation to the full-depth ST
floor. It is useful for `GapClosed`, but it is explicitly secondary and cannot block an otherwise
passing M3 result.

## Frozen crop sensitivity

Two H-optimus arms are evaluated:

| Arm | Construction | Interpretation |
|---|---|---|
| `target_55um` | Preserve the registered native 112-µm/0.5-mpp model canvas but whiten pixels outside the centered 55-µm spot | Target-matched regional signal |
| `context_112um` | Natural unmasked 112-µm registered field | Spot plus immediate architecture |

Both are generated with the exact frozen H-optimus checkpoint, official/local parity receipt,
qualified preprocessing, and CUDA inference. The crop supplement is keyed to the exact source spot
identities and source SHA-256. The UNI2-h secondary supplement independently generates the same two
registered physical fields plus its own blank-image vector. UNI2-h is bound to its exact manifest,
revision, checkpoint, config, adapter, base, factory, and builder hashes; no official/local UNI2-h
parity claim is made.

## NatCommun-specific preflight

The benchmark cannot start until a separately frozen preflight proves all of the following:

1. Exact source, model, protocol, and implementation hashes.
2. Passed official-versus-local H-optimus parity for the exact manifest; the UNI2-h secondary is
   instead explicitly hash-qualified without a parity claim.
3. Finite, nondegenerate features globally and in every primary section, independently for the
   55-µm and 112-µm arms of each encoder.
4. Current Space Ranger invocation, H&E, alignment JSON, and alignment-QC image hashes match the
   source receipt.
5. An independent visual review, blinded to ST/reference outcomes, passes the alignment for all 16
   sections and binds those same artifact hashes. The reviewer identity must transparently state
   whether the review was human or AI-assisted.
6. Donor-held-out H&E indication prediction exceeds its outer-training-majority baseline
   independently for every encoder/crop arm.
7. Blank and within-section deranged-image controls are constructible.
8. Matched, hard-wrong, and same-indication generic banks are supported in both bank conditions.
9. At least three fixed programs pass the split-half reliability gate in every outer-training fold.

The preflight gate is recomputed from its components, including refitting the deterministic visible
control during benchmark verification. A stored top-level `passed: true` or stored prediction is
never trusted. No single-nucleus geometry endpoint appears in this regional preflight.

The external registration review uses schema `heir.natcommun_registration_review.v1`, binds the
exact source SHA, names the reviewer, declares `review_blinded_to_ST_and_reference_outcomes: true`,
and contains one entry for every frozen section. Each section entry has `status: passed` plus the
exact `h_and_e_sha256`, `final_alignment_sha256`, and `alignment_qc_image_sha256` copied from—and
independently checked against—the source receipt. Placeholder or incomplete reviews fail closed.

## Execution sequence

After all 16 Space Ranger sections complete, build the frozen v1 molecular/source artifact, produce
both encoder supplements, complete the blinded registration review, and commit the frozen protocol
and implementation. Scientific execution deliberately refuses a dirty worktree. Then run:

```bash
export CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

PROTOCOL_SHA=$(sha256sum configs/natcommun_matched_regional_protocol_v2.json | cut -d' ' -f1)
RUNNER_SHA=$(sha256sum scripts/benchmark_natcommun_reference_fusion_v2.py | cut -d' ' -f1)
REF_V2_SHA=$(sha256sum src/heir/evaluation/reference_fusion_v2.py | cut -d' ' -f1)
SOURCE_SHA=$(sha256sum /mnt/seagate/HEIR_runs/natcommun_regional_source/source.npz | cut -d' ' -f1)
REVIEW_SHA=$(sha256sum /external/natcommun_registration_review.json | cut -d' ' -f1)
CROP_BUILDER_SHA=$(sha256sum scripts/build_natcommun_crop_sensitivity.py | cut -d' ' -f1)
UNI2_BUILDER_SHA=$(sha256sum scripts/build_natcommun_uni2_sensitivity.py | cut -d' ' -f1)
UNI2_ADAPTER_SHA=$(sha256sum src/heir/features/uni2h.py | cut -d' ' -f1)
ENCODER_BASE_SHA=$(sha256sum src/heir/features/base.py | cut -d' ' -f1)
ENCODER_FACTORY_SHA=$(sha256sum src/heir/features/__init__.py | cut -d' ' -f1)

.venv/bin/python scripts/build_natcommun_crop_sensitivity.py \
  --source /mnt/seagate/HEIR_runs/natcommun_regional_source/source.npz \
  --source-sha256 "$SOURCE_SHA" \
  --output /mnt/seagate/HEIR_runs/natcommun_regional_source/crop_sensitivity_55um.npz \
  --device cuda --batch-size 4

.venv/bin/python scripts/build_natcommun_uni2_sensitivity.py \
  --source /mnt/seagate/HEIR_runs/natcommun_regional_source/source.npz \
  --source-sha256 "$SOURCE_SHA" \
  --output /mnt/seagate/HEIR_runs/natcommun_regional_source/uni2h_sensitivity.npz \
  --device cuda --batch-size 4

CROP_SHA=$(sha256sum /mnt/seagate/HEIR_runs/natcommun_regional_source/crop_sensitivity_55um.npz | cut -d' ' -f1)
UNI2_SHA=$(sha256sum /mnt/seagate/HEIR_runs/natcommun_regional_source/uni2h_sensitivity.npz | cut -d' ' -f1)

.venv/bin/python scripts/benchmark_natcommun_reference_fusion_v2.py preflight-hoptimus \
  --source /mnt/seagate/HEIR_runs/natcommun_regional_source/source.npz \
  --expected-source-sha256 "$SOURCE_SHA" \
  --registration-review /external/natcommun_registration_review.json \
  --expected-registration-review-sha256 "$REVIEW_SHA" \
  --expected-protocol-sha256 "$PROTOCOL_SHA" \
  --expected-runner-sha256 "$RUNNER_SHA" \
  --expected-reference-v2-sha256 "$REF_V2_SHA" \
  --crop-55-supplement /mnt/seagate/HEIR_runs/natcommun_regional_source/crop_sensitivity_55um.npz \
  --expected-crop-55-supplement-sha256 "$CROP_SHA" \
  --expected-crop-builder-sha256 "$CROP_BUILDER_SHA" \
  --output /mnt/seagate/HEIR_runs/natcommun_regional_v2/hoptimus_preflight.json \
  --device cuda --cpu-threads 4

.venv/bin/python scripts/benchmark_natcommun_reference_fusion_v2.py preflight-uni2 \
  --source /mnt/seagate/HEIR_runs/natcommun_regional_source/source.npz \
  --expected-source-sha256 "$SOURCE_SHA" \
  --registration-review /external/natcommun_registration_review.json \
  --expected-registration-review-sha256 "$REVIEW_SHA" \
  --expected-protocol-sha256 "$PROTOCOL_SHA" \
  --expected-runner-sha256 "$RUNNER_SHA" \
  --expected-reference-v2-sha256 "$REF_V2_SHA" \
  --uni2-supplement /mnt/seagate/HEIR_runs/natcommun_regional_source/uni2h_sensitivity.npz \
  --expected-uni2-supplement-sha256 "$UNI2_SHA" \
  --expected-uni2-builder-sha256 "$UNI2_BUILDER_SHA" \
  --expected-uni2-adapter-sha256 "$UNI2_ADAPTER_SHA" \
  --expected-encoder-base-sha256 "$ENCODER_BASE_SHA" \
  --expected-encoder-factory-sha256 "$ENCODER_FACTORY_SHA" \
  --output /mnt/seagate/HEIR_runs/natcommun_regional_v2/uni2_preflight.json \
  --device cuda --cpu-threads 4
```

For each encoder whose own report passes, freeze that report hash and run the corresponding
benchmark in a separate output tree. A UNI2-h preflight failure cannot block or alter the
H-optimus primary run:

```bash
HOPT_PREFLIGHT_SHA=$(sha256sum /mnt/seagate/HEIR_runs/natcommun_regional_v2/hoptimus_preflight.json | cut -d' ' -f1)
UNI2_PREFLIGHT_SHA=$(sha256sum /mnt/seagate/HEIR_runs/natcommun_regional_v2/uni2_preflight.json | cut -d' ' -f1)

.venv/bin/python scripts/benchmark_natcommun_reference_fusion_v2.py benchmark-hoptimus \
  --source /mnt/seagate/HEIR_runs/natcommun_regional_source/source.npz \
  --expected-source-sha256 "$SOURCE_SHA" \
  --registration-review /external/natcommun_registration_review.json \
  --expected-registration-review-sha256 "$REVIEW_SHA" \
  --preflight-report /mnt/seagate/HEIR_runs/natcommun_regional_v2/hoptimus_preflight.json \
  --expected-preflight-report-sha256 "$HOPT_PREFLIGHT_SHA" \
  --crop-55-supplement /mnt/seagate/HEIR_runs/natcommun_regional_source/crop_sensitivity_55um.npz \
  --expected-crop-55-supplement-sha256 "$CROP_SHA" \
  --expected-crop-builder-sha256 "$CROP_BUILDER_SHA" \
  --expected-protocol-sha256 "$PROTOCOL_SHA" \
  --expected-runner-sha256 "$RUNNER_SHA" \
  --expected-reference-v2-sha256 "$REF_V2_SHA" \
  --output-dir /mnt/seagate/HEIR_runs/natcommun_regional_v2/hoptimus_primary \
  --device cuda --cpu-threads 4

.venv/bin/python scripts/benchmark_natcommun_reference_fusion_v2.py benchmark-uni2 \
  --source /mnt/seagate/HEIR_runs/natcommun_regional_source/source.npz \
  --expected-source-sha256 "$SOURCE_SHA" \
  --registration-review /external/natcommun_registration_review.json \
  --expected-registration-review-sha256 "$REVIEW_SHA" \
  --preflight-report /mnt/seagate/HEIR_runs/natcommun_regional_v2/uni2_preflight.json \
  --expected-preflight-report-sha256 "$UNI2_PREFLIGHT_SHA" \
  --uni2-supplement /mnt/seagate/HEIR_runs/natcommun_regional_source/uni2h_sensitivity.npz \
  --expected-uni2-supplement-sha256 "$UNI2_SHA" \
  --expected-uni2-builder-sha256 "$UNI2_BUILDER_SHA" \
  --expected-uni2-adapter-sha256 "$UNI2_ADAPTER_SHA" \
  --expected-encoder-base-sha256 "$ENCODER_BASE_SHA" \
  --expected-encoder-factory-sha256 "$ENCODER_FACTORY_SHA" \
  --expected-protocol-sha256 "$PROTOCOL_SHA" \
  --expected-runner-sha256 "$RUNNER_SHA" \
  --expected-reference-v2-sha256 "$REF_V2_SHA" \
  --output-dir /mnt/seagate/HEIR_runs/natcommun_regional_v2/uni2_secondary \
  --device cuda --cpu-threads 4
```

The runner executes one experiment at a time to bound CPU and memory use.

## Claim boundary

A robust pass can justify development of a scalable **regional research implementation**. A result
present only at 112 µm supports contextual/regional software, not a cell-state system. A pass does
not authorize production or clinical use, and independent replication remains required.

The cell-level molecular-state and spatial-cell-annotation hypotheses remain blocked until a future
cohort provides registered cell-resolved spatial truth and an independent matched sc/snRNA
reference.
