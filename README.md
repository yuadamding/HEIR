# HEIR

HEIR predicts relative spatial expression from H&E features, specimen-matched sc/snRNA,
measured proteins, and tissue geometry. Its shared pathway exports probabilistic RNA-state
assignments, reference transfer, bounded expression correction, and uncertainty indicators.
Spatial RNA supplies training supervision and subsequent evaluation; it is excluded from
ordinary prediction inputs.

This is a development prototype. Real HTAPP qualification, calibrated uncertainty, and biological
benefits from personalization or communication remain unestablished. See the
[workflow and scientific limits](docs/personalized_development.md).

The 13-biopsy working set at `/mnt/seagate/HnE/HTAPP_MBC` now has a
[tissue-region workflow](docs/personalized_development.md#htapp-tissue-region-workflow). It uses
biopsy-level matching and the published registered pixel frame without inventing section IDs.
Nucleus-level inputs remain supported by the original canonical workflow.

Version 0.3 fixes the numerical issues identified in the assessment of `d955c01`: fixed
normalization panels, support-independent transfer, and separate protein observation geometry.
Version 0.4 adds cohort preparation, patient-fold fitting, prediction, and replicate-wise evaluation.
Version 0.5 adds explicit interaction-radius units, a visual regional inspection, and verified
frozen histology caches alongside the RGB control. Pilot comparisons report transfer and correction
separately, including direct protein-marker outcomes versus the other outcomes.
Version 0.6 adds fixed-model diagnostics, separately fitted transfer and constant-reference
controls, and a frozen internal patient-held-out study with validation-only rank selection.
Version 0.7 preserves that study and adds error decomposition, cohort measurement consistency,
and explicitly target-informed reference reconstruction to diagnose its limitations.
Version 0.8 fixes reconstruction region-ID alignment and adds a measurement-only screen of
100/200/400 registered-pixel aggregation, fixed normalization panels, exposure, and displacement.
Version 0.9 adds a separate joint model with learned protein features, hierarchical RNA-state
assignment, and a multimodal correction, plus source-coordinate replay. Legacy bundles retain
their original meaning. Joint experiments use the historical 100-pixel view as exploratory
development; coordinate replay does not qualify biological alignment.
Version 0.10 makes the joint hierarchy and protein correspondence explicit in exports, adds
zero-initialized correction, and compares attention with a parameter-comparable reference
MLP and an availability/coverage-only control. See the
[HEIR-Joint architecture and control protocol](docs/personalized_development.md#version-010-heir-joint-controls).
Prediction exports use **schema v3**, with explicit spatial units; v2 predictions remain readable
as nucleus/µm outputs. Canonical specimen manifests and model bundles retain schema v2.

## Install and run

```bash
uv sync --frozen --extra dev --extra htapp --extra encoders
uv run heir demo --output /tmp/heir-demo
```

The `encoders` extra is optional when using cached image features. Environments and caches are
rebuildable and excluded from the repository; `uv sync` recreates the local environment.

For the available cohort, start with a real-data development run:

```bash
uv run heir cohort-inspect --root /mnt/seagate/HnE/HTAPP_MBC --output /tmp/heir-cohort
uv run heir cohort-demo --root /mnt/seagate/HnE/HTAPP_MBC --biopsy 313-932 \
  --config configs/personalized/cohort.json --output /tmp/heir-313-regions
```

The default uses nine shared outcome genes and handcrafted RGB control features. The demo fits
and scores the same biopsy; it tests the software, not held-out biological performance. The cohort
workflow also supplies `cohort-train`, `cohort-predict`, and `cohort-evaluate` commands.

Inspect assay coverage and cache the retained H-optimus-1 encoder on the same H&E regions:

```bash
uv run heir cohort-inspect-pilot --root /mnt/seagate/HnE/HTAPP_MBC --biopsy 313-932 \
  --output /tmp/heir-313-inspection
uv run heir cohort-features --root /mnt/seagate/HnE/HTAPP_MBC --biopsies 313-932 \
  --encoder-manifest manifests/encoders/hoptimus1.json \
  --model-dir /mnt/seagate/HnE/pretrained/H-optimus-1 \
  --device cuda --batch-size 1 --output /tmp/heir-313-hoptimus1
uv run heir cohort-demo --root /mnt/seagate/HnE/HTAPP_MBC --biopsy 313-932 \
  --config /tmp/heir-313-hoptimus1/config.json --epochs 100 \
  --output /tmp/heir-313-histology-pilot
```

The inspection writes a self-contained `index.html` and exact coverage tables. Feature extraction
reads H&E only; its generated configuration binds the image, grid, checkpoint, and feature cache.
Coordinates remain in registered pixels; physical calibration is unresolved. See the
[recorded pilot and its limits](docs/personalized_development.md#version-05-regional-pilot).

Diagnose an existing pilot or fit the three predefined controls with its original settings:

```bash
uv run heir cohort-diagnose --root /mnt/seagate/HnE/HTAPP_MBC \
  --run /tmp/heir-313-histology-pilot --output /tmp/heir-313-diagnostics
uv run heir cohort-controls --root /mnt/seagate/HnE/HTAPP_MBC \
  --config /tmp/heir-313-hoptimus1/config.json \
  --baseline-run /tmp/heir-313-histology-pilot --output /tmp/heir-313-controls
```

The [internal study protocol](configs/personalized/study.json) binds exact configuration and
patient folds. Its [workflow and results](docs/personalized_development.md#version-06-contribution-tests)
retain prior cohort exposure, target consistency, correction effects, and the observed null protein
increment. Neighborhood comparisons remain subsequent work.

Analyze the completed study under its separately bound capability protocol:

```bash
uv run heir cohort-capability --root /mnt/seagate/HnE/HTAPP_MBC \
  --study /storage/HE_GPT/HEIR-cohort-runs/v060-controls/internal-study \
  --protocol configs/personalized/capability.json --output /tmp/heir-capability
```

This command fits diagnostic RNA mixtures directly to spatial targets and evaluates shared-region
replicate transfers. Those reconstructions are target-informed analyses, not new prediction models
or certified performance bounds. See the [capability report](docs/personalized_development.md#version-07-capability-analysis).

Qualify the spatial measurements without fitting another model or loading image embeddings:

```bash
uv run heir cohort-measurement --root /mnt/seagate/HnE/HTAPP_MBC \
  --study /storage/HE_GPT/HEIR-cohort-runs/v060-controls/internal-study \
  --capability /storage/HE_GPT/HEIR-cohort-runs/v070-capability \
  --config configs/personalized/measurement.json --output /tmp/heir-measurement
```

The [measurement configuration](configs/personalized/measurement.json) fixes all six conditions,
support and eligibility rules, and the pixel displacement stress test. Raw counts are aggregated
before normalization. See the [measurement report](docs/personalized_development.md#version-08-measurement-qualification).

The current joint comparison uses frozen H-optimus-1 and UNI2-h features on identical crops.
Use `cohort-features` with each retained manifest and local checkpoint to create its cache.
Both pretrained weights remain frozen; their 1,536-dimensional outputs feed a trainable
32-dimensional projection. Model bundles reject features from a different encoder.

With the encoder caches available, prepare inputs separately from target tables and run:

```bash
DATA=/mnt/seagate/HnE/HTAPP_MBC
OUT=/storage/HE_GPT/HEIR-cohort-runs/v090-joint
uv run heir cohort-joint prepare --root "$DATA" --config configs/personalized/joint_v1.json \
  --folds configs/personalized/study_folds.csv --output "$OUT/prepared"
uv run heir cohort-joint prepare --root "$DATA" --config configs/personalized/joint_uni2h.json \
  --folds configs/personalized/study_folds.csv --output "$OUT/uni2h-prepared"
uv run heir cohort-joint pilot --prepared "$OUT/uni2h-prepared" --biopsies 313-932 514-6760 \
  --config configs/personalized/joint_uni2h.json --output "$OUT/uni2h-pilots"
uv run heir cohort-joint study --prepared "$OUT/prepared" \
  --config configs/personalized/joint_hoptimus1_study.json \
  --folds configs/personalized/study_folds.csv --seeds 0 --output "$OUT/hoptimus1-study"
uv run heir cohort-joint study --prepared "$OUT/uni2h-prepared" \
  --config configs/personalized/joint_uni2h_study.json \
  --folds configs/personalized/study_folds.csv --seeds 0 --output "$OUT/uni2h-study"
uv run heir cohort-joint compare-backbones --hoptimus-study "$OUT/hoptimus1-study" \
  --uni2-study "$OUT/uni2h-study" --config configs/personalized/joint_backbones_study.json \
  --output "$OUT/backbone-comparison"
```

The [backbone protocol](configs/personalized/joint_backbones_study.json) specifies **55 principal
fits plus 20 mechanistic fits**, all seed 0. RP is fitted once per fold and shared. Predictions
and fixed-model interventions precede each fold's test-target loading. Existing H-optimus-1
pilots are retained; same-biopsy pilots do not measure generalization. The earlier 100-fit
proposal and panel extensions remain separate, unexecuted protocols. New runs require new
output paths and corresponding protocol bindings; existing results are never overwritten.

The [completed seed-0 comparison](docs/personalized_development.md#version-09-completed-backbone-comparison)
retains all 75 cohort fits. Protein inputs affect predictions, but their average correlation
increment is small under both backbones; matched RNA does not improve mean correlation over
the generic-reference control. These results do not establish three-modality personalization.

Adapt the [specimen manifest](configs/personalized/htapp.example.json) and
[model configuration](configs/personalized/model.example.json), then run:

```bash
uv run heir prepare --manifest specimen.json --output prepared/
uv run heir train --manifests train-specimen.json --folds folds.csv \
  --outer-fold 0 --config model.json --output bundle/
uv run heir predict --manifest held-out-specimen.json --bundle bundle/ --output prediction.npz
uv run heir evaluate --prediction prediction.npz --manifest held-out-specimen.json \
  --genes evaluation-genes.json --output scores.json
```

## Development

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

Regression tests cover the supplied normalization, support, protein, and patient-scoring
counterexamples, cohort target isolation, spatial units, and replicate handling. Unit tests use
CPU and synthetic data; real cohort acceptance runs are documented separately. The active source
contains the personalized pipeline, optional frozen encoder adapters, and their configuration
files. The [comparison index](configs/personalized/comparisons.json)
describes the seven model variants; context variants require a fitted protein baseline.

## Historical work

Older experiments, scripts, protocols, reports, and pre-cleanup uncommitted work are preserved in
one [verified source archive](archive/README.md). Failed findings and scientific restrictions are
unchanged. Extract the archive into a separate directory to inspect or recover historical work.
