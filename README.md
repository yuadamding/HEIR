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
