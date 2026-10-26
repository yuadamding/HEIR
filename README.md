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
