# HEIR

HEIR predicts relative spatial expression from H&E features, specimen-matched sc/snRNA,
measured proteins, and tissue geometry. Its shared pathway exports probabilistic RNA-state
assignments, reference transfer, bounded expression correction, and uncertainty indicators.
Spatial RNA supplies training supervision and subsequent evaluation; it is excluded from
ordinary prediction inputs.

This is a development prototype. Real HTAPP qualification, calibrated uncertainty, and biological
benefits from personalization or communication remain unestablished. See the
[workflow and scientific limits](docs/personalized_development.md).

Version 0.3 fixes the numerical issues identified in the assessment of `d955c01`: fixed
normalization panels, support-independent transfer, and separate protein observation geometry.
Specimen manifests, model bundles, and prediction exports now use **schema v2**; older artifacts
must be regenerated after explicit configuration and metadata migration.

## Install and run

```bash
uv sync --frozen --extra dev --extra htapp --extra encoders
uv run heir demo --output /tmp/heir-demo
```

The `encoders` extra is optional when using cached image features. Environments and caches are
rebuildable and excluded from the repository; `uv sync` recreates the local environment.

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
counterexamples. Checks use CPU and synthetic data. The active source contains the personalized
pipeline, optional frozen encoder
adapters, and their configuration files. The [comparison index](configs/personalized/comparisons.json)
describes the seven model variants; context variants require a fitted protein baseline.

## Historical work

Older experiments, scripts, protocols, reports, and pre-cleanup uncommitted work are preserved in
one [verified source archive](archive/README.md). Failed findings and scientific restrictions are
unchanged. Extract the archive into a separate directory to inspect or recover historical work.
