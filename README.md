# HEIR

HEIR tests whether independently initialized H&E representations can map nuclei to
sample-matched single-nucleus RNA states without using target spatial expression for fitting.
The repository is intentionally limited to the core model, auditable artifact contracts, and
scientific benchmark code.

## Scientific status

The existing snPATHO results are negative development evidence, not a successful biological
claim:

- locked v0.2 macro median-gene Spearman: **-0.0054**;
- refinement matrix: **93/93 artifacts scored**, with **49/108 ordering checks passing**;
- refined minus round zero: **-0.00037** mean paired median-gene Spearman;
- refined minus image shuffle: **-0.00656**;
- refined minus wrong-prototype bank: **-0.00404**;
- unknown-mass study: **75/75 CUDA stages complete**, but no stable mass selected.

The revised frozen-target path has synthetic regression coverage but no new end-to-end biological
result. A primary run remains blocked until `MorphologyStateGate` passes and the run has both an
independently validated morphology initializer and a frozen molecular target artifact.
Uninitialized or live-student target runs are explicit negative controls.

See [scientific validation](docs/scientific_validation.md) for the hypotheses and contracts and
[benchmark status](docs/benchmark.md) for cohort readiness and reported results.

## Repository scope

```text
src/heir/       core data contracts, model, training, refinement, inference, evaluation
scripts/        strict artifact producers and benchmark runners/scorers
configs/        frozen model, experiment, and validation plans
manifests/      cohort, ontology, and frozen gene-panel identities
reports/        compact hash-bound benchmark evidence
tests/          scientific, numerical, provenance, and benchmark regressions
```

Raw downloads, derived cohort artifacts, run outputs, and pretrained checkpoints are external.
Set benchmark paths explicitly; do not add them to Git. Pretrained components belong under
`../HEIR_assets/pretrained` or `$HEIR_PRETRAINED_DIR`.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Optional cohort preparation uses `.[science]`; pathology feature extraction uses `.[pathology]`.

Verify the environment and the two local cohort ledgers:

```bash
heir doctor --require-files
```

Space Ranger is the default nucleus segmentation contract. Alternative segmentation code is not
part of the compact core.

## Strict validation workflow

The primary sequence is deliberately fail-closed:

```text
independent initializer evidence
        -> validated initialization receipt
        -> frozen teacher target artifact
        -> one fixed-target M-step phase by default
        -> immutable round-0 safety comparison
        -> controls and coverage-aware benchmark
```

Refinement is not part of the default experiment path. Explicit `heir refine` invocation runs one
fine-head phase by default. More than one phase over the same frozen artifact must be reported as a
`fixed_target_curriculum`, not iterative EM; live-student target recomputation is an excluded
negative control.

Relevant producers:

- `scripts/validate_initialization_checkpoint.py`
- `scripts/create_initialization_receipt.py`
- `scripts/create_molecular_e_step.py`
- `scripts/train_snpatho_scanvi.py`
- `scripts/prepare_snpatho_refinement_inputs.py`

Generate the current snPATHO refinement matrix with
`scripts/run_snpatho_refinement_benchmark.py`. Use `--artifact-root` to place all run products
outside the repository and `--prohibit-adoption` for a provenance-clean execution.

Score the scientific hypotheses with:

- `scripts/benchmark_broad_types.py` — independently reviewed broad-type prerequisite;
- `scripts/benchmark_morphology_state_gate.py` — learned donor-held-out within-type state gate;
- `scripts/benchmark_oracle_ladder.py` — decoder/type/state ceiling decomposition;
- `scripts/benchmark_snpatho_refinement_matrix.py` — image, graph, prototype, residual, and
  refinement controls;
- `scripts/benchmark_snpatho_unknown_mass.py` — fixed unknown-mass sensitivity;
- `scripts/benchmark_snpatho_deepbench.py` and `scripts/benchmark_snpatho.py` — immutable
  retrospective endpoints.

All report writers reject output/input aliases, including symlinks and hard links. Long-running
benchmark manifests bind source, runtime, stage inputs, stage outputs, and control transforms.

## Tests

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
```

The performance fixture is deterministic and source-bound. Larger CUDA studies remain explicit
benchmark jobs rather than ordinary unit tests.

## Claim boundary

HEIR does not claim full-transcriptome recovery, morphology-resolved fine immune states, or a
validated refinement gain. Until the independent initialization and external-cohort gates pass,
the defensible endpoint is round zero or a simple type/prototype baseline.
