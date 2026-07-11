# Frozen snPATHO benchmark pipeline

`scripts/run_snpatho_pipeline.py` is the resumable entry point for the v0.2
snPATHO benchmark. Its default mode is a dry run: it prints every command and
writes a status ledger without starting Space Ranger, OmiCLIP, training, or
benchmark jobs.

```bash
conda run -n hne python scripts/run_snpatho_pipeline.py --sample all
```

After the preflight and command plan have been reviewed, execute the missing
stages with:

```bash
conda run -n hne python scripts/run_snpatho_pipeline.py --sample all --execute
```

Use `--sample 4066` (or 4399/4411) for a single-case operational run. These are
locked cases, not a development set: a partial benchmark is explicitly labelled,
is not the primary three-donor result, and must not be used for model tuning.
`--stop-after STAGE` creates a safe checkpoint, for example:

```bash
conda run -n hne python scripts/run_snpatho_pipeline.py \
  --sample all --execute --stop-after pathology_features
```

## Frozen inputs

The default experiment is
`configs/experiments/snpatho_v0_2.yaml`. It binds the final 500-gene panel
`manifests/gene_panel_snpatho_500.tsv` (SHA-256
`22ddb91188b3b124d5cf3ec0f7ae81017399d141e39647b0dce80675119fe927`).
All 500 genes are available in the QC Visium variables for all three samples.

Execution also requires the following development-only artifacts to match the
same ordered panel and their embedded provenance:

- `artifacts/snpatho/prior500/shared_svd.npz`, a 32-dimensional transform fit
  on B1 only;
- `artifacts/snpatho/prior500/rna_decoder.pt`, trained on that exact transform
  and panel;
- `artifacts/snpatho/prior/omiclip_ood.npz`, fit on B1 OmiCLIP features only;
- `/storage/HE_GPT/Loki/checkpoint.pt`, the hash-bound OmiCLIP/Loki visual
  checkpoint.

The preflight loads these contracts. A stale transform or decoder stops
`--execute`; it is not silently accepted.

## Stage order and target isolation

This is a transductive personalized pipeline. Target H&E and non-expression
Visium metadata (spot positions and scalefactors) are used before training, while
target Visium expression/counts and derived labels remain locked until every
selected prediction passes validation.

The script runs each stage across every selected sample before advancing:

1. Space Ranger 4.1 nucleus segmentation (the default segmentation method).
2. Geometry-only filtering to in-tissue Visium disks. This stage reads only
   nucleus centroids, spot positions, and scalefactors—not expression.
3. Multi-scale OmiCLIP feature extraction on CUDA with mixed precision.
4. Histology graph preparation from the audited local full-resolution H&E.
5. Target-H&E-only calibration of the B1 detector's scalar OOD threshold. The
   B1 mean and precision remain unchanged, and the artifact records that target
   expression was not accessed.
6. A disjoint spatial-block train/validation split and matched snRNA
   reference/prototype preparation using the B1 latent transform.
7. Train/validation batch assembly with the calibrated OOD mask as explicit
   unknown-head supervision.
8. Personalized HEIR training and Monte Carlo prediction on CUDA. Training and
   inference telemetry are persisted.
9. Validation of every frozen prediction and checkpoint hash.
10. Only then, construction of locked spatial-truth artifacts from Visium
   expression and one-shot scoring against the prespecified baselines.

The locked truth phase cannot execute before the prediction phase has passed
validation. The final benchmark plan records a separate checkpoint hash for
each personalized model, the panel hash, config hash, prediction path, matched
reference, locked truth, and inference telemetry.

## Resume and failure behavior

An output is skipped only if every output from that stage exists and its HEIR
contract, identities, row order, source hashes, feature/latent space, and
leakage provenance validate. If only part of a stage exists, or a completed
file is invalid, the script stops and asks that the artifact be moved aside.
It never invokes `rm`, overwrites scientific artifacts, or treats file
existence alone as completion.

Completed external Space Ranger runs are imported from
`artifacts/snpatho/<sample>/spaceranger/snpatho_<sample>/outs/`. An incomplete
run directory is preserved and reported instead of being deleted or duplicated.

The append-only event ledger, latest status snapshot, per-command logs, frozen
benchmark plan, and final JSON/TSV reports are written below
`artifacts/snpatho/orchestration_v0_2/`.

The completed seed-17 result is summarized in
[`benchmark_report.md`](benchmark_report.md). The primary median-gene spatial
correlation endpoint is negative; the same locked cohort must not be used for
post-hoc model tuning and a replacement success claim.
