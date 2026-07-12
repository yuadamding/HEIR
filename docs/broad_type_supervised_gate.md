# Broad-cell-type supervised development gate

HEIR now has a deliberately separate gate for the failure that precedes molecular
refinement: mapping frozen H&E nucleus features to broad cell identities. The gate
does **not** create labels, use RNA-derived pseudo-labels, use graphs, or activate a
molecular residual. It is runnable only after independent nucleus/ROI annotations
arrive.

No current biological success is claimed. The committed example plan remains
`labels_pending` because reviewed nucleus labels are not present in the downloaded
cohorts.

## Prespecified tasks

The plan supports two development tasks with H&E-resolvable ontologies:

- `snpatho_4066_broad`: malignant epithelial, nonmalignant
  epithelial/myoepithelial, immune, fibroblast/stromal,
  endothelial/vascular, and other/unknown. Because 4066 is one specimen, it uses
  held-out ROI groups and must be described as within-specimen development.
- `snpatho_liver_metastases_broad`: metastatic malignant epithelial,
  hepatocyte, immune, fibroblast/stromal, endothelial/LSEC,
  ductal/cholangiocyte, and other/unknown. Sections 4399 and 4411 use
  leave-one-donor-out testing; calibration remains ROI-disjoint within the
  remaining donor.

The ordered ontology files are
[`snpatho_broad_4066.tsv`](../configs/ontologies/snpatho_broad_4066.tsv) and
[`snpatho_broad_liver_metastases.tsv`](../configs/ontologies/snpatho_broad_liver_metastases.tsv).
They are hash-bound by the example plan.

## Independent-label contract

Each dataset supplies a hash-bound TSV with the header in
[`broad_type_labels_template.tsv`](../manifests/broad_type_labels_template.tsv).
Every reviewed row must provide:

- section, nucleus, donor, and ROI identifiers;
- a nonblank pathology compartment;
- a broad type found in the task's prespecified ontology;
- reviewer confidence in `[0,1]` and reviewer count;
- an accepted adjudication status and a non-HEIR annotation source; and
- `independent_of_heir_predictions=true`.

Blank compartments, missing columns, duplicate nuclei, absent feature matches,
non-independent labels, stale hashes, insufficient class support, or fewer than
three distinct seeds block execution. Low-confidence or under-reviewed rows are
reported and excluded according to prespecified thresholds. The pipeline never
backfills excluded or missing labels.

The recommended annotation collection remains 1,000–2,000 nuclei per specimen
across 20–30 stratified ROIs, with two independent reviewers and adjudication.
ROIs should cover tumor core/margin, liver parenchyma where applicable, fibrosis,
immune-rich and vascular regions, necrosis, and staining-quality variation.

## Inspect now; run after labels arrive

Inspection verifies the plan, hashes, ontology, feature artifacts, and label TSV
headers without fitting a model:

```bash
python scripts/benchmark_broad_types.py \
  --plan configs/broad_type_supervised_gate.example.json \
  --output artifacts/snpatho/broad_type_gate/readiness.json \
  --inspect
```

The committed example must currently report `blocked_evidence`.

The current dry-run result is preserved in
[`snpatho_broad_type_gate_readiness.json`](../reports/snpatho_broad_type_gate_readiness.json):
all frozen feature and ontology hashes validate, while the three independently
reviewed label artifacts (4066, 4399, and 4411) have no registered hashes and the
plan therefore cannot train.

After independent annotations are deposited outside Git:

1. Fill each label artifact path and SHA-256.
2. Change plan `status` from `labels_pending` to `ready`.
3. Run the same command without `--inspect`:

```bash
python scripts/benchmark_broad_types.py \
  --plan configs/broad_type_supervised_gate.example.json \
  --output artifacts/snpatho/broad_type_gate/report.json \
  --device auto
```

`auto` uses CUDA for the small neural classifier when available. The logistic
probe remains a CPU scikit-learn baseline. Both operate on the same immutable
feature rows.

## Models, null, and endpoints

For each of at least three seeds, the gate fits:

- an inverse-frequency-balanced multinomial logistic/linear probe; and
- a one-hidden-layer classifier on frozen features, trained with
  inverse-frequency and reviewer-confidence weighting.

Temperature is fitted only on the calibration partition. Test reporting includes
macro-F1, balanced accuracy, per-class F1, ECE, Brier score, predicted and observed
occupancy overall and by compartment, and the full risk–coverage curve plus AURC.

The image-shuffle null permutes feature-to-nucleus assignments separately within
each split and donor. This preserves split/donor feature distributions while
destroying nucleus-level correspondence. The realized permutation is SHA-256
bound in every run. No graph is constructed, and the evaluator rejects any plan
that sets `graph.enabled` or `molecular_residual.enabled` to true.

The default development thresholds require, for both models:

- mean macro-F1 at least 0.65;
- mean real-minus-image-shuffle macro-F1 at least 0.05;
- mean ECE below 0.10; and
- at least 75% of ontology classes occupied in every seed's hard predictions;
- every seed × held-out-donor run independently satisfies all four thresholds
  (`minimum_seed_donor_run_pass_fraction = 1.0`), with at least three seeds.

The report separately records donor-fold run counts and the number of distinct
seeds that beat the shuffle null in every donor fold; these quantities are not
interchanged.

These are prespecified engineering gates, not evidence that HEIR's molecular
refinement succeeds. A completed negative gate is a valid result and must remain
reported as such.

## Schemas

- Plan: `heir.broad_type_supervised_plan.v1`, formalized in
  [`broad_type_supervised_plan.schema.json`](../configs/schemas/broad_type_supervised_plan.schema.json).
- Independent label rows: `heir.independent_broad_type_labels.v1`.
- Report: `heir.broad_type_supervised_report.v1`, formalized in
  [`broad_type_supervised_report.schema.json`](../configs/schemas/broad_type_supervised_report.schema.json).
- Dry-run inspection: `heir.broad_type_supervised_inspection.v1`.

The report binds the plan and all input artifacts by SHA-256 and records split
assignment hashes, donor/ROI membership, model/calibration recipes, seed-level
metrics, null permutation hashes, and task/overall gate decisions.
