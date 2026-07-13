# Coverage-aware spatial-expression evaluation

`heir.coverage_evaluation.v2` defines two prospective expression endpoints.
Both aggregate log1p cell expression in linear space with a prespecified,
positive RNA-mass weight for every spot-assigned cell.

The full-coverage endpoint replaces each abstained cell's expression with its
frozen hard-type mean, then aggregates all assigned cells. Both aggregation
endpoints require ordered, unique nucleus IDs and ordered, unique spot IDs.
Their reports bind those identities and the ordered gene identities to the
exact cell-to-spot assignment, cell-expression matrix, RNA-mass assignment,
resulting spot-mass vector, and resulting spot-expression matrix.
The full endpoint additionally binds the fallback matrix, identity-aligned
cell-to-type index, and identity-aligned abstention mask. A benchmark run can
require those hashes to match its locked plan. This prevents row reorderings,
mapping changes, or mass changes from retaining misleadingly identical
provenance.

The selective endpoint ignores a method's mutable binary abstention threshold.
It ranks assigned cells by the prespecified uncertainty score and retains the
exact requested fraction. The requested fraction must correspond to an integer
number of eligible cells; it is never silently rounded. Stable nucleus IDs break
uncertainty ties, and the selected-cell mask is hashed together with those IDs.
The helper can also require prespecified uncertainty-vector and RNA-mass-vector
hashes.

For comparative use, a benchmark plan must be externally hash-asserted before
truth values are loaded and must prespecify:

- the fallback type-mean artifact and its matrix-content hash;
- the cell-to-type mapping used for fallback;
- the RNA-mass artifact and its calibration recipe;
- the uncertainty definition and fixed selective coverage;
- the spot rows used to define the endpoint; and
- all method/comparison names.

RNA mass may combine independently calibrated broad-type library size, cell or
nucleus area, spot-overlap fraction, segmentation confidence, and an externally
calibrated known-state probability. The evaluator accepts the resulting mass
vector but does not estimate or tune it from locked spatial expression.

`build_truth_gene_mask` creates one truth-only mask from finite, spatially
variable genes. Its hash binds gene order, the selected mask, variance policy,
ordered spot identities, the prespecified truth-spot selection, and the truth
matrix used to construct it. Evaluation rejects a different spot order or truth
matrix. `evaluate_methods_on_truth_gene_mask` accepts only validated
`CoverageAggregation` objects, checks their spot and gene identities and
positive mass on every scored spot, then carries each endpoint's complete
coverage-provenance hash into the score report. Raw prediction matrices are not
accepted. The scorer uses the same truth-mask object for every method and every
paired per-gene difference.
When truth varies but a method's prediction is constant, correlation is scored
as zero instead of being discarded by `nanmedian`; every method is therefore
summarized over exactly the same genes.

## Prospective CLI endpoint

`heir evaluate-spatial-coverage` is the production path for these semantics.
It is separate from the historical `evaluate-spatial` command. Each method
requires a pickle-free `heir.coverage_endpoint_input` version 1 NPZ. The common
fields are:

- scalar `__contract__`, `__version__`, and `endpoint` declarations;
- `nucleus_ids`, `gene_names`, `spot_ids`, and `nucleus_spot_index`;
- a prespecified `evaluation_spot_mask` shared by compared methods;
- externally calibrated `cell_rna_mass`.

The full endpoint additionally requires `frozen_type_index`, `type_names`, and
`frozen_type_mean_log_expression`. The selective endpoint instead requires
`uncertainty` and an exactly attainable scalar `target_coverage`. The command
constructs and hashes the aggregation before loading spatial truth values, then
requires the locked truth artifact to have exactly the same nucleus, gene,
spot, and cell-to-spot identities. Dense overlap assignments are rejected: the
prospective artifact must contain an explicitly frozen `nucleus_spot_index`.

The single-method form below is a diagnostic invocation. Even when the endpoint
input hash is asserted, the runtime method name and absence of a locked
comparison list make it ineligible for paired method-comparison claims. Its
report states that limitation explicitly.

```bash
heir evaluate-spatial-coverage \
  --predictions predictions.npz \
  --endpoint-input frozen_coverage_endpoint.npz \
  --endpoint-input-sha256 "$ENDPOINT_SHA256" \
  --truth locked_spatial_truth.npz \
  --output prospective_coverage_metrics.json
```

Comparative scoring instead requires one externally hash-asserted
`heir.coverage_benchmark_plan.v1` JSON. The plan contains exactly the locked
truth path/hash, two or more method names, each method's prediction and endpoint
input path/hash, and all comparison pairs. Relative paths resolve from the plan
directory. The evaluator constructs every method aggregation before loading
truth values, requires the endpoint kind, coverage, ordered identities,
cell-to-spot mapping, evaluation-spot mask, RNA-mass vector, and expression
space to match across methods, and then scores all methods on one truth-derived
gene mask.

```json
{
  "schema": "heir.coverage_benchmark_plan.v1",
  "truth": "locked_spatial_truth.npz",
  "truth_sha256": "REPLACE_WITH_TRUTH_SHA256",
  "methods": [
    {
      "name": "HEIR",
      "predictions": "heir_predictions.npz",
      "predictions_sha256": "REPLACE_WITH_PREDICTION_SHA256",
      "endpoint_input": "heir_coverage_endpoint.npz",
      "endpoint_input_sha256": "REPLACE_WITH_ENDPOINT_SHA256"
    },
    {
      "name": "type_mean",
      "predictions": "type_mean_predictions.npz",
      "predictions_sha256": "REPLACE_WITH_BASELINE_SHA256",
      "endpoint_input": "type_mean_coverage_endpoint.npz",
      "endpoint_input_sha256": "REPLACE_WITH_BASELINE_ENDPOINT_SHA256"
    }
  ],
  "comparison_pairs": [["HEIR", "type_mean"]]
}
```

```bash
heir evaluate-spatial-coverage \
  --plan coverage_benchmark_plan.json \
  --plan-sha256 "$PLAN_SHA256" \
  --output prospective_comparative_metrics.json
```

## Oracle-ladder provenance

`scripts/benchmark_oracle_ladder.py` requires a pickle-free NPZ fixture, the
physical RNA-decoder checkpoint, and the physical HEIR checkpoint. It hashes
all three files before evaluation.
The NPZ must contain ordered cell IDs, gene names, a spot ID aligned to each
cell row, a positive frozen RNA-mass value for every cell, and its declared
decoder-checkpoint and HEIR-checkpoint digests, in addition to every numerical
oracle input. The script rejects either declared digest when it differs from
the supplied checkpoint file. The resulting
`heir.oracle_ladder.v5` report records hashes for every normalized array, every
ordered identity vector, the exact spot-truth-derived gene mask, and the
complete input bundle. Every rung is scored at cell, RNA-mass-weighted spot,
and RNA-mass-weighted pseudobulk resolution. The ladder requires explicit
same-checkpoint oracle-type-conditioned and residual-disabled HEIR forward-pass
predictions beside full HEIR. The evaluator does not approximate either control
by mixing already decoded prototype profiles.
Constant endpoint predictions are scored as correlation zero on that same gene
mask rather than removed from the median.

The decoder ceiling, oracle-type-conditioned HEIR prediction, residual-disabled
HEIR prediction, and full HEIR prediction are fixture-declared precomputed
outputs bound to their supplied checkpoint hashes. The evaluator does not rerun
either checkpoint, so the report marks the ladder diagnostic-only and
ineligible for primary performance claims; provenance binding is not itself a
claim that fixture generation was correct.

```bash
python scripts/benchmark_oracle_ladder.py \
  --input oracle_ladder_fixture.npz \
  --decoder-checkpoint frozen_rna_decoder.pt \
  --heir-checkpoint frozen_heir.pt \
  --output oracle_ladder_report.json
```

## Historical snPATHO reports

The existing `heir.snpatho_refinement_matrix.v1` artifacts are preserved as
historical endpoints. They did not pre-register a frozen fallback hash, a fixed
coverage, or a shared uncertainty rule, so these new endpoints are not
retroactively inserted into that matrix. A future versioned snPATHO plan can
adopt the helpers once those inputs are frozen before execution.
