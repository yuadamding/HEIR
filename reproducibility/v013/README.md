The 48 rows in [aggregates.tsv](aggregates.tsv) reproduce every cohort aggregate
from the completed v0.13 diagnosis. [provenance.json](provenance.json) binds the
original summary, executed reconstruction script, audit receipt, source commit,
and this publication. Patient records, predictions, targets, and large inventories
remain in the original run artifacts.

Run the independent aggregate check from the repository root, with HEIR's `htapp`
dependencies installed:

```bash
python reproducibility/v013/audit_residual_moments.py \
  --aggregates reproducibility/v013
```

This checks the published table's hash, unique group count, error identities, and
analytic multipliers. It cannot verify the underlying arrays or the historical
audit's reported maximum discrepancy of `1.4210854715202004e-13`.

The same script exposes the full independent reconstruction used for that audit.
It imports only existing input loaders and the saved model for forward evaluation;
normalization, moments, scores, and aggregation are implemented independently.
All ten original helper/class syntax trees are unchanged. Formatting, explicit CLI
paths, runtime checks, and aggregate verification were added for publication; both
script hashes and the unchanged helper names are recorded in the provenance file.
The complete biological audit was not repeated for this publication.

With the original artifacts and compatible numerical source bytes available, a
new reconstruction receipt can be produced explicitly:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
python reproducibility/v013/audit_residual_moments.py \
  --study /storage/HE_GPT/HEIR-cohort-runs/v0120-joint-residual/study \
  --diagnostics /storage/HE_GPT/HEIR-cohort-runs/v0130-saved-diagnostics/diagnostics \
  --output /path/to/new-audit.json
```

The output must not exist. Full reconstruction reads the original receipt-bound
prepared inputs, targets, bundles, exports, and diagnostic tables. Its data paths
remain those in the historical receipts; source-module relocation support does
not remap these data identities. It performs no fitting or predictor selection.

All multipliers remain retrospective, target-informed diagnostics. Negative
centered alignment concerns the expression error remaining after ridge, and does
not establish negative correlation with true expression or biological reversal.
