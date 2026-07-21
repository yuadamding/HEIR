# HEIR estimator versions

Estimator identity is separate from scientific authorization. A better estimator on exposed data
does not convert that data into prospective evidence, and a new estimator does not overwrite an
earlier negative result.

| Estimator | Status | Permitted interpretation |
|---|---|---|
| `ridge_v5` | Frozen | Existing H-CELL/H-INTRINSIC gate and its original negative or positive result |
| `regional_fusion_v2` | Frozen, not supported | Existing regional matched-reference result; favorable mean-loss estimates failed molecular-variance preservation |
| `nonlinear_qualification_v1` | Retrospective; core/smoke complete; biological run blocked | Engineering qualification only; may support registering a new prospective estimator protocol only after every registered arm and control can run |
| `prospective_nonlinear_v1` | Not yet authorized | May be registered only after retrospective qualification passes and before any pristine outcome is opened |

## Version rules

- Frozen estimators keep their original code, artifact hashes, reports, thresholds, and
  interpretation.
- `nonlinear_qualification_v1` has schema
  `heir.hest_nonlinear_qualification_protocol.v1` and analysis status
  `retrospective_exposed_non_authorizing`.
- V1 is blocked because the source lacks a receipt-bound blank-patch embedding and the frozen
  best-registration subset has no donor/type stratum at primary support 20. Inputs may not be
  fabricated and thresholds/bands may not be changed after results.
- Its deterministic neural probe, architecture-matched control/null machinery, source preflight,
  and synthetic 20-permutation smoke are implemented. The full B0-B3/N0-N7 biological runner and
  decision remain unavailable under the blocked v1 inputs; the smoke is not scientific evidence.
- A retrospective pass cannot authorize H-CELL, H-INTRINSIC, H-REF, or full HEIR.
- `prospective_nonlinear_v1` does not exist until a complete retrospective pass and a separately
  reviewed preregistration. Its protocol must be frozen before pristine molecular outcomes are
  accessed.
- Encoder changes, RNA-target changes, reference expansion, and estimator changes require distinct
  version identities; they must not be combined silently.

## Frozen predecessor evidence

The v1 nonlinear contract hash-binds the historical ridge runner, morphology gate, reference-fusion
v2 implementation, and their decision reports. These files are evidence, not templates to rewrite.
Their registered hashes are recorded in
`configs/hest_nonlinear_qualification_v1.json`.
