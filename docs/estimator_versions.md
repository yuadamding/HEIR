# HEIR estimator versions

Estimator identity is separate from scientific authorization. A better estimator on exposed data
does not convert that data into prospective evidence, and a new estimator does not overwrite an
earlier negative result.

| Estimator | Status | Permitted interpretation |
|---|---|---|
| `ridge_v5` | Frozen | Existing H-CELL/H-INTRINSIC gate and its original negative or positive result |
| `regional_fusion_v2` | Frozen, not supported | Existing regional matched-reference result; favorable mean-loss estimates failed molecular-variance preservation |
| `nonlinear_qualification_v1` | Retrospective; core/smoke complete; biological run blocked | Engineering qualification only; may support registering a new prospective estimator protocol only after every registered arm and control can run |
| `nonlinear_qualification_v2` | Retrospective; terminal procedural failure at B0 selection | No held-out effect estimate; D1/D2 not evaluable; no biological or representation-level conclusion |
| `prospective_nonlinear_v1` | Not authorized | Requires a complete retrospective qualification pass before any pristine outcome is opened |

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
- V2 resolved the outcome-blind inputs and passed preflight, implementation smoke, and 2,000-trial
  calibration. Its first biological fold stopped because B0 had no eligible ridge candidate. The
  target was opened once; no fold or effect estimate completed, and no post-exposure tuning or
  rerun was performed. The calibration did not exercise production B0 selection, and the rejected
  candidate registry was not persisted, so the failure is procedurally indeterminate. V2 is
  terminal, but it is not evidence that the representation lacks nonlinear molecular signal.
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
