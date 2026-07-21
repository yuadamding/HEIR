# HEST nonlinear qualification v2 execution

**Status:** stopped; retrospective, exposed, and nonauthorizing.

Preflight, CUDA implementation smoke, and all 2,000 frozen calibration trials passed at commit
`a5ca54d`. The gated loader then opened `nucleus_molecular_targets` once and last.

The first outer fold stopped during frozen B0 nested ridge selection because every registered
candidate was ineligible. No outer fold, held-out prediction, effect estimate, registration
sensitivity, or refitted-null permutation completed. The selector did not emit its rejected
candidate registry, so this result does not identify a biological effect or a specific rejection
criterion.

Per the frozen stopping rule, the model grid, thresholds, and protocol were not changed and the
target was not reopened. V2 therefore does not support D1 or D2 and stops SCHAF-style nonlinear
expansion on this representation. It authorizes neither H-CELL, H-INTRINSIC, H-REF, nor full HEIR.

The machine-readable result is
`manifests/studies/hest_nonlinear_qualification_v2.result.json`. The external failure receipt is
`/mnt/seagate/HEIR_runs/hest_nonlinear_v2/report.a5ca54d.failed.json` with SHA-256
`61c0a34da6a7da83571f8308d4534818a96669c13c266dd13f4b92eddc2780b9`.
