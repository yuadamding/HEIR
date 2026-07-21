# HEST nonlinear qualification v2 execution

**Status:** terminal procedural qualification failure; retrospective, exposed, and nonauthorizing.

This report is a post-execution scientific erratum to the original record at commit `a498d39`.
The original external receipts, frozen protocol, and preregistration manifest are unchanged.

Preflight, CUDA implementation smoke, and all 2,000 frozen calibration trials passed at commit
`a5ca54d`. The gated loader then opened `nucleus_molecular_targets` once and last.

The first outer fold stopped during frozen B0 nested ridge selection because every registered
candidate was ineligible. No outer fold, held-out prediction, effect estimate, registration
sensitivity, or refitted-null permutation completed. Before that split, the loader materialized all
18,062 evaluation rows and enforced source-level donor/section/type support. B0 candidate search
then ran on the first 14-donor outer-training split. The selector did not emit its rejected-candidate
registry, so the failure cannot be separated into a fold-local basis-ceiling rejection, variance
rejection, or caught fit/metric error without impermissibly reopening the target.

The 2,000-trial calibration covered the shared scoring, multiplicity, registration, and D1/D2 rules
using directly generated held-out predictions. It did not execute production B0 selection. The
separate implementation smoke exercised neural selection with support 5 and both eligibility
screens set to zero; it also did not execute B0. The passing calibration receipt is therefore a
decision-rule calibration, not an end-to-end qualification of the production path that failed.

Per the frozen stopping rule, the model grid, thresholds, and protocol were not changed and the
target was not reopened. Because zero folds completed, D1 and D2 are **not evaluable** and no final
engineering decision is available. This is not a negative biological result and does not establish
that the H-optimus representation lacks nonlinear molecular signal. The exposed v2 identity is
nevertheless terminal: it cannot be rerun, and any further test requires a new protocol on
unexposed data with production-equivalent B0 synthetic calibration and persistent failure receipts.
It authorizes neither H-CELL, H-INTRINSIC, H-REF, nor full HEIR.

The machine-readable result is
`manifests/studies/hest_nonlinear_qualification_v2.result.json`. The external failure receipt is
`/mnt/seagate/HEIR_runs/hest_nonlinear_v2/report.a5ca54d.failed.json` with SHA-256
`61c0a34da6a7da83571f8308d4534818a96669c13c266dd13f4b92eddc2780b9`.
