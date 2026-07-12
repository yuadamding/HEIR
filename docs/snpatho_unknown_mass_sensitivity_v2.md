# snPATHO unknown-mass sensitivity v2

The clean checkpoint-bound seed-17 grid is complete: **75 of 75 CUDA stages**
and **30 of 30 prediction endpoints** completed from an empty, adoption-prohibited
artifact root. All 15 specimen/mass cases were scored at fixed masses 0, 0.01,
0.05, 0.10, and 0.20.

The final-source run records `execution_mode=all_completed`: its manifest binds
315 stage-input records and 150 stage-output records, the complete 73-file HEIR
runtime source inventory, and the exact 140-package Python/CUDA environment.
The scorer independently rehashed the 5.9-GB endpoint family and reported zero
blockers.

PyTorch warned that CUDA multinomial sampling uses a cumsum kernel without a
deterministic implementation. The manifest therefore proves the realized
endpoint identities and recipe/environment, not guaranteed bitwise equality of
a future rerun.

The result is **unstable**. Using the prespecified practical paired median-gene
Spearman margin of 0.002:

| Specimen | Refined vs round 0 across masses | Refined vs hard type mean | Refined vs soft type mean |
|---|---|---|---|
| 4066 | tie at all masses | pass at all masses | pass at all masses |
| 4399 | tie at all masses | fail at all masses | fail at all masses |
| 4411 | tie at all masses | tie at all masses | pass, pass, pass, tie, tie |

Raw direction is also not stable overall: 4411 changes from a positive
refined-minus-round-zero sign at mass 0 to negative signs at the four nonzero
masses. Higher mass predictably changes abstention and mean unknown probability,
but it does not yield a reproducible practical refinement benefit.

Therefore:

- no unknown mass is selected;
- no setting is called optimal from a sub-threshold round-zero contrast;
- the grid does not rescue the failed refinement claim; and
- multiseed follow-up is deferred until independent broad-type labels pass the
  upstream no-graph/no-residual morphology gate.

The compact machine-readable record is
[`snpatho_unknown_mass_sensitivity_v2_summary.json`](../reports/snpatho_unknown_mass_sensitivity_v2_summary.json).
The ignored full report and hash-bound run manifest remain under
`artifacts/snpatho/unknown_mass_sensitivity_v2/`; the 5.9-GB endpoint family is
external at `/mnt/seagate/HEIR_runs/snpatho_unknown_mass_clean_v2_run4`.

This remains an opened-cohort development sensitivity. The R1 shared scANVI
backbone used published integrated-workflow annotations and saw all three
specimens, and snPATHO is not an untouched confirmation cohort.
