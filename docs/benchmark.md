# Benchmark and cohort status

## Current result

No current result supports a biological performance claim for refinement.

| Benchmark | Coverage | Result | Interpretation |
|---|---:|---:|---|
| snPATHO locked v0.2 | 3/3 specimens | macro median-gene Spearman **-0.0054** | negative historical endpoint |
| snPATHO DeepBench | 3/3 specimens | HEIR macro approximately **-0.0067** | type-mean baselines are better |
| refinement matrix | 93/93 artifacts | **49/108 pass**, **59/108 fail** | strict ordering fails |
| unknown-mass grid | 75/75 CUDA stages | no stable mass | tuning does not rescue refinement |
| revised frozen-target training | synthetic only | no cohort result | improvement not established |

Mean paired median-gene Spearman deltas for refined HEIR were:

| Contrast | Delta |
|---|---:|
| refined - image shuffle | **-0.00656** |
| refined - graph shuffle | **-0.00225** |
| refined - no graph | **+0.00054 mean**, **-0.00043 median** |
| refined - prototype only | **-0.00015** |
| refined - round zero | **-0.00037** |
| refined - wrong-prototype bank | **-0.00404** |

The fixed unknown-mass grid used masses `0`, `0.01`, `0.05`, `0.10`, and `0.20`, seed 17,
three specimens, 30 prediction endpoints, and 15 fixed-mass comparisons. It selected no mass.

## Downloaded cohorts

A complete benchmark needs calibrated H&E, stable nucleus instances, frozen H&E features, a
specimen-matched annotated single-nucleus reference, held-out registered spatial expression, a
frozen gene panel, donor-level splits, and target exclusion from all development choices.

| Cohort | Full clean benchmark? | Current valid role | Limitation |
|---|---|---|---|
| snPATHO, 3 specimens | No | historical and opened-cohort development | all spatial truths opened; labels are not an independent clean reannotation |
| NatCommun/MOSAIC, 15 sections and 13 usable donors | No | personalization, reference falsification, runtime | no reproducible processed registered spatial truth |
| spatial DLPFC snapshot | No | coordinate/format smoke tests | expression objects and full-resolution H&E missing |
| HEST prostate ST | No | spot-level baseline/pretraining | not nucleus resolved; no matched snRNA |
| generic prostate atlas | Not a benchmark cohort | unmatched-reference baseline | no paired H&E or spatial truth |
| TRACERx421 WES | Out of scope | none | no H&E/snRNA/spatial triplet |

No downloaded cohort currently supports a clean, untouched validation of the complete method.

## Local data contract

The canonical ledgers are `manifests/natcommun.tsv` and `manifests/snpatho.tsv`. Space Ranger is
the default segmentation method. Raw cohort files and all derived artifacts remain external to the
repository, normally under `/mnt/seagate`.

The three snPATHO specimens are 4066, 4399, and 4411. They have historical Visium truth and
native-scANVI development artifacts, but cannot be reused as untouched confirmation. The revised
primary path additionally requires a passing `MorphologyStateGate`, a validated initializer
receipt, and independent frozen target artifacts, which do not currently exist. Passing the gate
later would only qualify snPATHO for development because its spatial truth has already been opened
and its molecular labels are not an independent clean reannotation.

Use `scripts/benchmark_morphology_state_gate.py` for the compact direct-bridge experiment. Its
training and held-out NPZ inputs contain frozen features, registered latent targets, broad-type
indices, and donor identities; the held-out input additionally contains expression targets and may
contain ROI identities. CUDA is selected by default when available. A snPATHO result from this
command is opened-cohort development evidence only.

The machine-readable readiness decision is
`reports/snpatho_morphology_state_gate_readiness.json`; it records
`benchmark_executed=false` and the exact missing prerequisites.

## Evidence files

Compact machine-readable results live in `reports/`. Important immutable hashes include:

- `configs/experiments/snpatho_deepbench_v1.yaml`:
  `893e169dcb0ebc59577ec45c8f1dbf2ffee756dee20082f73058dfea0b8eac00`;
- `configs/experiments/snpatho_v0_2.yaml`:
  `1f572ed7d80255484fcf1f43abd8468e1f57a8a78394379d3bc2daeb7a6a029f`;
- `reports/snpatho_unknown_mass_sensitivity_v2_summary.json`:
  `8979a7c6882b0a34483420310113c762c18bf0e58580a5fca16e169fc7cd697a`.

Historical results remain distinct from the revised source tree. The new code should be described
as scientific and provenance hardening until an eligible end-to-end cohort run exists.
