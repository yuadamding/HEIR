# snPATHO unknown-mass sensitivity

Evaluation status: **blocked**. Stability conclusion: **blocked**.

Scored 1 of 15 required seed-17 cases.

## Blockers

- `4066` / mass `0p00`: round-zero checkpoint lacks serialized unknown-mass metadata; a post-hoc run manifest cannot prove the mass used to produce this endpoint
- `4066` / mass `0p01`: round-zero checkpoint lacks serialized unknown-mass metadata; a post-hoc run manifest cannot prove the mass used to produce this endpoint
- `4066` / mass `0p05`: round-zero checkpoint lacks serialized unknown-mass metadata; a post-hoc run manifest cannot prove the mass used to produce this endpoint
- `4066` / mass `0p10`: round-zero checkpoint lacks serialized unknown-mass metadata; a post-hoc run manifest cannot prove the mass used to produce this endpoint
- `4066` / mass `0p20`: round-zero checkpoint lacks serialized unknown-mass metadata; a post-hoc run manifest cannot prove the mass used to produce this endpoint
- `4399` / mass `0p00`: round-zero checkpoint lacks serialized unknown-mass metadata; a post-hoc run manifest cannot prove the mass used to produce this endpoint
- `4399` / mass `0p01`: round-zero checkpoint lacks serialized unknown-mass metadata; a post-hoc run manifest cannot prove the mass used to produce this endpoint
- `4399` / mass `0p05`: round-zero checkpoint lacks serialized unknown-mass metadata; a post-hoc run manifest cannot prove the mass used to produce this endpoint
- `4399` / mass `0p10`: round-zero checkpoint lacks serialized unknown-mass metadata; a post-hoc run manifest cannot prove the mass used to produce this endpoint
- `4399` / mass `0p20`: round-zero checkpoint lacks serialized unknown-mass metadata; a post-hoc run manifest cannot prove the mass used to produce this endpoint
- `4411` / mass `0p00`: round-zero checkpoint lacks serialized unknown-mass metadata; a post-hoc run manifest cannot prove the mass used to produce this endpoint
- `4411` / mass `0p01`: round-zero checkpoint lacks serialized unknown-mass metadata; a post-hoc run manifest cannot prove the mass used to produce this endpoint
- `4411` / mass `0p05`: round-zero checkpoint lacks serialized unknown-mass metadata; a post-hoc run manifest cannot prove the mass used to produce this endpoint
- `4411` / mass `0p10`: round-zero checkpoint lacks serialized unknown-mass metadata; a post-hoc run manifest cannot prove the mass used to produce this endpoint

## Compact results

| Sample | Mass | Round | Round0 rho | Refined rho | Refined-round0 | Refined-hard | Refined-soft |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4411 | 0.20 | 4 | -0.004448 | -0.003239 | -0.000474 | 0.000187 | 0.000371 |

## Endpoint behavior

| Sample | Mass | R0 abstain | Refined abstain | R0 public expr. | Refined public expr. | R0 mean unknown | Refined mean unknown |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4411 | 0.20 | 0.2315 | 0.2102 | 0.7685 | 0.7898 | 0.2354 | 0.1997 |

## Stability interpretation

blocked unless all 15 canonical cases validate; otherwise compare the signs of paired median per-gene Spearman deltas at all five masses
