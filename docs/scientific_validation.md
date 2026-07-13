# Scientific validation contract

## Hypotheses

HEIR is a falsifiable histology-to-transcriptome model. Its primary hypotheses are:

1. Matched H&E identity outperforms image-feature shuffle.
2. A matched molecular reference outperforms every valid wrong-reference bank.
3. Graph context outperforms no graph and graph shuffle when graph mode is enabled.
4. The molecular residual improves over residual-disabled prototype prediction.
5. Selected refinement improves over round zero; otherwise the system returns round zero.
6. These decisions transfer to one untouched cohort after all development choices are frozen.

Composition, pseudobulk, marker, and transport losses are regularizers. They do not by themselves
identify which RNA state belongs to a particular nucleus.

## MorphologyStateGate

`MorphologyStateGate` is the required independent bridge before molecular-state results may be
interpreted. It consumes frozen H&E features, independently reviewed broad types, registered RNA
latents, and a frozen decoder. Graphs, UOT, abundance priors, unknown heads, refinement, and weak
sample losses are absent. Training-only type centroids and low-rank residual bases remove the
between-type signal before a small residual MLP is fitted.

The checkpoint-executing scorer regenerates the decoder ceiling, oracle-type mean, oracle-type
image residual, predicted-type image residual, donor/type-preserving shuffle, optional
donor/type/ROI-preserving shuffle, and same-type state retrieval. A pass requires:

- within-type partial R² of at least `0.05` for every held-out donor;
- positive cosine and RMSE improvement over the within-type shuffle;
- positive decoded-expression improvement over the oracle type mean;
- retrieval top-1 and MRR above their uniform expectations;
- matched states closer than every available wrong-donor state bank;
- no predicted-type collapse; and
- donor-bootstrap lower bounds above zero for within-type R² and decoded-expression improvement.

Missing wrong-donor banks block the primary gate. A command-line waiver exists only for explicit
development diagnostics. The historical cross-type shuffle remains a broad-type control and is not
the decisive molecular-state null.

The committed snPATHO gate is `labels_pending`, so it does not pass. Moreover, all three snPATHO
spatial truths have already been opened and its molecular annotations are not an independent clean
reannotation. A future MorphologyStateGate pass on these specimens would support development only;
it could not turn snPATHO into untouched external confirmation.

## Frozen target construction and M-step

A frozen teacher generates one molecular target artifact before student fitting. Its artifact
records:

- exact teacher checkpoint, morphology, graph, and molecular evidence hashes;
- teacher type posterior and image-latent query;
- Gaussian/type/dustbin costs;
- raw and row-conditional unbalanced-transport plans;
- raw real row mass, raw dustbin row mass, and conditional known-state routing;
- source mass, effective target mass, realized Sinkhorn iterations, parameters, and telemetry.

The M-step consumes this fixed artifact. It cannot evaluate live UOT, use its own type or unknown
heads to define targets, mint pseudo-anchors, or update molecular priors. CPU float32 replay must
reconstruct the teacher output, costs, and transport plans within tight tolerances before training.
The default is a single M-step phase. If an explicit development plan runs more than one phase
against the same artifact, the protocol is named `fixed_target_curriculum`; it is not iterative or
generalized EM because no new E-step is computed. Live-student target recomputation exists only as
the excluded `live_student_e_step_negative_control`.

## Refinement safety

- Round-zero validation loss is immutable.
- Each candidate is compared with the round-zero ceiling.
- The best safe snapshot is retained; the final round is never selected merely because it is last.
- Round-level teacher updates default to an accepted-student copy rather than a step-scale
  `0.99` EMA applied four times.
- Fixed-target curriculum phases cannot convert self-confirmation into hard molecular routing.
- Parent-only rounds are reported as parent-head fitting, not broad molecular transport.

The residual branch uses a detached, continuous concentration gate. Residual-off is a required
endpoint. Graph mode is disabled by default; distance-only graph behavior is experimental and
starts behind a zero residual gate.

## Unknown and OOD states

Technical artifact, feature-domain OOD, transport unassignment, type confidence, and biological
unknown are distinct concepts. Model and prediction contracts expose
`transport_unassigned_probability`; `unknown_probability` is retained only as a checkpoint/file
compatibility alias. Target quantiles are descriptive telemetry only. Biological unknown requires
independent supervision and is assessed separately.

## Leakage and provenance

Target spatial expression is evaluation-only. Training, refinement, feature selection, gene-panel
selection, threshold selection, and stopping rules may not read it. Whole-specimen weak losses are
computed once after patch aggregation rather than attaching the same specimen target to each patch.

Every fitted input is frozen before use and rechecked before publication. Output paths must not
alias inputs, archive containers, or descendants of bound input directories. Stage manifests retain
pre-stage input hashes and validation-bracketed output hashes rather than recomputing identities only
at the end.

## Evaluation

All methods use the same nuclei, spot assignment, RNA-mass policy, truth-defined gene mask, and
comparison plan. Reports include:

- nucleus-level type metrics when independent labels exist;
- full-coverage expression with a prespecified fallback;
- fixed-coverage selective expression;
- spot-level correlation/error and pseudobulk metrics;
- donor-equal macro summaries and paired control deltas;
- oracle decoder, type, state, and residual-disabled ceilings;
- coverage, abstention, runtime, CUDA, and provenance telemetry.

The method remains blocked if a required control is unavailable or if different methods are scored
on different effective genes or nuclei.

## Pass criteria

MorphologyStateGate must pass before UOT or a one-pass molecular M-step is eligible. If the
oracle-type bridge fails, transport, graph, unknown-mass tuning, and repeated fixed-target phases
are not interpreted as remedies. If oracle type passes but predicted type fails, work remains on
the broad-type initializer. If the direct bridge passes but the frozen target path fails, diagnose
the transport cost, target masses, and prototype construction. Final confirmation still requires a
previously untouched cohort.
