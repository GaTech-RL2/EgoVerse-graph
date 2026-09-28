# 03 — Agent-guided curriculum and learner feedback

## Scientific question

Can Astra generate physically valid scenes and action supervision that teach a fully random HPT learner to follow object/target instructions? Does using learner feedback to choose those scenes and their stage allocation help compared with uniform allocation under the same accepted-data and optimizer-update budgets?

The linked [paper](https://arxiv.org/pdf/2609.30063) motivates a generator–learner feedback loop. The recovered paper notes describe a generator trained by reinforcement learning to produce programs and a byte-sequence learner trained on their outputs. Here Astra is frozen and adapts through context, the simulator supplies physics, and typed teacher programs supply demonstrations. These changes make this an adaptation of the idea rather than a reproduction of the whole method.

## Agent authority versus fixed engineering

The scene grammar, stages, simulator, validation rules, learner, acquisition budgets, and evaluation protocol are fixed. Astra chooses adaptive-arm stage allocations, task templates, legal scene parameters, teacher programs, and whether to advance, consolidate, or revisit. There is no mastery threshold or round-to-stage schedule in the adaptive implementation.

The first adaptive decision is made after common S1 warm-start and all-stage control probes. The prompt asks Astra to increase rigor as warranted by evidence and retain the option to revisit. It may choose zero samples from a stage, introduce S3 early, or return to S1. A validator may reject illegal decisions; it may not replace them with its own curriculum.

## Feedback report

Each adaptive call receives the current snapshot/round, per-stage probe successes and trial counts, wrong-object/target and constraint errors, displacement/overshoot failures, previous-report deltas, signs of forgetting, previous allocation and teacher-yield summaries, remaining budget, and legal schemas. Include up to six training-control frame sequences chosen by a frozen deterministic error-sampling rule.

Always show counts and denominators, not an unexplained competence score. A 0/12 result differs from missing probes. Mark inferred failure categories and provide evidence. Astra receives neither development nor sealed-test instances, metrics, nor frames. Raw gradients and full replay datasets are unnecessary provider inputs.

Before each call, publish the report and its source checkpoint hash. Store the exact provider request, response, model/version, decoding settings, response ID, usage, latency, and validation results without credentials. No public Astra API identifier or training endpoint is assumed; W01 must establish actual runtime access.

## Decision contract

Each decision selects integer accepted-episode quotas across S1/S2/S3 summing to 75 and exactly five positive-allocation task templates whose allocations sum to those quotas. Every template references or supplies a compilable scene/task/teacher contract, uses one stage, and has bounded geometry. At least two structurally new scene templates must survive the round novelty gate.

The response includes an evidence-linked rationale, a predicted next-report change, and `advance`, `consolidate`, or `revisit` intent for active stages. Those labels record choices; the runner does not interpret them as imperative promotion rules. Valid examples and the proposed prompt are in [examples](examples/README.md).

Acquisition fills the assigned template quotas. A template failure cannot silently donate its quota to another template or stage. Astra may revise unfilled task geometry or teacher instructions within the call budget, retaining the committed per-template allocations. Record revisions and accepted episodes against an immutable decision lineage; changed allocations require a new run specification rather than rewriting a partly collected round.

## Uniform comparison

U gets exactly 25 accepted episodes per stage per round. Astra still authors its scenes and teacher programs, receives schema/compile errors needed to repair validity, and sees a generation-only duplicate archive. It receives no learner performance, probe frames, or adaptive-arm decision history. A sees the learner report and chooses its allocation. Both arms have five templates, identical attempt and provider caps, the same teacher controller and grammar, and the same learning recipe.

Compilation errors and witness failures are generator-validity feedback, not student feedback; keep their fields separate. Maintain arm-local task archives and equal initial catalog access. Do not transmit adaptive-arm performance through the baseline's duplicate-detection metadata.

This comparison estimates the combined effect of learner-conditioned scene generation and allocation. It cannot isolate allocation from content changes. Matched accepted-data and update budgets do not mean equal wall-clock/API cost; report actual costs, including rejected scenes and failed teachers.

## Budgets and failures

Per arm per round: at most eight structured generation calls, 8,192 output tokens per call, at most two repair responses per invalid proposal, and 300 collection attempts for exactly 75 accepted episodes. Transport failures retry at most twice; record provider attempts separately from logical generation calls and count any provider-billed usage. Exhaustion yields a paused/incomplete run, not an automatic deterministic fallback.

For accounting completeness, the common six-scene commissioning gate and the common seed-collection phase each receive their own cap of eight logical generation calls under the same token/repair/transport rules. This adds at most 16 common calls to the 64 branch calls. These common-phase call caps are new explicit planning defaults introduced in version 2.0; they prevent unbounded pre-run generation costs.

Malformed schemas, impossible geometry, duplicate scenes, and teacher failures remain archived. A failed provider call cannot be presented as evidence that a curriculum would not learn. An incomplete acquisition arm prevents a matched final curriculum comparison, although engineering outcomes can still be reported.

## Post-run gradient diagnostic

Run this after the competence-only experiment. On A's final branch checkpoint, define the paper-inspired score

\[
r_\tau=\left|\sum_j \nabla_{\theta_j}L_\tau(\theta_{2000})\,\frac{\eta_j}{\sqrt{\hat v_{j,2000}}+\epsilon}\,(\theta_{j,1000}-\theta_{j,2000})\right|.
\]

The history starts at the common warm-start checkpoint, so 1,000/2,000 refer to branch updates. Retrieve each AdamW parameter's bias-corrected second moment using its actual saved step counter, which includes warm-start updates. Use the frozen final learning rate and actual parameter-group settings. AdamW weight decay is not part of this displayed diagonal preconditioner. Missing moments are a diagnostic failure rather than zero-filled evidence.

Choose twelve existing verified training tasks, four per stage. If A has no tasks for a stage, use U's training pool and label that provenance. Use four fixed windows per candidate and two fixed flow noise/time draws. Disable dropout and freeze running-statistic updates during score and transfer-loss measurements; use identical measurement conditions for all candidates. Log the signed inner products, absolute score, loss, and gradient norm. For two draws, compute each signed dot and take the mean of the two absolute values; do not silently substitute the absolute value of their mean.

Compute gradients with PyTorch autograd over named trainable parameters and accumulate the dot product in FP32 or higher precision. The direction uses parameter values, not BatchNorm buffers. Validate the dot product with a deterministic finite-difference fixture. Do not implement forward-mode JVP or distributed scoring in this pilot.

For each candidate, restore the same final learner/optimizer/scheduler/RNG/buffer snapshot, train twenty candidate-only updates with the regular training recipe, and measure loss reduction on a fixed training-only transfer bank of 48 windows, balanced across stages and disjoint from candidate episodes. Use the same transfer noise/time draws before and after each branch. Restore the complete snapshot between candidates, including running buffers and sampler state. These twelve branches are discarded after analysis and never enter final policy evaluation.

Report rank association between candidate score and transfer-loss improvement, alongside loss and gradient-norm comparators, ties/outliers, timings, and memory. This adds 240 updates. Twelve candidates cannot establish a validated online reward or a closed-loop policy gain; positive association only motivates a future rollout-transfer experiment.

## Deferred work

Online gradient feedback to Astra, generator weight updates, signed-score alternatives, ALP-GMM, fixed easy-to-hard arms, independent generation-versus-scheduling ablations, larger task families, additional learner seeds, and broader language generalization are future studies. None is needed to declare this pilot complete or report a negative result.
