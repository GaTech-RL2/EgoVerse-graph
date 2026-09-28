# 05 — Evaluation, budget accounting, and reporting

## What the experiment can establish

The first question is whether generated supervision trains the random HPT learner to use language in a bounded pushing world. The second is whether learner-informed generation/allocation improves that outcome under matched accepted-data and optimizer-update budgets. The gradient diagnostic asks only whether a historical-gradient score predicts a small supervised transfer-loss change.

Do not replace instruction grounding with training loss, teacher success, or success on a scene where only one action is feasible. One seed and this small dataset may produce no learning signal; that is a valid pilot outcome.

## Frozen partitions

| Partition | Definition | Number per checkpoint | Provider exposure |
|---|---|---:|---|
| Training-control | 12 single-instruction states per stage | 36 episodes | Aggregate counts and up to six selected frame sequences |
| Development | Four S2 and four S3 pairs; two reset states per pair | 32 episodes | None |
| Sealed test | Eight S2 and eight S3 pairs; three reset states per pair | 96 episodes = 48 pair instances | None |

Author development/test tasks independently from the trusted grammar; Astra does not author sealed instructions. Freeze definitions, reset states, instruction realizations, source revisions, and split hashes before the first production learner update. Use structural scene-family hashes and lineage grouping to keep generated variants of a held-out scene out of training. Control probes are also reset-state-disjoint from demonstrations.

Same asset classes and bounded grammar may recur across splits. Novel test scene layouts/combinations and disjoint linguistic realizations do not imply unseen physics or open-domain semantics. State the exact holdouts. Reject train proposals that collide with sealed structural families through a split firewall without disclosing held-out content to Astra.

For each counterfactual pair, restore the identical full initial simulation/controller state and change only the instruction. Choose complementary feasible referents/targets with preservation constraints; a single indiscriminate trajectory must not solve both. Evaluate with frozen per-pair policy-noise seeds shared across compared checkpoints, recording the seeds and executed actions.

## Evaluation schedule

Run control probes on the shared warm-start checkpoint and after each of four rounds in each arm: `36 × (1 + 2 × 4) = 324` episodes. Run one development check of the shared warm-start and both final models: `32 × 3 = 96` episodes. Run the three models on the same sealed bank once: `96 × 3 = 288` episodes. The planned evaluation total is **708 episodes**, separate from acquisition attempts.

Additional engineering/development evaluations are allowed to diagnose bugs before sealing results, but must be logged and must not become hidden hyperparameter search. A methodological change requires a new manifest/run version. Do not open sealed results and then extend training, simplify tasks, change thresholds, or adjust curricula to rescue the same run.

## Metrics and uncertainty

Primary: **joint pair success**, the count of the 48 sealed pair instances where both instructions succeed. Also report per-instruction success, per-stage counts, wrong-object/target outcomes, maximum preservation/lift violations, and completion steps. Include paired videos showing the same initial scene under different instructions.

The 48 pair instances arise from 16 scene/instruction templates with three resets each. Use a paired cluster bootstrap over those 16 templates, carrying all three resets and all compared checkpoints together. Report the bootstrap unit, number of resamples (10,000), fixed seed, and percentile interval. This addresses correlated resets within templates; it does not estimate training-seed variability. Always include raw counts and avoid statistical superiority claims from one training seed.

Blank or swapped instructions can reveal language insensitivity but are diagnostic only and require a separately logged evaluation budget. They do not replace counterfactual pair success and are not included in the 708-episode core total.

## Acquisition and training accounting

| Phase | Accepted demonstrations | Attempt cap | Optimizer updates |
|---|---:|---:|---:|
| Commissioning / normalization / novelty witnesses | 30, excluded from imitation training | 120 | 0 production updates |
| Common S1 seed pool | 120 | 480 | 1,000 |
| Uniform arm, four rounds | 300 | 1,200 | 2,000 |
| Adaptive arm, four rounds | 300 | 1,200 | 2,000 |
| Gradient branches | Reuse existing data | No new acquisition | 240 |
| **Total** | **750 including commissioning; 720 imitation episodes** | **3,000** | **5,240** |

Each final learner sees 420 imitation episodes: 120 common and 300 arm-specific. Do not double-count the shared seed pool when reporting distinct acquisition. A partial round does not satisfy the per-round budget; if either arm cannot obtain its exact 75 successful episodes, the matched comparison is incomplete.

Generation caps: eight logical calls each for common commissioning and seed collection, plus eight per arm/round, totaling 80 logical calls maximum. At 8,192 output tokens per call, the logical response cap is 655,360 output tokens; input tokens and billed transport retry usage are additional and must be reported separately. This is not a price estimate. Report per-arm actual API tokens, compute, simulator attempts, rejected scenes, failed witnesses, latency, and storage.

## Signs-of-life decision rules

1. **Learning signal:** a final arm solves at least six more of 48 pairs than warm-start, with successful pairs in both S2 and S3.
2. **Adaptive signal:** A solves at least five more pairs than U at the completed matched budgets, and the saved decision rationales show that measured learner failures changed allocations or task parameters.
3. **Gradient feasibility:** scores fit memory, pass the numerical check, and show positive exploratory rank association with supervised transfer-loss improvement.

These thresholds are triage rules inherited from the earlier pilot, not power calculations or guaranteed improvements. A tie with warm-start is a negative learning signal at this budget. Learning in both arms with no A–U difference supports generated supervision while leaving curriculum benefit unresolved. S2-only gains support a narrower color/target-binding claim. Failed teacher generation identifies a supervision pipeline issue, not a learner-curriculum result.

## Final report artifacts

Provide the manifest and lock hashes; initialization audit; six-scene gate evidence; source and rejection archives; accepted-data and attempt counts; warm-start/U/A complete checkpoints; stage allocation charts and evidence-linked agent decisions; raw evaluation outcomes and cluster intervals; paired videos; gradient-score/transfer table; and actual resource ledger.

Separate engineering validity, teacher generation, learner learning, adaptive benefit, and gradient evidence. Include missing gates and incomplete rounds. The report must not describe a planned artifact, illustrative JSON, or stub implementation as measured evidence.
