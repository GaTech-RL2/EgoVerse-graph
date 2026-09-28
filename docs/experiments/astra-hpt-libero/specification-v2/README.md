# Astra-guided curricula for an HPT learner trained from scratch

**Consolidated plan · version 2.0 · 27 September 2026 UTC**

Build a small LIBERO experiment in which Astra creates novel pushing scenes, instructions, and teacher programs; a randomly initialized EgoVerse HPT policy learns from verified demonstrations; and Astra uses the learner's results to choose its next curriculum. Compare this against uniform stage allocation with the same data and training budgets.

This folder is the current planning package. It contains specifications and illustrative contracts, not an implemented simulator, trained model, or experimental result.

## Reading order

| Read | Document | Purpose |
|---|---|---|
| 1 | [Implementation manifest](00-IMPLEMENTATION-MANIFEST.md) | Authoritative decisions, budgets, milestones, and completion criteria |
| 2 | [Scenes and teachers](01-scenes-and-teachers.md) | Three stages, genuine scene generation, teacher execution, independent success checks |
| 3 | [HPT learner and data](02-hpt-learner-and-data.md) | Fully random initialization, fixed architecture, action/language integration, dataset contracts |
| 4 | [Curriculum and feedback](03-curriculum-and-feedback.md) | Agent authority, uniform comparison, reports, paper-inspired gradient diagnostic |
| 5 | [Codebase and execution](04-codebase-and-execution.md) | One repository layout, process boundaries, work items, acceptance gates, resources |
| 6 | [Evaluation and reporting](05-evaluation-and-reporting.md) | Same-scene instruction pairs, split isolation, metrics, stopping rules, budget accounting |
| 7 | [Sources and decision history](06-sources-and-decisions.md) | Evidence, inspected upstream files, historical changes, unresolved implementation checks |

The [machine-readable manifest](implementation_manifest.yaml) mirrors the numerical decisions. [Examples](examples/README.md) define the proposed data and agent contracts. Neither is a drop-in upstream Hydra configuration.

## Fixed choices

- **Environment:** custom `AstraPush` tasks on LIBERO/MuJoCo/robosuite; Panda robot; two RGB cameras; seven normalized OSC actions.
- **Learner:** EgoVerse HPT with random vision, language, trunk, and flow action-head weights; full-parameter training.
- **Curriculum:** S1 basic pushing, S2 object/target selection, S3 spatial-reference grounding. Astra chooses when to advance, consolidate, or revisit.
- **Study:** 120 common seed demonstrations, then four rounds of 75 demonstrations per arm; one training seed and two arms.
- **Main evidence:** successful instruction changes in identical physical scenes, plus the matched adaptive-versus-uniform comparison.
- **Gradient mechanism:** a small post-run diagnostic, not an online curriculum dependency in this pilot.

## Start here when implementing

Read the manifest and learner specification first. W01 must prove that the complete learner initializes without pretrained weights, that the simulator renders, and that Astra can return the structured contract. Before seed training, six genuinely new scene bundles must load and have successful witness trajectories. Only then collect the shared seed data and run the two arms.

All gates are currently **not run**. The fixed numerical choices are proposed pilot defaults; they do not imply tested throughput or guaranteed learning. Any method change creates a new manifest version before sealed testing.

## What changed from the recovered documents

The previous OpenVLA and π₀.₅ configurations have been replaced throughout this active plan. The older research documents and Astra Reversal plan remain in the workspace's historical `docs/` directory. They are referenced for provenance and future ideas, not as competing instructions for this pilot. See [decision history](06-sources-and-decisions.md).
