# Implementation manifest — Astra × EgoVerse HPT × LIBERO

**Version:** 2.0 — consolidated HPT-from-scratch plan, 27 September 2026 (UTC).

**Status:** build specification; no implementation or experimental result is claimed.  
**Scope:** one simulator, one learner, three task stages, two curriculum conditions, one training seed.  
**Authority:** this manifest and `implementation_manifest.yaml` govern implementation. The numbered documents in this folder give active implementation detail. Recovered documents outside this folder are historical background. Resolve any disagreement within the active package before running the affected work item.

## 1. The experiment to build

Build a small system in which Astra generates a pushing-task curriculum and executable teacher programs, a fully randomly initialized EgoVerse HPT VLA learns from verified demonstrations, and Astra observes the learner's results before deciding what to generate next.

**The engineering plan is fixed. The adaptive curriculum is agent-controlled.** Stage definitions, interfaces, budgets, and evaluation are fixed; stage allocations, task parameters, and decisions to advance or revisit belong to Astra. Do not implement a fixed stage schedule or an automatic mastery-threshold promotion rule for the adaptive arm.

The first result must answer two questions:

1. Can generated demonstrations teach a VLA to select the correct object and target from language in a simple pushing world?
2. Does Astra's learner-informed allocation show a useful signal against uniform sampling at the same accepted-data and update budgets?

A small gradient diagnostic tests whether the paper-inspired score is worth integrating next. It is not a third full training arm and does not gate the competence-guided curriculum.

## 2. Locked engineering decisions

| Component | Decision |
|---|---|
| Simulator | LIBERO's MuJoCo/robosuite stack, with one custom tabletop pushing environment |
| Backend name | `libero_push` |
| New task family | `AstraPush`; do not label custom results as official LIBERO benchmark results |
| Robot | One Franka Panda arm, fixed embodiment |
| Scene assets | Table, 40 mm cubes, two flat target markings, one fixed cylindrical reference marker |
| Student observations | External RGB, wrist RGB, proprioception, natural-language instruction |
| Image resolution | 224 × 224 for both cameras, with deterministic RGB scaling to [-1, 1] |
| Action space | Seven normalized `OSC_POSE` inputs: three translation, three axis-angle rotation, one gripper command |
| Control rate / horizon | 10 Hz / maximum 150 control steps; physics substeps remain backend configuration |
| Policy actions | Predict chunks of 10; execute only the first action and replan |
| Learner | EgoVerse HPT in PyTorch, single `libero_push` domain; all vision, language, trunk, and action-head weights initialized randomly |
| Training | Full-parameter training; no frozen representation, pretrained encoders, LoRA, distillation, or EMA |
| Teacher | Astra-authored typed skill program; a closed-loop pushing controller executes it using privileged state |
| Astra adaptation | Frozen model with updated context and a persisted decision archive; no generator weight training |
| Accepted training data | Successful demonstrations verified by an independent semantic evaluator |
| Data on disk | HDF5 episode capture plus JSON provenance; a direct HPT PyTorch dataset adapter; no mandatory LeRobot or RLDS export |
| Orchestration | Synchronous Python coordinator, local files, single-writer SQLite ledger |
| Simulator/trainer isolation | Two separately locked Python environments; a small local policy process exchanges versioned arrays/JSON with the simulator |
| Compute target | One GPU with at least 40 GB VRAM, batch size 4; do not silently change the model if profiling fails |
| Parallelism | One simulator worker in this pilot; no distributed learner training |
| Seed | 17, used to derive named subsystem seeds |

Do not implement Isaac Lab, ManiSkill, OpenVLA, π₀.₅, a second VLA, a replacement small-policy baseline, multi-GPU training, generated Python rewards, online VLA RL, arbitrary scene-code execution, or a generic multi-backend framework in this version.

Source/package revisions must be resolved once in work item W01 and written to a lock record. This is an explicit reproducibility task, not permission to choose another stack. Runtime Astra access and its actual model/version must also be recorded; no invented public API identifier is assumed. If required access or hardware is unavailable, stop the affected work item and report the concrete blocker.

## 3. Exactly three curriculum stages

These are selectable task families, not three fixed time intervals.

| Stage | Scene | Instruction examples | Required skill |
|---|---|---|---|
| S1 — basic push | One colored block, one target | “Push the block into the target.” | Approach, contact, push, stop |
| S2 — object and target choice | Red and blue blocks, left and right targets | “Push the blue block to the left target. Leave the red block in place.” | Bind color/object and target words to the correct action |
| S3 — spatial reference | Two same-colored blocks on opposite sides of a fixed marker, two targets | “Push the block left of the marker to the right target. Leave the other block in place.” | Resolve a spatial referent without relying on color |

Coordinate convention: “left” means negative table Y as seen by the fixed external camera. A camera calibration fixture must verify that rendering and instruction semantics agree. S3 placement guarantees exactly one block satisfies each side relation with a separation margin of 5 cm from the marker center. Block size is 4 cm. The marker is outside the teacher's direct push corridor; no obstacle routing is required.

The agent can vary initial positions, push distance in 5–15 cm, target half-width in 4–6 cm, requested referent/target, colors from the fixed catalog, and instruction realization within each stage's grammar. Validate geometry and uniqueness after every proposal. Keep physics/materials and cameras fixed across the pilot.

For S2 and S3, task success requires the requested block to be fully inside the target for 10 consecutive control steps, nonrequested-block displacement never exceeding 1.5 cm, and requested-block lift never exceeding 1 cm. S1 applies the target and lift conditions. Wrong-object, transient-goal, and lifted-object trajectories fail regardless of dense progress.

No obstacles, grasping, drawers, long-horizon sequences, new assets, or dynamics randomization are in scope. The point is to expose language dependence before increasing manipulation complexity.

### New LIBERO scenes are required, not just reset randomization

Astra emits a separate `SceneSpec` with object/fixture composition, placement-region geometry, target-region layout, and initial spatial constraints, plus a `TaskSpec` with instructions and goals. The compiler builds a registered `InitialSceneTemplates` subclass and calls LIBERO's task registration/BDDL-generation utilities. Each scene gets an immutable bundle containing scene JSON, task JSON, BDDL, registration metadata, resolved MuJoCo XML, asset hashes, preview images, and verified reset states. New primitive assets are implemented once during setup; Astra composes that fixed catalog rather than generating meshes.

The generator may compose approved noninteractive fixtures outside the push corridor, change static reference/fixture placement, and create new target and object-placement regions. Those are scene-construction choices. Sampling a different initial block coordinate from an unchanged region is reset randomization and does not count as a new scene.

Create a structural signature from the asset/fixture multiset, static geometry, target-region layout, and initial-region relation graph. Exclude names, instructions, request IDs, textures alone, and realized random reset coordinates. A new signature is a new authored scene **within this bounded simulator grammar**, not a new physics engine or evidence of unrestricted world generation.

**Before any learner training, Astra must generate six loadable novel scenes, two per stage.** All six must have unique structural signatures, differ structurally from the hand-built starter fixtures and shipped LIBERO scenes, and pass independent physical/semantic checks plus witness rollouts. At least three must change object/fixture composition or the region-relation graph as well as spatial layout. If this fails, stop and fix the generator/compiler; do not substitute stock LIBERO tasks and call the requirement met.

During the two-arm run, archive all generated scenes and report unique structural signatures, novelty yield, and duplicate/rejection rates. Require at least two structurally new scene templates per completed round; duplicate feedback returns to Astra without changing its requested stage quotas. A novelty shortfall makes the round incomplete. Full compiler contracts and checks are in [the scene and teacher specification](01-scenes-and-teachers.md).

## 4. Who controls the curriculum

### Inputs to Astra before each adaptive round

Supply a versioned JSON report with:

- Current learner snapshot ID and training round.
- Success counts and trial counts for each stage on the fixed training-control probes.
- Error counts: wrong object, wrong target, insufficient displacement, overshoot, and constraint violation.
- Change from the previous probe report and evidence of forgetting.
- Previously requested/accepted task allocations and teacher failure/yield by stage.
- Up to six fixed-budget rollout frame sequences selected by deterministic error sampling.
- Remaining data/API budget and the approved task/skill schemas.

The report contains no development or sealed-test examples. The first production version uses competence/failure feedback. Gradient evidence is collected separately at the end of the pilot.

### Outputs Astra must choose

Return one `CurriculumDecision` containing:

1. Integer accepted-episode allocations across S1, S2, and S3 summing to 75.
2. Five task templates, each assigned one of the three stages and an allocation summing to that stage's quota.
3. Scene parameter ranges and instruction/teacher programs for those templates.
4. A rationale tied to measured learner failures or uncertainty.
5. A prediction of what should improve in the next probe report.
6. Explicit `advance`, `consolidate`, or `revisit` intent for the active stages; these are logged choices, not automatic transitions.

Astra may allocate zero episodes to any stage, return to S1 after using S3, or introduce S3 early if it judges that useful. The software enforces validity and budgets, **not pedagogical choices**. The prompt asks Astra to begin with easier tasks and increase rigor as the evidence warrants, while retaining the option to revisit. It does not contain a numeric promotion threshold.

Exactly five templates can include multiple templates from one stage. Every returned template must have a positive integer allocation. The compiler samples reset states inside the selected ranges and executes the teacher until that template's accepted-data quota is met or the round attempt cap is reached.

### Guardrails and failures

Reject schema errors, unsupported stages/assets/skills, invalid allocations, impossible referents, and invalid geometry. Return those errors to Astra for repair. A validator cannot change S3 into S1 or rebalance stage quotas on its own.

Allow at most two repair responses after a failed proposal and at most eight generation calls per arm per round, including the initial call. Enforce an output-token cap of 8,192 per call. Commissioning and common seed collection each have a separate eight-call cap, giving at most 80 logical generation calls overall. Log billed transport retries separately. If a decision remains invalid, pause the run with a logged failure. Do not silently substitute a deterministic curriculum.

An actual provider timeout is retried at the transport layer at most twice, then pauses the run. Log both provider attempts and curriculum calls separately. Provider nondeterminism is recorded; replay uses saved decisions rather than assuming re-calling the model reproduces them.

## 5. Comparison condition

Run two arms from the **same complete warm-start checkpoint**, including optimizer state:

- **U — uniform:** 25 accepted episodes from each stage per round. Astra generates valid task templates/teacher programs under those quotas but receives no learner report and no previous performance feedback.
- **A — agent-guided:** Astra receives the learner report and chooses all 75 allocations and the corresponding task parameters/templates.

Both arms use the same model, teacher executor, allowed grammar, five-template proposal budget, maximum generation calls, attempt cap, learner hyperparameters, and accepted-data quota. Both use agent-generated demonstrations; the experimental difference is learner-informed task generation and allocation.

This small comparison estimates the combined effect of adaptive generation and scheduling. It does **not** isolate those components or prove the gradient mechanism. Do not introduce ALP, fixed easy-to-hard, reward-learning, or multiple-teacher training arms in this pilot.

## 6. Fixed training and data budget

1. Generate **120 common S1 demonstrations** using Astra-authored teacher programs and independently verify them. Cap seed collection at 480 attempts.
2. Warm-start the learner for **1,000 optimizer updates**.
3. Fork the complete checkpoint into U and A.
4. Run **four rounds per arm**, collecting **75 accepted demonstrations per round**, then training **500 updates**.
5. Run the training-control probes after warm-start and after each completed round.
6. Evaluate the shared warm-start model and the two final models once on the final test manifest.

Totals:

- 120 shared episodes + 300 U episodes + 300 A episodes = **720 distinct accepted episodes maximum**.
- Each final learner has access to 420 episodes, including the shared seed pool.
- 1,000 shared updates + 2 × 4 × 500 branch updates = **5,000 optimizer updates** across the experiment.
- **300 simulator attempts per arm per round**, including failed task resets, teacher failures, and accepted episodes. Together with the seed cap, collection is capped at **2,880 attempts**.

Training: batch size 4 without accumulation, learning rate `1e-4`, a 100-update warmup during the common seed phase followed by constant learning rate, AdamW betas `[0.9, 0.999]`, epsilon `1e-8`, weight decay `0.01`, gradient norm clip `1.0`, FP32 parameters/optimizer states with BF16 autocast, no EMA, and no image augmentation. These are proposed pilot settings, not a verified upstream training recipe. One GPU with at least 40 GB VRAM is the planning target, not a measured requirement. Architecture and masking details are fixed in [the learner specification](02-hpt-learner-and-data.md).

Use two independent random ResNet-18 image encoders (`weights=None`, `freeze_backbone=False`), a new random byte-level language transformer, the EgoVerse HPT trunk (width 256, 16 blocks, 8 heads), and its flow-matching action head adapted to seven-dimensional actions and horizon 10. Every component is trainable. Set all action-horizon fields consistently to 10. Do not load Qwen/T5, ImageNet, HPT, or robotics checkpoints. `pretrained: false` alone is insufficient.

Use masked flow-matching MSE on valid action timesteps. The upstream head uses an unmasked MSE; explicitly propagate the episode padding mask into a local head adapter. Match the same loss and mask in training, transfer probes, and gradient scoring. This integration must pass a real forward/backward update before data acquisition proceeds beyond commissioning.

Use the controller's bounded normalized seven-dimensional command as the canonical action and train directly in those units. Do not apply a second delta transform, convert to 6D rotation, or pad to 32 action dimensions. Record the resolved OSC controller scale, frame, quaternion convention, and gripper polarity. Normalize the nine-dimensional proprioception vector with fixed mean/std from the 30 commissioning episodes; clamp the divisor to at least `1e-6`. Scale RGB with `x / 127.5 - 1`. Preserve raw observations and actions and verify conversion round trips. Commissioning data are training-side calibration only and never evaluation data.

In the warm start, sample S1 episodes uniformly. During adaptive rounds, exactly two windows in each batch come from the latest round and two from the cumulative historical pool, episode-balanced within each pool. Historical data include the common seed data. Padding cannot cross episode boundaries. Commissioning rollouts supply normalization and validation only; they do not enter imitation minibatches.

The 30 commissioning rollouts have a separate cap of 120 attempts. Total accepted data including them are capped at **750 episodes**; total commissioning-plus-training acquisition is capped at **3,000 attempts**. The six-scene novelty gate uses these commissioning rollouts, not an unaccounted extra data pool.

If an arm cannot collect its exact 75 accepted episodes within its cap, mark that round incomplete and stop the matched comparison. Do not train on mismatched quotas, add extra attempts, or quietly replace teacher failures with another stage.

## 7. Tiny evaluation design

Freeze evaluation task definitions and simulator states before the first learner update. Evaluation definitions are independently authored from the trusted grammar; Astra does not author the sealed instructions.

| Partition | Size | Exposure |
|---|---|---|
| Training-control | 12 single-instruction reset states per stage = 36 episodes/checkpoint | Visible to Astra in aggregate and selected frames |
| Development fixtures | 4 S2 and 4 S3 instruction pairs, 2 states/pair = 32 episodes/checkpoint | Integration/semantic debugging only; no hyperparameter search |
| Sealed final test | 8 S2 and 8 S3 instruction pairs, 3 states/pair = 96 episodes/checkpoint | Final evaluation only |

The training-control bank uses fixed states disjoint from collected demonstration states. S1 starts simple but all three stages are probed from warm-start onward so the agent can see current limitations. Every counterfactual pair restores exactly the same scene and changes only the requested object/target. Use complementary feasible instructions and preservation constraints so a single indiscriminate trajectory cannot satisfy both.

The primary metric is **joint pair success**: both instructions in a same-scene pair succeed. Also report single-instruction success, wrong-object rate, wrong-target rate, constraint violations, and stage-specific results. Evaluate blank/swapped instructions only as diagnostics, never as substitutes for pair success.

Run the warm-start and both final models on the same 48 sealed pairs: **288 final-test episodes total**. Training-control evaluation is 36 × (1 + 2 × 4) = **324 episodes total**. One development check of the three models is **96 episodes**; additional debugging evaluations must be logged and cannot drive hidden hyperparameter sweeps.

With one training seed and 48 final pair instances, report counts and paired cluster bootstrap intervals over the 16 scene/instruction templates, carrying each template's three reset states together. Use 10,000 resamples with seed 17, explicitly conditional on the trained seed. These intervals do not capture training-run variability and do not support a statistical superiority claim.

## 8. Paper-inspired gradient diagnostic

After the competence-only experiment, use A's final checkpoint at branch update 2,000 and its saved branch-update-1,000 checkpoint. The gradient history for this diagnostic begins at the shared warm-start checkpoint, so the half-history window is defined on branch updates, not on an ambiguously mixed global counter.

Select **12 existing verified training tasks**, four from each stage. If an adaptive stage has no accepted data, use matching tasks from U's training pool and label that provenance. Do not generate extra candidate data. Compute the paper-form score on trainable parameters:

\[
r_\tau=\left|\nabla_\theta L_{BC}(D_\tau;\theta_{2000})^T\left[\frac{\eta}{\sqrt{\hat v}+\epsilon}\odot(\theta_{1000}-\theta_{2000})\right]\right|.
\]

Use four fixed action windows per task and average scores over two fixed flow-time/noise draws. Use PyTorch `torch.autograd.grad` and a serial named-parameter dot product over all trainable learner parameters; log the signed dot products before taking their absolute values. Reuse the same stochastic draws in candidate comparisons and transfer-loss evaluation. Log gradient norm and loss as cheap comparators. Do not implement JVP, distributed scoring, signed-score selection, or full gradient feedback to Astra yet.

For each candidate, restore exactly the same final learner/optimizer/RNG snapshot, perform **20 training updates**, then measure action-loss change on a fixed **48-window training-only transfer probe**, balanced across stages and disjoint from candidate episodes. Record rank correlation between score and loss improvement and report sensitivity to outliers. This adds **240 optimizer updates**, bringing the full pilot including diagnostic to **5,240**.

This small supervised-loss test is a feasibility signal. It does not establish closed-loop policy gain or a validated curriculum reward. If the score appears informative, the next experiment must test rollout transfer before making it an online agent input.

## 9. What counts as a sign of life

Report three outcomes separately:

1. **Learning signal:** at least one final learner solves at least six more of the 48 sealed pairs than the common warm-start model, with successful pairs in both S2 and S3.
2. **Adaptive-curriculum signal:** A solves at least five more pairs than U under the same completed budgets, and its logged decisions show that it used observed failures to change its task allocations/parameters. Include examples; do not infer adaptation from a changing histogram alone.
3. **Gradient feasibility:** the score is computable within the allotted GPU memory, numerically checked on a small fixture, and shows positive exploratory rank association with transfer-loss improvement. Report the actual association; do not claim significance from 12 candidates.

The first two pair-count thresholds are **triage rules**, not statistical claims or guaranteed outcomes. No result means “proven” with one seed. Failure to meet a threshold is reported directly; do not extend training or change task difficulty after opening the sealed test to rescue the result.

If learning is present but A and U tie, conclude that generated supervision works and agent curriculum benefit is unresolved. If only S2 improves, narrow the claim to object/target binding. If demonstrations fail, conclude the generation/control pipeline is not ready; do not attribute this to learner curriculum quality.

## 10. Ordered implementation work items

| ID | Build | Completion check |
|---|---|---|
| W01 | Repository, two environment locks, upstream revisions, Astra adapter access check | Imports, offscreen render, HPT forward pass and one full-parameter update with audited random initialization, structured provider response; all versions recorded |
| W02 | Single `AstraPush` environment and canonical action adapter | S1/S2/S3 instantiate; 20 reset seeds each are stable; action/gripper conversion fixture passes |
| W03 | Trusted semantics and frozen evaluation manifests | Wrong-object, wrong-target, lift, transient-goal, and preservation fixtures fail; correct witnesses pass |
| W04 | Scene compiler, novelty gate, closed-loop push executor and typed Astra teacher program | Six new Astra scenes, two per stage; 30 commissioning witness rollouts and frozen normalization; fresh-process bundle reload succeeds |
| W05 | HDF5 capture and direct HPT dataset adapter | One real episode round-trips with matching images/timestamps/actions; HPT loads a real batch with text and padding masks; privileged fields excluded |
| W06 | Seed collection and warm-start training | 120 verified S1 episodes, 1,000 updates, complete checkpoint, 36 control probes |
| W07 | Astra decision/report contracts and uniform baseline | Dry-run agent decisions change quotas under two contrasting fixture reports; malformed quotas fail; no mastery threshold in code |
| W08 | Four-round synchronous experiment runner | U and A complete prescribed quotas/updates or produce an explicit incomplete-run report; exact resume passes |
| W09 | Sealed evaluation and comparison report | Counts for warm-start/U/A, pair videos, decision history, accepted/failed attempts, API/compute ledger |
| W10 | Twelve-candidate gradient diagnostic | Scores, 12 × 20 branch updates, transfer-loss deltas, timings, caveats |

Do not begin W08 before W01–W07 pass. Development bugs found before the sealed test are repaired and logged. A methodological change creates a new manifest version; it is not silently treated as the same run.

## 11. Final deliverables

The pilot is complete when it has a runnable repository, locked dependencies and model ancestry, all task/decision/data manifests, three comparison checkpoints, the two-arm result table, same-scene paired-instruction videos, the 12-candidate gradient report, and an actual cost ledger.

The next study is chosen from those results. There is no automatic simulator migration, scale-up, extra arm, or continued training in this manifest. A new study gets a new specification and test partition.


## 12. Authority, interpretation, and changes from the recovered plan

This manifest and `implementation_manifest.yaml` are the active pilot specification. The companion documents explain implementation details; historical plans outside this folder do not override them. If active files disagree, stop the affected work item and resolve the discrepancy before running it.

The simulator and code structure are fixed, but the adaptive curriculum is chosen by Astra. Do not confuse deterministic artifact replay with deterministic pedagogical scheduling. A failed agent decision is a failed decision, not permission to substitute a hand-coded promotion rule.

“From scratch” covers the complete learner, including visual and language representations. Astra is still a pretrained generator, the environment is MuJoCo, and the teacher executes human-engineered control primitives. The supported claim is learning a VLA from generated supervision with no pretrained learner weights or external demonstration dataset. It is not zero-prior learning for the whole system or a reproduction of the linked paper's generator-training method.

The pretrained π₀.₅/OpenVLA choices, JAX/LoRA setup, 32-dimensional action padding, and required LeRobot export in earlier drafts are superseded. The old Astra Reversal project is unrelated to the active method. Budgets, stage definitions, novelty gate, and the two-arm comparison are retained as proposed signs-of-life defaults. No learning outcome is guaranteed at this deliberately small data budget.

The paper interpretation is inherited from the recovered research/source manifest. It was not re-fetched for this consolidation. Verify the linked version and Equation 2 during W01 before claiming an exact reproduction of the score.
