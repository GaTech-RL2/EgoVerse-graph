# 01 — Scenes, task semantics, and teacher execution

## Environment contract

Implement one custom tabletop pushing environment named `AstraPush` on LIBERO's MuJoCo/robosuite stack. It is a new task family using LIBERO infrastructure, not a result on the official LIBERO benchmark. Use a Franka Panda, fixed physics and cameras, 224 × 224 external/wrist RGB, and a 10 Hz controller with a 150-step horizon. Keep the gripper closed for pushing after the commissioning polarity check.

The student receives images, a natural-language instruction, and measured robot proprioception. The teacher and evaluator additionally receive object poses, target geometry, contacts, and reset state. These privileged fields never enter a student batch or policy request.

Use a bounded primitive asset catalog: 40 mm red/blue cubes, flat noncolliding goal markings, a fixed cylindrical reference marker, and at most one approved static reference fixture outside the push corridor. No generated mesh, arbitrary Python scene code, arbitrary reward function, grasping, drawers, or obstacle navigation is part of this version.

## Three stages

| Stage | Scene | Language requirement | Example |
|---|---|---|---|
| S1 | One cube, one goal marking | Basic instruction-conditioned pushing | Push the block into the target. |
| S2 | Red and blue cubes, left/right targets | Select requested color and destination; preserve the other cube | Push the blue block to the left target. Leave the red block in place. |
| S3 | Two same-colored cubes on opposite sides of a marker, left/right targets | Resolve the spatial referent and destination; preserve the other cube | Push the block left of the marker to the right target. Leave the other block in place. |

Left means negative table Y in the calibrated external-camera view. Verify this with rendered labels before collecting data. S3 cubes must be separated from the marker's centerline by at least 5 cm and have exactly one valid referent per instruction. Choose the marker position so it does not obstruct either requested direct push. Reject geometries that cannot support both instructions of an evaluation pair.

Allowed difficulty controls are push distance 5–15 cm, target half-width 4–6 cm, initial regions, relative placements, asset composition within the catalog, and grammar-valid language. Keep camera, friction, robot, and controller settings fixed. Astra may select any stage at any adaptive round; these are task families rather than a time schedule.

## SceneSpec, TaskSpec, and the compiler

`SceneSpec` describes assets and fixed fixtures, placement regions and their relations, target geometry, camera/controller references, and initial constraints. `TaskSpec` references an immutable scene hash and specifies a semantic referent, destination, language realization, preservation constraints, and typed teacher program. Validate units, bounds, unique names, unambiguous referents, reachability margins, collision-free reset geometry, visible targets, and direct push corridors before simulation.

Translate an accepted scene into a trusted registered subclass of LIBERO's `InitialSceneTemplates`. Use its scene registry and `register_task_info` / `generate_bddl_from_task_info` to generate BDDL. Generated text is data consumed by fixed compiler code. The language model does not supply executable Python. Primitive asset registrations and custom environment support are implementation work, not already available stock LIBERO functionality.

BDDL expresses the environment's objects, regions, and initial/goal relations. Temporal hold, lift, and preservation rules belong to an independent runtime evaluator; BDDL completion alone is insufficient.

Archive each immutable bundle with:

- Original scene/task JSON and canonical hashes.
- BDDL and scene registration metadata.
- Resolved MuJoCo XML, asset hashes, package revisions, controller and camera configuration.
- Verified reset states, preview images from both cameras, and semantic calibration evidence.
- Teacher programs, witness trajectories, validation outcomes, and rejection reasons.

A fresh process must restore a bundle without depending on unrecorded registry state from generation. A modified task or controller creates a new bundle hash.

## Genuine novelty gate

Compute a canonical structural signature from asset/fixture composition, static geometry, target-region layout, and the initial-region relation graph. Canonicalize object names and geometric precision; ignore arbitrary IDs, renamed tasks, instruction wording, textures alone, and realized random reset positions. Freeze the numeric canonicalization precision during W01 and save the algorithm version.

Before learner training, require six agent-authored bundles, two per stage, with distinct signatures that differ from the starter fixtures and shipped LIBERO scenes. At least three must alter composition or the region relation graph in addition to layout. For stock-scene comparison, use a canonical inventory of asset types, regions, and relations; record any unsupported stock-scene representation instead of claiming it was compared successfully.

Each bundle must load, render, survive reset stability checks, and support successful witnesses. W04 collects 30 accepted commissioning demonstrations, ten per stage and five per gate scene, within 120 total attempts. These supply the novelty witnesses and proprioception statistics; they never enter imitation minibatches. W02 separately checks 20 reset seeds per stage, recording failures and simulator time as engineering checks.

Every collection round needs at least two new structural templates. A shortfall is an incomplete round. Return duplicate explanations to Astra; do not silently relabel layouts or change its allocations. Common seed collection uses agent-authored S1 bundles from the approved catalog and does not introduce another unbudgeted novelty gate.

## Teacher supervision

Astra writes a typed skill program using a fixed API: `approach`, `align_behind`, `contact`, `push_toward`, and `settle`. Each call names valid semantic objects/targets and bounded numerical parameters. No free-form control code is executed. A closed-loop controller converts this program into normalized OSC commands using privileged state.

This is agent-authored high-level supervision with engineered low-level control. It is not evidence that Astra directly predicts accurate continuous motor actions. Testing raw Astra action sequences would be a separate experiment.

Controller behavior is fixed across arms. Log commands before clipping, executed commands, saturation counts, contacts, controller state, and all failures. A reset or teacher attempt consumes an attempt slot even when no accepted episode results. A reset failure must not enter a retry loop outside the budget.

## Independent success checker

Accept an episode only if the requested block lies fully within the destination for ten consecutive control steps. Throughout the rollout, its lift above initial height must not exceed 1 cm; in S2/S3 the nonrequested cube must never move more than 1.5 cm from its initial position. Record maximum violations over the entire trajectory, not just the final state. Dense distance progress is debugging feedback only.

An independent implementation resolves the instruction semantics and measures success. Validate it using correct witnesses and adversarial wrong-object, wrong-target, transient-goal, lift, and move-then-return trajectories. Failures remain visible to the curriculum and cost ledger, but become neither successful demonstrations nor positive imitation labels.

## Risks this gate resolves

Scene novelty does not imply physical solvability. A generated goal can be valid BDDL but unreachable, visually ambiguous, or outside the controller's capability. Conversely, a successful scripted trajectory does not demonstrate learner competence. Track compiler acceptance, stable resets, teacher success, scene novelty, and student success separately so a failed pilot can be diagnosed.
