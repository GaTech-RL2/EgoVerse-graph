# Generic Pipeline, Training, and Evaluation Contract

Status: normative architecture contract for shared EgoVerse graph code.

This document defines what “model-agnostic” and “data-agnostic” mean in this
repository. It applies to shared pipeline execution, Lightning orchestration,
checkpoint loading, inference-artifact export, evaluation lifecycle code, and
shared visualization utilities. Existing exceptions are listed below as
migration debt; they are not precedent for adding another exception.

The objective is not to erase meaningful specialization. ARC codecs, diffusion
and flow samplers, embodiment transforms, robot adapters, and task-specific
metrics may remain specialized. The requirement is that shared code discovers
and invokes those components through declared interfaces rather than learning
their concrete classes or data layouts.

## Non-negotiable boundaries

1. **Shared code depends on capabilities, not model families.** Generic modules
   must not branch on concrete stage classes, import paths, model-family names,
   task names, robot names, or codec names. A new model must be integrated by
   configuration or a declared plugin/capability, not by another `if model ==`
   branch in shared orchestration.
2. **Every stage is a configured `dict -> dict` transform.** A stage declares
   its `reads` and `writes`, validates its own inputs, and receives semantic
   choices through constructor arguments. It must not inspect neighboring
   stages by concrete type or reach into datasets, evaluators, launchers, or
   robot state.
3. **Model YAML owns model semantics.** Observation history, native output
   representation and shape, decoder, timing semantics, and safe runtime
   controls belong to a model-owned YAML contract. The generic artifact writer
   validates and serializes that declaration; it must not reconstruct model
   meaning from a whitelist of stage classes.
4. **The data layer owns data semantics.** Dataset construction, normalization,
   source identity, sampling, episode identity, ordering, and collation belong
   to the configured DataModule or data adapter. Training and evaluation entry
   points must not inspect `resolver.key_map`, `.zarr` paths, embodiment
   registries, `index_map`, or family-specific samplers.
5. **Evaluators declare requirements explicitly.** An evaluator may require
   ordered samples, complete episodes, particular metadata, or trainer-loop
   settings, but those requirements must be exposed through the evaluator
   interface. Shared orchestration must not probe arbitrary evaluator fields or
   know the names of individual evaluators.
6. **Generic failures are fail-closed and actionable.** Contract mismatches are
   rejected before expensive execution. Unsupported models receive a precise
   compatibility result; shared code must not guess a decoder, frame, timing
   convention, normalization state, or tensor meaning.
7. **Existing specialization is not contagious.** When modifying a currently
   coupled file, do not add another hardcoded case. Either use an existing
   capability boundary or introduce the smallest reusable interface and move
   the specialization behind it.

## Stage contract

A reusable stage should satisfy all of the following:

- `reads`, `writes`, `reads_by_mode`, and `writes_by_mode` fully describe its
  graph dependencies.
- Input/output keys, dimensions, horizons, encodings, and modes are constructor
  arguments when they vary between recipes.
- It validates shape and semantic invariants locally and reports the configured
  values in errors.
- It has no knowledge of Hydra experiment names, dataset SQL filters, task
  descriptions, camera serials, station calibration, or scheduler settings.
- It does not discover another stage using `isinstance`, `_target_` strings, or
  class-name matching. Cross-stage behavior uses a declared capability, stable
  stage identifier, or an explicit object supplied by configuration.
- It does not import an embodiment registry merely to choose a model branch.
  If an embodiment-specific mapping is essential, put it in a named adapter
  stage and configure that adapter explicitly.
- Training-only diagnostics and losses write namespaced `loss/*` or `log/*`
  values. Inference writes the graph’s declared canonical output key.

Family-specific stage modules such as `stages_flow.py`, `stages_diffusion.py`,
or `stages_arc.py` are allowed to implement family semantics. Their presence is
not a genericity violation. A violation occurs when the generic runner,
trainer, exporter, or evaluator must know which family was selected.

## Model-owned inference contract

Every deployable model should declare, in YAML, enough information to map its
inference graph from current observations to a canonical action sequence:

- required observation keys and history length;
- native output key, representation, horizon, and width;
- optional configured decoder and its exact timing/representation contract;
- canonical output representation, horizon, and width;
- runtime controls the model intentionally exposes, including type, bounds,
  default, UI label, and a stable capability/setting target;
- compatibility fields whose hashes bind the artifact to the resolved model,
  tokenizer, normalizer schema, and decoder.

The generic exporter should only validate, hash, and serialize this subtree.
It should support typed controls (`integer`, `number`, `boolean`, and `enum`)
without assuming that every model has Euler steps, diffusion steps, or a
`replan_every` field. Station calibration, camera routing, prompts, safety
limits, and hardware settings remain outside the model artifact.

ARC timing is a hard compatibility boundary. New ARC representations encode
timing per waypoint or segment, normally as nonnegative interval durations.
Chunk-level mean timing cannot reconstruct dwell, acceleration, or stop-and-go
motion and must remain legacy/diagnostic rather than being silently adapted for
full-trajectory rollout.

## DataModule and data-context contract

The desired shared boundary is a configured DataModule that can provide:

- train and validation loaders;
- a model-facing data context, including normalization state and sample schema;
- a way to bind that context to its datasets and the model graph;
- validation source/group names without requiring embodiment-registry lookup;
- optional episode identity and ordering capabilities;
- configured sampler and collation factories.

The exact Python protocol may evolve, but `trainHydra.py` should eventually be
limited to composition and lifecycle calls. It should not recreate datasets,
rebuild keymaps, or call concrete `MultiDataset` methods. A self-contained
evaluation bundle should carry or reference immutable normalization state so
evaluation does not require reopening the original training corpus.

## Evaluator capabilities

Two explicit methods replace the current implicit evaluator attributes:

```python
@dataclass(frozen=True)
class EvaluationDataRequirements:
    ordered: bool = False
    complete_episodes: bool = False
    max_episodes: int | None = None
    sample_id_key: str | None = None
    frame_index_key: str | None = None
    source_fps: float | None = None


class Eval:
    def data_requirements(self) -> EvaluationDataRequirements:
        return EvaluationDataRequirements()

    def trainer_overrides(self) -> dict[str, object]:
        return {}
```

`data_requirements()` describes what the evaluator needs from validation data.
For example, a scalar teacher-forced metric can accept ordinary shuffled
batches, while an episode rollout evaluator can require ordered complete
episodes, `episode_hash`, `frame_index`, and a source frame rate. The DataModule
either satisfies those requirements or fails during preflight. `trainHydra`
does not need to know why they are required.

`trainer_overrides()` describes evaluator-owned validation-loop behavior, such
as `limit_val_batches` or `num_sanity_val_steps`. It replaces the undeclared
`override_dict` convention with an interface shared by every evaluator. It must
not choose cluster resources such as accelerator, device count, node count, or
strategy; those remain trainer/launcher policy.

These methods are a target interface, not a claim that the migration is already
complete.

## Current exceptions and migration debt

Audit date: 2026-09-23. The following are known violations to remove, not APIs
to copy:

1. `pipeline/inference_config.py` recognizes concrete Flow and Diffusion stage
   targets, `actions_cartesian`, E1 variants, a 14D Cartesian output, and
   `BimanualArcDecoder`. It currently guarantees that every run writes an
   artifact, not that every model receives a usable inference contract.
2. `trainHydra.py` constructs datasets itself, requires
   `MultiDataModuleWrapper`, rebuilds `resolver.key_map`, and owns
   `MultiDataset` normalization lifecycle.
3. `pl_utils/pl_data_utils.py` uses registered embodiment names to infer config
   shape, assumes `index_map`/`episode_path` for episode limiting, and directly
   imports the E1 anchor sampler.
4. `Eval` does not yet declare data requirements or trainer overrides, while
   `trainHydra.py` reads evaluator-specific fields and `override_dict`.
5. `eval/video.py` assumes `episode_hash`, 30 Hz input, ordered MultiDataset
   traversal, `max_size_cycle`, and a particular distributed frame stride.
6. Standalone evaluation forces one device/node and reconstructs normalization
   through the training-data path.

The weighted/homogeneous training work improves mixed-source execution and
sample-weighted reductions, but it does not remove these data-orchestration
couplings.

Recommended migration order:

1. Move inference semantics into a model-owned YAML subtree and make the
   exporter a family-blind validator/serializer.
2. Add evaluator capability methods and consume them generically.
3. Move normalization and episode/sampler behavior behind the DataModule/data
   context.
4. Parameterize shared video identity, ordering, and frame-rate behavior.
5. Remove the forced single-device evaluation policy.

## Planned integration back into EgoVerse

EgoVerse-graph is expected to be integrated back into
`GaTech-RL2/EgoVerse`. The destination must have one graph runtime, not a
permanent legacy runtime beside a graph runtime. The graph implementation is
the intended canonical model execution path; old data, processing,
visualization, robot, and user-facing capabilities remain requirements until
they are explicitly replaced or deprecated.

Audit baseline (2026-09-23):

- EgoVerse-graph `main`: `161e3a0c`;
- EgoVerse `main`: `ec5c903c`;
- this is a source/config/test capability audit, not a completed merge or a
  claim that the combined tree passes.

The merge-back rules are:

1. Port an old recipe by expressing it as `PipelineAlgo` stages and model-owned
   YAML. Do not restore `egomimic.algo.*` dispatch inside shared code.
2. Preserve old behavior until a parity test or an explicit deprecation record
   says otherwise. A missing filename in EgoVerse-graph is not evidence that
   the behavior is obsolete, and a renamed file is not proof of parity.
3. Temporary import/config aliases may ease checkpoint or command migration,
   but they must delegate to one implementation, warn clearly, have a removal
   owner, and never become a second execution path.
4. Every deployable HPT and PI recipe must contain a graph model definition and
   a model-owned inference contract from observations to canonical actions.
   Training, evaluation, and rollout must consume the same declared semantics.
5. Do not delete an EgoVerse feature during cutover until the parity manifest
   classifies it as **ported**, **replaced**, **preserved outside the model
   runtime**, or **intentionally deprecated** with an owner and evidence.

### Required HPT graph coverage

The graph repository has real HPT primitives: shared and per-domain stems,
domain embeddings, a transformer trunk, flow stages, pooled and per-token Qwen
encoders, Cartesian/ARC evaluation, and current Yam/human campaigns. That is
the engine, not yet complete EgoVerse recipe parity.

| Legacy HPT capability | Current graph status | Merge-back requirement |
| --- | --- | --- |
| Single-domain Cartesian flow for EVA and human data sourced from Aria, Mecka, and Scale | The generic graph pieces exist, but the old five recipe names/settings are not all represented | Add graph YAML for each supported data source and preserve its action width, cameras, transforms, optimizer, scheduler, and validation semantics |
| Shared-head robot/human cotraining | Current Yam/human 14D padded recipes use a shared graph head | Prove equivalence for the old shared-head recipes or add explicit graph variants; do not infer equivalence from the shared class name |
| Separate robot/human heads with different 14D/12D outputs | No equivalent configured graph recipe was found | Add a configured multi-head stage/capability if this experiment remains supported, or explicitly deprecate it |
| Pooled Qwen language conditioning | Present in current graph campaigns | Add general recipe coverage outside the current ABC campaign and verify annotation fallbacks and overlays |
| Per-token Qwen conditioning | `QwenPerTokenEncoder` exists, but no complete current recipe was found | Add and test a graph model/data/evaluator recipe |
| 138D MANO keypoint prediction in head and wrist frames | Human keypoint transforms remain, but the model, top-level train configs, and keypoint evaluators are absent | Port `train_zarr_keypoints`, `train_zarr_keypoint_wrist`, their model/data configs, and keypoint visualization through graph stages |
| EgoBridge representation alignment (OT/DTW, warm starts, representation freezing) | No graph equivalent was found | Decide whether it is active research; port it as configured loss/diagnostic stages or record an intentional deprecation |
| Pretrained HPT trunk load and partial freeze/unfreeze | No graph recipe/API equivalent was found | Add a generic weights-initialization/freeze capability if still required; do not revive family checks in the trainer |
| Direct loading of old HPT checkpoints | Intentionally not generally supported by the graph runtime | Keep immutable old-runtime reproduction or provide an explicit one-way converter; never silently partial-load |
| Rollout inference contract | Current exporter reconstructs only recognized Flow/Diffusion Cartesian cases | Put input history, native output, decoder, canonical output, timing, and typed controls in every HPT model YAML |

### Required PI graph coverage

The graph repository has `PI05Stage`, the PI policy adapter, action encodings,
prompt assembly, shared Cartesian evaluation, and several EVA, Mecka,
EVA/Aria, ABC, wrist-frame, and 6D recipes. The remaining work is recipe and
boundary parity:

| Legacy PI capability | Current graph status | Merge-back requirement |
| --- | --- | --- |
| Generic EVA PI | Graph recipe exists | Retain exact action/camera/prompt behavior and add full train-step parity |
| Generic Aria PI | Only specialized current Aria fold/gripper recipes were found | Add a graph equivalent for the old generic Aria recipe if it remains supported |
| Generic Mecka PI | Current 6D Mecka recipes exist | Verify the old Cartesian recipe or explicitly migrate it to the new 6D representation |
| Generic Scale PI and Mecka+Scale training | No equivalent current model recipe was found | Add graph recipes or record a deliberate retirement; preserve the collapsed `human_bimanual` data semantics |
| EVA+Aria cotraining | Current 6D graph recipe exists | Verify old prompt, action, camera, scheduler, and validation behavior rather than assuming name-level parity |
| Canonical dataset camera names mapped to OpenPI slots inside the model adapter | Regressed: current PI data configs and Yam keymaps can emit OpenPI names such as `base_0_rgb` | Restore one canonical dataset naming scheme and perform PI slot mapping inside the PI adapter; port the old camera-slot tests |
| Warning once when annotations are missing/empty and an empty default prompt is used | Present in old PI tests, not found in the graph adapter | Restore the warning and its tests so language-free training cannot happen silently |
| PI attention/latent extraction with random, paired, and custom episode modes plus PCA/UMAP/t-SNE/CSV rebuild | Legacy latent evaluator was intentionally not restored in the graph consolidation | Port it as a configured diagnostic provider/data requirement if still needed, or explicitly deprecate it |
| Old cosine schedule and current campaign-specific schedules | Recipes differ | Preserve schedule semantics per recipe in YAML and test resolved optimizer/scheduler values; do not normalize them accidentally during porting |
| Rollout inference contract | `PI05Stage` is not recognized by the current inference-artifact builder | Declare PI observation history, prompts/state preparation, native 32D output conversion, canonical action output, sampling steps, and replanning controls in PI YAML |

“Add graph configs for HPT and PI” therefore means more than renaming the old
files. Each retained recipe must compose to `ModelWrapper -> PipelineAlgo`,
declare its stages and inference contract, bind normalization through the
shared data context, and use an evaluator that consumes canonical graph
outputs. Dataset selection stays in data/experiment YAML, not in model Python.

### Old-feature parity findings

The following findings are based on behavior and tests in the audited trees.
They must be converted into a maintained parity manifest before cutover.

| Capability | Finding | Required disposition |
| --- | --- | --- |
| Shared Cartesian metrics, stochastic coverage metrics, coordinate reversion, and episode videos | Replaced by graph-native shared evaluators; current code is broader than the old HPT/PI evaluator split | Retain and validate metric names/units needed by existing dashboards |
| Language text in prediction overlays | Present in graph Cartesian visualization | Preserve; add end-to-end HPT and PI annotation tests |
| Standalone language-video command (`viz_language.py`) | Not found in EgoVerse-graph | Preserve the EgoVerse tool or rebuild it over the shared evaluator/video APIs |
| Keypoint training and keypoint overlays | Model/config/evaluator path missing as described above | Port before deleting the old HPT path |
| PI latent-attention analysis and latent-specific sampling | Missing by design from the prior consolidation | Make an explicit port/deprecation decision |
| HDF5-to-Zarr conversion utility | `egomimic/rldb/zarr/hdf5_to_zarr.py` is absent | Preserve it as a data tool; it must not depend on the model runtime |
| Programmatic dataset visualization notebook/script | The web `latent_inspector` remains, but the old simple script/notebook is absent | Preserve useful nonduplicate workflows or document their supported replacement |
| Local training imports without cloud packages | Regressed: graph `trainHydra.py` imports `aws_data_utils`, which imports cloud libraries eagerly; EgoVerse deliberately split `utils/env.py` | Keep the EgoVerse cloud-free loader and lazy cloud imports; restore the old import test |
| Resolver memoization (`resolve_once`) | Not found in the graph train path | Measure duplicate resolver/SQL work and either retain the optimization behind the DataModule or document why it is unnecessary |
| Installed-wheel robot resources | Regressed: graph package data omits `egomimic/resources/*` even though EVA code resolves `model_x5.xml` from the installed package | Restore package data and the wheel-content/fresh-install tests |
| OpenPI installation | Source adapter is ported, but dependency metadata is not currently solvable | Resolve the `av` dependency policy and make the lock reproducible before merge |
| Old HPT Hugging Face trunk helper and warmup/cosine helper | No direct graph counterpart was found | Replace with generic initialization/scheduler configuration where required by retained recipes; do not copy unused utilities automatically |
| Legacy algorithm/evaluator dispatch | Deliberately removed by graph `pipeline-only` tests | Keep removed at final cutover, but only after HPT/PI/keypoint/diagnostic parity gates pass |
| CI | EgoVerse has lint, full unit, and wheel jobs; EgoVerse-graph has no equivalent `ci.yml` | Use EgoVerse CI as the destination baseline and extend it for graph configs rather than dropping it |

### Merge blockers and cleanup

These are blocking or required cleanup items, not optional polish:

1. **Dependency resolution is currently broken in EgoVerse-graph.** On the
   audit baseline, `uv lock --check` fails because base `av==12.0.0` conflicts
   with OpenPI/Lerobot's `av>=14.2.0`; current `uv.lock` does not contain the
   declared OpenPI dependency. EgoVerse's lock check passes. Choose a safe
   environment policy (compatible global AV, a truly isolated PI environment,
   or documented no-dependency installation) and regenerate/verify the lock.
   Do not hide the conflict with an unverified override.
2. **The cutover test conflicts with the destination tree.** Graph's
   `test_pipeline_only_surface.py` asserts that legacy algo/evaluator files do
   not exist, while EgoVerse still needs them until parity is complete. Stage
   the merge so the test becomes mandatory only in the final removal layer;
   do not weaken it permanently.
3. **Inference export is not yet model-agnostic.** Replace the hardcoded
   Flow/Diffusion/Yam/E1 inspection before requiring artifacts from all HPT and
   PI recipes. Every deployable config must validate its own declared
   observation-to-action graph.
4. **PI camera naming leaks model semantics into data configs.** Move OpenPI
   slot mapping back behind the PI adapter before consolidating keymaps.
5. **Packaging metadata is incomplete.** Restore robot resources, build a
   wheel, install it outside the checkout, and load every packaged resource
   without relying on source-relative paths.
6. **Config coverage is incomplete.** Most graph configs are nested, while the
   general data-config test scans only top-level YAML. Add recursive discovery
   for every top-level, data, model, evaluator, visualization, experiment,
   trainer, callback, and launcher config. Strictly xfail only named,
   owner-tracked migration debt.
7. **Recipe defaults must be reconciled.** EgoVerse has working defaults and
   compatibility commands; EgoVerse-graph intentionally requires explicit
   model/data/evaluator choices. Define the destination defaults and aliases
   once, then update README, training docs, launchers, and examples together.
8. **Compatibility modules need one owner.** Retain only thin, tested aliases
   that are needed for saved configs or commands. Remove duplicate PI campaign,
   evaluator, utility, and action-encoding implementations after imports are
   rewritten to canonical paths.
9. **Checkpoint policy must be explicit.** Test strict graph resume and
   weights-only initialization. For legacy HPT/PI checkpoints, either provide
   a hash-verified converter or preserve the exact old source/environment;
   never rely on `strict=False` or shape-based guessing.
10. **Perform a real combined-tree audit.** Before recommending the final PR
    stack, run a read-only merge-tree/conflict inventory against fresh heads,
    then validate the assembled integration worktree. Per-PR green status in
    either repository is not evidence that the combined tree works.

The AV/OpenPI conflict is not fixed in this documentation change because its
safe resolution determines which runtime and binary dependency set PI users
receive. That requires an explicit environment decision; the durable
prevention is to make `uv lock --check` and the PI import/train smoke mandatory
merge gates.

### Required integration test matrix

The final merge-back is not ready until all applicable rows pass on the same
assembled source revision.

**Static and packaging gates**

- `uv lock --check`, pinned Ruff check/format, and a clean generated lock;
- recursive compose-and-resolve over every shipped Hydra YAML, including all
  nested HPT/PI models and experiments;
- instantiate graph stages, data keymaps/transforms, evaluators, and filters
  without touching SQL/S3, downloading model weights, or importing optional
  backends when they are not selected;
- validate a ready model-owned inference contract for every deployable HPT and
  PI config, and a precise fail-closed result for nondeployable configs;
- build the wheel, assert all YAML/JSON/dashboard/robot resources are present,
  install it in a fresh environment outside the checkout, and import local-only
  functionality with cloud and PI packages blocked;
- assert that surviving model configs use only `PipelineAlgo`, and at final
  cutover assert there is no second first-party policy runtime.

**CPU functional gates**

- two real optimizer steps on synthetic episodes for HPT and stubbed-PI across
  EVA, Aria, Mecka, and Scale, plus the retained Yam recipes;
- forward/inference shape, finite loss, gradient reachability, strict
  checkpoint round-trip, weights-only initialization, and normalizer binding;
- canonical PI camera-slot mapping, missing-camera masks, BHWC/BCHW handling,
  prompt assembly/fallback warnings, and all supported action encodings;
- keypoint head/wrist transforms and visualization when that path is retained;
- finite normalization in the presence of bad rows, train/validation split
  integrity, annotation collation, and source/embodiment isolation;
- evaluator capability preflight, metric units/names, ordered full-episode
  video output, language overlays, and repeated-sample metrics.

**Scheduled GPU gates**

- five-step real HPT and PI recipes for each retained data source using real
  weights/data, with at least one actual validation batch and a strict-loadable
  checkpoint;
- one multi-source HPT and one multi-source PI run, including homogeneous batch
  routing and weighted reduction where configured;
- one distributed smoke for the destination trainer/launcher semantics;
- no-hardware inference from the emitted artifact through the same graph used
  by rollout, including HPT flow controls and PI sampling/replanning controls;
- legacy-checkpoint conversion tests only for explicitly supported converters.

**Cutover gates**

- a feature-parity manifest with an owner and disposition for every old-only
  model/config/evaluator/data/tool capability;
- fixed-fixture comparisons between old and graph equivalents for input keys,
  output representation, transforms, normalization, and headline metrics;
- clean merge-tree/conflict evidence for the intended Graphite stack;
- updated installation, training, evaluation, data-tool, and checkpoint
  migration documentation;
- removal of legacy runtime files only after all retained HPT/PI/keypoint and
  diagnostic paths use the graph and no shipped config imports them.

Recommended stack order: generic contracts and dependency/packaging fixes;
HPT graph recipe parity; PI graph recipe parity; evaluator/data-tool parity;
recursive CI and GPU smoke coverage; then the final legacy-runtime cutover.
Keeping these as reviewable layers makes a missing feature visible before the
deletion layer can hide it.

## Review checklist

Before accepting a shared training/evaluation change, answer all of these:

- Can a new model use it without editing generic Python dispatch?
- Can a new dataset backend use it without pretending to be `MultiDataset` or
  registering an embodiment solely for loader routing?
- Are all varying semantics declared in YAML or a typed capability object?
- Does each stage remain independently testable from a flat input mapping?
- Does evaluation validate required metadata before the first batch?
- Are normalization, representation, horizon, decoder, and timing compatibility
  checked exactly?
- Are model/data-specific imports confined to explicitly named adapters,
  stages, evaluators, or campaign tools?
- Did the change reduce or preserve the known exception list rather than expand
  it?
