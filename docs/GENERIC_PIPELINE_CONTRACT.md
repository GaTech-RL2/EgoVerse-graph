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

