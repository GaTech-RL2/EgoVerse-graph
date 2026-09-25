# Remaining legacy capability decisions

Owner: graph integration. This inventory refers to EgoVerse
`ec5c903c067bf1b29bbff781bc414c6df2bcf1f2`. The machine-readable path/blob map
is `parity-manifest.json`. A replacement entry identifies an implementation and
its evidence; it does not assert that GPU or assembled-tree gates have passed.

| Source capability | Graph implementation and evidence |
| --- | --- |
| Algorithm base, HPT training and multi-domain routing | `PipelineAlgo`, `ModelWrapper`, configured HPT/flow/routing stages; retained source optimizer tests, shared/separate heads, keypoints, language, and EgoBridge loss/gradient parity |
| HPT network primitives and deterministic decoder | `models/cores/hpt_transformer.py`, `models/stems/`, `models/heads/hpt_decoder.py`; core/retained-variant tests and pinned trunk output/gradient comparison |
| Denoising and flow policy wrappers | `stages_diffusion.py`, `stages_flow.py` and configured noising/loss/sampling stages; inference, optimizer and strict checkpoint tests |
| HPT evaluator and shared videos | `BimanualCartesianEval` and `EvalVideo`; source metric references, real encoding/annotation tests, ordered episode and distributed-spool tests |
| PI action converters | `utils/action_encoding.py`; all retained layouts, normalized-YPR and 6D contracts, camera masks and round trips |
| HPT pretrained trunk helper | Generic strict initialization with exact stage/module/source namespace, source hash and immutable Hugging Face revision; actual source-format trunk test below |
| Hydra output-path resolvers | Explicit `hydra.run.dir`/`hydra.sweep.dir` YAML and diagnostic output paths; recursive resolution and diagnostic export tests. Automatic naming derived from checkpoint directory substrings is not retained. Existing log directories are preserved. |
| Tensor utilities | The two imported HPT helpers live in `models/cores/hpt_utils.py`. The old wrapper's loss detachment is handled directly by `ModelWrapper`; its tests verify logged scalar handling. Other utilities have no retained first-party consumers at the audited source. |

`models/act_nets.py` has no import or Hydra target consumer in the audited
repository. `scripts/evaluation/eval.py` likewise has no consumer and contains
an abstract interface with no evaluation implementation. They are classified
as unreferenced source, not silently advertised as tested model features. The
audit covers in-repository references; unknown downstream imports are outside
that claim. Their original blobs and entire source revision remain available.
No deletion is performed by this inventory update.

The corresponding source scan is recorded in
`evidence/legacy-unused-surface-audit.json`. The tensor-utility scan finds only
the two HPT imports and `TensorUtils.detach` in the old Lightning wrapper.
The graph counterparts are explicit above; copying the unused utility collection
would create another implementation without preserving an active recipe.

## HPT trunk reference

`tests/fixtures/legacy_hpt_trunk/` contains generated CPU parameters, not a
pretrained model or historical ARC checkpoint. The probe executes the original
`HPTModel` factory and `load_trunk`, then records a masked forward pass, both
block outputs, a scalar loss, input gradients and all 28 parameter gradients.
The graph test obtains its core from the retained HPT YAML, changes only its
test dimensions, and loads the exact `trunk.` source namespace through the
generic initializer. Both local-file and pinned-Hub transport paths are tested;
only the Hub download transport is replaced by a local fixture.

The legacy core runs sequence-first internally and the graph core runs
batch-first. The reference records block outputs in the common batch-first
layout. Tolerances are 2e-6 absolute and 2e-5 relative. File hashes, constructor
arguments, seed and source revision are recorded next to the fixture.

Regenerate in a new directory using `scripts/integration/trunk_reference_probe.py`
with `--source-root` pointing at the pinned legacy checkout and that checkout on
`PYTHONPATH`. Its optional imports use the source-locked `overrides==7.7.0` and
`projectaria-tools==2.0.0`, installed outside the graph environment. The probe
refuses another source revision or an existing output directory.

For a compatible actual trunk file, the equivalent declaration is:

```yaml
pipeline:
  initialization:
    - stage_id: trunk
      module_path: trunk
      source: /absolute/path/to/trunk.pth
      source_prefix: trunk.
      sha256: <full-file-sha256>
```

This block belongs inside the selected model configuration and requires its
declared `trunk` stage. For Hub input, replace `source` with a mapping containing
`repo_id`, `filename`, and a full immutable `revision`. Shapes, keys and dtypes
must match exactly. Freeze/unfreeze uses the separate explicit trainability
schedule; no architecture is guessed from a checkpoint.

Real-weight/data GPU execution and combined-tree verification still block
cutover. In particular, a complete inventory with CPU evidence cannot authorize
removing the destination's legacy runtime.
