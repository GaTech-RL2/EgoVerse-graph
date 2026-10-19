# Repo Agent Rules

## Absolute-duration ablation

`benchmark/pusher_absolute_duration.yaml` selects one grouped-XY ARC arm at
seeds 42/43/44. Use `launch_pusher_holdout.py --campaign
egomimic/hydra_configs/benchmark/pusher_absolute_duration.yaml` for four L40S
training GPUs and one dependent evaluation shard per level per seed. This changes
only duration targets to per-stream cumulative waypoint seconds from the chunk
anchor. Keep the parent's data release, geometry, network, 240k updates, frozen
resets, simulator and fixed8 execution. Fit fresh training-only normalization.
Tests and launch evidence live in `../pusher-absolute-duration-20261018-artifacts/`.

## Pusher holdout campaign

This branch launches pusher-only co-training with triangle obstacle data held
out. See `egomimic/hydra_configs/benchmark/pusher_triangle_holdout.yaml`.
The selected sources are pinned by digest; training and normalization exclude
triangle obstacle demonstrations. Triangle clean demonstrations remain in the
primary experiment. Completely unseen-embodiment transfer is a separate study.
Use `scripts/benchmarks/launch_pusher_holdout.py` for one regime/seed workflow:
six arms, four L40S per training run, three dependent triangle evaluation shards.
The full matrix uses co-training and clean-only controls at seeds 42/43/44.
Data gates verify the immutable release, family splits and reset overlap before
allocating GPUs; each training job runs a fused four-GPU smoke before 240k updates.
The primary evaluation uses the same fixed eight-step replanning for DP and ARC.
RGB caches use lossless Blosc/LZ4 frame indexing to bound storage and decoding.
`launch_pusher_evaluation.py` recovers evaluation from completed training receipts.
It keeps the policy checkout at the training commit and pins the evaluation
harness separately. Simulator capsules must not shadow the graph's `egomimic`
package; `planar_imports.py` checks both module locations before restoring weights.
`launch_pusher_evaluation_scaled.py` partitions remaining levels across extra
L40S workers. `planar_eval_resume.py` verifies the original configuration,
checkpoint, normalizer, protocol and artifact hashes before reusing scores.
Stop the superseded evaluation workers before starting overlapping partitions;
keep their R2 artifacts. Verification-only smoke jobs never emit benchmark scores.
Keep tests, resolved configurations, source manifests and preparation receipts
outside this runtime branch, in `../pusher-holdout-20261012-artifacts/`.
Do not launch the older four-source launcher with the new pusher recipes; its
data, domain, effective-batch and evaluation assertions describe another study.

## Parent four-source Planar stream campaign

This isolated experiment branch preserves the previous obstacle comparison's
PaperConditionalUnet1D backbone, 240k updates, effective batch 128, and frozen
31-layout / 50-reset evaluation. Its nine arms are configured in
`egomimic/hydra_configs/benchmark/planar_streams.yaml`. The launcher is
`scripts/benchmarks/launch_planar_streams.py`; it uses only L40S GPUs and gates
training on the checksummed data release and a four-GPU smoke test.

The grouped layout is XY / SO(2) / continuous gripper aperture. The scalar
layout is X / Y / SO(2) / gripper. Shape columns precede the per-stream timing
columns. Gripper total variation is uncapped; duration retains zero-motion
waits, while STK has no hidden time channel. This is a planar adaptation of
the LIBERO stream findings, not a three-axis rotation ablation.

The parent campaign's tests and launch receipts remain on
`codex/obstacle-stream-validation-20261007`.
Do not edit or cancel other agents' worktrees or workflows. The training data
and evaluation artifacts are under `s3://rldb/experiments/obstacle-streams-20261007/`.
The new loader uses decoded NumPy caches, pinned family splits, source-specific
pre/post-step offsets, and one fused forward pass for equal-size embodiment
batches. Normalization is fit only on the pinned training episodes.

## Shell / Command Execution
To run commands in the interactive shell, source `emimic/bin/activate` when it
exists. In worktrees where `emimic` is absent and the checked project
environment is `.venv`, source `.venv/bin/activate` instead; verify the selected
activation file exists before running project Python tooling.

Apply this before running project Python tooling (for example: `python`, `pytest`, `pip`).

## Model settings
use plan mode for anything except extremely simple tasks

## Slurm rules
If you're on a slurm cluster, request a GPU before running or testing training.
On sky1/sky2: salloc -p rl2-lab -A rl2-lab --gres=gpu:a40:1 -c 12 --mem=30G

## Find the implementation

This is the EgoVerse graph runtime. Start with the file for the component's
role, then follow `_target_` references in the selected Hydra YAML. Folder names
under `hydra_configs/data`, `model`, and `experiment` identify recipes; they do
not imply a separate runtime for that task or model.

| Work area | Start here |
| --- | --- |
| Train/eval entry point, normalizer binding, checkpoint resume | [egomimic/trainHydra.py](egomimic/trainHydra.py) |
| Model graph execution and stage contracts | [pipeline/core.py](egomimic/pipeline/core.py), [pipeline/algo.py](egomimic/pipeline/algo.py) |
| HPT, flow, ARC, PI and other graph stages | `egomimic/pipeline/stages_*.py` |
| Neural network implementations | `egomimic/models/`; HPT stems in `models/stems/`, optional PI backend in `models/pi05/` |
| Shared Lightning training and validation hooks | [pl_utils/pl_model.py](egomimic/pl_utils/pl_model.py), [pl_utils/pl_data_utils.py](egomimic/pl_utils/pl_data_utils.py) |
| Episode loading, metadata, normalization, bounds and sampling | [rldb/zarr/zarr_dataset_multi.py](egomimic/rldb/zarr/zarr_dataset_multi.py) |
| Embodiment keymaps and coordinate frames | `egomimic/rldb/embodiment/{human,eva,yam}.py`; see [data guide](egomimic/rldb/AGENTS.md) |
| Generic bimanual ARC inputs used by E1 | [embodiment/bimanual_arc.py](egomimic/rldb/embodiment/bimanual_arc.py) |
| Pose transforms and action encodings | [zarr/action_chunk_transforms.py](egomimic/rldb/zarr/action_chunk_transforms.py), [utils/pose_utils.py](egomimic/utils/pose_utils.py), [utils/action_encoding.py](egomimic/utils/action_encoding.py) |
| Shared HPT/PI robot metrics and videos | [eval/bimanual_cartesian_eval.py](egomimic/eval/bimanual_cartesian_eval.py), [eval/video.py](egomimic/eval/video.py); see [evaluation guide](egomimic/eval/AGENTS.md) |
| ARC and tempo evaluation | `egomimic/eval/{arc_bimanual_cartesian_eval,bimanual_tempo_eval,arc_metrics,e1_metrics}.py` |
| Dataset filters, task prompts, camera/frame choices and experiment parameters | `egomimic/hydra_configs/`; calibration matrices in `hydra_configs/calibration/` |
| Shared Eva/Yam collection, local graph rollout, Zarr replay and upload | [robot/AGENTS.md](egomimic/robot/AGENTS.md), [docs/YAM_RUNTIME.md](docs/YAM_RUNTIME.md) |
| Offline normalization export for any model | [scripts/data/precompute_norm_stats.py](scripts/data/precompute_norm_stats.py), or `trainHydra.py norm_stats_only=true` |
| Focused CPU regression checks | `tests/test_pipeline*.py`, `test_hpt*.py`, `test_arc*.py`, `test_e1*.py`, `test_pi05*.py`, `test_wrist6d_roundtrip.py`, `test_robot_runtime.py`, `test_robot_graph_policy.py` |

Useful searches from the repository root:

```bash
rg --files egomimic/pipeline egomimic/rldb egomimic/eval egomimic/robot tests
rg -n '_target_:|action_mode:|coord_frame:|rotation_mode:' egomimic/hydra_configs
rg -n 'class BimanualCartesianEval|class PI05Stage|class MultiDataset' egomimic
```

## Component boundaries

- Put task selection, SQL filters, task prompts and experiment-specific numeric
  choices in YAML. Reuse the generic keymap/transform APIs; do not add Python
  modules for individual manipulation tasks.
- Models use `PipelineAlgo` and the shared Lightning/data/evaluation system.
  Keep model-specific networks and graph stages in their corresponding folders.
  Add general data, metric and video capabilities to the shared components.
- `egomimic/campaigns/pi05/`, `e1_fold.py`, and `pi05_graph_eval.py` retain old
  import names. Their implementations now live at the canonical paths above.
  Use current YAML examples when migrating older serialized configuration APIs.
- Read the relevant tests before changing normalization, pose conventions or
  checkpoint loading. Verify behavior with small CPU fixtures before a cluster
  run; hardware and pretrained-model validation are separate from those tests.

## Preserve work and verify the target

Check `git status`, `git worktree list`, and `git remote -v` before making changes.
Local worktrees can share a Git directory with EgoVerse while targeting
EgoVerse-graph; use the intended repository explicitly when creating PRs.
Preserve other people's branches, working changes, checkpoints and datasets.
Use an isolated worktree for consolidation or substantial review follow-ups.

The consolidated PR/source map is in [docs/ARC_STACK.md](docs/ARC_STACK.md) and
[docs/arc_consolidation.json](docs/arc_consolidation.json). PI environment and
graph behavior are documented in [docs/PI05_GRAPH.md](docs/PI05_GRAPH.md).
