# Evaluation navigation

The root AGENTS.md applies here. Configure evaluators in
`../hydra_configs/evaluator/`; use the same graph contract for every robot model.

- `eval.py`: evaluator lifecycle interface.
- `bimanual_cartesian_eval.py`: shared HPT/PI Cartesian evaluation over
  `model.forward_eval(batch)[source]["pred_action"]`. It owns normalization,
  validation-group names, pose scores, optional repeated-sample scores and overlays.
- `video.py`: shared `EvalVideo` episode/chunk buffering, file output, distributed
  playback FPS and WandB video logging. Only rank zero buffers/writes videos.
- `cartesian_metrics.py`, `distribution_metrics.py`: model-independent pose,
  DTW, Fréchet, reverse-KL and coverage calculations.
- `arc_bimanual_cartesian_eval.py`, `arc_metrics.py`: ARC-specific decoding and
  trajectory alignment layered on the shared Cartesian evaluator.
- `bimanual_tempo_eval.py`, `e1_metrics.py`: tempo/duration scoring.
- `planar_action_eval.py`, `synthetic_trajectory_eval.py`: nonrobot evaluation.
- `checkpoint_loading.py`: strict graph checkpoint restoration and EMA selection.

PI-specific backend calls do not belong in the evaluator. Its graph stage emits
the same normalized `pred_action` as HPT. Validation groups use `Valid/` for
`valid`, and `Valid_<group>/` otherwise. Group limits and repeated-sample counts
are YAML settings, including the `train_viz` group.

Frame reversion must match the data configuration. Rotation encodings and
gripper padding affect both actions and proprio; test both sides of the round
trip. Focused regression tests are `test_shared_robot_components.py`,
`test_eval_overlay_annotations.py`, `test_wrist6d_roundtrip.py`,
`test_pi05_graph.py`, and the root ARC/tempo evaluation tests.

## Generic evaluator contract

Read `../../docs/GENERIC_PIPELINE_CONTRACT.md` before changing shared evaluator
or video infrastructure.

- Evaluators expose data needs through `data_requirements()` and validation-loop
  settings through `trainer_overrides()`. These are target interface methods;
  when migrating current code, remove implicit `override_dict` and
  evaluator-field probing rather than adding more probes.
- Data requirements describe ordering, complete episodes, sample/frame identity,
  episode limits, and source frame rate. The DataModule satisfies or rejects
  them before validation; shared training code must not know the evaluator’s
  concrete class.
- Trainer overrides may control evaluation-loop behavior such as batch limits
  or sanity steps. They must not select devices, nodes, accelerators, or cluster
  strategy.
- Shared video code receives episode identity, ordering semantics, and frame
  rate explicitly. Do not hardcode `episode_hash`, 30 Hz, MultiDataset order,
  or a particular distributed sampler in a generic video base.
- Model-specific metrics and diagnostics belong in configured evaluator/provider
  plugins. Shared evaluator lifecycle code dispatches through interfaces, not
  concrete model or stage classes.
- During the EgoVerse merge-back, preserve retained keypoint visualization and
  PI latent-analysis behavior as configured evaluator/diagnostic providers with
  explicit data requirements. Do not reintroduce HPT/PI family dispatch in the
  shared evaluator base to recover a legacy feature.
