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
