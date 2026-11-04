# Recorded-episode open-loop DTW

The current metric version is `arc_arrival_global_dtw_v7`. Older scores are
not directly comparable after changing target origin from frame t to t+1.

## Shared execution for scores and videos

For ARC, retain exactly M × execute_fraction waypoints (an integer, at least
two). Decode every stream from measured current proprio using its own clock.
Replan when the first moving stream exhausts that prefix, bounded only by the
remaining episode or an explicitly declared caller horizon. No selectable
race/joint-distance mode and no decoder-imposed hold duration remain.

The recorded observation at t predicts targets t+1 onward. At each replan,
store its native same-frame `arc.current_action` and its EEF-to-world metric
matrices. The former is the decoder's initial state; the latter restores
predictions and GT to one world coordinate system. Neither is inferred from
predicted waypoint zero. Missing ARC initial states fail explicitly.

The baseline retains same-index targets and its configured fixed control-step
prefix. This is offline recorded-observation segment evaluation—not a physical
closed-loop simulation. At each boundary the policy sees a new recorded
observation, not the physical consequence of its previous prediction.

The paired MSE and DTW paths walk the same predicted ARC chunk boundaries.
Default executed-chunk video overlays hold the actual decoded chunk while the
recorded video advances, then replace it at the next replan. The optional
`arc_video_trajectory_cap_mode: distance` crops diagnostic overlays only; the
default `execution_horizon` shows the full executed prefix.

## One global alignment

Concatenate predictions and GT in world XYZ, then compute one endpoint-anchored,
monotone DTW over all six position coordinates. The path permits horizontal,
vertical and diagonal moves. Reporting averages matches within each GT
timestamp first, then averages GT timestamps. Dataset aggregation weights by
GT frame count; an episode-macro score is also logged.

`Valid/open_loop_sim/Distance_DTW/` reports XYZ_MSE, Episode_XYZ_MSE,
Segments, GT_Frames, Predicted_Samples, GT_Coverage, Prediction_Coverage and
Duration_Ratio. Under this shared episode-time walk, the total rollout sample
count is capped at episode coverage, so Duration_Ratio is normally one; it is
not an independent policy-speed score. Inspect per-segment control steps and
paired time-indexed MSE to understand timing errors. Good DTW alone is not
proof of correct speed or successful robot behavior.

The nominal source-prefix distance remains D × (K−1)/(M−1). Source supports
span the future path; current-proprio-to-first-target travel is an additional
incoming segment, not included in that source D budget. This number is a
diagnostic, not a guarantee about predicted physical travel.

## Resources and compatibility

Exact DTW uses O(N×M) work and backpointers. `dtw_max_cells` guards the matrix
size. Recorded episode ends bound rollout steps; caller-supplied horizons also
bound all-static PW tokens, which contain no duration information.

Checkpoint sweeps require matching source, model, normalization and metric
contracts. Legacy tensors or caches cannot be loaded as arrival-v2 merely
because their shapes match. Keep old checkpoints on their recorded source.
