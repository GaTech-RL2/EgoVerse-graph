# Execution-horizon validation

`execution_horizon_global_dtw_v4` uses the same temporal replanning schedule
as the pointwise open-loop metric. At an observation, decode one executed
prediction prefix, score it through its control-frame duration, then obtain the
next prediction at that boundary. Observations come from the recorded episode;
this evaluates open-loop segments and does not simulate environment dynamics.

With `execute_fraction=0.30`, the baseline keeps 30 of 100 control frames.
ARC keeps 30 of 100 waypoints plus their matching rate rows. The ARC control
frame duration depends on its predicted per-arm translation/rotation clocks.
There is no additional `0.30*D` or `0.30*R` crop after this prefix. Only the
recorded episode end can shorten the executed segment. A short stored GT chunk
does not force an earlier replan: consecutive episode GT frames are recomposed
into the execution anchor's wrist frames.

Videos retain the same predicted and GT paths for every frame in that segment,
then replace both paths at the next execution boundary. Current images and
camera intrinsics remain current. Wrist/world metadata reprojects the retained
paths into a moving camera rather than attaching them to the latest wrist pose.
The renderer buffers one segment to apply the same final-episode cutoff as the
metric. `arc_video_trajectory_cap_mode=execution_horizon` is the default; old
saved video distance-cap names migrate to this behavior.

The W&B `Distance_DTW` namespace is retained for checkpoint/config compatibility,
but metadata identifies v4. V3 chose replanning observations from GT distance
milestones; v4 replans from predicted execution duration. Scores across these
versions should be distinguished by their metric version. The old explicit
`score_gt_distance_dtw_episode` helper remains available for reproducing v3.

Regression coverage includes Mx28 race/multistream/joint-distance tokens, all 30
waypoint/rate pairs reaching the decoder, speed-dependent replanning, shared
pointwise/DTW boundaries, full GT/prediction coverage, GT windows shorter than
execution duration, camera motion, held video plans, final tails, and overshoot
remaining visible beyond the former distance cap.
