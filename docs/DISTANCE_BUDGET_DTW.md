# Distance-budgeted episode DTW

`eval_open_loop_sim` enables `distance_dtw_enabled: true`. This adds a new
metric; existing timestamp-MSE metrics and execution-capped, per-frame videos
are unchanged. Video-only evaluation does not run DTW.

## Rollout

For recorded GT positions in one world frame:

```
L_GT = sum_t (||left[t+1]-left[t]|| + ||right[t+1]-right[t]||)
K_ARC = ceil(L_GT / d_exec)
```

This is the sum of both arms' travelled XYZ distances, not the distance of an
average hand position or a six-dimensional Euclidean arc length. Wrist-relative
predictions and GT commands are restored using raw observation EEF-to-world
matrices preserved by YAM/human transforms. Human head/camera motion therefore
cannot change the metric coordinate frame. Missing anchors fail explicitly.

Default **M-based execution** keeps exactly `m = M * execute_fraction`
waypoints and their per-waypoint velocities before detokenization. `m` must be
integral and at least two. Because M waypoints include both endpoints, the
nominal source-distance budget is `d_exec = D * (m-1)/(M-1)`.
For D=0.40 m, M=100, m=30 this is **0.1171717 m**, not exactly 0.12 m.
Separate **D-based execution** uses `d_exec = D * execute_fraction` with
the existing joint-distance boundary interpolation.

Select predictions at the first recorded frame reaching GT progress
`0, d_exec, 2*d_exec, ...`. If a frame crosses multiple milestones, repeated
anchors are allowed. The last chunk uses only the remaining nominal distance
budget, interpolating its terminal waypoint before detokenization (SLERP for
rotation). This final partial episode boundary is the only fractional M-prefix.
A stationary episode executes one normal prefix, so all stationary GT frames
and any hallucinated predicted motion are still scored.

Detokenize all executed prefixes on their **predicted** clocks, including
independent translation/rotation clocks in hybrid ARC, and concatenate in world
coordinates. Slower prefixes retain **more** samples than GT; faster ones retain
**fewer**. No clipping to remaining GT time or spatial scaling is applied.
Samples use the existing control-frequency detokenizer's endpoint/rounding
convention. Nominal distance budgets do not guarantee physical predicted travel.

The baseline retains 30-of-100 control-frame chunking, with the last prefix
clipped at episode end. It uses the same global DTW and weights, but need not
have the same chunk count as ARC.

These are **offline oracle-observation** rollouts: each anchor uses the recorded
observation and world pose, not a simulated observation or predicted physical
state. DTW does not itself change chunk count; the distance-budget rule does.

## One global alignment, complete coverage

Perform exactly **one endpoint-anchored, monotone DTW** between the concatenated
prediction and the complete GT episode. Local cost is mean squared error over
six XYZ coordinates, in m². Both arms share one alignment. Horizontal, vertical,
and diagonal moves cover every predicted sample and every GT timestamp,
including predictions slower than GT. No per-chunk alignment, resampling,
approximation, or alignment band is used.

DTW minimizes summed path cost. Reporting then averages all matches **within
each GT timestamp**, followed by an average over GT timestamps. This is not
path cost divided by path length; the reporting normalization is not the path
optimization objective. Dataset aggregation weights episodes by GT frame count;
an episode-macro score is also logged. Longer predictions do not gain extra
dataset weight merely by containing more samples.

`Valid/open_loop_sim/Distance_DTW/` contains:

- `XYZ_MSE`: GT-frame-weighted score in m².
- `Episode_XYZ_MSE`: unweighted average of episode scores.
- `Segments`: predictions under the new rollout rule.
- `GT_Frames`, `Predicted_Samples`, `GT_Coverage`, `Prediction_Coverage`.
- `Duration_Ratio`: predicted sample count / GT frame count. Above 1 means a
  slower rollout; below 1 means a faster rollout.

Per-episode JSON includes GT joint distance, nominal budgets, anchor frames,
per-segment sample counts, durations, and version `joint_distance_global_dtw_v1`.
Existing `XYZ_MSE`, `Segments`, etc. outside `Distance_DTW/` retain their legacy
timestamp-MSE meanings. Low DTW does not establish correct speed; retain timing
diagnostics and legacy MSE alongside it.

## Resource limits and checkpoint sweeps

Exact DTW takes O(N*M) work and one byte per cell for backpointers. Defaults:
50,000,000 cells and 30,000 samples per predicted prefix. Exceeding either bound,
or predicting moving waypoints with unusable velocity, raises an explicit error.
No GT frames or slow prediction samples are silently dropped. Raise
`evaluator.dtw_max_cells` / `evaluator.dtw_max_prediction_steps` explicitly when
the allocated host resources support it.

Sweep resume requires a completion signature matching the command, checkpoint
identity, source commit, and metric version. Legacy MSE JSON alone is not a cache hit. Old
weights work with the updated data transforms. Cached relative predictions
without observation world poses cannot directly produce the new metric.
