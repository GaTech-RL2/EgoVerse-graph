# OSMO obstacle co-training comparison

This campaign co-trains one model across corrected clean U-Socket (2,999
episodes) and ChainGripper (3,000 clean plus 1,920 obstacle episodes). Each
optimizer step contains one batch from each embodiment and averages the two
source losses. ChainGripper clean/obstacle sampling follows their frame counts.
All five recipes use the same split, seed 42, normalizer procedure, optimizer,
PaperConditionalUnet1D architecture, observation horizon 2, causal targets,
240,000 updates, EMA settings, and effective batch size 128 (32 per embodiment
per GPU, two GPUs). Validation is deferred; training loss is not a policy score.

| Recipe suffix after `planar_v2_cotrain_obstacle_` | Target |
| --- | --- |
| `paper_dp` | 16 native actions in common-five coordinates |
| `arc_duration_D80_M16_R26deg_paper` | 16 seven-channel supports with durations |
| `arc_duration_D80_M56_R26deg_paper` | 56 seven-channel supports with durations |
| `arc_stacked_D80_M16_R26deg_paper` | 16 seven-channel supports with local velocities |
| `arc_stacked_D80_M56_R26deg_paper` | 56 seven-channel supports with local velocities |

All ARC allocations are uniform. D80 is the translation-distance budget, not
80 executed actions. The separate native future window is 80 controls; R26 is
26 degrees. Translation and rotation have independent clocks. Grip is sampled
on the translation clock. Duration supports reserve stationary boundaries
within their fixed support budget; purely stationary grip changes retain time.
The stacked-velocity representation cannot uniquely encode the duration of a
zero-distance interval. That limitation is part of this ablation. The proposed
lead candidate is M56/duration; no closed-loop advantage has been established.
Baseline execution is the first 8/16 future controls. ARC execution uses half
the geometric supports with predicted clocks determining the native prefix;
closed-loop evaluation must implement and record this rule, not substitute a
fixed number of native ticks for half the support count or distance budget.

The recipes and seven-channel codec originate in ElmoPA's commit
`c642bb81750a52d1e7858098803cdbf7dc0f611f` (PR #77). This port retains the newer
causal target slicing and duration hold handling. It does not replace the
existing five-channel ARC implementation. Decoder output starts at its causal
anchor; moving intervals with unusable predicted clocks cannot teleport.

## Corpus identity

The source data is under
`s3://rldb/datasets/pushshapes/arc-duration-handoff-20260916/`.
Clean corpora retain the previously audited content manifests. Obstacle data
comes from `chain_gripper_obstacle_mimicgen_128_20260826_output_128/`, selecting
episode indices 0–63 from each of levels 1–30, including manual and generated
episodes. This is a new pinned 1,920-episode cohort. Its identity has **not**
been verified against the historical `frozen1920` folder. The older recipe's
scripted U-Socket corpus is also replaced by the corrected handoff corpus.

The source-pinned split is
`egomimic/hydra_configs/data/pusht/planar_v2_cotrain_obstacle_7919_split_seed42_v1.json`:
U-Socket 2,970 train / 29 validation; ChainGripper 4,871 train / 49 validation.
The combined ChainGripper split is newly drawn over its 4,920 episodes, so it
does not preserve the prior clean-only ChainGripper validation assignment.
It is episode-disjoint, not grouped by source demonstration; generated siblings
can appear across this split. Generalization claims require fresh closed-loop
seeds, not these validation reconstruction scores alone.

All runs start from step zero. No historical checkpoint is used for
initialization. OSMO launchers verify the data archive, record resolved config
and normalizer hashes, smoke two-GPU training before the full run, and upload
checkpoints under content-addressed paths. Existing datasets, branches and
checkpoints are preserved.

## Observation/action alignment

Clean episodes store the observation before executing the same-index action;
obstacle episodes store the observation after that action. Physics replay of
manual and generated obstacle episodes confirms the latter convention. With
two observation rows, clean targets start at loader action index 1 and obstacle
targets at index 2. Both therefore begin with the action following the latest
observed state. Slicing happens before either DP encoding or ARC tokenization.

`CompositeEpisodeResolver` merges the two ChainGripper sources while retaining
their key maps and transforms, before applying one pinned episode split.
The launch creates `chain_clean` and `chain_obstacle` symlink views of the
unchanged source bytes. Per-source offsets and storage conventions are recorded
in run provenance. Live rollout always executes decoded action index zero.
