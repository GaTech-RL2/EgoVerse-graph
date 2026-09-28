# Bimanual ARC chunking modes

Every retained `abc_arc/robot_bc` and `abc_arc/human_bc` visual experiment
accepts `abc.arc_chunking_mode=race`, `multistream`, or `joint_distance`.
The shared default is `joint_distance`. ARC recipes use the hybrid action
representation, a positive rotation target R, and `per_waypoint` timing.
Cartesian baseline recipes accept the override for evaluation and provenance;
their training targets remain Cartesian.

Supported experiment entrypoints are listed in
`egomimic/hydra_configs/experiment/abc_arc/README.md`. Unreferenced legacy ABC
Qwen, diffusion-policy, non-hybrid, and cotrain templates were removed; the
historical versions remain recoverable from Git. Other experiment families and
the data component still consumed by `experiment/abc` are unchanged.

Retained recipes default to `norm_stats.sample_frac=0.10`. Cached statistics are
immutable artifacts: this default does not convert a historical 20% cache into
a 10% cache. Use a verified matching cache or compute a new one. An intentional
historical-cache launch must explicitly declare its actual sampling fraction;
the loader's ARC action-contract check is not a dataset/sampling provenance check.

| Mode | Translation target and timing |
| --- | --- |
| `race` | End when the first arm travels D. Interpolate each arm by its own distance through that shared end time. |
| `multistream` | Each arm targets D with its own translation clock and resamples its available path to M waypoints. An arm with less than D travels its available distance, then holds its last absolute pose. |
| `joint_distance` | Sum both arms' travel to D using one shared translation clock. |

For example, with D=40 cm and M=100, an arm that travels only 10 cm within
the bounded source window resamples that actual 10 cm across **all 100 motion
waypoints**. There is no null-waypoint grid padding. Detokenization uses the
predicted velocities to execute the 10 cm, then holds the last absolute pose
through the remaining bounded rollout horizon. The two arms resample their
own paths independently to the same M.

All three modes keep the existing **separate R target and independent joint
rotation clock**, which sums both arms' angular travel. Translation reaching D
does not truncate the raw rotation trajectory.

The generic Zarr horizon resolver reads the bounded native source buffer and
selects enough rows to cover both translation and rotation endpoints. A missing
endpoint uses the available bounded prefix. The default buffer remains 600
frames; episode bounds and the existing repeat-last tail behavior still apply.
Explicit Human ARC modes retain native rows and use `dt=1/30`, bypassing legacy
stride/interpolation. Their untokenized evaluation copy has 100 native rows,
with repeat-last padding at short tails.

The public embodiment APIs and tokenizer use `arc_chunking_mode`. Omission
remains `None` when forwarded to the codec: R present infers `joint_distance`;
no R infers `multistream`. Any explicitly supplied mode requires a positive
finite R and `velocity_mode=per_waypoint`. Legacy omitted-mode no-R APIs remain
usable. Human configs that omit both mode and R retain
their existing fixed source window and stride settings.
Old no-R resolver specs with `require_all_arms: false` retain race selection.

The shared visual recipe saves representation, mode, D, R, waypoint count,
timing mode, and control interval in `run_provenance.action_contract`, which
training retains in Lightning checkpoint configuration. The full Hydra config
also records the mode passed to each resolver, transform, and evaluator.

Changing modes changes action semantics even when tensor shapes are equal.
The existing cache had no transform-config hash. Retained ARC recipes now pass
their resolved action contract from `trainHydra` to normalization. New caches
store it per embodiment in `provenance.action_contracts`; normalizer state also
retains it across checkpoint/state round trips and cache rewrites.

When loading precomputed statistics, a tagged contract must match the requested
representation, mode, D, R, waypoint count, timing mode, and control interval.
A mismatch raises before any statistics are applied. Race and multistream
reject untagged caches. Existing normalization-mode and key-set checks still
apply.

Legacy **joint_distance** caches without action-contract metadata remain
eligible, with a warning, so established compatible runs do not need to
recompute normalization. Their existing data, D/R, and representation
compatibility must already have been verified; missing metadata cannot prove
those facts. Baseline and old configurations without an ARC contract retain
their prior behavior.

For migration to race or multistream, compute statistics with that recipe into
a new cache directory (omit `norm_stats.precomputed_norm_path` for that pass),
then use the resulting `norm_stats/norm_stats.json`. Do not relabel a joint
cache as a different mode. Existing compatible joint caches can continue using
their current path; a subsequent cache write records the configured contract.

External caches of transformed targets and resume signatures must also
distinguish the complete action contract. Pure image feature caches need no
action-mode distinction if they contain no transformed actions or
action-dependent sample selection.
