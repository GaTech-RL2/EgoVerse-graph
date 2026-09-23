# Goal-conditioned Q-chunking and ARC

`egomimic/trainRL.py` runs transition-based offline RL through the existing
`PipelineAlgo`, `ModelWrapper`, and Lightning optimizer/checkpoint lifecycle.
The base configuration is `hydra_configs/benchmark/dqc.yaml`; task selection,
published hyperparameters, seeds, and replay calibration grids live in
`hydra_configs/benchmark/dqc_suite.yaml`.

Use an isolated Python 3.11 environment with `requirements-goal-rl.txt` for the
audited benchmark. Numerical parity tests additionally use JAX 0.4.38, Flax
0.10.4, Distrax 0.1.5 and ml-collections 1.1.0 with the pinned upstream checkout.

This implements **Decoupled Q-chunking**, the repository explicitly requested:
https://github.com/ColinQiyangLi/dqc at
`df898256a77f3594b54a7268bd5f89915981da35` (MIT; `third_party/dqc/LICENSE`).
It is the offline, goal-conditioned method from arXiv:2512.10926. The original
arXiv:2507.07969 paper instead uses offline-to-online QC; those online experiments
are not claimed by this offline runner.

## Comparison

- **native_reference**: five-action DQC with policy horizon 5, critic horizon 25,
  batch 4096, width 1024 x 4, two Q heads, flow 10 steps, best of 32, gamma .999,
  Adam 3e-4, one million optimizer updates, 50 episodes per evaluation task.
- **native_window**: retain every native control within a bounded spatial
  window and predict its native duration. This controls for the changed
  replanning boundaries separately from compression.
- **arc**: same bounded window as native_window, uniformly spaced curve
  supports plus per-interval duration. Only the short policy representation
  changes between these two arms. Both distill from the unchanged native
  25-step critic. Network input/output sizes necessarily differ with encoding.

Do not attribute a difference between ARC and native_reference solely to the
tokenizer: their chunk boundaries also differ. Report the matched native_window
comparison alongside the fixed five-action reference.

The paper's selected configuration (arXiv:2512.10926, Table 7) uses policy
horizon **5** for Cube triple/quadruple/octuple and Puzzle 4x5, but **1** for
Humanoid giant and Puzzle 4x6. All six use critic horizon **25**. The existing
all-horizon-5 baseline is one of the paper's ablations, not its selected
configuration for every task. These are native environment actions executed
before replanning; they are not ARC waypoint counts. Keep that distinction in
result labels and include the selected reference in future per-domain studies.

The manipulation action space is relative XYZ (0.05m/unit), yaw (0.3rad/unit),
and relative gripper opening. ARC integrates the relative commands, samples
the resulting control path, and differentiates the decoded native-rate path
back into commands. **D is a spatial distance in meters**, R is radians, and M
is the number of supports. A separate native horizon caps the represented
window. The first native command is anchored exactly; duration preserves
stationary intervals. Humanoid actions are torques, so that recipe uses a
control-space arc metric and has **no physical rotation threshold**.

## Contracts and validation

OGBench transition replay uses s[t] with a[t], without the observation-horizon
offset used by robot sequence datasets. Replay excludes windows crossing
episode boundaries. Bellman rewards and discount exponents always refer to
native environment timesteps, including when a sampled goal shortens a backup.
In the fixed DQC comparison, the critic's long native action chunk is never replaced by an approximate
decoded chunk. The short critic conditions on the encoded behavior prefix.

Losses and gradients are checked against the pinned JAX implementation in
`tests/test_q_chunking_upstream_parity.py`; set `DQC_REFERENCE` to that checkout.
The port uses upstream's pre-optimizer target update ordering. It clamps value
logits at numerical saturation and avoids upstream's flattened-action index
into temporal validity masks (all admitted upstream chunk windows are valid).
Backend initialization and PRNG streams differ from JAX; this is a numerically
audited PyTorch port, not a claim of bit-identical training trajectories.

`eval/control_replay.py` selects M/D/R from held-out dataset trajectories,
checks native action reconstruction, and compares simulator replay from
identical states. Its qpos replay errors are reconstruction diagnostics, not
goal success. Selection must pass the preregistered gates and never uses
learned-policy evaluation. Report the best passing **tested** configuration,
not a globally optimal setting.

The initial and expanded humanoid sweeps failed the 1.25x compression gate.
A separately labeled follow-up permits expansion while keeping the same
action and physics fidelity thresholds. Its scalar cost is reported. Such a
selection is an ARC representation control, not successful compression.

`eval/goal_rollout.py` evaluates graph outputs under the environment's original
termination/time limits, starting decoded execution at action zero. It executes
the complete represented chunk. Shared reset seeds, raw per-rollout results,
action traces, and a small fixed set of preview videos support paired analysis.
This DQC protocol does not reuse PushShapes-specific M/2 or D/2 replanning.
Evaluation seeds both the environment and its action space: Gym's reset seed
does not seed action-space sampling, which Cube environments use when settling
their goal states. Manipulation environments also return a fresh Box on every
property access; a temporary per-instance subclass caches one space during
evaluation so the reset's internal samples use that seeded RNG. The original
class is restored on exit, without globally modifying OGBench. Legacy
evaluations before this fix may share initial physics
states but differ slightly in their goal vectors. Check stored reset hashes
before claiming exact paired resets; retain those legacy results separately.

Checkpoints contain optimizer state, replay update alignment, and Torch RNG.
Every checkpoint remains in durable artifact storage. With remote storage
enabled, `local_checkpoint_keep` bounds only the scratch copies created by the
current process, after rechecking the local hash and remote hash/size. Local-only
runs and pre-existing files are retained.
Replay sampling is derived from (training seed, update), and shard replacement
occurs every 1000 updates. Outputs include resolved config, environment-bound
normalizer metadata, data hashes, compute runtime, checkpoints, and scores.
An optional shared data registry atomically pins each shard's content hash;
paired jobs fail before consuming a shard if its bytes differ. Generated data
use a verified manifest. Scratch caches can be bounded without modifying the
source objects or deleting pre-existing user files.

## Prediction logging

New runs log detached prediction diagnostics to both W&B and CSV alongside the
losses, every `log_interval` updates (1,000 in the benchmark). The configured
`TargetNetworkBehavior.prediction_log_interval` follows that interval and uses
Lightning's restored global step, including after a mid-interval resume.
Metrics reuse the current training pass: no extra network forwards, actor
samples, or RNG draws. They are `log/*` outputs, excluded from the loss sum.

Each group below has `mean`, `std` (population), `min`, and `max` under
`Train/DQC/<group>/<stat>`, with per-source copies ending in `/ogbench`:

| Group | Meaning |
| --- | --- |
| `q_action` | Online short-policy critic on encoded replay actions |
| `q_chunk` | Long native-action critic; absent when that critic is disabled |
| `q_target` | Frozen short-policy critic target used to train V |
| `v` | Current observation/goal value prediction |
| `v_next` | Next-observation value used in the Bellman backup |
| `bellman_target` | Discounted, terminal-masked TD target (long critic in DQC; action critic in direct TD) |

Q/V diagnostics are sigmoid probabilities, not logits. Critic distributions
aggregate ensemble heads using configured `q_agg` (`mean` or `min`), matching
the value-target convention. `q_action/ensemble_std` and
`q_chunk/ensemble_std` additionally report the mean per-example disagreement
between heads. `bellman_target/entropy` is mean Bernoulli target entropy in
nats, including exact zero/one targets without infinities. It helps distinguish
a moving BCE entropy floor from prediction error.

These statistics describe the logged replay minibatch, including hindsight
goals. They are not held-out calibration, selected-action Q values, or rollout
success probabilities measured against outcomes. Plot against
`trainer/global_step` in W&B, rather than the W&B event index. Jobs already
running a pinned source archive retain their original logging; the additional
metrics begin with launches using this updated source and recipe.

## DQC-matched median-duration comparison

`hydra_configs/benchmark/dqc_median_comparison.yaml` defines a separate fresh
comparison of standard DQC, matched native-window and uniform ARC. The replay
calibrator first measures cumulative distance and rotation over five native
actions. It tests relative rotation budgets and scales each pair of spatial
thresholds until the selected window has median duration five. The 25-action
cap remains above that target: fast segments, slow segments and stationary
holds keep different durations. This does **not** force every decoded chunk
to five actions, set M to five, or reinterpret spatial D as a timestep count.

Only held-out replay is used. Candidates must have median exactly five and
pass the same action/physics fidelity gates. Among passing candidates the
calibrator prefers fewer encoded scalars, then a mean duration closer to five;
it no longer rewards long tails for improving mean-based compression. Reports
include duration histograms, quantiles, cap fraction and both median- and
mean-based scalar compression. Compression is measured rather than required
at the expense of matching duration. In particular, a humanoid fidelity-only
expansion must remain labeled as an expansion. Learned-policy duration may
drift from the replay median and must also be measured during evaluation.

The **fixed matching TD** variant uses DQC's original native 25-action teacher
and its fixed backup. Its reward, endpoint, mask and discount exponent are
unchanged; goals within the 25-step window terminate the backup early under
the original convention. Short-policy duration affects neither that teacher
nor the number of optimizer updates. All three arms retain the per-domain
kappa values, batch 4096 and one-million-update budget. The paired native-window
and ARC configurations differ only in codec kind; standard DQC has a fixed
five-action codec, so its boundaries necessarily differ. Network input/output
dimensions follow the representation; the hidden architecture is shared.

`Train/Chunk/native_duration/{mean,min,max,p10,p50,p90}` measures replay prefix
durations in both TD modes. `Train/TD/native_backup_horizon/mean` includes early
goals; `Train/TD/nonterminal_backup_horizon/mean` verifies 25 for the fixed
variant. The real-data backup audit now supports both modes and checks all
three fixed-DQC representations against independently indexed native targets.
Older jobs, calibration reports and results remain separate.

## Shape and time factorization

`ShapeTimeControlChunkCodec` is a separate, incompatible ARC checkpoint format.
It addresses two weaknesses of the earlier representation: geometry was
scaled by the native cap, and total duration summed M predicted timing values.
Increasing M therefore changed timing sensitivity as well as shape resolution.
The original `ControlChunkCodec` and earlier recipes retain their behavior.

For delta actions, define the commanded path `q[0]=0`,
`q[t+1]=q[t]+a[t]`. Uniform geometric supports describe this path independently
of its timestamps. Each channel has a separate displacement extent, so a large
gripper or yaw coordinate does not shrink the translation targets. For direct
controls such as torques, geometry describes a curve in control space and has
a separate initial control anchor. This is not measured Cartesian motion.
The geometry metric is explicitly configured with `geometry_units`; the new
recipe uses equal units in the normalized native control space, while its
spatial window still uses the frozen physical D/R thresholds.

The factor layout is:

| Factor | Scalars | Representation |
| --- | --- | --- |
| Geometry | M x action_dim | Per-channel normalized curve supports at uniform arc progress |
| Extents | action_dim | Log-scaled per-channel displacement amplitudes |
| Clock | native cap | Square-root-coded progress increments at native control boundaries |
| Duration | 1 | Log of total native steps relative to a configured reference |
| Initial anchor, direct controls only | action_dim | First native control |

The decoder sums nonnegative clock increments into monotonic arc progress,
interpolates the geometric curve, and differentiates the commanded delta path
back into native actions. Zero progress increments preserve holds. Entirely
stationary paths use zero extent and a canonical uniform clock. Execution
starts at decoded index zero and lasts the separately predicted duration,
subject to the environment's termination and native cap. M never specifies
the number of environment steps executed. Clock padding is masked after that
duration and canonicalized before Q ranking.

No global [-1,1] clamp is applied to log extent/duration coordinates. The codec
projects each factor into its own valid domain before the critic ranks actor
samples. Actor flow matching gives each factor equal total loss weight by
default; increasing M cannot silently reduce duration/clock supervision. The
fixed native 25-step critic, reward endpoint, discount, and target updates are
unchanged. This remains an approximate finite-M path representation: sharp
corners can be smoothed, and inconsistent predicted geometry/timing can lead
to native-control clipping. Replay fidelity and learned-policy scores must
both be measured; neither is implied by this factorization.

`hydra_configs/benchmark/dqc_shape_time.yaml` defines a fresh fidelity sweep.
Freeze the old median-five D/R, validation hash and sampled indices; merge its
`codec_overrides` into that selected codec and fill `geometry_units` from the
environment's `action_scale` (ones for direct controls). The grid increases M
without changing the represented windows. Selection minimizes native-control
reconstruction error, with no compression requirement, and retains the action
and physics gates. Reports include per-channel and first-control errors.
Do not interpret the lowest-error tested M as the optimal learned-policy size.

Focused tests cover independent retiming, holds, synchronized rotation/grip,
prefix causality, native-cap invariance, per-factor bounds, duration independence
from M, critic projection and graph gradients. Previous trained weights cannot
be loaded into this format to fix an existing policy; train fresh checkpoints.

## Table 7 reference and stricter M selection

`hydra_configs/benchmark/dqc_table7.yaml` records each selected paper tuple
`(h, ha, kappa_b, kappa_d)`. Apply the domain's `ha` to the native-reference
codec and to `protocol.reference_policy_native_steps`; it is 1 for Humanoid
and Puzzle 4x6 and 5 for the other domains. The fixed teacher remains 25.
The short reference actor must actually predict/execute that many actions;
changing an evaluation label or truncating a differently trained actor is not
the same reference. The shape/time median-five recipe also explicitly records
its reference horizon, required by the real-data TD audit.

For the paired spatial-window controls, fit the replay median to the domain's
`ha`. Their individual durations remain variable. Freeze D/R and sample indices
before sweeping M; changing M cannot shorten the motion window to hide error.
The Table 7 recipe sweeps M3 through M8192 in bounded CPU batches and measures
p50/p90/p95/p99/max native RMSE, first-action errors, channel errors, maximum
absolute control error, exact duration recovery, and paired physics replay.
It selects the smallest M passing every configured fidelity gate, without a
scalar compression objective. Physics can be evaluated for all M, including
candidates failing the control-error gates. Cache keys retain exact decoded
controls rather than rounding away small differences.

The stricter recipe requires p99 native and first-action RMSE <=0.001,
maximum absolute native error <=0.01, physics qpos p90 <=0.001 and maximum
qpos error <=0.01. These are replay acceptance thresholds, not guarantees of
policy success. `exclude_indices` excludes entire previously sampled windows
including endpoint states for confirmation. If confirmation fails and guides
another choice of M, preserve that failure and use a new disjoint set before
claiming confirmation. Log native/native repeatability and native/recorded
drift when inspecting sensitive contact cases.

For direct controls, the first decoded control is always the separate anchor,
even with a noisy predicted clock. A one-action direct-control chunk has no
subsequent geometric path; its shape/extent/clock are canonicalized before Q
ranking. Delta actions continue to describe integrated commanded motion.

## Separate variable-duration TD comparison

`hydra_configs/benchmark/smdp_comparison.yaml` specifies a fresh two-arm
comparison: **native_window versus uniform ARC**, with the same replay-selected
M/D/R, native cap, data, seeds, optimizer, training budget, and evaluation.
Apply its dotted overrides to the base recipe *after* the domain settings.
Within this pair, only `codec.kind` differs in the model/data configuration;
run IDs and artifact paths differ for bookkeeping. The original fixed-25-step
DQC experiment continues separately and is not resumed into this variant.

`backup_mode=policy_window` directly trains the policy action critic at the
represented window's endpoint. It disables the long critic/distillation teacher
and requires `kappa_d=0.5` (symmetric BCE, with the existing constant 0.5 loss
weight). It retains DQC's quantile value fit, target-action critic update, flow
actor, and best-of-N action selection. This is an offline goal-conditioned
semi-Markov QC variant, not the unchanged DQC algorithm or the original paper's
offline-to-online experiment. Differences from fixed DQC include removal of
the long teacher; this is not a discount-exponent-only ablation.

For a recorded prefix of **tau native controls**, replay provides the observed
state trajectory, and the codec selects tau using the same spatial/rotation
rule in both arms. TD changes the endpoint, accumulated goal reward, mask and
discount together. Tau is neither M nor a fixed 25. If no goal occurs within
the prefix, the target is `gamma**tau * V(s[t+tau], g)`. A goal at native offset
delta inside the prefix gives `gamma**delta` with no bootstrap. This preserves
upstream's **state-based, half-open [t,t+tau) reward convention**: a current goal
gives reward 1 at offset 0, and a goal exactly at the endpoint bootstraps V at
that goal state. It is not shifted to a transition-arrival reward. Admitted
replay windows remain within one episode, including their endpoint.

Stationary controls still consume native time. ARC's geometry approximation
does not create an exact simulator transition: training uses the recorded
behavior endpoint, with approximation error bounded only by the replay
calibration measurements. At inference each arm executes its full predicted
decoded duration, stopping early if the environment terminates or truncates.

`Train/SMDP/duration/{mean,min,max}` logs represented native duration;
`backup_horizon` logs elapsed time up to the first goal or the window endpoint;
`bootstrap_discount` logs gamma raised to that elapsed time (before masking).
`goal_terminal_fraction` records how often the bootstrap is masked. Q/V metrics
remain available; `q_chunk` is absent because this variant has no long teacher.
The real-data `scripts/benchmark/audit_goal_backup.py` verifies equal native/ARC
targets, decoded durations, independently indexed endpoints and rewards, and
publishes data/target hashes. CPU fixtures additionally cover goal boundaries,
stationary holds, unchanged fixed-backup parity, and exact checkpoint resume.

For generated corpora, `scripts/data_download/wait_goal_manifest.py` is a CPU
gate on the existing verified dataset manifest. It checks its content hash,
generator, unique shard count, and transition count before releasing dependent
GPU tasks; it does not generate another dataset or modify the source objects.

## Dataset scope

The six domains and dataset sizes match the linked reproduction script.
Triple/quadruple use the hosted 100M datasets. Humanoid giant and puzzle 4x5
use default hosted datasets. Octuple and puzzle 4x6 require generated 1B corpora;
the hosted 100M substitutes must not be labeled as 1B reproductions.
`scripts/data_download/generate_goal_data.py` runs the pinned upstream generator
in isolated processes with explicit NumPy/reset seeds and immutable shard
receipts. Data generation and reduced smoke jobs are separate from final RL.

## Fixed-interpolation Cube M/D/R pilot

`hydra_configs/benchmark/dqc_cube_mdr.yaml` varies only M, translation distance
and rotation budget in the existing `ShapeTimeControlChunkCodec`. The codec,
uniform interpolation, geometry metric and native cap 25 remain unchanged.
Distance and rotation proposals preserve a calibration median of five native
actions; testing the threshold interval between the fifth and sixth actions
explores D/R without changing that median into a fixed execution length.

The selected pilot is M56, D = 0.18813843364466518 metres of cumulative commanded
translation, R = 1.5359093226505232 radians (88.0011 degrees). Native execution is
the entire decoded variable-duration chunk, starting at index 0. M is geometric
resolution, not the number of environment actions executed. M96 also passed
the physical-motion confirmation and is retained as an unlaunched alternative.

This pilot uses a declared **physical-motion** criterion, not the previous
near-exact action/qpos criterion. Native and decoded actions are replayed from
identical states. Measure cube and effector displacement and gripper opening
after every native step. The recipe records explicit tail and maximum limits;
also retain raw cube orientation and mixed-unit qpos errors. Cube's published
goal uses object XYZ, but orientation differences can affect later contacts,
so positional fidelity does not imply closed-loop success.

On the previously unused shard 003 confirmation (2048 control windows and 256
paired physics replays), M56 had p99 action RMSE 0.01155 and maximum absolute
control error 0.06235. Cube trajectory error was p99 2.211 mm / max 4.833 mm;
effector error p99 0.897 mm / max 1.311 mm. Gripper opening error was p99 1.133% /
max 1.514% of range. Raw cube orientation error reached 13.43 degrees. This
does **not** pass the old strict action or mixed-coordinate qpos gates. The
earlier failed M/D/R combinations and all gate definitions remain evidence.

Train fresh ARC and matched native-window pilots for three shared seeds on
Cube quadruple 100M; reuse the existing matching standard DQC references.
Keep the Table 7 h25/ha5 reference, kappa_b=.93, kappa_d=.8, batch 4096 and 1M
updates. Compare the fixed final checkpoint across all five goal tasks with 50
rollouts each, identical reset banks and source/config/data receipts. Treat
three-seed results as development evidence; a robust improvement needs wider
seed confirmation. Neither passing replay nor submitting training is a win.

### Full-trajectory replay success audit

The M56 pilot **fails to preserve full-trajectory replay success**, despite the
short-chunk physical-motion checks above. Freeze M/D/R and replay every trace
from standard DQC seed 100001's final evaluation at 1M updates: five goals with 50
rollouts each, including failed reference rollouts. Encode/decode consecutive
spatial chunks using the existing interpolation, preserving the original
native action budget. The final chunk is capped at the remaining recorded
actions. Never reset the simulator between chunks or add actions after the
recorded trace ends.

OSMO workflow `dqc-cube-mdr-replay-sr-20260922-v1-1` completed this audit with
identical full simulator states and success targets for each replay pair.
Native replay matched every recorded reset hash and success/failure outcome.

| Cube goal | Native replay | ARC M56 replay |
| --- | ---: | ---: |
| 1 | 50/50 (100%) | 48/50 (96%) |
| 2 | 50/50 (100%) | 44/50 (88%) |
| 3 | 49/50 (98%) | 26/50 (52%) |
| 4 | 33/50 (66%) | 5/50 (10%) |
| 5 | 46/50 (92%) | 44/50 (88%) |
| All goals | 228/250 (91.2%) | 167/250 (66.8%) |

ARC retained 166/228 native successes (72.8%), losing 62 and gaining 1. This is
open-loop replay of recorded actions, **not learned ARC policy SR**. The
reference traces stop at first success or timeout; no extra time is granted
to reconstructed actions. The earlier local macOS audit had 16 native replay
disagreements and is retained as a diagnostic; use the OSMO result above.

Evidence, including per-rollout outcomes, configs, source hashes, recorded
actions and replay states, is stored under
`s3://rldb/experiments/qchunk-arc-benchmark-20260921/cube-mdr-v1/replay-sr-audit/`.
The archive SHA-256 is
`edbeffb30d73ec99380b32713f3cf73d1a98634edc1c5131e4ff793a233ed753`.
The training pilots do not establish replay preservation or a performance win;
future M/D/R acceptance needs a full-trajectory replay-success check using a
separate development bank before final evaluation.

### Expanded M/D/R search and numerical replay controls

The 2026-09-23 search found **no accepted replacement configuration**. The
original 250 trajectories above are now development data. A fixed first-five
per-goal subset contains 25 trajectories and 21 native successes. Test 98
median-five settings: M16/32/56/96/128/256/512 crossed with 14 translation/yaw
budget choices. The best retains 17/21 native successes. An 18-setting extension
with shorter budgets, M1024/M4096 and a one-action numerical control also fails
the declared gates. Shorter budgets change the duration comparison; large M
and single-action controls are diagnostics, not recommended RL settings.

The gates were declared before this search: at least 98% retention of native
successes, at most two percentage points overall SR loss, and at most four
points loss in any goal. They are tolerance gates, not exact-equivalence claims.
All candidate outcomes, including failures, remain recorded. A fresh 250-trace
bank from the frozen standard DQC checkpoint, seed start 7100000, is ready for
future confirmation and has not been used to select ARC configurations.

OSMO workflow `dqc-cube-replay-numerical-20260923-v1-1` then tested whether
full-trajectory open-loop SR can reliably distinguish very small codec errors
from numerical sensitivity. On the same 25 development trajectories:

| Diagnostic | Native successes retained | Maximum action error |
| --- | ---: | ---: |
| Copied native actions | 21/21 | 0 |
| Native actions shifted up one float32 increment | 18/21 | 5.96e-8 |
| Native actions shifted down one float32 increment | 17/21 | 5.96e-8 |
| Native actions shifted one increment in seeded random directions | 18/21 | 5.96e-8 |
| One-action ARC, existing float32 computation | 17/21 | 5.96e-8 |
| One-action ARC, float64 arithmetic then cast back to float32 | 21/21 | 0 |
| Existing M56 / D0.188138m / R88.001 degrees | 16/21 | 0.04265 |
| M512 with the same D/R | 17/21 | 0.004934 |

Both zero-error controls reproduce every simulator trajectory bit for bit.
The one-action float32 failures eventually show 30–38cm maximum cube position
divergence, despite action errors no larger than one float32 increment. Thus
the losses are not simply just-missed success thresholds at the original
cutoff. This is evidence of numerical sensitivity in long open-loop replay;
it does not invalidate the native reference or establish a learned ARC win.
It also does not show how much of the full 250-trace M56 gap is attributable
to roundoff. Float64 here is a diagnostic only. The production codec and
piecewise-linear interpolation remain byte-identical, with codec SHA-256
`57b9ed420252ab434360204979cc80e8adc0d63545abc197da1a90599d32b17e`.

Learned-policy results remain separate. All nine final evaluations are complete
at 1M updates, using five goals and 50 rollouts per goal. The exact same 250
reset hashes were verified across all models and training seeds.

| Model | Seed 100001 | Seed 200002 | Seed 300003 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Standard DQC | 91.2% | 90.8% | 90.4% | 90.8% |
| Native spatial window | 31.2% | 21.6% | 36.0% | 29.6% |
| ARC M56 | 34.8% | 30.4% | 30.4% | 31.9% |

ARC's paired differences from native-window are +3.6, +8.8 and -5.6 percentage
points, averaging +2.3 points. Three seeds do not establish a robust advantage;
ARC still trails standard DQC by 58.9 points on average. Native-window failure
also means codec reconstruction error alone cannot explain the learned-policy
gap. No replacement M/D/R configuration has passed the declared replay gates.
At that stage no replacement training run had been launched. The later
bounded-window investigation below remains exploratory.

Timing in this Cube codec is **native-step geometric progress**, not a single
average velocity or a per-waypoint velocity. The clock stores
`2*sqrt(normalized_progress_increment)-1`; total duration stores `log(tau/5)`.
Progress uses the commanded five-channel XYZ/yaw/gripper path in normalized
control units, not measured Cartesian motion. Decoding emits relative native
controls at 20Hz and preserves zero-progress holds. Window selection uses
`max(cumulative_translation/D, cumulative_abs_yaw/R) <= 1`, with a minimum of
one action. There is no residual plan or time carried between chunks, and no
named `hybrid` or `carry` decoder mode is invoked by this DQC implementation.

Search results, compute receipts, scripts, configs and numerical trajectories
are under
`s3://rldb/experiments/qchunk-arc-benchmark-20260921/cube-mdr-sr-search-v1/`.

### Native-window Q input consistency

The frozen native-window pilots above ranked actor candidates after clipping,
but their Q inputs could still contain nonzero controls after the predicted
duration and continuous duration codes between the integer execution lengths.
Replay training supplies zero padding and discrete duration codes. Thus Q could
rank two encodings differently even though they execute identical actions.

`ControlChunkCodec.project_latent` now zeroes unused native-window controls and
uses the decoded integer duration before Q ranking. `ShapeTimeControlChunkCodec`
delegates its native/native-window cases to this projection. Each candidate's
valid controls and duration are preserved exactly, as are encoded replay inputs;
only candidate ranking can change. Fixed-length native behavior and all ARC
geometry, timing, interpolation and projection behavior remain unchanged.

The completed OSMO comparison `dqc-cube-q-ranking-20260923-v1-1` reproduced
every original outcome and reset hash across all three checkpoints and the
250-reset bank before evaluating the correction, without training:

| Native-window inference | Seed 100001 | Seed 200002 | Seed 300003 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Original Q inputs | 31.2% | 21.6% | 36.0% | 29.6% |
| Canonical Q inputs | 28.8% | 24.0% | 35.2% | 29.3% |

The correction changes the winning candidate on 11.1–14.6% of the same 512
held-out states per checkpoint while preserving every candidate's executed
prefix exactly. It does **not** recover performance. Across 750 rollouts it
loses 91 original successes and gains 89. ARC and fixed-native rankings are
unchanged. Similar held-out Q/V values and critic losses do not imply similar
closed-loop behavior; actor losses across representations are not comparable.
These scores remain separate from the original policy results above. Artifacts
are under
`s3://rldb/experiments/qchunk-arc-benchmark-20260921/cube-q-ranking-v1/`.

### Frozen component diagnostics

OSMO workflow `dqc-cube-components-20260923-v1-1` isolated remaining causes on
the frozen seed-100001 checkpoints and the first ten previously evaluated
resets per goal. All eight 50-rollout controls completed, reproducing every
native/ARC identity outcome. Raw rows and reset hashes were verified:

| Diagnostic policy | Successes / 50 | SR |
| --- | ---: | ---: |
| Standard DQC identity | 42 | 84% |
| Native actor/Q, ARC round trip at five steps | 43 | 86% |
| Native actor/Q, ARC round trip after spatial cutoff | 46 | 92% |
| Native actor, ARC Q, decoded spatial prefix | 44 | 88% |
| Native actor, native-window Q, native spatial prefix | 45 | 90% |
| Learned ARC identity | 15 | 30% |
| Learned ARC, execution capped at five | 17 | 34% |
| Learned native-window, execution capped at five | 17 | 34% |

The unchanged M56 codec and learned ARC Q can support strong behavior when
supplied with reference-actor proposals. Capping execution alone does not
recover the learned ARC policy. This narrows the failure to proposal generation
and how Q ranks those proposals, without establishing that either component
alone is the cause. This small development screen does not show an ARC win.

The reference actor supplies five-action proposals, so spatial-prefix controls
cannot represent longer windows. This distribution change is explicit, and
the reference-Q spatial-prefix arm controls for it. The execution-cap arms
also change protocol. These are diagnostic interventions, not learned ARC
benchmark wins, and they do not replace full-trajectory replay gates. The
unused fresh confirmation bank remains untouched. Source, script, checkpoint,
config, reset, CPU semantic checks and GPU sampling-parity receipts accompany
the results under
`s3://rldb/experiments/qchunk-arc-benchmark-20260921/cube-components-v1/`.

The completed frozen-checkpoint workflow,
`dqc-cube-actor-sampling-20260923-v2-1`, tested 50 actor-flow steps instead of 10
for all three model types, pure ARC/native-window critic substitutions, and
ARC execution-based Q inputs. The latter generates the usual candidates `z`,
decodes native actions `a` and duration `tau`, scores `Q(s, E(a, tau))`, and
executes the original `a`. It does not decode the re-encoded representation.
The candidates, durations, M/D/R and path interpolation remain unchanged.

A CPU fixture demonstrates the input ambiguity: changing an unvisited M56
geometry support changes the direct Q input by 0.9143 while native controls
and duration remain bit-identical. Re-encoding the executed controls removes
that difference. However, this correction did not recover learned-policy SR.
V1 stopped before rollout because GPU checks found
up to 1.82e-6 differences between batched candidate decoding and the original
single-candidate decoder. V2 uses the original single-candidate decoding shape
for each proposal; exact GPU sampler/RNG/decoded-action checks pass. Failed
checks remain under `cube-actor-sampling-v1/`; new results are separate under
`s3://rldb/experiments/qchunk-arc-benchmark-20260921/cube-actor-sampling-v2/`.

All six variants completed on the same 50 development resets. Raw rows,
archive hashes, and reset hashes were independently checked:

| Frozen intervention | Successes / 50 | SR |
| --- | ---: | ---: |
| Standard DQC, 50 flow steps | 48 | 96% |
| ARC, 50 flow steps | 16 | 32% |
| Native-window, 50 flow steps | 20 | 40% |
| ARC proposals, native-window Q | 23 | 46% |
| Native-window proposals, ARC Q | 13 | 26% |
| ARC proposals, Q on re-encoded executed actions | 14 | 28% |

The critic swaps preserve every candidate's executed controls and duration;
only ranking changes. Native-window Q improves ARC on this screen, but the
result remains far below the reference. More flow steps and execution-based
Q inputs do not solve the failure. None is an accepted benchmark remedy.

### Smaller representation investigation

The next investigation reduces the native cap and waypoint count while
retaining the same physical distance/rotation budgets and piecewise-linear
interpolation. Cap selection used 16,384 source-pinned validation windows,
without policy scores: among caps retaining median five, cap eight gives the
mean closest to five (4.96094). This shortens 19.34% of the old windows and
reduces native-window padding from 77.91% to 37.99%. Eight is a native action
cap; D remains a distance, 0.1881384336m, and R remains 88.001 degrees.

OSMO `dqc-cube-small-m-20260923-v1-1` screened M8/16/32/56 after the frozen
reference actor/Q selects a five-action proposal, versus its exact raw spatial
prefix. It uses the same 50 development resets and single-candidate decoding.
The declared closed-loop gate allows at most four percentage points overall
loss and ten points loss per goal relative to that raw-prefix control. These
are exploratory training-selection gates, separate from the failed
full-trajectory open-loop gates. Short-window reconstruction and physics are
also measured for up to eight actions; the reference actor can only test up
to five. The fresh confirmation reset bank remains unused.

All 250 rollout rows and reset hashes were verified. The exact raw spatial
prefix scored 45/50 (90%), M8 46/50 (92%), M16 47/50 (94%), M32 48/50 (96%)
and M56 48/50 (96%). The predeclared smallest-passing-M rule selects M8.
These remain reference-actor/Q controls, not learned ARC successes.

Short-window replay retains a material tradeoff: M8's action RMSE p99 is
0.0582, maximum absolute control error 0.2993, and it fails the earlier
control and physical-motion gates. M16 passes motion gates but fails control
error limits. M32 passes both on the first bank, but its p90 action RMSE
0.0100016 misses the 0.01 limit on a second bank without shared native
windows; that bank has mean duration 4.9238 and median four. M56 meets the
control and motion limits on both banks, but also misses the second bank's
exact median-five rule. These failures remain recorded and are not called
replay-equivalence passes.

`dqc_cube_bounded.yaml` defines an exploratory fresh-step-zero pilot: native
spatial-window control, selected ARC M8, and an additional ARC M32 arm to
separate representation size from geometric fidelity. All use native cap8,
the same D/R, unchanged interpolation, the original native25 TD teacher,
seed100001, 1M updates, batch4096, and five goals with 50 rollouts per goal at
each 100k updates. Policy dimensions are 41 for native-window, 54 for M8,
and 174 for M32, versus 126 and 311 in the old native-window/M56 runs.
M32 is an added diagnostic arm, not the predeclared winner. Recovery and a
multi-seed performance advantage have not been demonstrated.

OSMO workflow `dqc-cube-bounded-20260923-v1-1` was submitted to `groot-l40-04`
with source `222dcdd38444b8382a4b01de596e8b07e85cf013`. Each full training task
depends on all three 100-update GPU smoke tasks and native-time target audits.
It uses one L40 per arm. Checkpoints, resolved configs, normalizer and data
hashes, Q/V metrics and periodic rollouts are preserved under
`s3://rldb/experiments/qchunk-arc-benchmark-20260921/cube-bounded-v1/`.

Selection, raw action traces, fixed preview videos, simulator diagnostics and
GPU parity receipts are under
`s3://rldb/experiments/qchunk-arc-benchmark-20260921/cube-small-m-v1/`.
