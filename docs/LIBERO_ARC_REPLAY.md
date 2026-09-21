# LIBERO ARC replay calibration

Calibrate the codec before training an ARC policy. This experiment replays
reconstructed demonstration commands through the pinned LIBERO simulator; its
success rates are **not trained-policy benchmark scores**.

The dated [STK/DUR results and support-count curves](results/libero_arc_timed_replay_20260921.md)
record all ten completed searches, selected R/D/M, and independently audited
evidence, including 3,900 final episode records across all five suites.

The [five configurations shared across suites](results/libero_arc_common_configs_20260921.md)
provide two STK recipes, two DUR recipes, and one R/D/M recipe for both modes.
These are calibration-ranked sweep choices; validation across the larger
selection split is incomplete. They do not replace the existing per-suite
confirmation results or automatically change the running training campaign.

Launch one frozen sweep member with `launch_libero_osmo.py --arc-profile
stk_1 --arc-modes stk --calibration-parent PARENT_REPLAY_RUN
--oat-reference-run ORIGINAL_FULL_RUN --mode full` and the normal immutable
commit, run ID, suite and output arguments. Other profile IDs are `stk_2`,
`dur_1`, `dur_2`, and `shared`; run `shared` separately in STK and DUR. All six
mode/profile combinations over five suites give 30 independent L40S workflows.
Each first reuses its matching audited calibration for the single frozen
candidate, runs selection demos 2–9, and confirms on fresh demos **10–24**.
These demos are separate from earlier confirmations. Parameters are fixed
before testing; failed controls stop the workflow. A completed replay with a
measured raw gap is allowed under the existing declared protocol.

The workflow then trains only its ARC policy and evaluates all 50 trials per
task over five repeats. The original OAT jobs provide the shared comparison
reference; the sweep does not retrain their tokenizer or policy. Its artifacts
record the exact profile, replay source and intended OAT reference run.
`ARC_POLICIES_COMPLETE` means ARC policy evaluation finished; paired OAT
comparisons remain explicitly incomplete until both sets of records exist
and pass the existing protocol/reset identity checks. Each workflow has
separate source, data staging, checkpoints and artifact prefixes.

## STK and DUR comparison

The LIBERO comparison now tests both native timed ARC variants independently.
`stk` means stacked **velocity** ARC; `dur` stores interval durations. Both use
separate translation and rotation supports/clocks, following
`TokenizePlanarArcTimed` and `PlanarArcTimedNativeDecoder` from the obstacle
campaign at `02db41a01cfdd018785b59c06a1485c2717ad683`. The SE(3) adaptation is
`LiberoArcTimedCodec`: M×12 rows contain xyz, translation timing, rotation6d,
rotation timing, and gripper. D caps translation alone; R caps rotation alone.
Gripper follows the translation clock and is held as a discrete OSC command.
Velocity timing uses m/s and rad/s; duration timing uses seconds.

STK policy targets divide rate magnitudes by the three-axis OSC bound
`sqrt(3) * controller_scale / dt`, so valid diagonal commands fit the diffusion
sampler's [-1,1] clipping range. The bound is recorded in the Hydra config and
restored with the checkpoint. It does not change the native replay codec or
the R/D/M selection. Older checkpoints without that field retain their scale.

[STK](../egomimic/hydra_configs/benchmark/libero_arc_replay_stk.yaml) and
[DUR](../egomimic/hydra_configs/benchmark/libero_arc_replay_dur.yaml) each test
the 336-candidate grid below on all five suites, with calibration demos 0–1,
selection 2–9, and final testing 35–49. Each mode freezes its own R/D/M choice.
These specs use 16 workers, recycling after four episodes, and 128 GiB memory.
Use `--replay-spec libero_arc_replay_stk` or `libero_arc_replay_dur` when rendering
an OSMO workflow. Their corresponding policy recipes are
`oat/libero_arc_stk_policy` and `oat/libero_arc_dur_policy`.

The earlier LIBERO codec/results are explicitly **`joint_dur`**, a shared-clock
M×11 baseline. They cannot be reused as STK or independent-clock DUR evidence.
That codec remains byte-for-byte unchanged. Its M=33 reconstruction still
serves as the numerical simulator control; a separate `mode_dense` result
measures each new variant at M=33. Uniform arc sampling can be lossy even at
that support count. STK also cannot encode a stationary dwell using zero rate;
its clock loss is measured rather than repaired with unrecorded timing data.
Coverage uses the shortest active stream clock, with gripper included in the
translation stream. Training checks both the mode and all codec source files.

Tests cover separate R/D endpoints, clock units and time scaling, duration
holds, velocity dwell loss, rotation across pi, native normalization and
bfloat16 inference, and 12-channel policy forward/backward for all tested M.
An additional read-only comparison against the native planar codec passed 240
tokenization cases (both modes, six M values, twenty paths); maximum absolute
difference was 2.98e-8.

## Shared-clock baseline and common replay protocol

`R` is the accumulated SO(3) geodesic rotation horizon in degrees. `D` is the
accumulated translation horizon in metres. In `joint_dur`, the first budget crossing truncates
the command path, including a fractional last control interval. These SE(3)
physical budgets are explicit; they do not reuse the planar implementation's
whole-window angular/translation ratio. `rotation_radius=0.05` is the separate
fixed weight in the support-selection metric. `M` counts all support rows,
including the initial anchor. A null R or D means no corresponding cap within
the 32-command lookahead.

The checked-in [specification](../egomimic/hydra_configs/benchmark/libero_arc_replay.yaml)
crosses R={12,24,48,96,128,192,384,uncapped},
D={0.05,0.1,0.2,0.4,0.8,1.6,uncapped}, and
M={4,8,16,24,32,36}: 336 candidates. M values fit the policy UNet. M=36 permits a
fully sampled 32-command path with padding; the separate dense control uses
33 supports. The original M=16, uncapped bridge is also tested at confirmation.

The float32 pilot uses demo IDs 0–1 for calibration, 2–3 for selection, and
30–34 for confirmation, per task. The larger
[refinement specification](../egomimic/hydra_configs/benchmark/libero_arc_replay_refine.yaml)
reuses only the completed calibration, selects on demos **2–9**, and tests the
frozen choice on fresh demos **35–49**. This gives 80 selection and 150 final
episodes per ten-task suite, or 720 and 1,350 for LIBERO-90. A parent calibration
must match the data revision, codec bytes, execution protocol, precision, and
simulator/library versions. Its artifact hashes are recorded; its selection
and confirmation results are never loaded by the refinement selector. These
are codec experiment splits within the official training demonstrations.
Normal policy benchmark rollouts use independent initial states and seeds.

Each replay restores the demonstration's saved model XML and initial simulator
state. Only asset paths are relocated in the XML. Four LIBERO-90 living-room
tasks contain the old `salad_dressing_1` model while the pinned BDDL expects
`new_salad_dressing_1`. For those recordings, replay binds BDDL object references
and type to the recorded asset, preserving the saved geometry and the same
logical goal. Original/effective task-definition hashes and that mapping are
recorded per episode; recordings with the current asset remain unchanged.
Replay then executes actions without state
injection, corrections, or extra settling steps. Commands are cast to float32,
matching the released replay, converter, and graph input. Both original source
commands and their cast values are hashed. Four controls run: float32 raw,
an actual repeat of that raw simulation (never a cached result), original source
precision raw, and dense float32 ARC. Repeated raw simulator states must be
identical, and every dense episode must have command MSE at most 1e-12. All
controls' task outcomes are reported, including raw successes lost or gained
after a precision change. The pilot required 80% raw success; refinement
measures that baseline without assuming a particular success rate (zero raw
successes still stops the run). LIBERO-10's pilot selection baseline was 15/20
despite deterministic repeated replay. The pinned OAT release also uses
MuJoCo 3.4.0 and robosuite 1.4.0; changing physics versions to improve this
baseline would change the comparison.

The pilot exposed numerical sensitivity: on LIBERO-10's kitchen scene 4 drawer
task, demo 2, identical source actions replayed identically twice, but casting
them from float64 to float32 changed success to failure (command MSE 3.23e-17).
Dense float32 ARC also failed, while a float64 diagnostic reconstruction
succeeded. Dense action equality to numerical precision therefore cannot
guarantee the same contact outcome. The pilot's requirement that dense ARC
retain 95% of individual raw successes was replaced by explicit numerical and
deterministic-reset controls before the fresh confirmation split was evaluated.

Every candidate encodes 32 future commands and executes 16, exactly as the
policy adapter does. Short R/D horizons hold after their recorded duration;
they cannot obtain extra replanning opportunities or stretch time. Calibration
command coverage below 99% screens a candidate out before expensive simulator
replay; all exclusions and reconstruction metrics remain in the evidence.

The pilot preferred the lowest M matching aggregate raw success. Refinement
instead prioritizes **highest replay success, then lowest M**, with command
MSE as the next tie breaker. The best calibrated R/D at each M advances to the
larger selection split. All screened candidates can compete regardless of
their gap to raw; the selector does not change its parameters after final
testing. Every paired loss and gain is reported. `CONFIRMED` means the frozen
choice matches aggregate raw success on that final split; `REPLAY_EVALUATED`
means testing completed but a measured gap remains. Neither means all the same
demonstrations succeeded. The result is the best tested choice under this
protocol, not a proof of a global optimum or statistical equivalence.

Codec tests cover translation, rotation, mixed motion, grip-only transitions,
dwells, closed paths, and rotation across pi; fractional R/D truncation must
preserve the original clock and leading holds. Pipeline checks cover action
normalization, bfloat16 predictions, degenerate predicted rotations/durations,
and policy forward/backward at every tested M. Simulator controls add actual
reset repeatability, source-precision sensitivity, dense reconstruction, and
the legacy object/task binding described above.

Run in the pinned simulator environment:

```bash
source emimic/bin/activate
python -m egomimic.benchmarks.libero.replay \
  --root /workspace/libero --suite libero_10 --run-id UNIQUE_RUN_ID
```

For OSMO, render `scripts/benchmarks/launch_libero_osmo.py --replay` with an
immutable pushed commit and unique run ID. Each isolated workflow requests one
L40S. The pilot uses eight independent physics workers; the updated refinement
spec uses 32, with the CPU and memory requests derived from that specification.
Workers are recycled after four episodes to bound retained simulator allocations;
the larger refinement requests 192 GiB. No rendering or policy model
is needed for the replay scores. Raw HDF5 downloads are pinned by revision and
verified against their LFS SHA-256 hashes. Per-episode results, split/spec,
source revision, controls, selection, and confirmation are uploaded to the
run's separate artifact prefix. Existing benchmark worktrees and jobs are not
used as sweep scratch space.

Use `--replay-spec libero_arc_replay_refine --calibration-parent PARENT_RUN_ID`
to reuse a completed float32 calibration and run the larger selection and
fresh final test. Omitting the parent runs calibration from scratch.
`--raw-cache /absolute/path` optionally reads already staged demonstrations.
Every cached file must match the official LFS SHA-256; the cache is never
modified or silently repaired. This avoids repeating downloads during recovery.

Full training defaults to both independent modes. The OSMO renderer takes
`--arc-replay-runs-file replays.json`, containing a JSON mapping from `stk` and
`dur` to their respective replay run IDs. The container runner accepts that
mapping through `--arc-replay-runs` or `ARC_REPLAY_RUNS_JSON`.
The legacy `--arc-replay-run` flag applies only to `--arc-modes joint_dur`.
Before the
ARC stage, the runner verifies the completed result, suite, split/spec hash,
32/16 execution cadence, all final episode counts, numerical controls, and
exact codec source bytes. It loads R/D/M from that result into both graph
stages. A measured gap is accepted only when the checked-in protocol explicitly
prioritizes success and permits reporting that gap. Missing, incomplete, or
invalid evidence leaves the run in `AWAITING_ARC_CALIBRATION` after preserving
its OAT checkpoints; it cannot start ARC with default parameters. The replay
gap remains part of the evidence when reporting subsequent policy scores.

`--resume-from-run` restores completed and partial training checkpoints from
immutable artifact receipts, verifies SHA-256, size, suite and the full global
batch/epoch budget, and resumes optimizer/EMA/normalizer state through the
shared training entry point. Completed stages are skipped. A partial ARC
checkpoint must also match the confirmed codec parameters. This allows the
existing OAT training to move to the gated runner without restarting training
from random weights. `--evaluate-from-run` remains restricted to completed
training checkpoints.
