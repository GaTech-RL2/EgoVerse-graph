# LIBERO ARC replay calibration

Calibrate the codec before training an ARC policy. This experiment replays
reconstructed demonstration commands through the pinned LIBERO simulator; its
success rates are **not trained-policy benchmark scores**.

`R` is the accumulated SO(3) geodesic rotation horizon in degrees. `D` is the
accumulated translation horizon in metres. The first budget crossing truncates
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

For each task, demo IDs 0–1 are calibration, 2–3 are selection, and 10–14 are
confirmation in the original pilot; the final float32 protocol reserves demo
IDs **30–34** instead. No confirmation result affects the selected parameters. These
are codec experiment splits within the official training demonstrations, not
claims of an unseen policy evaluation dataset. Normal benchmark rollouts use
the independent benchmark initial states and seeds.

Each replay restores the demonstration's saved model XML and initial simulator
state. Only asset paths are relocated. It then executes actions without state
injection, corrections, or extra settling steps. Commands are cast to float32,
matching the released replay, converter, and graph input. Both original source
commands and their cast values are hashed. Four controls run: float32 raw,
an actual repeat of that raw simulation (never a cached result), original source
precision raw, and dense float32 ARC. Raw replay must succeed on at least 80%
of the split, its repeated simulator states must be identical, and every dense
episode must have command MSE at most 1e-12. All controls' task outcomes are
reported, including raw successes lost or gained after a precision change.

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

Among candidates matching the overall float32 raw calibration success rate,
the best R/D at each M advances to selection. Selection chooses the lowest M
with no decrease in overall success rate; ties prefer higher success then lower
command MSE. Every paired loss and gain is reported; this criterion does not
claim that the same demonstrations always succeed. Set `minimum_retention=1`
to request that stricter criterion explicitly. Confirmation checks the frozen
choice. A failed confirmation is reported as unconfirmed and must not unlock
ARC training. The result is the best tested choice under this protocol, not a
proof of a global optimum or statistical equivalence in the population.

Run in the pinned simulator environment:

```bash
source emimic/bin/activate
python -m egomimic.benchmarks.libero.replay \
  --root /workspace/libero --suite libero_10 --run-id UNIQUE_RUN_ID
```

For OSMO, render `scripts/benchmarks/launch_libero_osmo.py --replay` with an
immutable pushed commit and unique run ID. Each isolated workflow requests one
L40S and runs eight independent physics workers. No rendering or policy model
is needed for the replay scores. Raw HDF5 downloads are pinned by revision and
verified against their LFS SHA-256 hashes. Per-episode results, split/spec,
source revision, controls, selection, and confirmation are uploaded to the
run's separate artifact prefix. Existing benchmark worktrees and jobs are not
used as sweep scratch space.

Full training now requires `--arc-replay-run` (or `ARC_REPLAY_RUN`). Before the
ARC stage, the runner verifies the confirmed result, suite, split/spec hash,
32/16 execution cadence, success criteria, and exact codec source bytes. It
loads R/D/M from that result into both graph stages. Missing or failed
confirmation leaves the run in `AWAITING_ARC_CALIBRATION` after preserving its
OAT checkpoints; it cannot start ARC with default parameters.

`--resume-from-run` restores completed and partial training checkpoints from
immutable artifact receipts, verifies SHA-256, size, suite and the full global
batch/epoch budget, and resumes optimizer/EMA/normalizer state through the
shared training entry point. Completed stages are skipped. A partial ARC
checkpoint must also match the confirmed codec parameters. This allows the
existing OAT training to move to the gated runner without restarting training
from random weights. `--evaluate-from-run` remains restricted to completed
training checkpoints.
