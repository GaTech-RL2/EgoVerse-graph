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

- **native_reference**: published DQC with policy horizon 5, critic horizon 25,
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
comparison alongside the published reference.

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
The critic's long native action chunk is never replaced by an approximate
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

Checkpoints contain optimizer state, replay update alignment, and Torch RNG.
Replay sampling is derived from (training seed, update), and shard replacement
occurs every 1000 updates. Outputs include resolved config, environment-bound
normalizer metadata, data hashes, compute runtime, checkpoints, and scores.
An optional shared data registry atomically pins each shard's content hash;
paired jobs fail before consuming a shard if its bytes differ. Generated data
use a verified manifest. Scratch caches can be bounded without modifying the
source objects or deleting pre-existing user files.

## Dataset scope

The six domains and dataset sizes match the linked reproduction script.
Triple/quadruple use the hosted 100M datasets. Humanoid giant and puzzle 4x5
use default hosted datasets. Octuple and puzzle 4x6 require generated 1B corpora;
the hosted 100M substitutes must not be labeled as 1B reproductions.
`scripts/data_download/generate_goal_data.py` runs the pinned upstream generator
in isolated processes with explicit NumPy/reset seeds and immutable shard
receipts. Data generation and reduced smoke jobs are separate from final RL.
