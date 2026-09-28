# AstraPush pilot

The version 2 specification is preserved in `specification-v2/`. The user
explicitly selected **EgoVerse-graph**, overriding its proposed AstraExps
repository. This work builds on integration PR #159; it does not replace or
modify the previous Astra-reversal or DQC experiments.

## Implemented foundation

- The shared HPT graph has two independently initialized ResNet18 encoders,
  a random UTF-8 byte language stem, proprioception, the 16-block HPT trunk,
  and the six-block flow head. All 45,934,983 parameters are trainable.
- The YAML declares ten predicted commands at 10 Hz, 50 Euler inference
  steps, and execution of **one command before replanning**. No ARC tokenizer
  or pretrained language/vision weights are used in this pilot.
- The language encoder masks padding in self-attention and latent pooling.
  Masked flow stages canonicalize padded targets before joint-horizon
  attention and divide loss by the number of valid action scalars. They
  support explicit flow draws for the later diagnostic.
- Strict scene/task/teacher schemas feed a trusted LIBERO BDDL compiler.
  The independent semantic checker measures ten-step full-cube containment
  and maximum lift/preservation violations over the entire trajectory.
- Immutable HDF5 episodes align observation t with executed command t.
  The dataset reads only student features, pads within each episode, excludes
  commissioning/evaluation data, and provides resumable episode-balanced
  current/history sampling.
- Provider calls use the configured Astra gateway without substitution and
  archive requests, raw responses, validation failures and transport attempts.
  The preflight proposal consumes commissioning logical call 1; it is not
  an extra unbudgeted call or an accepted novelty witness.

## Sources and environments

LIBERO is pinned to `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` from the
locally available, inspected checkout. Its `register_problem` decorator
returns None, so the subclass uses the class retained in `TASK_MAPPING`.
No upstream files are patched. Every bundle reconstructs registrations by
importing the trusted compiler in a fresh process.

The learner uses the repository `uv.lock`. The separately generated Linux
Python 3.11 simulator lock is `scripts/astra_push/simulator.lock`, with
MuJoCo 3.2.3, robosuite 1.4.1 and NumPy 1.26.4. The local render preview uses
an existing read-only simulator environment; it is engineering evidence,
not proof that the locked Linux environment has run. OSMO records the actual
package lists, source/lock hashes, GPU and EGL settings.

The [source paper](https://arxiv.org/html/2609.30063v1) was checked against
the supplied gradient equation. The frozen-generator, robot imitation
pilot is an adaptation, not a reproduction of the paper's generator RL
experiment. No paper benchmark results are claimed.

The structural signature algorithm is version 1 with 0.1 mm numerical
precision. It includes primitive composition, initial/target region geometry,
static fixture geometry and marker relations, and ignores labels/colors alone.
Stock-scene inventory comparison and the six generated-scene witness gate
remain separate requirements.

## Validation and limitations

CPU checks: 49 focused tests passed, covering temporal alignment, masking,
language gradients, instruction semantics, preservation/lift failures,
immutable episodes and exact replay-sampler restoration. Full-size offline
construction succeeded with seed 17 and no network or weight-loading calls.
S1 and S3 engineering starters compiled and rendered in both cameras locally.

OSMO workflow `astra-hpt-foundation-20260928-v1-1` ran on **one L40 GPU**.
The synthetic full-size update/inference passed: all six component groups
received gradients, update time was 1.141 seconds, and peak allocated CUDA
memory was 950,507,520 bytes. This is a single engineering measurement, not a
steady-state training or evaluation throughput estimate. Actual hardware was
an NVIDIA L40 with 49,140 MiB reported memory and driver 580.95.05.

The workflow failed overall: catalog GET returned HTTP 429 before any
generation call, and upstream LIBERO's editable package did not expose its
namespace without an explicit source path. The follow-up adds bounded catalog
retries, the verified import path, and readable artifact permissions for OSMO.
R2 preserved all fourteen v1 artifacts independently of the output collector.

The follow-up teacher implements the five typed phases with measured robot
state, bounded world-frame OSC deltas, a closed gripper and wrist alignment
with the initial push direction. It stores executed/raw commands, contacts,
controller targets, torques and independent whole-trajectory metrics. No
model-generated control code runs. Successful local engineering fixtures used
123 steps for S1 and 130 for S2/S3; S3 preserved the other cube to below 1e-8 m.
These are hand-authored fixtures; they are **not** Astra commissioning data.

Local physical calibration passed 20 reset seeds per stage. Full state includes
MuJoCo integration/warm-start fields, controller targets, gripper state, robot
buffers, observation caches/timing and RNG. Eight subsequent commands replayed
with zero state error and identical camera frames after restoration. The
external-camera overlay confirms negative Y is left. Positive-axis control
and closed-gripper polarity are measured in the saved receipts.

The revised OSMO preflight runs these checks under the Linux lock and then
tests the real three-stage HDF5 batch on the exact learner. Its identity
proprioception transform is explicitly engineering-only; W04 must still freeze
statistics from thirty accepted generated commissioning demonstrations.

No production training is enabled here. Astra commissioning, the frozen split
firewall, complete checkpoint/round coordinator, paired policy evaluation and
gradient diagnostic still need implementation and measured gates.

From an activated project environment:

```bash
python -m scripts.astra_push.build_workflow --source-commit FULL_SHA \
  --name astra-hpt-foundation-UNIQUE_ID --output /tmp/astra-workflow.yaml
osmo workflow submit /tmp/astra-workflow.yaml --pool groot-l40-01
```

Artifacts are conditionally published to a new
`s3://rldb/experiments/astra-hpt-libero-20260928/...` prefix. Existing objects,
branches, datasets and checkpoints are never replaced.
