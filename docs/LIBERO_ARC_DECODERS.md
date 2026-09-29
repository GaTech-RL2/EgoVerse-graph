# ARC shape and timing decoder comparison

This experiment tests whether separating the prediction of ARC geometry from
its timing improves a physical LIBERO policy. It starts on LIBERO Spatial with
the completed ARC/OAT-DP configuration `stk_2`: STK, R=192 degrees, D=0.8 m,
M=32. At the selection snapshot (2026-09-29 04:33 UTC), STK1 and STK2 both score
76.16% over 2500 trials. STK2 has fewer supports (32 versus 36), so it is the
fixed choice for every arm. No tokenizer search or retraining occurs here.

The historical reference is `arc-dpr-20260925-stk2-spatial`, checkpoint SHA256
`57bd8c96d93de388380767edb18e5a43308a1b4fbdd6cf37b5be6542c86a7df4`.
The audited demonstration replay is
`arc-five-v2-20260921-stk2-libero-spatial-replay`. Its recorded reconstruction
gap remains part of the existing replay evidence; this experiment does not
claim lossless replay.

## Three matched arms

The external timed ARC tensor remains **M x 12**:
`[xyz(3), translation timing, rotation6d(6), rotation timing, gripper]`.
The velocity components here are the existing two normalized timing channels,
at indices 3 and 10, used by the unchanged STK physical decoder. All other ten
channels, including gripper, are the shape stream.

The denoiser projects these into separate shape and timing tokens, producing
2M internal tokens. This tokenization of the denoiser input is necessary for a
token attention mask: the historical model packed both components into each
of its M tokens. The new shared arm controls for that input-interface change.

| Variant | Decoder | Action-token dependencies |
| --- | --- | --- |
| `shared` | One 4-layer stack | Both streams attend to both streams |
| `separate` | Two independent 2-layer stacks | Each stream attends only to itself |
| `shape_masked` | One 4-layer stack | Shape cannot attend to timing; timing can attend to shape |

All attention remains causal by **waypoint**, including across the two
streams. A token can attend to its own waypoint in either permitted stream.
Both streams receive the same original observation/time memory and waypoint
position embedding, with separate learned stream embeddings. The final
LayerNorm is shared and operates independently on each token. Loss remains
the same epsilon MSE over the original 12 output channels; there is no new
per-stream reweighting, diffusion schedule, or timing target.

Every arm has exactly **4,791,564 action-network parameters**, including the
original observation-conditioning MLP. Each shares the unchanged 22,394,248
parameter observation encoder, for **27,185,812 total**. This is 768 more than
the historical joint-feature ARC policy. The new shared arm is therefore the
primary control. Separate decoders halve depth per stream to hold total
parameters fixed; this is not a same-depth comparison with doubled capacity.
Matched seeds give identical initialization before splitting the decoder
layers, but topology-dependent training trajectories need not match.

## Fixed training and evaluation

- Original OAT-DP width 256, four heads, feedforward width 1024, dropout 0.1.
- Same camera/state encoder, dataset checksum, train/validation split and
  decoded replay loader as the completed ARC benchmark.
- AdamW: action LR 5e-5, observation LR 1e-5, betas (0.9, 0.95), weight decay 0.
- Seed 42, 5001 epochs, effective batch 1024 (8 L40S GPUs x microbatch 128).
  Spatial has 54 optimizer updates per epoch, **270,054 updates** total.
- Same EMA, epsilon loss, 100 training diffusion steps, 10 DDIM inference steps.
- Two observation frames, predict 32 dense actions, execute 16 before replanning.
- Final EMA checkpoint only. Evaluation has 10 tasks x 50 trials x 5 repetitions
  = **2500 episodes**, with the original 550-step limit.
- Each evaluation validates the protocol, episode identities, data context and
  actual initial-state hashes against the completed STK2 reference. Scores are
  saved before comparison, but a pairing mismatch fails the comparison rather
  than being reported as a fully matched result.

The three new arms isolate decoder topology under the same input interface and
parameter budget. Their comparison with the historical 76.16% run additionally
includes the split input interface. These are single training seeds with five
evaluation repetitions, not five independently trained policies.

## Launch and artifacts

Use a pushed 40-character source commit and change the variant/run ID for each
arm. The checked-in launcher builds a train-to-evaluate OSMO dependency:

```bash
source emimic/bin/activate
PYTHONPATH=. python scripts/benchmarks/launch_libero_osmo.py \
  --commit "$ARC_DECODER_COMMIT" \
  --run-id arc-decoder-20260928-shared-spatial \
  --suite libero_spatial --mode full --epochs 5001 --gpus 8 --gpu-type L40S \
  --arc-decoder-variant shared --arc-profile stk_2 --arc-modes stk \
  --arc-replay-run arc-five-v2-20260921-stk2-libero-spatial-replay \
  --arc-decoder-reference-run arc-dpr-20260925-stk2-spatial \
  --output /tmp/arc-decoder-shared.yaml
osmo workflow submit /tmp/arc-decoder-shared.yaml \
  --pool groot-l40s-03 --priority NORMAL --format-type json
```

Campaign run IDs:

| Variant | Run ID |
| --- | --- |
| Shared | `arc-decoder-20260928-shared-spatial` |
| Separate | `arc-decoder-20260928-separate-spatial` |
| Shape masked | `arc-decoder-20260928-masked-spatial` |

Submitted workflow IDs, immutable source/spec hashes and evaluation prefixes are
recorded in [the launch manifest](arc_decoder_runs_20260928.json). All three
were allocated eight L40S GPUs in `groot-l40s-03`. The separate decoder workflow
uses 72 GiB local storage after replacing a never-started 240 GiB request;
measured Spatial raw data is 6.24 GB and the decoded cache is 6.12 GB. This
resource change does not change the policy or optimizer configuration.

Training runs a two-update GPU preflight and short simulator rollout before
starting the full fresh budget. It releases its eight L40S GPUs on completion;
the dependent `evaluate` task requests one L40S, five workers and 120 GiB RAM.
The evaluation run ID appends `-eval`. Artifacts live under
`s3://rldb/experiments/arc-oat-20260919/<run-id>/`: training runtime,
replay provenance, preflight proof, optimizer budget and checkpoint receipts;
evaluation `scores.json`, `paired-comparison.json`, and
`arc_stk/libero_spatial/{protocol.json,episodes.jsonl}`.

## Verification

The launch change passed 197 CPU regression tests across the new model and
campaign tests plus OAT parity, LIBERO rollout/evaluation, cluster orchestration,
and ARC sweep tests. The pinned OAT reference source was available for parity
checks. New tests verify channel reconstruction, equal parameter counts,
optimizer coverage, exact forbidden-attention gradients, permitted reverse
dependencies, waypoint causality, and absence of timing-to-shape influence
through all ten DDIM steps. All three arms complete CPU training, EMA save/load,
optimizer resume and action decoding, including a bfloat16 training fixture.
GPU and simulator preflight evidence is generated separately by each job.
