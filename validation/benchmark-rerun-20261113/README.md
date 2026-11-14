# Pusher + LIBERO rerun — November 13, 2026

68 OSMO workflows accepted in `groot-l40s-01`. No H100 allocation was requested. All run IDs are new; old results and checkpoints are preserved. This is a launch/validation report, not new benchmark scores.

## Submitted matrix

| Family | Policy runs | Evaluation per run | Training |
| --- | ---: | ---: | --- |
| Pusher DP16 / ARC velocity / ARC absolute duration | 18 | 1,550 resets: 50 each at L0–L30 | cotrain + clean, seeds42/43/44, 240k updates, batch128 |
| LIBERO original ARC/OAT/FAST/DP comparison | 64 | 2,500 episodes | four suites, seed42, 5001 epochs, batch1024 |
| LIBERO grouping / component-clock sweep | 88 | 2,500 episodes | 11 variants × STK/DUR × four suites |
| LIBERO Spatial decoder / MoT ablations | 6 additional | 2,500 episodes | shared, separate, masked, three MoT variants; STK2 |

Eight shared-profile ARC/U-Net cells are identical across the two LIBERO tables and use the same fresh reference runs, yielding **150 unique LIBERO policies**. LIBERO suites are Spatial, Object, Goal and LIBERO-10. LIBERO-90 and the previously deferred ARC+OAT hybrid remain excluded. Total scheduled evaluation budget is 402,900 episodes including pushers.

Each training allocation uses four L40S GPUs; each dependent evaluation uses one. Pusher evaluations are independently sharded by level. LIBERO stream/profile workflows serialize their own training/evaluation chains while separate suites/profiles may overlap. Training budgets remain Spatial270054, Object325065, Goal280056, LIBERO-10 605121 optimizer/EMA updates. The original 32 predicted / 16 executed LIBERO controls and fixed8 pusher execution are unchanged.

At the captured status check, all six pusher workflows were RUNNING (data gates completed, training allocations pending), and all 62 LIBERO workflows were PENDING. No new training completion or SR is claimed. Pool quota was nearly exhausted. This limits start times; no reliable wall-clock completion ETA is available.

## Implementation and gates

- Pusher source: `505c3eed6a6e62ad3fc985ab4bdee6ebe830f0d8`, runtime branch `codex/pusher-main-arrival-20261105`.
- LIBERO ARC and native OAT: `9a054f6` (full SHA in launch plan). Spatial decoder/MoT additions: `6138cd6`. Runtime branch `codex/libero-main-arrival-20261113`.
- Unchanged DP/FAST baselines: historical runtime `9c36037a5da54d5a84cd753fe57353be91e436a8`, preserving their training and inference configurations.
- Main checked live remains `c7e110461650d09a88b2c2dff8f02db54b1deace`.

LIBERO required its own migration: the old codec stored interval durations. The new adapter uses main's future-only supports, incoming rates, absolute arrival timestamps and unbounded isotonic correction. The canonical module is copied from main with its helper import redirected to AST-identical extracted geometry functions. Relative OSC geometry uses a zero-displacement / identity-rotation / neutral-command anchor. This is explicitly command-space geometry, not a measured Cartesian pose. Separate-gripper events and coupled-gripper interpolation follow main. The gripper event and pause parity tests compare directly against the canonical module. Scalar/angular/component ablations preserve their explicit geometry and grouping while adopting the new timing rules. No projection or metric speed bound was introduced.

Clock format is explicit in new model-stage configurations. Old same-width checkpoints are rejected on this runtime and must use their recorded code. Fresh normalization/targets/checkpoints are required. MoT and split-decoder shape/timing indices were remapped to the trailing clock channels (shape0–9, timing10–11); xyz/rotation/gripper MoT uses [0,1,2,10] / [3,4,5,6,7,8,11] / [9]. Parameter budgets were checked against the prior matched models.

**99 CPU checks passed**, covering main parity, stationary/event/pause paths, angular noncommutativity, component clocks, corrected predictions, all profile/backbone recipes, decoder/MoT channel and parameter budgets, and dependency-graph rendering. Code compilation, shell syntax and diff whitespace checks passed. Pusher's prior 55 CPU checks remain recorded on its validation branch. These CPU checks do not establish GPU or simulator success: each ARC workflow first runs paired raw/dense/candidate simulator replay, then a fresh GPU training/checkpoint/rollout smoke before its full training budget. Native OAT now has a tokenizer/policy GPU smoke and releases training GPUs before dependent evaluation.

## Previous results checked

These are **dated archived snapshots**, not new results or a claim that all later R2 records were re-audited.

Pusher Oct23 completed subset, matched cotrain seed42:

| Method | L0 SR80 | L1–30 SR80 |
| --- | ---: | ---: |
| DP16 | 12.00% (50) | 4.20% (1,500) |
| ARC absolute duration, first-delay version | 18.00% (50) | 6.67% (1,500) |

The completed absolute-duration cotrain subset across seeds42/44 was 25.00% L0 and 7.93% L1–30. This must not be compared as a matched-seed aggregate against the single completed DP16 seed. The full archived subset is in `previous-pusher-results.csv`.

LIBERO Sept28 original table:

| Method | Spatial | Object | Goal | LIBERO-10 |
| --- | ---: | ---: | ---: | ---: |
| Native OAT | 61.68 | 0.36 | 75.52 | 59.44 |
| Plain U-Net DP | 61.44 | 52.88 | 85.68 | 50.72 |
| Shared ARC DUR / U-Net | 68.08 | 59.64 | 88.96 | 53.52 |
| Plain OAT-DP | 64.52 | 0.00 | 83.12 | 60.52 |

The old OAT/OAT-DP Object near-zero scores remain an unresolved anomaly, not a settled tokenizer advantage. That snapshot had 57/64 complete cells including FAST; partial scores are not treated as full-suite SR. The Oct7 stream snapshot includes complete Spatial/Object/Goal scores except one partial cell, while many LIBERO-10 tails were pending. Separate angular-axis DUR reached 70.40 / 67.92 / 91.28 on Spatial/Object/Goal in that snapshot. The Sept30 Spatial architecture audit was complete for all six ablations. Full archived tables are included alongside this report; they remain separated by their original contracts.

Sources:
- https://github.com/GaTech-RL2/EgoVerse-graph/blob/codex/libero-stream-validation-20261001/docs/results/arc_vs_oat_libero_20260928.md
- https://github.com/GaTech-RL2/EgoVerse-graph/blob/codex/libero-stream-validation-20261001/results/libero_arc_streams_20261007/status_041459/score-table.csv
- https://github.com/GaTech-RL2/EgoVerse-graph/blob/codex/libero-stream-validation-20261001/docs/results/libero_arc_mot_spatial_20260930.md

## Receipts and reproduction

`submission-manifest.json` maps all 68 accepted workflow names to live overview URLs and SHA256 hashes of their submitted specs. `submitted-workflow-specs.tar.gz` contains the exact YAMLs (credential names only, no credential values). Launch plans pin source revisions; status JSONs are point-in-time snapshots. Tests and reports are isolated on `codex/libero-main-arrival-validation-20261113`.

From repository root with the project environment activated:

```bash
source emimic/bin/activate
PYTHONPATH=. pytest -q validation/test_libero_arrival.py
```

Do not resubmit the archived specs without inspecting the accepted workflow IDs first; doing so would duplicate runs. The pusher artifact namespace remains `experiments/pusher-main-arrival-20261105/`; fresh LIBERO IDs live under the established `experiments/arc-oat-20260919/` storage root. This launch does not delete any artifacts.
