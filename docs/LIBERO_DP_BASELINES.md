# Plain diffusion-policy controls for LIBERO

These controls predict 32×7 normalized raw actions directly, with no ARC encode,
decode, or learned OAT tokenizer. They isolate ARC's effect within each of the
two existing policy architectures. Spatial, Object, Goal and LIBERO-10 use the
same pinned demonstrations, split, decoded replay loader and observation
encoder as the ARC/OAT comparisons. LIBERO-90 remains deferred.

| Setting | Current DP control | OAT-release DP control |
| --- | --- | --- |
| Total parameters | 62,733,455 | 27,182,479 |
| Action-network parameters | 40,339,207 | 4,788,231 |
| Backbone | Conditional U-Net, widths 256/512/1024 | Transformer, 4 layers, width 256, 4 heads |
| Hydra experiment | `oat/libero_dp_unet_policy` | `oat/libero_dp_oat_dp_policy` |
| Evaluation method | `dp_unet` | `dp_oat` |
| Sampling | 100 DDIM steps, existing local scheduler | 10 DDIM steps, released scheduler |
| Noise objective | 100 training timesteps, epsilon MSE | 100 training timesteps, epsilon MSE |
| Optimizer |AdamW, action 5e-5/observation 1e-5, betas 0.9/0.95, WD 0 |Same, with released parameter grouping |
| Training | 5001 epochs, global batch 1024, seed 42, EMA |Same |
| Observations / prediction / execution | 2 frames / 32 steps / 16 steps |Same |
| Evaluation | 10 tasks × 50 trials × 5 repetitions, 550 step cap |Same |

Eight-L40S training uses 128 examples per rank and one accumulation step. The
global drop-last optimizer budget remains 270054 Spatial, 325065 Object,
280056 Goal and 605121 LIBERO-10. Device count does not shorten the budget.

Each full job first runs two separate preflight optimizer updates with its real
architecture, bf16 and requested DDP layout, reloads EMA, and executes a short
simulator rollout on every task. Full training then starts fresh in a separate
process/output directory; preflight weights and updates are never reused.
Checkpoints preserve optimizer, EMA, normalization and dataset identity. Raw
DP policies carry explicit representation/backbone metadata and cannot be
labeled as ARC in rollout records.

The workflow has a native train→evaluate dependency. Training must finish all
epochs and optimizer/EMA updates, upload a content-addressed checkpoint, and
export its validated receipt before evaluation can start. The training task
then releases its eight GPUs; evaluation requests one L40S and five independent
repetition workers. No workstation watcher is required. Evaluation artifacts
use the training run ID with an `-eval` suffix. Saved initial-state hashes must
still match the reference before claiming a paired comparison.

Render a full run with a pushed immutable source commit:

```bash
source emimic/bin/activate
python scripts/benchmarks/launch_libero_osmo.py \
  --commit <40-character-commit> --run-id <unique-run> \
  --suite libero_spatial --mode full --epochs 5001 \
  --dp-backbone oat_dp --gpus 8 --gpu-type L40S --output run.yaml
```

Use `unet` for the other control. `--resume-from-run` restores compatible
optimizer/EMA state; no tokenizer calibration or replay selection is needed.
`tests/test_libero_dp_baseline.py` covers real graph training, EMA reload,
resume, target-free inference, and job dependencies. `test_oat_diffusion.py`
also checks 32×7 released-network outputs, epsilon gradients and all 10 DDIM
updates against the pinned upstream source.

The launcher also recognizes L40 (`ovx-l40`) as a distinct device family and
rejects an L40/L40S mismatch. This is an optional capacity fallback, not a
change to the current L40S-only campaign: L40 jobs require the user's hardware
exception before submission.
