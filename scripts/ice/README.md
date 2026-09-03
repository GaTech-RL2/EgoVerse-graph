# ICE training entrypoints

`launch_planar_v2_dp_cotrain_clean.sbatch` is the single authoritative ICE
entrypoint for the clean Planar V2 standard-DP campaign. Its modes are:

- `MODE=preflight`: resolve and validate source, data, split, normalization,
  pipeline stages, parameter count, and run identity without allocating a GPU.
- `MODE=smoke`: run two real optimizer steps, scheduled validation,
  EnergyScore@32, checkpoint save, and strict reload on one H100 or H200.
- `MODE=full`: start a 240,000-step run from scratch on one H100 or H200 using
  a passed smoke's exact split and normalization artifacts.

Full mode deliberately rejects `ICE_INITIAL_CHECKPOINT`,
`ICE_INITIAL_CHECKPOINT_SHA256`, and `CKPT_PATH`. Its resolved Hydra config must
contain `ckpt_path: null`. A Slurm requeue may use only the scheduler-generated
HPC checkpoint from the same job; it is not an experiment checkpoint restore.

Submit the GPU modes with exactly one compatible GPU and the cluster's current
eight-hour limit, for example:

```bash
sbatch --nodes=1 --ntasks=1 --cpus-per-task=16 --mem=128G \
  --gres=gpu:1 --constraint='H100|H200' --time=08:00:00 \
  --requeue --signal=B:USR1@600 --export=ALL,MODE=full,... \
  scripts/ice/launch_planar_v2_dp_cotrain_clean.sbatch
```

Always run the exact command through `sbatch --test-only` immediately before
the real submission. Record the actual GPU from `nvidia-smi`; H100 and H200 are
interchangeable for this campaign and do not by themselves invalidate a passed
smoke. Do not infer full-run completion from a completed smoke: the W&B ID and
output directory include `smoke` or `full` explicitly.

The trainer publishes aggregate and per-domain `Train/MSE` on optimizer steps
as well as at epoch end. A running full job is not considered healthy until its
W&B history contains a finite step-level MSE; scheduler `RUNNING` alone is not
enough.
