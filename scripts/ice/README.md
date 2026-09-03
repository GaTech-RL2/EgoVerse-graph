# ICE training tools

These tools provide a portable, fail-closed path for one-GPU Planar V2 BC
training on PACE ICE. Site paths, source identity, dataset identity, output,
W&B identity, GPU type, and checkpoint validation are inputs; none are embedded
in the launcher.

## Prepare an immutable source/runtime

Use a clean, detached or branch checkout on the current scratch path reported
by `pace-whoami`. Capture the environment after installing from the repository
lock:

```bash
python scripts/ice/capture_runtime_lock.py \
  --repo /absolute/repo \
  --expected-head 0123456789abcdef0123456789abcdef01234567 \
  --lock-input /absolute/repo/uv.lock \
  --output /absolute/run-bundles/runtime-lock.json
```

The command refuses a dirty checkout and refuses to overwrite a lock.

## Required launcher inputs

Export these before `sbatch --export=ALL`:

- `ICE_REPO`, `ICE_EXPECTED_HEAD`, and `ICE_PYTHON`
- `ICE_TOOL_DIR` (optional only when the tools are at
  `$ICE_REPO/scripts/ice`)
- `ICE_CHECKPOINT_VALIDATOR`, an absolute executable path to
  `validate_lightning_checkpoint.py`
- `ICE_EXPECTED_SPLIT_MANIFEST_SHA256`, matching the exact committed Planar
  U-Socket split manifest (override `ICE_SPLIT_MANIFEST` only with identical
  bytes at another absolute path). Its current committed SHA-256 is
  `3683e3461596eef8df2432fa865779b3c77b2a2057dabd0fea125595729cf313`.
- `ICE_DATASET_DIR` and a new, absolute `ICE_OUTPUT_DIR`
- `ICE_EXPERIMENT`, exactly
  `pusht/planar_v2_usocket_direct_bc` or
  `pusht/planar_v2_usocket_arc_bc`
- `ICE_WANDB_ENTITY`, `ICE_WANDB_PROJECT`, and a unique
  `ICE_WANDB_RUN_ID`
- `ICE_EXPECTED_GPU_NAME`
- `ICE_MAX_STEPS`, `ICE_VAL_CHECK_INTERVAL`,
  `ICE_LIMIT_TRAIN_BATCHES`, `ICE_LIMIT_VAL_BATCHES`,
  `ICE_CHECKPOINT_EVERY_N_STEPS`, `ICE_TRAIN_BATCH_SIZE`, and
  `ICE_VALID_BATCH_SIZE`

Optional settings include `ICE_WANDB_NAME`, `ICE_WANDB_GROUP`,
`ICE_WANDB_TAGS`, `ICE_NORM_STATS_PATH`, `ICE_SEED`, and
`ICE_MAX_RESTARTS`. Authentication remains in the user's normal W&B
environment; never place credentials in a submission script.

On the first attempt, omitting `ICE_NORM_STATS_PATH` computes fresh statistics
from the pinned training split and writes
`$ICE_OUTPUT_DIR/norm_stats/norm_stats.json`. Every Slurm restart must reuse
that exact file; it will not recompute or overwrite normalization. Supplying a
precomputed file or directory likewise sets the cache destination to `null`.

The safe default is `ICE_LAUNCH_MODE=preflight`. It validates source and input
paths, resolves the full Hydra config, checks the one-GPU/runner-owned contract,
re-enumerates the physical dataset through the training resolver, reproduces
the seed-42 1% episode split with zero ID/path overlap and complete union, and
dry-runs the generic checkpoint runner without creating the requested run
directory. `ICE_LAUNCH_MODE=dry-run` is an alias. Set
`ICE_LAUNCH_MODE=run` only after reviewing the emitted preflight record.
The validator records new resolved-path hashes on ICE; it does not compare the
manifest's Skynet-specific absolute-path hashes. Episode IDs and their hashes
must match exactly.

Probe the exact real allocation before submission, for example:

```bash
sbatch --test-only --account=coc --partition=ice-gpu --qos=coc-ice \
  --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=96G \
  --time=16:00:00 --gres=gpu:h200:1 scripts/ice/launch_planar_bc.sbatch
```

Then submit exactly one selected allocation. The batch job records the Slurm
allocation and full `nvidia-smi` report, rejects nonzero uncorrectable ECC or a
pending repair/reset, performs a real BF16 forward/backward pass, and completes
a one-rank NCCL all-reduce before training. Training uses world size one,
`strategy=auto`, immutable
epoch/optimizer-step checkpoints, and the generic requeue runner as the sole
owner of `scontrol requeue`.

The runner validates every candidate checkpoint, forwards `SIGUSR2` to request
a fresh full-state checkpoint at the `USR1` boundary, resumes the same W&B ID,
and writes `COMPLETE.json` only after successful training and terminal
checkpoint validation. It never opts into unvalidated checkpoints. For a long
run, launch the separate CPU checkpoint mirror described by
`ice_checkpoint_mirror.sbatch`; do not put mirroring inside the GPU job.
