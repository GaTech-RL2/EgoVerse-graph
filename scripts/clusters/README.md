# Cluster adapters

Model code and Hydra experiment graphs are cluster-neutral.  This directory
contains the small environment boundary that binds those graphs to a storage
layout.  Scheduler account, partition, QoS, GPU type, and availability are
live facts and are deliberately **not** stored in these profiles.

Load exactly one profile before composing or launching a cluster job:

```bash
source scripts/clusters/common/load_profile.sh skynet
```

For ICE and Phoenix, bind the verified task-local dataset and scratch roots
first.  A source config or historical run name is not evidence that dataset
bytes exist on that cluster.

```bash
export PACE_SCRATCH=/storage/ice1/<group>/<user>
export PUSHSHAPES_DATA_ROOT="$PACE_SCRATCH/datasets/Tsim_v2"
source scripts/clusters/common/load_profile.sh ice
```

The profiles expose one shared interface:

- `EGOVERSE_CLUSTER_PROFILE`
- `PUSHSHAPES_DATA_ROOT`
- `EGOVERSE_RUN_ROOT`
- `EGOVERSE_SOURCE_ROOT`

Cluster launchers may consume these paths, but must still query live Slurm
associations and use the maintained training or evaluation launcher.  Skynet's
live `EVAL_PROTOCOL.md` and `bf_eval_par.sbatch` remain canonical; do not copy
them into this repository as another authority.

When invoking Slurm over non-interactive SSH, use a login shell so the cluster
adds its Slurm installation to `PATH`:

```bash
ssh sky2 'bash -lc "sbatch --test-only /absolute/path/to/job.sbatch"'
```

Fail closed if `command -v sbatch` is empty.  A plain `ssh host 'sbatch ...'`
may run with a reduced `PATH` even though Slurm works in an interactive shell.

## Frozen Python runtime

Do not reuse a shared environment merely because its Python and PyTorch
versions look compatible. Reconcile an explicit task runtime from the checked-in
lock before capturing its runtime lock or launching training:

```bash
srun --account=<account> --partition=<cpu-partition> --time=00:30:00 \
  scripts/clusters/common/sync_runtime.sh \
  /absolute/clean/source /absolute/task/runtime
```

The command runs `uv sync --frozen`, so the task environment cannot silently
omit a locked dependency. It then verifies the exact `torchdiffeq==0.2.5`
required by EgoVerse. The command refuses to run outside a scheduled allocation;
login nodes remain orchestration-only.
