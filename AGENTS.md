# Repo Agent Rules

## Shell / Command Execution
To run commands in the interactive shell, source `emimic/bin/activate` when it
exists. In worktrees where `emimic` is absent and the checked project
environment is `.venv`, source `.venv/bin/activate` instead; verify the selected
activation file exists before running project Python tooling.

Apply this before running project Python tooling (for example: `python`, `pytest`, `pip`).

## Pipeline and cluster operations

Read [docs/cluster-pipeline.md](docs/cluster-pipeline.md) before selecting a
training checkout, modifying a launcher, or cleaning cluster task files.
The generic graph runner and PipelineAlgo must interpret only stage contracts
and opaque batch keys; model-specific semantics belong in configured stages.

Resolve current source from `GaTech-RL2/EgoVerse-graph` and preserve each
existing run's recorded commit. Some model families live on review branches;
a newer `main` does not authorize replacing their source.

Use the maintained cluster workflow for allocations. Query live account,
partition, and scheduler state instead of copying a fixed `salloc` command.
PACE Phoenix (`pacerh9`) and ICE (`Ice`) are different clusters. Their login
nodes are for orchestration: run Python, including CPU validation, through a
scheduled allocation. Keep scheduler logs and output outside source checkouts.

Before cleanup, inventory exact candidate paths and their consumers. Retain
datasets, normalizers, environments, recorded source, and unverified checkpoints.
Use bounded task directories; never discover files by recursively scanning a
whole scratch/project root. Candidate inventory alone does not authorize deletion.

Publish source from the authenticated Skynet account with Graphite `gt submit`,
preserving branch parents. Do not merge without an explicit user request.
