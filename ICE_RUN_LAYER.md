# ICE ARC run layer

This branch is layer 16 on top of the ARC consolidation and robot stack:

- `70dcb4196aad44e22668d0a9587e0b0760b2f7de` (PR #92), stack validation;
- `907a027de682f528df2ed6290fd3b25913cf3f5c` (PR #96), shared Eva/Yam
  collection and local graph rollout; and
- `129bd34f5308852f6cb94e81689354cc695dd056` (PR #97), repository navigation
  and component-boundary guidance.

It carries only the ICE and active ABC campaign additions required by the
September 12 training campaign:

- clock-head and mixture-HPT graph recipes;
- shared shape/clock flow stages;
- annotation wiring required by the mixture prompt stage;
- exact W&B identity enforcement and identity sidecars for Slurm resumes;
- stable metric names with `add_dataloader_idx=False` by default;
- the bounded four-minute ICE campaign monitor; and
- timestamped Slurm/W&B run snapshots under `docs/run_snapshots/`.

The consolidated ARC duration codec and vectorized resampling implementation
remain unchanged from layers 04 and 05. Existing checkpoint resumes must use
their original W&B ID with `resume=must`; runs without a recoverable checkpoint
must start a new W&B identity instead of pretending to be continuations.
