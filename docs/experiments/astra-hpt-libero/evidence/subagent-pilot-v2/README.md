# Corrected Astra subagent commissioning evidence

The one-L40 OSMO workflow
[`astra-hpt-subagent-commissioning-20260928-v2-1`](https://us-west-2-aws.osmo.nvidia.com/workflows/astra-hpt-subagent-commissioning-20260928-v2-1)
completed in 717.992 seconds, including environment setup and preservation.
Executable source: `41778ab32bb84d10bea8745763bfe8107f152b56`.

| Check | Measured result |
| --- | --- |
| Generated commissioning | 30 accepted / 31 attempts, five per template, ten per stage |
| Retained failure | Attempt 26: S1 approach timeout at 50 commands |
| Total acquisition budget | 31 earlier + 31 corrected = 62 / 120 attempts |
| Fresh-process replay | Six scenes; next eight commands restore with zero state error and identical camera frames |
| Normalizer | Frozen from the corrected 30 accepted episodes only |
| Full random HPT | 45,934,983 trainable parameters; gradients in all six component groups |
| Discarded engineering update | Loss 1.784012; 0.797 seconds; 950,507,520 peak allocated CUDA bytes |
| Inference | `[4,10,7]`; contract predicts ten commands and executes one before replanning |
| Production optimizer updates / student policy rollouts | 0 / 0 |
| Preserved artifacts | 523; all 58 selected downloads match uploaded hashes |

The yellow marker is now visually distinct from the optional gray fixture.
Geometry, dynamics, teacher logic and saved Astra proposals are unchanged.
The earlier run remains immutable and charged; its visual ambiguity supersedes
it as the current commissioning corpus. These teacher results and the single
HPT engineering update do not measure student policy success or curriculum
benefit.

The original commissioning proposal and three synthetic-report decision tests
are in [`../subagent-pilot-v1/generation/`](../subagent-pilot-v1/generation/).
All four actual Codex `gpt-6-astra` children returned valid responses without
repair. The uniform child had no learner report or adaptive archive. The
transport exposes no resolved provider snapshot, hard token-cap enforcement,
provider usage or billing; those fields remain explicitly unavailable.

The full immutable output prefix is:

```text
s3://rldb/experiments/astra-hpt-libero-20260928/astra-hpt-subagent-commissioning-20260928-v2/41778ab32bb84d10bea8745763bfe8107f152b56/preflight/
```

It contains every attempt's HDF5 data, initial state, resolved XML, controller
trace, video and receipt, plus the initial checkpoint and full model audit.
The compact files here link those outputs by hash. The upload index was
reconstructed from the completed uploader's log, then checked against all
58 files exported through the authenticated OSMO execution channel. It is
explicitly a log-derived index, not a new independent read of every R2 object.

The self-contained HTML embeds six representative successful videos and the
retained failure. Its HTML/ZIP paths and hashes are in
[`preview-artifacts.json`](preview-artifacts.json); all seven videos were
reviewed in external and wrist views. The first bulk export was truncated by
the terminal transport; a paced retry succeeded and every file's hash matched.
Both transfer diagnostics remain in local scratch; no simulation was repeated.

Repository CI for the executable source passed 1,489 CPU tests, configuration
and installed-wheel checks, static checks and lock validation in
[run 36463070158](https://github.com/GaTech-RL2/EgoVerse-graph/actions/runs/36463070158).
The following documentation/evidence commit changes no executable source.

Frozen evaluation partitions, common seed data/training, the full resumable
round coordinator, matched U/A policy evaluation and the gradient diagnostic
remain unfinished.
