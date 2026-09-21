# Infrastructure: OSMO and R2

This file is mostly a list of things that cost hours. None of it is obvious from
the tooling's own help text.

## R2 credentials — get this right first

`s3://rldb` is **Cloudflare R2**, not AWS S3. `~/.egoverse_env` exports `R2_*`
names, but `s5cmd` and the aws CLI only read `AWS_*`:

```bash
set -a; . /Users/rpunamiya/.egoverse_env; set +a
export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
unset AWS_SESSION_TOKEN                          # R2: InvalidArgument X-Amz-Security-Token
export AWS_REGION=auto AWS_DEFAULT_REGION=auto   # file says us-east-2; R2 rejects it
```

Three separate traps: the missing `AWS_*` names; the session token in the file is
for AWS STS and R2 rejects it; and the file's `AWS_DEFAULT_REGION=us-east-2` is
invalid for R2 (`Must be one of: wnam, enam, weur, eeur, apac, oc, auto`).

**When it is right, a missing key answers `no object found` in under a second.**
Any multi-minute hang from `s5cmd ls` is an auth failure, not an empty prefix. I
concluded "the prefix is empty, so nothing was banked" from exactly that hang and
was wrong.

`tools/collect_artic_results.py` does this remapping internally, which is why it
worked while raw `s5cmd` calls silently failed.

## OSMO: submission mechanics

| thing | reality |
|---|---|
| `--set` | `nargs="+"` with plain store — a **second `--set` REPLACES the first**. Combine into one. |
| `--set-string` | same; also use it for anything with units (`400Gi`) |
| `--set-env` | **does NOT override the `environment:` block.** Its `--help` claims it does. Verified by dry-run. |
| `default-values:` | top-level block; supplies defaults for `{{params}}`. Omitted -> default, `--set-string` -> override. **This is the mechanism that works.** |
| `--dry-run` | renders the spec and **skips resource assertions** — good for template checks, does not prove schedulability |
| empty values | `--set-string x=` is not reliably parseable. Use a sentinel (`none`) and map it in the script. |

## OSMO: pools

| pool | constraint |
|---|---|
| `groot-h100-01` | **requires `gpu=8` exactly**; anything else is `400 Assertion failed: GPU value must be 8` |
| `groot-l40s-03`, `groot-l40-04`, `groot-l40-03` | accept `gpu=4`; used for all per-cell rollouts |
| `groot-h100-ci-02` | shows huge free quota but is **shared** with `groot-h100-02`; physical nodes are full, so jobs sit PENDING forever |

**Free quota is not free hardware.** A `(shared)` capacity column means the
quota is illusory. Check `Total Usage / Total Capacity`, not `Quota Used / Quota Limit`.

Transient `503` on submit happens; retry succeeds. Pool name is `groot-h100-01`,
not `h100-01` — the short name returns a bare `403`.

## `FAILED_EVICTED` means DISK, not preemption

The status column carries no cause and renders preemption and disk-overrun
identically. **Run `osmo workflow events <id>` and read the termination reason
before forming any theory about contention:**

```
Evicted: Container train exceeded its local ephemeral storage limit "800Gi"
```

Two distinct instances in this project, both initially misread as preemption:

1. **The checkpoint pull.** The rollout copied `s3://.../<job>/*` wholesale.
   Those prefixes hold every periodic checkpoint of every arm:
   `articotrain-arc-18` = 45 objects / **141 GB**, `articotrain-dp-8` = 16 /
   **51 GB**. 192 GB copied for a job that reads ~4. Fixed by pulling only
   `<job>/<run>/checkpoints/last.ckpt` and `<job>/<run>/norm_stats/*`.
2. **`save_top_k: -1`.** 48 checkpoints/run x 7 runs x 3.92 GB = **1.3 TB**.
   Fixed by explicit pruning (see `docs/02_TRAINING.md`).

Cost of the misdiagnosis: hours spent shrinking chunk sizes, migrating across
four GPU pools, and cutting `storage` 400Gi -> 200Gi -> 100Gi — which made things
**strictly worse**, because storage was the binding constraint the whole time.

**Do the arithmetic before submitting:** checkpoints/run x runs x size, plus
anything copied from S3/R2, plus staged corpora, plus ~14 GB image. If that
exceeds the limit the job is unfinishable regardless of pool.

## Logs are slow, and why

`osmo workflow logs` on a multi-hour job regularly exceeds ten minutes and times
out. Main cause found and fixed: `drain_pids` waits on training PIDs with
`kill -0` every 30 s under the script-wide `set -x`, emitting two trace lines per
run per 30 s for the entire run — most of the log. Now wrapped in `set +x`
(commit `81ab11f`).

Prefer reading **R2 object listings** for progress. Checkpoint count and max
epoch are a better progress signal than the log, and answer in seconds:

```bash
s5cmd --endpoint-url "$R2_ENDPOINT_URL" ls 's3://rldb/staged/articulated_cotrain/<job>/*' \
  | grep -oE 'epoch_epoch=[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1
```

Use `osmo workflow logs <id> -t <task> -n <N>` to fetch a tail server-side;
streaming the whole log truncates around 40 KB.

## YAML block-scalar trap

The launcher scripts live inside a `contents: |` block. A multi-line
`python -c "..."` or a nested heredoc **breaks the scalar** and the YAML parses
into nonsense. Hit four times. Keep every embedded python to one line.

Lint before submitting:

```bash
/Users/rpunamiya/Desktop/GEAR/sim_run/venv/bin/python \
  /Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/tools/lint_osmo_launcher.py \
  /Users/rpunamiya/Desktop/GEAR/sim_run/wt_codec/osmo/articulated_rollout.yaml
```

plus a `bash -n` on the extracted script body — both are cheap and have each
caught real breakage.

## zsh vs bash

The local shell is zsh; containers run bash.

- zsh arrays are **1-indexed**. `A=( "" "first" )` with `${A[$i]}` gave an empty
  value and a `bad array subscript` failure thirty minutes into a job.
- zsh does **not word-split** unquoted variables. `$EXPS` holding a
  space-separated list arrives as **one** argument. Pass literals separately.
- Backticks in a `git commit -m "..."` string get command-substituted. Use
  `git commit -F -` with a heredoc.

## Corpus

```
s3://rldb/staged/pushshapes_articulated/articulated-20260909/
  <emb>/<gap>/<emb>/shardNNN.tar     24 shards x 125 episodes = 3,000 per cell
  162,000 episodes = 9 embodiments x 6 control modes x 3,000
  manifest sha256 78de5167ec165c8916e1d9a8768085f6765cbba34edfe13ad183e13ccf4df94e
```

Only the `ideal` control mode is used by this study.

Staged layout must be `cells/ideal/<emb>/episode_<shard>_<rest>.zarr`. Two
details that each broke a run: the per-shard episode index restarts at zero, so
the shard name **must** be in the link name or 24 shards collapse onto 125
episodes; and the link must start with `episode_` because `episode_budget.py`
globs `episode_*.zarr`. Guard with the same glob the consumer uses — a guard
that counted `find -mindepth 1` reported 3,000 while the consumer saw none.
