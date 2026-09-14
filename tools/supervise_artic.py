#!/usr/bin/env python
"""Keep the articulated sweep moving without a human in the loop.

These pools have preempted this sweep five times. Each preemption needs the same
three steps -- notice, pick a resume source that actually has checkpoints,
resubmit -- and none of them need judgement. This does them.

Behaviour:
  * poll the training job; when it dies, resubmit resuming from the newest
    prior job that has checkpoints. If a resubmit dies quickly with the
    launcher's "no checkpoint" FATAL, fall back to the next older candidate --
    a job can run for hours and still write nothing if it never crosses a
    checkpoint boundary, which is how arc-16 left an empty prefix.
  * when training completes, launch the full rollout over all five arms.
  * stop after MAX_RELAUNCH resubmissions so a systematic failure cannot spin.

Everything it does is appended to results/supervisor.log.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(REPO, "results", "supervisor.log")
ARMS = [
    "artic_cotrain7_arc_dur_D80_M56_R26deg",
    "artic_cotrain7_arc_stk_D80_M56_R26deg",
    "artic_cotrain7_arc_dur_D80_M16_R26deg",
    "artic_cotrain7_arc_stk_D80_M16_R26deg",
]
ALL_ARMS = ARMS + ["artic_cotrain7_dp_paper"]
EMBS = ("u_socket gripper chain_gripper suction triangle flipper spring umi scoop")
MAX_RELAUNCH = 12


def say(msg: str) -> None:
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def osmo(*args: str, timeout: int = 900) -> str:
    try:
        r = subprocess.run(["osmo", *args], capture_output=True, text=True,
                           timeout=timeout)
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"


def status(workflow: str) -> str:
    m = re.search(r"^Status\s*:\s*(\S+)", osmo("workflow", "query", workflow), re.M)
    return m.group(1) if m else "UNKNOWN"


def submit_train(name: str, resume_from: str, ckpt_every: int, pool: str) -> bool:
    out = osmo(
        "workflow", "submit", "osmo/articulated_cotrain_sweep.yaml",
        "--pool", pool, "--priority", "HIGH",
        "--set", "gpus_per_run=2", "num_gpu=8", "cpu=88", "max_steps=240000",
        f"ckpt_every={ckpt_every}", "batch_size=32", "val_batches=8",
        "val_interval=20000",
        "--set-string", f"job_name={name}", "branch=codec-replay-rotfix",
        f"resume_job={resume_from}", "memory=512Gi", "storage=600Gi",
        "wandb_entity=rl2-group", f"experiments={' '.join(ARMS)}",
    )
    ok = "submit successful" in out
    say(f"submit train {name} (resume from {resume_from}, pool {pool}): "
        f"{'OK' if ok else 'FAILED'}")
    if not ok:
        say("    " + out.strip().splitlines()[-1][:200] if out.strip() else "    no output")
    return ok


def submit_eval(name: str, ckpt_jobs: str, pool: str) -> bool:
    out = osmo(
        "workflow", "submit", "osmo/articulated_rollout.yaml",
        "--pool", pool, "--priority", "HIGH",
        "--set", "n_episodes=40", "replan_every=8", "chunk_start=0", "num_gpu=1",
        "cpu=12", "shards_per_cell=24",
        "--set-string", f"job_name={name}", "branch=codec-replay-rotfix",
        "sim_commit=f952ca0d", "memory=96Gi", "storage=350Gi",
        f"ckpt_job={ckpt_jobs}", f"experiments={' '.join(ALL_ARMS)}",
        f"embodiments={EMBS}",
    )
    ok = "submit successful" in out
    say(f"submit eval {name} (ckpts from {ckpt_jobs}): {'OK' if ok else 'FAILED'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-job", required=True, help="currently running training job")
    ap.add_argument("--resume-candidates", nargs="+", required=True,
                    help="newest first; jobs believed to hold checkpoints")
    ap.add_argument("--pool", default="groot-h100-01")
    ap.add_argument("--eval-pool", default="groot-l40s-03")
    ap.add_argument("--poll", type=int, default=600)
    a = ap.parse_args()

    current = a.train_job
    candidates = list(a.resume_candidates)
    relaunches = 0
    say(f"supervisor up. watching {current}; resume candidates {candidates}")

    while True:
        st = status(current)
        if st in ("RUNNING", "PENDING"):
            say(f"{current}: {st}")
            time.sleep(a.poll)
            continue

        if st == "COMPLETED":
            say(f"{current}: COMPLETED -- training finished")
            ck = " ".join([current] + candidates)
            stamp = datetime.datetime.now().strftime("%m%d%H%M")
            submit_eval(f"articroll-all5-{stamp}", ck, a.eval_pool)
            say("supervisor done: eval submitted for all five arms")
            return 0

        if st == "UNKNOWN":
            say(f"{current}: status unreadable (API hiccup) -- retrying")
            time.sleep(120)
            continue

        # Terminal failure.
        say(f"{current}: {st}")
        relaunches += 1
        if relaunches > MAX_RELAUNCH:
            say(f"hit MAX_RELAUNCH={MAX_RELAUNCH}; stopping rather than spinning")
            return 1

        # The job that just died is the freshest resume source IF it wrote
        # anything; put it at the head and let the launcher's fatal check sort
        # it out.
        if current not in candidates:
            candidates.insert(0, current)
        stamp = datetime.datetime.now().strftime("%m%d%H%M")
        nxt = f"articotrain-arc-s{stamp}"
        placed = False
        for src in list(candidates):
            if submit_train(nxt, src, 5000, a.pool):
                time.sleep(900)   # long enough for the resume check to run
                s2 = status(nxt)
                if s2 not in ("FAILED", "FAILED_CANCELED"):
                    say(f"{nxt} alive after resume check (status {s2}); "
                        f"resumed from {src}")
                    current, placed = nxt, True
                    break
                say(f"{nxt} died during startup with resume source {src}; "
                    f"trying an older checkpoint")
                nxt = f"articotrain-arc-s{datetime.datetime.now():%m%d%H%M}"
            else:
                say(f"submit rejected for {src}; trying next candidate")
        if not placed:
            say("no resume candidate worked; stopping")
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
