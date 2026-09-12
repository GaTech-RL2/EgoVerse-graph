#!/usr/bin/env python
"""Catch shell variables a launcher uses but never sets.

These scripts run under `set -u`, so an undefined variable aborts the job at the
moment it is expanded -- after cloning, installing and staging tens of
gigabytes. A `|| true` on the line does not help: the failure happens during
expansion, before the command runs.

This exists because $S5 was copied into articulated_cotrain_sweep.yaml from the
rollout launcher without its definition, and the job died 50 minutes in.
"""

from __future__ import annotations

import re
import sys
import yaml

# Set by the container, by sourcing ~/.egoverse_env, or by the credential block.
# Set by the container, by sourcing ~/.egoverse_env, by the credential block,
# or by the shell/awk itself.
AMBIENT = {
    "HOME", "PATH", "PWD", "IFS", "PYTHONPATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES",
    "GITHUB_TOKEN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION",
    "AWS_REGION", "AWS_SESSION_TOKEN", "WANDB_API_KEY", "MASTER_PORT",
    "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT_URL",
    "DEBIAN_FRONTEND", "RANDOM", "SECONDS", "LINENO", "PPID", "UID", "BASH_SOURCE",
    "FUNCNAME", "OSTYPE", "BASH_REMATCH",
    "NF", "NR", "FS", "OFS",           # awk, inside single-quoted programs
}

# Any NAME= that is not itself a variable reference. Deliberately loose: it
# accepts `local A=$1 B=$2`, `... & PID=$!`, and `export X=y`, and will also
# accept a stray `--flag=value`. Over-accepting causes a missed warning; being
# strict caused eleven false positives, which is worse -- a linter that cries
# wolf gets ignored.
ASSIGN = re.compile(r"(?<![\$\w])([A-Za-z_]\w*)=")
DECLARE = re.compile(r"^\s*declare\s+-\w+\s+(.+)$", re.M)
LOOPVAR = re.compile(r"\bfor\s+([A-Za-z_]\w*)\s+in\b")
# `read -r -d '' EP` -- the variable is the last bare word on the line.
READVAR = re.compile(r"\bread\b[^|;\n]*?([A-Za-z_]\w*)\s*(?:;|$)", re.M)
# $VAR / ${VAR}, but NOT ${hydra.interpolations} which contain a dot.
USE = re.compile(r"\$\{?([A-Za-z_]\w*)(?![\w.])")


def check(path: str) -> int:
    src = open(path, encoding="utf-8").read()
    doc = yaml.safe_load(re.sub(r'(?<!")\{\{(\w+)\}\}(?!")', r"PLACEHOLDER_\1", src))
    problems = 0
    for task in doc["workflow"]["tasks"]:
        env = set(task.get("environment") or {})
        for f in task.get("files") or []:
            body = f["contents"]
            defined = set(AMBIENT) | env
            defined |= set(ASSIGN.findall(body))
            defined |= set(LOOPVAR.findall(body))
            defined |= set(READVAR.findall(body))
            for group in DECLARE.findall(body):
                defined |= set(re.findall(r"[A-Za-z_]\w*", group))
            missing = sorted(u for u in set(USE.findall(body)) - defined)
            for m in missing:
                line = next((i + 1 for i, l in enumerate(body.splitlines())
                             if re.search(r"\$\{?" + m + r"\b", l)), "?")
                print(f"  {path}:{f['path']} line {line}: ${m} is used but never set")
                problems += 1
    print(f"{path}: {'OK' if not problems else str(problems) + ' undefined variable(s)'}")
    return problems


if __name__ == "__main__":
    raise SystemExit(min(1, sum(check(p) for p in sys.argv[1:])))
