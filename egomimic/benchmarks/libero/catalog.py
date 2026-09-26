"""Complete task inventory of OAT's pinned LIBERO submodule."""

import json
from pathlib import Path

OAT_COMMIT = "1da92695ef12c23b7000a0b1a76cab0aef4750e6"
LIBERO_COMMIT = "6090ff21837566fed47b7c9061c9899f4749d36b"
TASKS = json.loads(Path(__file__).with_name("tasks.json").read_text())
ALIASES = {"libero10": "libero_10", "libero90": "libero_90"}
# Iteration order is part of upstream's task_uid observation, not alphabetical.
TASK_IDS = {
    task: (suite, local_id, uid)
    for uid, (suite, local_id, task) in enumerate(
        (suite, local_id, task)
        for suite, tasks in TASKS.items()
        for local_id, task in enumerate(tasks)
    )
}


def get_tasks(name):
    name = ALIASES.get(name, name)
    if name in TASKS:
        return list(TASKS[name])
    if name in TASK_IDS:
        return [name]
    raise ValueError(f"Unknown LIBERO benchmark {name!r}")


def validate_task_coverage(task_uids, suite):
    expected = {TASK_IDS[task][2] for task in get_tasks(suite)}
    actual = {int(uid) for uid in task_uids}
    if actual != expected:
        raise ValueError(
            f"Task coverage mismatch: missing={expected - actual}, extra={actual - expected}"
        )
