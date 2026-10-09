"""Exact native versus canonical-explicit operational reachability guard.

No permissive missing-key fallback. root_dir is the only allowlisted addition:
the immutable paths/default.yaml comments it out and canonical BASE uses ++.
"""

OPS = (
    "trainer.accelerator",
    "trainer.strategy",
    "trainer.sync_batchnorm",
    "trainer.num_nodes",
    "trainer.max_epochs",
    "trainer.min_epochs",
    "trainer.check_val_every_n_epoch",
    "trainer.num_sanity_val_steps",
    "norm_stats.precomputed_norm_path",
    "norm_stats.save_cache_dir",
    "paths.root_dir",
    "paths.work_dir",
    "paths.output_dir",
)
ADDITIONS = {"paths.root_dir": "++"}


def validate_native(flat, common):
    # Audit every native model/data/evaluator/checkpoint binding, not just the
    # next operational field. Missing values remain errors even when ++ exists.
    mandatory = set(OPS) - set(ADDITIONS)
    mandatory.update(common)
    missing = sorted(mandatory - set(flat))
    if missing:
        raise ValueError(("native required keys absent", missing))
    if (
        flat["trainer.strategy"] != "auto"
        or flat["trainer.sync_batchnorm"] is not False
        or flat["trainer.num_nodes"] != 1
    ):
        raise ValueError("reviewed one-device operational profile required")
    return True


def validate_argv(flat, argv, common):
    validate_native(flat, common)
    seen = {}
    for argument in argv:
        if "=" not in argument or argument.startswith("--"):
            continue
        key = argument.split("=", 1)[0]
        prefix = key[: len(key) - len(key.lstrip("+"))]
        key = key.lstrip("+")
        if key in seen:
            raise ValueError(("duplicate override", key))
        seen[key] = prefix
        if key in ADDITIONS:
            if prefix != ADDITIONS[key]:
                raise ValueError(("canonical explicit addition prefix mismatch", key))
        elif (
            key not in flat
            and not key.startswith(("hydra.", "hydra/"))
            and key != "experiment"
        ):
            raise ValueError(("unknown override key", key))
    if seen.get("paths.root_dir") != "++":
        raise ValueError("canonical root_dir addition missing")
    return True
