#!/usr/bin/env python
"""Stage one campaign's dataset union once, before dependent Slurm jobs.

Always ask s5cmd to synchronize selected episodes, including partial directories
from interrupted attempts. A receipt is emitted only after every sync succeeds.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def sync_lines(paths, folder, bucket="rldb"):
    lines = []
    for source, episode in paths:
        if not episode or Path(episode).name != episode or episode in {".", ".."}:
            raise ValueError(f"Unsafe episode identifier: {episode!r}")
        source = source.rstrip("/")
        if not source.startswith("s3://"):
            source = f"s3://{bucket}/{source.lstrip('/')}"
        # json quoting is accepted by s5cmd's batch-file tokenizer.
        lines.append(f"sync {json.dumps(source + '/*')} {json.dumps(str(folder / episode) + '/')}")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Dataset staging must run in a Slurm allocation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import hydra
    import egomimic.trainHydra  # register config resolvers
    from egomimic.rldb.zarr.zarr_dataset_multi import S3EpisodeResolver
    from egomimic.utils.aws.aws_sql import load_env

    load_env()
    env = os.environ.copy()
    for key in ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
        if not env.get(key):
            raise RuntimeError(f"Missing required environment variable {key}")
    env.update(AWS_ACCESS_KEY_ID=env["R2_ACCESS_KEY_ID"],
               AWS_SECRET_ACCESS_KEY=env["R2_SECRET_ACCESS_KEY"],
               AWS_DEFAULT_REGION="auto", AWS_REGION="auto")
    groups, union = [], set()
    for experiment in args.experiment:
        with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "egomimic/hydra_configs")):
            cfg = hydra.compose(config_name="train_zarr_cartesian", overrides=[f"+experiment={experiment}"])
        for name, dataset in cfg.data.train_datasets.items():
            folder = Path(dataset.resolver.folder_path)
            if not folder.is_absolute() or not folder.is_relative_to("/workspace/users/ani-cheluva"):
                raise ValueError(f"Dataset root outside private Lambda workspace: {folder}")
            paths = S3EpisodeResolver._get_filtered_paths(hydra.utils.instantiate(dataset.filters))
            if not paths:
                raise RuntimeError(f"Empty dataset for {experiment}")
            folder.mkdir(parents=True, exist_ok=True)
            new_paths = [(source, episode) for source, episode in paths if (str(folder), episode) not in union]
            lines = sync_lines(new_paths, folder, dataset.resolver.get("bucket_name", "rldb"))
            if lines:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".s5cmd", dir=args.output.parent) as batch:
                    batch.write("\n".join(lines) + "\n")
                    batch.flush()
                    print(f"Staging {experiment}: {len(new_paths)} episodes", flush=True)
                    subprocess.run(["s5cmd", "--log", "error", "--endpoint-url", env["R2_ENDPOINT_URL"], "--numworkers", "32", "run", batch.name], env=env, check=True)
            missing = [episode for _, episode in paths if not (folder / episode).is_dir()]
            if missing:
                raise RuntimeError(f"Missing {len(missing)} episode directories after successful sync")
            union.update((str(folder), episode) for _, episode in paths)
            groups.append(dict(experiment=experiment, embodiment=name, episodes=len(paths),
                               folder=str(folder), catalog_sha256=hashlib.sha256(json.dumps(sorted(paths)).encode()).hexdigest()))
    receipt = dict(job=os.environ["SLURM_JOB_ID"],
                   commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   groups=groups, episodes=len(union), status="sync_complete")
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=2) + "\n")
    tmp.replace(args.output)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
