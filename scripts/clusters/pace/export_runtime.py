"""Export explicit readonly runtime mounts from an allocated CPU-only Slurm step.

No preparation, ML imports, recursive discovery outside those roots, or transfer.
Archive names preserve logical container paths; symlinks are never dereferenced.
"""

import argparse
import hashlib
import json
import os
import stat
import tarfile
import time
from pathlib import Path, PurePosixPath

DENIED = {
    ".ssh",
    ".aws",
    ".netrc",
    ".git",
    ".env",
    ".git-credentials",
    ".npmrc",
    ".pypirc",
    "credentials",
    "secrets",
    "logs",
    "log-mirror",
    "wandb",
    "wandb-cache",
    "__pycache__",
}
INTERPRETERS = {"/usr/bin/python", "/usr/bin/python3", "/usr/bin/python3.12"}


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def omitted(path):
    name = path.name
    need(
        not name.startswith((".env.", "id_rsa", "id_ed25519")),
        "Credential-like runtime input",
    )
    return (
        name in DENIED
        or name.endswith((".log", ".out", ".err"))
        or (
            name.endswith((".pem", ".key"))
            and not (name == "cacert.pem" and path.parent.name == "certifi")
        )
    )


def inventory(mappings):
    entries, omitted_names, directory_names = {}, [], set()
    for cached, logical in mappings:
        cached, logical = Path(cached), Path(logical)
        need(
            cached.is_dir() and not cached.is_symlink(),
            "Runtime root is not a directory",
        )
        for directory, dirs, files in os.walk(cached, followlinks=False):
            d = Path(directory)
            logical_dir = logical / d.relative_to(cached)
            directory_names.add(str(logical_dir))
            for name in list(dirs):
                path = d / name
                if omitted(path):
                    omitted_names.append(str(logical_dir / name))
                    dirs.remove(name)
                elif path.is_symlink():
                    files.append(name)
                    dirs.remove(name)
            for name in files:
                path, target = d / name, str(logical_dir / name)
                if omitted(path):
                    omitted_names.append(target)
                    continue
                need(
                    target.startswith("/workspace/")
                    and not set(Path(target).parts) & DENIED,
                    "Unsafe logical runtime path",
                )
                need(
                    not target.endswith(".ckpt"),
                    "Runtime contains a training checkpoint",
                )
                mode = path.lstat().st_mode
                need(stat.S_ISREG(mode) or stat.S_ISLNK(mode), "Special runtime file")
                need(target not in entries, "Overlapping runtime mounts")
                entries[target] = path
    links = []
    for target, path in entries.items():
        if path.is_symlink():
            literal = os.readlink(path)
            resolved = os.path.normpath(os.path.join(str(Path(target).parent), literal))
            need(not any(c in literal for c in "\n\r\t\0"), "Unsafe symlink")
            need(
                resolved in INTERPRETERS
                or resolved in entries
                or resolved in directory_names,
                "Symlink leaves the exact runtime closure: "
                + target
                + " -> "
                + resolved,
            )
            links.append(dict(path=target.lstrip("/"), target=literal))
    return entries, links, omitted_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mounts", type=Path, required=True)
    parser.add_argument("--mounts-sha256", required=True)
    parser.add_argument("--cache-prefix", required=True)
    parser.add_argument("--exclude-logical", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    need(
        os.environ.get("SLURM_JOB_ID", "").isdigit()
        and os.environ.get("SLURM_STEP_ID", "").isdigit()
        and os.environ.get("SLURM_CPUS_PER_TASK") == "2",
        "Allocated two-CPU Slurm step required",
    )
    need(
        not os.environ.get("SLURM_STEP_GPUS")
        and os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "NoDevFiles", "-1"),
        "CPU-only export must not expose GPUs",
    )
    need(sha(args.mounts) == args.mounts_sha256, "Mount binding changed")
    mappings = []
    for item in args.mounts.read_text().strip().split(","):
        cached, logical, mode = item.split(":")
        need(
            mode == "ro"
            and cached.startswith(args.cache_prefix.rstrip("/") + "/workspace/")
            and logical.startswith("/workspace/")
            and str(PurePosixPath(logical)) == logical
            and ".." not in Path(logical).parts,
            "Unapproved runtime mount",
        )
        if logical not in args.exclude_logical:
            mappings.append((cached, logical))
    need(mappings, "No runtime mounts")
    args.output.mkdir()
    started = time.time()
    (args.output / "started.json").write_text(
        json.dumps(
            dict(
                job_id=os.environ["SLURM_JOB_ID"],
                step_id=os.environ["SLURM_STEP_ID"],
                mappings=mappings,
                training=False,
            )
        )
        + "\n"
    )
    entries, links, omissions = inventory(mappings)
    archive, manifest = (
        args.output / "runtime.tar.gz",
        args.output / "runtime.files.sha256",
    )
    total_bytes = 0
    with (
        tarfile.open(archive, "x:gz", compresslevel=1, dereference=False) as tar,
        manifest.open("x") as hashes,
    ):
        for logical, path in sorted(entries.items()):
            before = path.lstat()
            if path.is_symlink():
                tar.add(path, arcname=logical.lstrip("/"), recursive=False)
            else:
                digest = sha(path)
                hashes.write(digest + "  " + logical.lstrip("/") + "\n")
                tar.add(path, arcname=logical.lstrip("/"), recursive=False)
                total_bytes += before.st_size
            after = path.lstat()
            need(
                (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                == (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                "Runtime changed during export",
            )
    receipt = dict(
        schema="readonly-runtime-export-v1",
        status="PASS",
        mappings=mappings,
        job_id=os.environ["SLURM_JOB_ID"],
        step_id=os.environ["SLURM_STEP_ID"],
        files=len(entries),
        regular_bytes=total_bytes,
        symlinks=links,
        omissions=omissions,
        archive=dict(
            path=str(archive), bytes=archive.stat().st_size, sha256=sha(archive)
        ),
        manifest=dict(path=str(manifest), sha256=sha(manifest)),
        seconds=time.time() - started,
    )
    with (args.output / "receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(dict(status="PASS", files=len(entries), bytes=total_bytes)),
        flush=True,
    )


if __name__ == "__main__":
    main()
