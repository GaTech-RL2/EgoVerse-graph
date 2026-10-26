"""Bootstrap the reviewed optional runtime in every trainer Python process."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import site
import sys


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    root = Path(os.environ["PIXEL_FA4_ROOT"])
    binding = json.loads(Path(__file__).with_name("runtime-binding.json").read_text())
    if str(root) != binding["root"]:
        raise ValueError("Unexpected FA4 runtime location")
    for name, expected in binding["file_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Runtime receipt or API changed: {name}")
    # Process both the CuTe namespace hook and CUDA bindings redirector hook.
    site.addsitedir(str(root / "runtime"))
    for path in reversed([root / "torch-runtime", root / "runtime"]):
        name = str(path)
        if name in sys.path:
            sys.path.remove(name)
        sys.path.insert(0, name)
    fa3_name = os.environ.get("PIXEL_FA3_ROOT")
    if fa3_name:
        fa3_root = Path(fa3_name)
        fa3_binding = json.loads(Path(__file__).with_name("fa3-binding.json").read_text())
        if str(fa3_root) != fa3_binding["root"]:
            raise ValueError("Unexpected FA3 runtime location")
        for name, expected in fa3_binding["file_sha256"].items():
            if digest(fa3_root / name) != expected:
                raise ValueError(f"FA3 installed file changed: {name}")
        sys.path.insert(0, str(fa3_root))
        for name, expected in fa3_binding["versions"].items():
            if importlib.metadata.version(name) != expected:
                raise ValueError(f"FA3 runtime version changed: {name}")
    source = os.environ.get("PIXEL_SOURCE_ROOT", str(Path(os.environ["PIXEL_TASK"]) / "source"))
    sys.path.insert(0, source)
    for name, expected in binding["versions"].items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise ValueError(f"Runtime version changed for {name}: {actual}")
    import torch
    if not Path(torch.__file__).is_relative_to(root / "torch-runtime"):
        raise ValueError("Torch imported outside the isolated FA4 overlay")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    args = sys.argv[1:]
    if args[0] == "-m":
        sys.argv = [args[1], *args[2:]]
        runpy.run_module(args[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv = args
        sys.path.insert(0, str(Path(args[0]).resolve().parent))
        runpy.run_path(args[0], run_name="__main__")


if __name__ == "__main__":
    main()
