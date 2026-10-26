"""Load a hash-bound operational service beside an immutable scientific source.

The runtime bootstrap still owns Torch and dependency admission. This second
stage selects operational tools and their explicit cluster dependencies, never
egomimic or its configs.
Use it in every child process so the old source cannot shadow newer recovery
and archive admission code through Python's namespace-package search order.
"""

import hashlib
import importlib
import json
import runpy
import sys
from pathlib import Path


# Dependencies must ship with the service, not be assumed present in historical
# scientific checkouts. Keep this allowlist narrow; model/data code stays there.
DEPENDENCIES = {
    "scripts/clusters/pace/__init__.py",
    "scripts/clusters/pace/gpu_ownership.py",
    "scripts/clusters/pace/gpu_admission.py",
    "scripts/clusters/pace/pace_podman.py",
    "scripts/ice/ice_checkpoint_mirror.py",
}


def service_files(root):
    """Exact export closure shared by packaging and admission regression checks."""
    root = Path(root)
    names = {str(p.relative_to(root)) for p in
             (root / "scripts/train/joint").iterdir()
             if p.is_file() and p.suffix in (".py", ".sbatch", ".sh")}
    names |= DEPENDENCIES
    for name in names:
        if not (root / name).is_file() or (root / name).is_symlink():
            raise ValueError("Missing operational dependency: " + name)
    return sorted(names)


def activate(request):
    binding = request["operational_service"]
    root = Path(binding["root"])
    manifest = Path(binding["manifest"])
    if not root.is_absolute() or manifest.parent != root:
        raise ValueError("Absolute content-bound service root required")
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != binding["manifest_sha256"]:
        raise ValueError("Operational service manifest changed")
    names = set()
    for line in manifest.read_text().splitlines():
        expected, name = line.split("  ", 1)
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or (relative.parts[:3] != ("scripts", "train", "joint") and name not in DEPENDENCIES):
            raise ValueError("Service contains an unapproved operational path")
        path = root / relative
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Operational service changed: " + name)
        names.add(name)
    required = {"scripts/train/joint/"+n+".py" for n in (
        "service_bootstrap", "run", "contract", "continuation", "allocation_boundary", "retention", "verify_full",
    )}
    if not (required | DEPENDENCIES) <= names:
        raise ValueError("Incomplete joint service")
    # A namespace may already be imported by run_module. Select this audited
    # service first, while preserving source-owned modules not included here.
    sys.path.insert(0, str(root))
    for name in ("scripts", "scripts.train", "scripts.train.joint",
                 "scripts.clusters", "scripts.clusters.pace", "scripts.ice"):
        package = importlib.import_module(name)
        local = str(root.joinpath(*name.split(".")))
        package.__path__ = [local] + [str(p) for p in package.__path__ if str(p) != local]
    owned_modules = {str(Path(n).with_suffix("")).replace("/", ".") for n in names
                     if n.endswith(".py")}
    for name, module in list(sys.modules.items()):
        if name in owned_modules:
            filename = getattr(module, "__file__", None)
            if filename and not Path(filename).resolve().is_relative_to(root.resolve()):
                raise ValueError("Service imported before operational admission: " + name)
    return root


def command(request, request_path, runtime_bootstrap, arguments):
    prefix = [sys.executable, str(runtime_bootstrap)]
    if request.get("operational_service"):
        prefix += [str(Path(request["operational_service"]["root"])/"scripts/train/joint/service_bootstrap.py"),
                   str(request_path), "--"]
    return prefix + list(arguments)


def main():
    request_path, delimiter, *arguments = sys.argv[1:]
    if delimiter != "--" or not arguments:
        raise ValueError("Expected request -- module-or-script")
    root = activate(json.loads(Path(request_path).read_text()))
    if arguments in (["--check-only"], ["--check-dependencies"]):
        tools = (("contract",) if arguments == ["--check-dependencies"] else
                 ("contract", "continuation", "retention", "allocation_boundary"))
        paths = {name: importlib.import_module("scripts.train.joint."+name).__file__ for name in tools}
        for name in DEPENDENCIES:
            module = name.removesuffix(".py").replace("/", ".")
            paths[module] = importlib.import_module(module).__file__
        if not all(Path(p).resolve().is_relative_to(root.resolve()) for p in paths.values()):
            raise ValueError("Operational tool import drift")
        print(json.dumps(dict(status="PASS", operational_imports=paths,
                              model_package=importlib.import_module("egomimic").__file__)))
        return
    if arguments[0] == "-m":
        sys.argv = [arguments[1], *arguments[2:]]
        runpy.run_module(arguments[1], run_name="__main__", alter_sys=True)
    else:
        sys.argv = arguments
        runpy.run_path(arguments[0], run_name="__main__")


if __name__ == "__main__":
    main()
