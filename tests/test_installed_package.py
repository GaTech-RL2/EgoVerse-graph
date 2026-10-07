"""Build independently and check wheel resources without checkout fallbacks."""

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def installed_wheel(tmp_path_factory):
    work = tmp_path_factory.mktemp("package")
    source = work / "source"
    source.mkdir()
    shutil.copytree(
        ROOT / "egomimic",
        source / "egomimic",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        if (ROOT / name).exists():
            shutil.copy2(ROOT / name, source / name)
    proc = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(work / "dist")],
        cwd=source,
        text=True,
        capture_output=True,
        timeout=240,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    wheel = next((work / "dist").glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    subprocess.run(
        ["uv", "venv", str(work / "env"), "--python", sys.executable],
        check=True,
        capture_output=True,
    )
    python = work / "env/bin/python"
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)],
        check=True,
        capture_output=True,
    )
    return work, python, names


def test_all_declared_resources_are_in_wheel(installed_wheel):
    _, _, names = installed_wheel
    required = set()
    for folder in (
        "hydra_configs",
        "resources",
        "robot/teleop_dashboard_static",
        "robot/rollout_dashboard_static",
    ):
        required.update(
            str(path.relative_to(ROOT))
            for path in (ROOT / "egomimic" / folder).rglob("*")
            if path.suffix
            in {".yaml", ".json", ".xml", ".urdf", ".html", ".css", ".js"}
        )
    required.update(
        str(path.relative_to(ROOT))
        for path in (ROOT / "egomimic/robot/eva/stanford_repo/models/meshes/X5").glob(
            "*.STL"
        )
    )
    assert required <= set(names), sorted(required - set(names))
    assert not any(name.startswith(("tests/", "external/")) for name in names)


def test_fresh_install_loads_resources_outside_checkout(installed_wheel):
    work, python, names = installed_wheel
    resources = [name for name in names if name.startswith("egomimic/resources/")]
    code = """
import egomimic, json, pathlib, sys, xml.etree.ElementTree as ET
root = pathlib.Path(egomimic.__file__).parent
assert 'site-packages' in str(root), root
for name in json.loads(sys.argv[1]):
    path = root.parent / name
    ET.parse(path)
tree = ET.parse(root / 'resources/model_x5.xml')
meshdir = root / 'resources' / tree.find('compiler').attrib['meshdir']
for mesh in tree.findall('asset/mesh'):
    assert (meshdir / mesh.attrib['file']).is_file(), mesh.attrib
"""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [str(python), "-c", code, json.dumps(resources)],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
