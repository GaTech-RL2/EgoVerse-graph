"""Regression checks for inherited BC recipes and scheduled config generation."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

REPO = Path(os.environ.get("EGO_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO / "scripts/e1"))
import recipe_builders as builders  # noqa: E402 - script entry points share this directory
from build_slowpace_aria_cotrain import human_leaf, robot_base  # noqa: E402


@pytest.mark.parametrize("variant", ["time", "arcdur", "arcdurhyb", "arcvel"])
def test_frozen_elmoaidan_rebuild_preserves_contract(variant):
    data = builders.elmoaidan_data(variant)
    cfg = builders.compose_recipe("scratch_rl2_stattempo_elmoaidan_" + variant)
    assert data == OmegaConf.to_container(cfg.data, resolve=True)
    for group in ("train_datasets", "valid_datasets"):
        resolver = data[group]["yam_bimanual"]["resolver"]
        assert resolver["key_map"]["yam_source_frames"] == 100
        if variant.endswith("hyb"):
            assert (
                resolver["transform_list"]["rotation_distance_unit"]
                == builders.ROTATION_DISTANCE_UNIT
            )
            assert (
                cfg.model.inference.compatibility.tokenizer.rotation_distance_unit
                == builders.ROTATION_DISTANCE_UNIT
            )


@pytest.mark.parametrize(
    "variant", ["time", "arcdur", "arcdurhyb", "arcvel", "arcvelhyb"]
)
def test_frozen_aria37_rebuild_preserves_both_domain_contracts(variant):
    manifest = json.loads(
        (REPO / "scripts/e1/stationery_slowpace_aria_manifest.json").read_text()
    )
    assert len(set(manifest["episodes"])) == manifest["n"] == 37
    data = OmegaConf.to_container(robot_base(variant), resolve=True)
    data["train_datasets"]["human_bimanual"] = human_leaf(
        variant, manifest["episodes"], set(manifest["embodiment"])
    )
    data["train_dataloader_params"]["human_bimanual"] = {
        "batch_size": 32,
        "num_workers": 6,
        "persistent_workers": True,
    }
    cfg = builders.compose_recipe("cotrain_rl2_stattempo_slowpace_aria_" + variant)
    assert data == OmegaConf.to_container(cfg.data, resolve=True)
    assert set(cfg.model.data_requirements.preprocessing) == {"3", "7"}
    human = data["train_datasets"]["human_bimanual"]["resolver"]
    assert human["key_map"]["horizon"] == 100
    assert human["key_map"]["drop_wrist_images"] is True
    assert data["source_fps"] == 30
    if variant.endswith("hyb"):
        assert (
            human["transform_list"]["rotation_distance_unit"]
            == builders.ROTATION_DISTANCE_UNIT
        )
        decoder = cfg.model.inference.profiles.default.adapter.decoder
        assert decoder.rotation_distance_unit == builders.ROTATION_DISTANCE_UNIT


def test_pace_login_guard_requires_a_srun_step(monkeypatch):
    monkeypatch.setattr(builders.socket, "gethostname", lambda: "phoenix-login-1")
    monkeypatch.delenv("SLURM_STEP_ID", raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    with pytest.raises(RuntimeError, match="srun"):
        builders.require_compute_node()
    monkeypatch.setenv("SLURM_STEP_ID", "0")
    builders.require_compute_node()


@pytest.mark.parametrize("allocation", [False, True])
def test_launcher_schedules_every_python_command(tmp_path, allocation):
    # The stubs execute no Python and submit no jobs. They show that even the
    # build and metadata probes take the scheduler path on a login shell.
    bindir = tmp_path / "bin"
    bindir.mkdir()
    py = bindir / "fake-python"
    py.write_text("""#!/bin/bash
printf '%s\\n' "$*" >> "$CALLS/python"
if [ "${1:-}" = -c ]; then echo '37 time arcdur arcdurhyb arcvel arcvelhyb'; else echo CHECK_OK; fi
""")
    srun = bindir / "srun"
    srun.write_text("""#!/bin/bash
printf '%s\\n' "$*" >> "$CALLS/srun"
while [ "$#" -gt 0 ] && [ "$1" != "$PY" ]; do shift; done
[ "$#" -gt 0 ] || exit 91
exec "$@"
""")
    sbatch = bindir / "sbatch"
    sbatch.write_text("""#!/bin/bash
printf '%s\\n' "$*" >> "$CALLS/sbatch"
echo 12345
""")
    for executable in (py, srun, sbatch):
        executable.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(bindir) + os.pathsep + os.environ["PATH"],
        "PY": str(py),
        "OUT": str(tmp_path / "runs"),
        "CALLS": str(tmp_path),
        "MODE": "launch",
    }
    if allocation:
        env["SLURM_JOB_ID"] = "777"
    else:
        env.pop("SLURM_JOB_ID", None)
    subprocess.run(
        ["bash", str(REPO / "scripts/e1/launch_slowpace_aria_cotrain.sh")],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert len((tmp_path / "python").read_text().splitlines()) == 3
    assert len((tmp_path / "srun").read_text().splitlines()) == 3
    assert len((tmp_path / "sbatch").read_text().splitlines()) == 5
    if allocation:
        assert all(
            line.startswith("--ntasks=1 ")
            for line in (tmp_path / "srun").read_text().splitlines()
        )
    else:
        assert all(
            "-p cpu-small" in line
            for line in (tmp_path / "srun").read_text().splitlines()
        )
