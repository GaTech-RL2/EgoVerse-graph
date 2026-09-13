import hashlib
import tomllib
from pathlib import Path

import yaml

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
ROOT = PYPROJECT.parent


def _extra_conflict_pairs(conflicts: list[list[dict[str, str]]]) -> set[frozenset[str]]:
    pairs = set()
    for group in conflicts:
        selected = [requirement["extra"] for requirement in group]
        if len(selected) == 2:
            pairs.add(frozenset(selected))
    return pairs


def test_yam_dependencies_are_isolated_from_aria_and_pi05() -> None:
    with PYPROJECT.open("rb") as file:
        project = tomllib.load(file)

    base_dependencies = project["project"]["dependencies"]
    extras = project["project"]["optional-dependencies"]
    conflicts = project["tool"]["uv"]["conflicts"]

    assert not any(
        dependency.startswith("projectaria-tools") for dependency in base_dependencies
    )
    assert extras["aria"] == ["projectaria-tools[all]==2.0.0"]
    assert "mink==1.1.0" in base_dependencies

    conflict_pairs = _extra_conflict_pairs(conflicts)
    assert frozenset(("aria", "yam")) in conflict_pairs
    assert frozenset(("pi05", "yam")) in conflict_pairs


def test_yam_pipeline_control_and_calibration_copies_are_byte_exact() -> None:
    expected = {
        "egomimic/robot/yam/quest_mapper.py": (
            "a94d333cc3467da41670b49e77dd3e37f565f1f82579c28a6ad31f0f7d2a5b8d"
        ),
        "egomimic/robot/yam/streaming_ik.py": (
            "488a453ab6623d05d3c690d3008970f83700a149240a8715c205f43ff46b8add"
        ),
        "egomimic/hydra_configs/robot/yam_rl2_agentview_extrinsics.yaml": (
            "5795bd2480e3ef31a1ffcf993834613bd3bf267082e24c23c30094848fbd122c"
        ),
        "egomimic/hydra_configs/robot/yam_rl2_workspace.yaml": (
            "bef2471d3a6d759728bcb7d85eb2943fd9f0377cf65d450d4712f2ad43f9a9b9"
        ),
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_rl2_yam_profile_locks_reference_yaw_rate_and_streaming_ik() -> None:
    profile = yaml.safe_load(
        (ROOT / "egomimic/hydra_configs/robot/yam_rl2_collect.yaml").read_text()
    )
    assert profile["teleop"]["headset_yaw_degrees"] == 180.0
    assert profile["teleop"]["orientation_rx_degrees"] == 0.0
    assert profile["frequency"] == 60
    assert profile["recording"]["rate_hz"] == 30
    assert profile["max_joint_velocity"] == 2.5
    assert profile["robot"]["teleop_kinematics"] == {
        "ee_site": "tcp_site",
        "n_arm": 6,
        "dt": 1 / 60,
        "steps": 4,
        "gain": 0.5,
        "damping": 0.01,
        "posture_cost": 0.08,
        "max_joint_vel": 2.5,
    }


def test_rl2_yam_compatibility_calibration_matches_reference_copy() -> None:
    reference = yaml.safe_load(
        (
            ROOT / "egomimic/hydra_configs/robot/yam_rl2_agentview_extrinsics.yaml"
        ).read_text()
    )
    calibration = yaml.safe_load(
        (ROOT / "egomimic/hydra_configs/calibration/rl2yam.yaml").read_text()
    )
    assert calibration["camera_serial"] == reference["serial"]
    assert calibration["convention"] == reference["convention"]
    assert calibration["distortion"] == reference["dist"]
    assert calibration["intrinsics"] == [row + [0.0] for row in reference["K"]]
    assert calibration["extrinsics"] == {
        "left": reference["channels"]["can_follower_l"]["base_T_camera"],
        "right": reference["channels"]["can_follower_r"]["base_T_camera"],
    }
