import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from egomimic.benchmarks.libero.arc_sweep import fixed_replay_spec, profile_settings
from egomimic.benchmarks.libero.replay import candidates_from_spec


@pytest.mark.parametrize(
    "profile,mode",
    [
        ("stk_1", "stk"),
        ("stk_2", "stk"),
        ("dur_1", "dur"),
        ("dur_2", "dur"),
        ("shared", "stk"),
        ("shared", "dur"),
    ],
)
def test_frozen_sweep_replay_preserves_choice_and_uses_fresh_confirmation(
    profile, mode
):
    settings = profile_settings(profile, mode)
    spec = fixed_replay_spec(profile, mode)
    candidates = candidates_from_spec(spec)
    assert len(candidates) == 1
    candidate = next(iter(candidates.values()))
    assert candidate == {
        "mode": mode,
        "num_waypoints": settings["arc_waypoints"],
        "max_rotation_degrees": settings["arc_max_rotation_degrees"],
        "max_translation": settings["arc_max_translation"],
    }
    assert spec["calibration_demos"] == [0, 1]
    assert spec["selection_demos"] == list(range(2, 10))
    assert spec["confirmation_demos"] == list(range(10, 25))
    assert not set(spec["confirmation_demos"]) & set(range(30, 50))
    assert spec["minimum_execution_coverage"] == 0.99
    assert spec["max_raw_repeat_state_error"] == 0
    assert spec["max_dense_action_mse"] == 1e-12


def test_profile_rejects_wrong_mode_and_workflow_omissions():
    with pytest.raises(ValueError, match="incompatible"):
        profile_settings("stk_1", "dur")
    path = Path(__file__).parents[1] / "scripts/benchmarks/launch_libero_osmo.py"
    spec = importlib.util.spec_from_file_location("sweep_osmo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = dict(
        commit="a" * 40,
        run_id="sweep-stk1-10",
        suite="libero_10",
        mode="full",
        arc_modes=["stk"],
        arc_profile="stk_1",
    )
    with pytest.raises(ValueError, match="requires calibration"):
        module.workflow(**args)
    result = module.workflow(
        **args, calibration_parent="parent", oat_reference_run="baseline"
    )["workflow"]
    env = result["tasks"][0]["environment"]
    assert env["RUN_KIND"] == "arc_sweep" and env["ARC_PROFILE"] == "stk_1"
    assert env["OAT_REFERENCE_RUN"] == "baseline"
    assert result["resources"]["default"]["gpu"] == 1
    assert result["resources"]["default"]["memory"] == "128Gi"
    assert result["resources"]["default"]["cpu"] == 20


@pytest.mark.parametrize("replay_failure", [False, True])
def test_policy_launch_follows_replay_and_stops_on_preflight_failure(
    tmp_path, monkeypatch, replay_failure
):
    from egomimic.benchmarks.libero import arc_sweep, cluster

    calls = []

    def execute(argv, log):
        calls.append(argv)
        if replay_failure:
            raise RuntimeError("preflight failed")

    monkeypatch.setattr(cluster, "execute", execute)
    monkeypatch.setenv("REPLAY_CALIBRATION_PARENT", "parent-stk")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sweep",
            "--root",
            str(tmp_path),
            "--suite",
            "libero_10",
            "--run-id",
            "sweep-stk1-10",
            "--profile",
            "stk_1",
            "--arc-mode",
            "stk",
        ],
    )
    if replay_failure:
        with pytest.raises(RuntimeError, match="preflight failed"):
            arc_sweep.main()
        assert len(calls) == 1
    else:
        arc_sweep.main()
        assert len(calls) == 2 and "--arc-only" in calls[1]
        assert calls[0][2].endswith(".replay") and calls[1][2].endswith(".cluster")
    assert (
        json.loads((tmp_path / "fixed-replay-spec.json").read_text())["frozen_profile"]
        == "stk_1"
    )


@pytest.mark.parametrize(
    "run_mode,bad_replay", [("smoke", False), ("full", False), ("full", True)]
)
def test_arc_only_never_trains_oat_and_checks_frozen_profile(
    tmp_path, monkeypatch, run_mode, bad_replay
):
    import torch

    from egomimic.benchmarks.libero import cluster
    from egomimic.benchmarks.libero.rollout import rollout_plan, run_rollouts
    from tests.test_libero_benchmark import FakeEnvironment, FakePolicy

    monkeypatch.setenv("SOURCE_COMMIT", "a" * 40)
    for name in (
        "CAMPAIGN_ID",
        "CAMPAIGN_RUNS_JSON",
        "RESUME_FROM_RUN",
        "EVALUATE_FROM_RUN",
        "ARC_REPLAY_RUN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ARC_MODES_JSON", '["stk"]')
    monkeypatch.setenv("ARC_REPLAY_RUNS_JSON", '{"stk":"verified-replay"}')
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "L40S")
    monkeypatch.setattr(cluster.subprocess, "check_output", lambda *a, **k: "a" * 40)

    def no_op(*a, **k):
        pass

    monkeypatch.setattr(torch.cuda, "synchronize", no_op)

    class Uploader:
        def __init__(self, *a):
            self.client = None
            self.prefix = "test/"
            self.thread = SimpleNamespace(start=no_op, join=no_op)
            self.stop = SimpleNamespace(set=no_op)

        upload = no_op

    monkeypatch.setattr(cluster, "ArtifactUploader", Uploader)
    monkeypatch.setattr(cluster, "stage_dataset", lambda *a: tmp_path / "data.zarr")
    monkeypatch.setattr(cluster, "configure_simulator", no_op)
    settings = profile_settings("stk_1", "stk")
    monkeypatch.setattr(
        cluster,
        "load_arc_calibration",
        lambda *a, **k: settings | ({"arc_waypoints": 16} if bad_replay else {}),
    )
    calls = []

    def execute(argv, log):
        calls.append(argv)
        if "egomimic.trainHydra" in argv:
            assert "+experiment=oat/libero_arc_stk_policy" in argv
            assert "benchmark.arc_waypoints=36" in argv
            checkpoint = tmp_path / "evidence/training/arc_stk/checkpoints/last.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
        if "rollout" in argv:
            output = Path(argv[argv.index("--output") + 1])
            run_rollouts(
                FakePolicy(),
                rollout_plan("libero_10")
                if run_mode == "full"
                else rollout_plan("libero_10", trials_per_task=1, repetitions=1),
                output,
                max_episode_steps=550 if run_mode == "full" else 4,
                env_factory=FakeEnvironment,
                metadata={
                    "suite": "libero_10",
                    "method": "arc",
                    "representation": {
                        "mode": "stk",
                        "waypoints": 36,
                        "max_translation": 1.6,
                        "max_rotation_degrees": 192,
                    },
                    "horizon": 32,
                    "n_obs_steps": 2,
                    "n_action_steps": 16,
                    "use_ema": True,
                },
            )

    monkeypatch.setattr(cluster, "execute", execute)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cluster",
            "--root",
            str(tmp_path),
            "--suite",
            "libero_10",
            "--run-id",
            "sweep",
            "--mode",
            run_mode,
            "--arc-only",
            "--arc-profile",
            "stk_1",
        ],
    )
    if bad_replay:
        with pytest.raises(ValueError, match="differs from frozen"):
            cluster.main()
        assert not any("egomimic.trainHydra" in c for c in calls)
    else:
        cluster.main()
        assert sum("egomimic.trainHydra" in c for c in calls) == 1
        assert not any("reconstruct" in c for c in calls)
        result = json.loads((tmp_path / "evidence/arc-results.json").read_text())
        assert set(result["variants"]) == {"stk"}
        assert result["paired_oat_comparison_complete"] is False
        assert json.loads((tmp_path / "evidence/status.json").read_text())["state"] == (
            "ARC_POLICIES_COMPLETE" if run_mode == "full" else "ARC_SMOKE_PASSED"
        )
