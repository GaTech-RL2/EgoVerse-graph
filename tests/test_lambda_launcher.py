from pathlib import Path

import hydra


def test_lambda_launcher_composes_whole_node_pyxis_contract(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv(
        "EGOVERSE_ABC_DATASET_DIR", "/workspace/users/ani-cheluva/datasets/arc_abc"
    )
    monkeypatch.setenv("PROJECT_ROOT", str(root))
    with hydra.initialize_config_dir(
        version_base=None, config_dir=str(root / "egomimic/hydra_configs")
    ):
        cfg = hydra.compose(
            config_name="train_zarr_cartesian",
            return_hydra_config=True,
            overrides=[
                "+experiment=abc_arc/abc_multitask4_hpt300_baseline_visual_openloop",
                "hydra/launcher=submitit_lambda_h100",
                "launch_params.gpus_per_node=8",
            ],
        )
    launch = cfg.hydra.launcher
    assert launch.account == launch.partition == "loaner"
    assert launch.tasks_per_node == 8
    assert launch.gres == "gpu:h100:8"
    assert launch.additional_parameters.exclusive
    assert launch.additional_parameters.requeue
    assert "--container-mounts=/workspace:/workspace:rw" in launch.srun_args
    instance = hydra.utils.instantiate(launch)
    assert instance.params["srun_args"] == list(launch.srun_args)
    assert instance.params["mem_gb"] >= 128
    assert cfg.norm_stats.sample_frac == 0.20
