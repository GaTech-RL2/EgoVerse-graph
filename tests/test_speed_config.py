from pathlib import Path
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def test_pair_config_diff_is_only_encoding(monkeypatch):
    monkeypatch.setenv("PUSHSHAPES_USOCKET_ROOT", "/unused/u")
    monkeypatch.setenv("PUSHSHAPES_CHAIN_GRIPPER_ROOT", "/unused/c")
    root = Path(__file__).resolve().parents[1] / "egomimic/hydra_configs"
    configs = []
    with initialize_config_dir(config_dir=str(root), version_base="1.3"):
        for encoding in ("scalar", "fourier"):
            c = compose(config_name="train_zarr_cartesian", overrides=[
                "hydra/launcher=basic", "+experiment=pusht/action_flow_cotrain_uc_speed_interpolation",
                "speed_diagnostic.speed_reference=84.11600368466028",
                f"speed_diagnostic.encoding={encoding}",
            ])
            assert c.model.hidden_dim == 696
            assert c.model.pipeline._recursive_ is False
            assert c.trainer.max_steps == 80000
            assert c.data.train_datasets.pushshapes_sim_u_socket.resolver.key_map.rates == [1.]
            assert c.data.train_datasets.pushshapes_sim_chain_gripper.resolver.key_map.rates == [1,1.25,1.5,1.75,2]
            # Normalize the intentional differing value before comparison.
            c.speed_diagnostic.encoding = "scalar"
            configs.append(OmegaConf.to_container(c, resolve=False))
    assert configs[0] == configs[1]


def test_factory_places_speed_inside_diagnostic_prefix(monkeypatch):
    import hydra.utils
    from egomimic.pipeline.core import Stage
    from egomimic.pipeline.stages_speed import build_speed_conditioned_pipeline, SharedSpeedCondition

    class Marker(Stage):
        def __init__(self, config):
            super().__init__()
            self.config = config

    monkeypatch.setattr(hydra.utils, "instantiate", lambda c: Marker(c))
    configs = OmegaConf.create([
        {"_target_": "unused.FusedObsEncoder"},
        {"_target_": "unused.RoutedContentEncoderStage"},
        {"_target_": "unused.LatentBridgeStage"},
        {"_target_": "unused.ConditionalVelocityStage"},
    ])
    pipeline = build_speed_conditioned_pipeline(configs, 84.116, device="cpu")
    stages = pipeline.pipeline.stages
    assert isinstance(stages[1], SharedSpeedCondition)
    assert stages[2].config["_target_"].endswith("RoutedContentEncoderStage")
    assert stages[3].config["condition_key"] == "speed_condition"
    assert stages[4].config["inference_condition_key"] == "speed_condition"
