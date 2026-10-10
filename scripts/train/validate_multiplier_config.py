"""Fail-closed typed validation for the approved interpolation-only pair.

Importable by the canonical launcher; command-line mode composes the exact
launcher overrides without loading data. No historical co-training bypass.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

EXPERIMENT = "pusht/action_flow_cotrain_uc_multiplier_interpolation"
SOURCE = "344dd3baf5ba942d947e0a0e136c8990025d91e5"
REFERENCE = 84.11600368466028
ROUTES = ("pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper")
COUNTS = {"scalar": 471549402, "fourier": 471549530}


def validate_speed_contract(cfg, encoding):
    assert encoding in COUNTS
    assert cfg.name == EXPERIMENT.split("/")[-1]
    assert cfg.speed_diagnostic.encoding == encoding
    assert cfg.speed_diagnostic.get("speed_reference") is None
    assert cfg.speed_diagnostic.fps == 30
    assert list(cfg.speed_diagnostic.rates_chain) == [1, 1.25, 1.5, 1.75, 2]
    assert (
        cfg.model.pipeline._target_
        == "egomimic.pipeline.stages_speed.build_speed_conditioned_pipeline"
    )
    assert cfg.model.pipeline._recursive_ is False
    assert cfg.model.pipeline.encoding == encoding
    assert cfg.model.pipeline.get("speed_reference") is None
    assert cfg.model.pipeline.conditioning_input == "retiming_multiplier"
    assert cfg.model.hidden_dim == 696
    assert cfg.model.action_horizon == 16
    assert cfg.model.num_latent_tokens == 8 and cfg.model.latent_dim == 16
    assert cfg.model.condition_dim == 128
    assert cfg.model.reconstruction_weight == cfg.model.flow_weight == 1
    assert cfg.model.flow_samples_per_content == 14
    assert cfg.model.num_inference_steps == 50
    assert cfg.seed == 42 and cfg.ckpt_path is None
    assert str(cfg.trainer.precision) == "bf16"
    assert (
        cfg.trainer.devices
        == cfg.trainer.num_nodes
        == cfg.trainer.accumulate_grad_batches
        == 1
    )
    assert (
        cfg.run_provenance.comparison_label
        == "NON-PROTOCOL_INTERPOLATION_ONLY_DIAGNOSTIC"
    )
    assert cfg.run_provenance.speed_conditioning.contact_physics_verified is False
    assert cfg.evaluator.energy_score_distance is None
    assert cfg.evaluator.energy_score_distances_by_embodiment[ROUTES[1]] is None
    assert OmegaConf.to_container(
        cfg.evaluator.energy_score_provenance.distance_contract, resolve=True
    ) == OmegaConf.to_container(
        cfg.evaluator.energy_score_distances_by_embodiment, resolve=True
    )
    for split in ("train", "valid"):
        datasets = cfg.data[f"{split}_datasets"]
        assert set(datasets) == set(ROUTES)
        for route, rates, representation in zip(
            ROUTES, ([1], [1, 1.25, 1.5, 1.75, 2]), ("rotvec4", "points6")
        ):
            resolver = datasets[route].resolver
            keymap, transforms = resolver.key_map, resolver.transform_list
            assert keymap._target_.endswith(".get_retimed_planar_keymap")
            assert transforms._target_.endswith(".get_retimed_planar_transforms")
            assert list(keymap.rates) == list(transforms.rates) == rates
            assert transforms.representation == representation
            assert keymap.action_horizon == transforms.action_horizon == 16
            assert keymap.fps == transforms.fps == 30
            assert cfg.data[f"{split}_dataloader_params"][route].batch_size == 32
    stages = cfg.model.pipeline.stages
    assert stages[4]._target_.endswith("RoutedContentEncoderStage")
    assert stages[7]._target_.endswith("RoutedContentDecoderStage")
    assert stages[6].flow_clean_gradient_mode == "all_stopgrad"
    assert stages[8].action_velocity_weight == 1
    assert stages[8].flow_aggregation == "sum_samples"
    assert cfg.model.optimizer._target_.endswith("ReleasedUniteCompositeOptimizer")
    assert cfg.model.optimizer.lr == 1e-4
    sch = cfg.model.scheduler
    assert sch.warmup_steps == 8000
    assert sch.decay_start_1_steps == 12000 and sch.decay_end_1_steps == 20000
    assert sch.final_lr == 5e-5
    assert cfg.callbacks.model_checkpoint.save_top_k == -1
    # Launcher-owned metadata must describe the actual phase, not inherited
    # H384/10k labels. Source-only constructor probes lack launcher_sha256.
    if "launcher_sha256" in cfg.run_provenance:
        arch = cfg.run_provenance.architecture
        assert arch.family == "action_flow_private_codecs_shared_field_h696d12h12"
        assert arch.tokenizer == "two_private_UniteDiTBackbone_h696_d12_h12"
        assert arch.denoiser == "one_shared_UniteDiTBackbone_h696_d12_h12"
        assert arch.decoder == "two_private_UniteActionDecoder_h696_d12_h12"
        assert (
            cfg.run_provenance.validation.interval_optimizer_steps
            == cfg.trainer.val_check_interval
        )
        assert (
            cfg.run_provenance.validation.limit_batches == cfg.trainer.limit_val_batches
        )


def validate_speed_pipeline(cfg, pipeline):
    stages = tuple(pipeline.pipeline.stages)

    def one(name):
        matches = [s for s in stages if type(s).__name__ == name]
        assert len(matches) == 1, (name, len(matches))
        return matches[0]

    speed = one("SharedSpeedCondition")
    encoder = one("RoutedContentEncoderStage")
    decoder = one("RoutedContentDecoderStage")
    field = one("ConditionalVelocityStage")
    bridge = one("LatentBridgeStage")
    assert stages.index(speed) < stages.index(encoder)
    assert (
        bridge.condition_key
        == field.inference_condition_key
        == speed.output_key
        == "speed_condition"
    )
    assert speed.encoding == cfg.speed_diagnostic.encoding
    assert speed.conditioning_input == "retiming_multiplier"
    assert not hasattr(speed, "speed_reference")
    assert field.flow_clean_gradient_mode == "all_stopgrad"
    assert field.field.backbone.hidden_dim == 696
    assert field.field.backbone.depth == field.field.backbone.num_heads == 12
    for route, dim in zip(ROUTES, (4, 6)):
        assert (
            encoder.encoder[route].input_dim == decoder.decoder[route].action_dim == dim
        )
    count = sum(p.numel() for p in pipeline.nets.parameters())
    assert count == COUNTS[speed.encoding], (count, COUNTS[speed.encoding])
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--instantiate", action="store_true")
    args = parser.parse_args()
    assert args.experiment == EXPERIMENT
    with initialize_config_dir(
        config_dir=str(args.config_root.resolve()), version_base="1.3"
    ):
        cfg = compose(
            config_name="train_zarr_cartesian",
            overrides=["+experiment=" + EXPERIMENT, *args.override],
            return_hydra_config=True,
        )
    # Mirror the maintained config validator's runtime initialization, but
    # retain the exact output/run identities rather than comparison sentinels.
    with open_dict(cfg.hydra.runtime):
        cfg.hydra.runtime.cwd = str(Path.cwd())
        cfg.hydra.runtime.output_dir = str(cfg.hydra.run.dir)
    HydraConfig.instance().set_config(cfg)
    cfg = OmegaConf.masked_copy(cfg, [k for k in cfg if k != "hydra"])
    OmegaConf.resolve(cfg)
    encoding = os.environ["AF_SPEED_ENCODING"]
    validate_speed_contract(cfg, encoding)
    count = None
    if args.instantiate:
        from hydra.utils import instantiate

        count = validate_speed_pipeline(
            cfg, instantiate(cfg.model.pipeline, device="cpu")
        )
    resolved = OmegaConf.to_yaml(cfg, resolve=True)
    payload = dict(
        status="PASS",
        experiment=EXPERIMENT,
        encoding=encoding,
        resolved_config_sha256=hashlib.sha256(resolved.encode()).hexdigest(),
        parameter_count=count,
        model_instantiated=args.instantiate,
        validation_mode="typed_speed_contract",
    )
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
