"""Historical native LIBERO recipes, not current OAT-matched launch defaults.

Legacy public names remain aliases for checkpoint-bound callers.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NativeLaunchProfile:
    suite: str
    name: str
    replay_environment: str
    task_uids: tuple
    action_velocity_weight: float = 1.0

    @property
    def experiment(self):
        return "libero_historical/" + self.name


HISTORICAL_NATIVE_PROFILES = {
    "libero10": NativeLaunchProfile(
        "libero10",
        "action_flow_libero10_h240_euler50_dithalf_80k_s42",
        "LIBERO10_REPLAY_ROOT",
        tuple(range(30, 40)),
    ),
    "libero_object": NativeLaunchProfile(
        "libero_object",
        "action_flow_libero_object_h240_euler50_dithalf_80k_s42",
        "LIBERO_OBJECT_REPLAY_ROOT",
        tuple(range(10, 20)),
    ),
    "libero_goal": NativeLaunchProfile(
        "libero_goal",
        "action_flow_libero_goal_h240_euler50_dithalf_80k_s42",
        "LIBERO_GOAL_REPLAY_ROOT",
        tuple(range(20, 30)),
    ),
    "libero_spatial": NativeLaunchProfile(
        "libero_spatial",
        "action_flow_libero_spatial_h240_euler50_dithalf_80k_s42",
        "LIBERO_SPATIAL_REPLAY_ROOT",
        tuple(range(10)),
    ),
}

AV0_PROFILES = {
    suite: NativeLaunchProfile(
        p.suite,
        p.name.replace("_80k_s42", "_80k_av0_s42"),
        p.replay_environment,
        p.task_uids,
        0.0,
    )
    for suite, p in HISTORICAL_NATIVE_PROFILES.items()
}


def historical_profile_for_suite(suite):
    try:
        return HISTORICAL_NATIVE_PROFILES[suite]
    except (KeyError, TypeError):
        raise ValueError("unsupported native LIBERO suite: " + str(suite)) from None


def profile_for_experiment(experiment):
    for profile in (*HISTORICAL_NATIVE_PROFILES.values(), *AV0_PROFILES.values()):
        if profile.experiment == experiment:
            return profile
    raise ValueError("unsupported native LIBERO experiment: " + str(experiment))


def profile_for_config(config):
    profile = profile_for_experiment("libero_historical/" + str(config.get("name")))
    if config["benchmark"]["suite"] != profile.suite:
        raise ValueError("native suite/recipe identity mismatch")
    expected_ablation = (
        "action_velocity_off" if profile.action_velocity_weight == 0.0 else None
    )
    if config.get("run_provenance", {}).get("ablation") != expected_ablation:
        raise ValueError("native ablation provenance/profile mismatch")
    if (
        config.get("run_provenance", {})
        .get("objective", {})
        .get(
            "action_velocity_weight",
            1.0 if profile.action_velocity_weight == 1.0 else None,
        )
        != profile.action_velocity_weight
    ):
        raise ValueError("native ablation objective provenance mismatch")
    for mode in ("train", "valid"):
        if (
            config["data"][mode + "_datasets"]["libero_panda"]["resolver"]["suite"]
            != profile.suite
        ):
            raise ValueError("native suite/resolver identity mismatch")
    return profile


def profile_for_argv(argv):
    selected = [
        arg.split("=", 1)[1]
        for arg in argv
        if arg.split("=", 1)[0].lstrip("+") == "experiment" and "=" in arg
    ]
    if len(selected) != 1:
        raise ValueError("exactly one native experiment override required")
    return profile_for_experiment(selected[0])


# Compatibility aliases preserve historical checkpoint and external launcher callers.
PROFILES = HISTORICAL_NATIVE_PROFILES
profile_for_suite = historical_profile_for_suite


@dataclass(frozen=True)
class JointNativeLaunchProfile:
    """Explicit unequal-width recipe; never an alias of historical native7."""

    name: str
    suite: str
    sources: tuple

    @property
    def experiment(self):
        return "libero/" + self.name


SPATIAL_HAND_JOINT_PROFILE = JointNativeLaunchProfile(
    "action_flow_libero_spatial_sharpa_gripper_keypoints_s42",
    "libero_spatial",
    (("libero_panda", 21, 7, 2), ("libero_panda_sharpa", 22, 69, 22)),
)


def joint_phase_argv(
    *,
    phase,
    source_root,
    original_root,
    sharpa_root,
    output_root,
    context_path,
    split_path,
    split_sha256,
    normalization_sha256,
    source_commit,
    diagnostic_seed_bank,
    wandb_entity,
    wandb_project,
    wandb_run_id,
    data_preparation_source_sha256=None,
):
    """One exact Hydra argv factory for the explicit 7/69 operational route.

    The maintained launcher must use this same array for composition and
    execution. This only describes a phase: source, corpus/context, smoke,
    allocation and safe-requeue gates are independently required before run.
    No historical single-body selector or default is relaxed here.
    """
    import json
    import re
    from pathlib import Path

    if phase not in {"normalization", "preflight", "smoke", "full"}:
        raise ValueError("unsupported joint phase")
    data_only_snapshot = phase == "normalization" and source_commit is None
    if data_preparation_source_sha256 is not None or data_only_snapshot:
        if (
            not data_only_snapshot
            or not isinstance(data_preparation_source_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", data_preparation_source_sha256)
        ):
            raise ValueError(
                "data-only source snapshots require normalization phase and an exact SHA"
            )
    for label, value in (
        ("source_commit", source_commit),
        ("split_sha256", split_sha256),
        ("normalization_sha256", normalization_sha256),
    ):
        if (
            label == "normalization_sha256"
            and phase == "normalization"
            and value is None
        ):
            continue  # The fitting phase creates this identity; never invent one.
        if label == "source_commit" and data_only_snapshot:
            continue  # An audited data-only snapshot is not a clean training checkout.
        width = 40 if label == "source_commit" else 64
        if not isinstance(value, str) or not re.fullmatch(
            r"[0-9a-f]{%d}" % width, value
        ):
            raise ValueError("invalid joint " + label)
    for value in (
        source_root,
        original_root,
        sharpa_root,
        output_root,
        context_path,
        split_path,
        diagnostic_seed_bank,
    ):
        if (
            not isinstance(value, str)
            or not Path(value).is_absolute()
            or any(character in value for character in ("\n", "\r", "\0"))
        ):
            raise ValueError(
                "joint artifact paths must be absolute single-line strings"
            )
    for value in (wandb_entity, wandb_project, wandb_run_id):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise ValueError("unsafe joint W&B identity")

    def text(key, value):
        return key + "=" + json.dumps(value)

    argv = [
        "--config-name=train_zarr_cartesian",
        "hydra/launcher=basic",
        "+experiment=" + SPATIAL_HAND_JOINT_PROFILE.experiment,
        "mode=train",
        "ckpt_path=null",
        text("hydra.run.dir", output_root),
        text("++paths.root_dir", source_root),
        text("paths.work_dir", source_root),
        text("paths.output_dir", output_root),
        text("trainer.default_root_dir", output_root),
        text("benchmark.dataset", original_root),
        text("benchmark.sharpa_dataset", sharpa_root),
        "launch_params.gpus_per_node=1",
        "launch_params.nodes=1",
        "trainer.devices=1",
        "++trainer.num_nodes=1",
        "++trainer.strategy=auto",
        "trainer.precision=bf16-mixed",
        "trainer.accumulate_grad_batches=1",
        "trainer.max_epochs=-1",
        "trainer.min_epochs=null",
        "trainer.check_val_every_n_epoch=null",
        "trainer.num_sanity_val_steps=0",
        "trainer.limit_train_batches=1.0",
        "trainer.limit_val_batches=1.0",
        "callbacks.model_checkpoint.save_top_k=-1",
        "callbacks.model_checkpoint.save_last=false",
        "callbacks.model_checkpoint.every_n_epochs=null",
        "++callbacks.model_checkpoint.train_time_interval=null",
        "callbacks.model_checkpoint.save_on_train_epoch_end=false",
        text(
            "callbacks.model_checkpoint.dirpath", str(Path(output_root) / "checkpoints")
        ),
        "runtime.slurm_requeue_owner=runner",
        "runtime.slurm_save_signal=SIGUSR2",
        text(
            "runtime.slurm_signal_checkpoint_dir",
            str(Path(output_root) / "checkpoints"),
        ),
        text("++run_provenance.split_manifest_path", split_path),
        text("++run_provenance.split_manifest_sha256", split_sha256),
        text("run_provenance.diagnostic_seed_bank_path", diagnostic_seed_bank),
        text("++run_provenance.run_kind", phase),
        text("++logger.wandb.entity", wandb_entity),
        text("++logger.wandb.project", wandb_project),
        text("++logger.wandb.id", wandb_run_id),
        text("++logger.wandb.name", wandb_run_id),
        text("++logger.wandb.save_dir", output_root),
        "++logger.wandb.offline=false",
        "++logger.wandb.mode=online",
        "++logger.wandb.resume=allow"
        if phase == "full"
        else "++logger.wandb.resume=never",
    ]
    if data_only_snapshot:
        argv.append("++run_provenance.source_commit=null")
        argv.append(
            text(
                "++run_provenance.data_preparation_source_snapshot_sha256",
                data_preparation_source_sha256,
            )
        )
    else:
        argv.append(text("++run_provenance.source_commit", source_commit))
    if normalization_sha256 is not None:
        argv.append(text("++run_provenance.normalization_sha256", normalization_sha256))
    if phase == "normalization":
        # Caller exports context.snapshot(), not just the numeric stats cache.
        argv = [arg for arg in argv if not arg.lstrip("+").startswith("logger.")]
        argv += [
            "norm_stats_only=true",
            "norm_stats.precomputed_norm_path=null",
            text("++norm_stats.export_context_path", context_path),
            text("norm_stats.save_cache_dir", output_root),
            "~logger",
            "trainer.accelerator=cpu",
            "trainer.max_steps=120000",
            "trainer.val_check_interval=15000",
        ]
    else:
        argv += [
            "norm_stats_only=false",
            "norm_stats.save_cache_dir=null",
            text("norm_stats.precomputed_norm_path", context_path),
            "trainer.accelerator=cpu"
            if phase == "preflight"
            else "trainer.accelerator=gpu",
            "trainer.max_steps=8" if phase == "smoke" else "trainer.max_steps=120000",
            "trainer.val_check_interval=8"
            if phase == "smoke"
            else "trainer.val_check_interval=15000",
            "callbacks.model_checkpoint.every_n_train_steps=8"
            if phase == "smoke"
            else "callbacks.model_checkpoint.every_n_train_steps=15000",
        ]
    if phase == "smoke":
        argv.append("trainer.log_every_n_steps=1")
    return argv


def joint_profile_for_config(config):
    """Validate the joint neural/data interface without constructing a model.

    Full context, source, runtime, smoke, storage and allocation proofs remain
    separate gates. This function alone never issues a launch-ready verdict.
    """
    profile = SPATIAL_HAND_JOINT_PROFILE

    def require(condition, message):
        if not condition:
            raise ValueError("joint LIBERO: " + message)

    require(config.get("name") == profile.name, "unsupported experiment")
    require(
        config["benchmark"]["suite"] == profile.suite and config["seed"] == 42,
        "suite/seed mismatch",
    )
    stages = config["model"]["pipeline"]["stages"]
    names = [stage["_target_"].rsplit(".", 1)[-1] for stage in stages]
    require(
        names
        == [
            "KeyedFeatureProjection",
            "OATObservationStage",
            "ActionTargetBuilder",
            "GaussianLatentNoise",
            "RoutedContentEncoderStage",
            "LatentBridgeStage",
            "ConditionalVelocityStage",
            "RoutedContentDecoderStage",
            "ActionFlowObjectiveStage",
        ],
        "private-codec/shared-field stage topology changed",
    )
    source_names = {source[0] for source in profile.sources}
    require(
        all(
            set(config["data"][mode + "_datasets"]) == source_names
            for mode in ("train", "valid")
        ),
        "dataset source identities differ",
    )
    require(
        set(stages[4]["encoders"]) == set(stages[7]["decoders"]) == source_names,
        "private action codec routes differ",
    )
    require(
        (stages[3]["num_tokens"], stages[3]["latent_dim"]) == (16, 16),
        "shared latent shape differs",
    )
    for name, identity, action_dim, hand_dim in profile.sources:
        require(
            stages[4]["route_aliases"].get(
                str(identity), stages[4]["route_aliases"].get(identity)
            )
            == name
            and stages[7]["route_aliases"].get(
                str(identity), stages[7]["route_aliases"].get(identity)
            )
            == name,
            "embodiment route identity differs",
        )
        encoder, decoder = stages[4]["encoders"][name], stages[7]["decoders"][name]
        for codec in (encoder, decoder):
            require(
                (
                    codec["action_dim"],
                    codec["action_horizon"],
                    codec["latent_dim"],
                    codec["num_latent_tokens"],
                )
                == (action_dim, 32, 16, 16),
                "private codec dimensions differ",
            )
        for backbone in (encoder["backbone"], decoder):
            require(
                (backbone["hidden_dim"], backbone["depth"], backbone["num_heads"])
                == (240, 12, 8),
                "private codec backbone differs",
            )
        require(
            "condition_dim" not in decoder and "condition_key" not in decoder,
            "action decoder must remain context-free",
        )
        for mode in ("train", "valid"):
            dataset = config["data"][mode + "_datasets"][name]
            resolver = dataset["resolver"]
            require(
                dataset["valid_ratio"] == 0.1 and dataset["split_seed"] == 42,
                "episode split differs",
            )
            require(resolver["suite"] == profile.suite, "resolver suite differs")
            require(
                resolver.get("action_representation", "native")
                == ("native" if action_dim == 7 else "wrist_hand_keypoints"),
                "target representation differs",
            )
            require(
                resolver.get("action_dim", 7) == action_dim
                and resolver.get("gripper_state_dim", 2) == hand_dim,
                "resolver dimensions differ",
            )
        params = config["data"]["train_dataloader_params"][name]
        require(
            params["batch_size"] == 16 and params["drop_last"],
            "fixed per-source batch differs",
        )
    field = stages[6]
    require(
        (
            field["field"]["backbone"]["hidden_dim"],
            field["field"]["backbone"]["depth"],
            field["field"]["backbone"]["num_heads"],
        )
        == (240, 12, 8),
        "shared backbone differs",
    )
    require(
        (config["benchmark"]["n_obs_steps"], config["benchmark"]["horizon"]) == (2, 32),
        "history/horizon differs",
    )
    require(
        "fields" not in field
        and "experts" not in field
        and field["field"]["_target_"].endswith(".UniteActionFlowVelocityField"),
        "one shared conditional field required",
    )
    require(
        (field["inference_method"], field["num_inference_steps"], field["cfg_scale"])
        == ("euler", 50, 4.0),
        "inference contract differs",
    )
    require(
        stages[8]["action_velocity_weight"] == 1.0
        and stages[8]["flow_aggregation"] == "sum_samples"
        and stages[5]["samples_per_content"] == 14,
        "AV1/FM14sum objective differs",
    )
    require(
        config["normalizer"]["_target_"].endswith(".LiberoMultiEmbodimentNormalizer")
        and config["norm_stats"]["norm_mode"] == "libero_train_limits",
        "train-only private normalization required",
    )
    require(
        config["evaluator"]["action_flow_diagnostics"]["provenance"].get("latent_shape")
        == [16, 16],
        "diagnostic latent shape must match shared16x16 graph",
    )
    trainer = config["trainer"]
    require(
        trainer["precision"] == "bf16-mixed"
        and trainer["devices"] == 1
        and trainer["accumulate_grad_batches"] == 1
        and trainer["gradient_clip_val"] == 3.0,
        "precision/world/batch/clip differs",
    )
    return profile
