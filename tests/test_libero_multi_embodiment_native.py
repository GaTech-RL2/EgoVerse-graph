import json

import h5py
import numpy as np
import pytest
import torch
import zarr

from egomimic.benchmarks.libero.catalog import TASK_IDS, get_tasks
from egomimic.benchmarks.libero.convert import (
    convert_collection_manifest,
    native_episode_values,
)
from egomimic.pipeline.stages_sampler import KeyedFeatureProjection
from egomimic.rldb.zarr.libero_dataset import (
    LiberoDataset,
    LiberoMultiEmbodimentNormalizer,
    LiberoReplayResolver,
    keymap,
)


def replay(path, action_dim, hand_dim, demos_per_task=2):
    tasks = get_tasks("libero_spatial")
    ids = np.repeat([TASK_IDS[t][2] for t in tasks for _ in range(demos_per_task)], 2)
    n = len(ids)
    rng = np.random.default_rng(4)
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    data = group.create_group("data")
    for key, values in {
        "action": rng.uniform(-1, 1, (n, action_dim)).astype("float32"),
        "robot0_gripper_qpos": rng.uniform(0, 1, (n, hand_dim)).astype("float32"),
        "robot0_eef_pos": rng.uniform(0, 1, (n, 3)).astype("float32"),
        "robot0_eef_quat": np.tile(np.array([0, 0, 0, 1], dtype="float32"), (n, 1)),
        "task_uid": ids[:, None],
        "agentview_rgb": rng.integers(0, 255, (n, 2, 2, 3), dtype="uint8"),
        "robot0_eye_in_hand_rgb": rng.integers(0, 255, (n, 2, 2, 3), dtype="uint8"),
    }.items():
        data.create_array(key, data=values)
    group.create_group("meta").create_array("episode_ends", data=np.arange(2, n + 1, 2))
    return group


def resolver(path, sharpa=False):
    return LiberoReplayResolver(
        path,
        keymap(2, 32),
        "libero_spatial",
        **(
            dict(action_dim=28, gripper_state_dim=22, embodiment="libero_panda_sharpa")
            if sharpa
            else {}
        ),
    )


def test_distinct_native_widths_and_reject_sharpa_as_original(tmp_path):
    path = tmp_path / "sharpa.zarr"
    replay(path, 28, 22)
    with pytest.raises(ValueError, match="action must"):
        resolver(path).resolve()
    sample = next(iter(resolver(path, True).resolve().values()))[0]
    assert sample["embodiment"] == 22
    assert sample["actions"].shape == (32, 28)
    assert sample["robot0_gripper_qpos"].shape == (2, 22)
    assert torch.equal(sample["actions"][1], sample["actions"][-1])
    with pytest.raises(ValueError, match="Sharpa contract"):
        LiberoReplayResolver(
            path,
            keymap(),
            "libero_spatial",
            action_dim=7,
            gripper_state_dim=22,
            embodiment="libero_panda_sharpa",
        )


def test_configured_replay_split_proves_rows_and_rejects_leakage(tmp_path):
    from egomimic.rldb.zarr.libero_saved_state import make_replay_split_proof

    path = tmp_path / "replay.zarr"
    replay(path, 7, 2)
    r = resolver(path)
    train = LiberoDataset._from_resolver(r, mode="train", valid_ratio=0.1)
    valid = LiberoDataset._from_resolver(r, mode="valid", valid_ratio=0.1)
    kwargs = dict(expected_episodes=20, valid_ratio=0.1, seed=42)
    proof = make_replay_split_proof(train, valid, **kwargs)
    assert len(proof["train"]) == 18 and len(proof["valid"]) == 2
    rows = sorted(proof["train"] + proof["valid"], key=lambda row: row["start"])
    assert [(row["start"], row["end"]) for row in rows] == [
        (i, i + 2) for i in range(0, 40, 2)
    ]
    assert set(row["physical_id"] for row in proof["train"]).isdisjoint(
        row["physical_id"] for row in proof["valid"]
    )
    with pytest.raises(ValueError, match="coverage"):
        make_replay_split_proof(train, valid, **{**kwargs, "expected_episodes": 21})
    key = next(iter(valid.datasets))
    valid.datasets[key] = next(iter(train.datasets.values()))
    with pytest.raises(ValueError, match="physical rows"):
        make_replay_split_proof(train, valid, **kwargs)


def test_joint_operational_split_materializes_450_50_and_rejects_wrong_binding(
    tmp_path, monkeypatch
):
    import copy
    from pathlib import Path

    import hydra
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    from egomimic.benchmarks.libero.native_launch_profiles import (
        SPATIAL_HAND_JOINT_PROFILE,
    )
    from scripts.train.libero_native_launch_contract import (
        joint_split_contract,
        normalize_joint_context,
        sha,
    )

    root = Path(__file__).resolve().parents[1]
    manifests = {}
    for name, width, hand in (("libero_panda", 7, 2), ("libero_panda_sharpa", 69, 22)):
        path = tmp_path / name
        replay(path, width, hand, demos_per_task=50)
        manifest = dict(complete=True, action_dim=width)
        if width == 69:
            manifest.update(
                action_representation="wrist_hand_keypoints",
                keypoint_frame="wrist_relative_Quest_XYZ_m",
                observation_alignment="causal_pre_action_raw_lossless",
            )
        (path / "benchmark_manifest.json").write_text(json.dumps(manifest))
        manifests[name] = sha(path / "benchmark_manifest.json")
    monkeypatch.setenv("LIBERO_ORIGINAL_SPATIAL_ROOT", str(tmp_path / "libero_panda"))
    monkeypatch.setenv(
        "LIBERO_SHARPA_SPATIAL_ROOT", str(tmp_path / "libero_panda_sharpa")
    )
    monkeypatch.setenv("LIBERO_DIAGNOSTIC_SEED_BANK", str(tmp_path / "seeds.json"))
    with hydra.initialize_config_dir(
        version_base="1.2", config_dir=str(root / "egomimic/hydra_configs")
    ):
        cfg = hydra.compose(
            config_name="train_zarr_cartesian",
            return_hydra_config=True,
            overrides=[
                "hydra/launcher=basic",
                "+experiment=" + SPATIAL_HAND_JOINT_PROFILE.experiment,
                "trainer.precision=bf16-mixed",
                "trainer.devices=1",
            ],
        )
        cfg.hydra.runtime.output_dir = str(tmp_path / "output")
        HydraConfig.instance().set_config(cfg)
        resolved = OmegaConf.to_container(
            OmegaConf.masked_copy(cfg, [k for k in cfg if k != "hydra"]), resolve=True
        )
    result = joint_split_contract(resolved, manifests, expected_episodes=500)
    assert result["status"] == "PASS" and result["launch_ready"] is False
    for proof in result["sources"].values():
        assert len(proof["train"]) == 450 and len(proof["valid"]) == 50
        assert proof["frames"] == 1000 and proof["complete_episode_and_frame_coverage"]
    with pytest.raises(ValueError, match="manifest SHA"):
        joint_split_contract(
            resolved, {**manifests, "libero_panda": "0" * 64}, expected_episodes=500
        )
    wrong = copy.deepcopy(resolved)
    for mode in ("train", "valid"):
        wrong["data"][mode + "_datasets"]["libero_panda_sharpa"]["resolver"][
            "folder_path"
        ] = str(tmp_path / "libero_panda")
    with pytest.raises(ValueError, match="distinct"):
        joint_split_contract(wrong, manifests, expected_episodes=500)
    # Real operational handoff: physical proof from a YAML config -> configured
    # DataModule fitting -> complete context, with no model or trainer.
    original_config = tmp_path / "split-config.yaml"
    OmegaConf.save(OmegaConf.create(resolved), original_config)
    result["resolved_config"] = dict(
        path=str(original_config), sha256=sha(original_config)
    )
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(result))
    fit = copy.deepcopy(resolved)
    fit["norm_stats_only"] = True
    fit["norm_stats"].update(
        precomputed_norm_path=None,
        save_cache_dir=str(tmp_path / "stats"),
        export_context_path=str(tmp_path / "context.json"),
    )
    fit["run_provenance"].update(
        split_manifest_path=str(split_path), split_manifest_sha256=sha(split_path)
    )
    fit["data"]["run_provenance"] = copy.deepcopy(fit["run_provenance"])
    norm = normalize_joint_context(fit)
    assert norm["status"] == "PASS" and norm["launch_ready"] is False
    context = json.loads((tmp_path / "context.json").read_text())["data_context"]
    assert set(context["normalizer_state"]["norm_stats"]) == {"21", "22"}
    assert (
        len(context["normalizer_state"]["norm_stats"]["22"]["actions"]["scale"]) == 69
    )
    # The checkpoint verifier must restore the same complete state without data IO.
    from egomimic.pipeline.checkpoint_binding import checkpoint_binding
    from egomimic.rldb.zarr.data_module import load_data_context
    from scripts.train.verify_libero_native_action_flow_smoke import (
        validate_joint_checkpoint_context,
    )

    checkpoint_config = copy.deepcopy(fit)
    checkpoint_config["norm_stats"]["precomputed_norm_path"] = str(
        tmp_path / "context.json"
    )
    checkpoint_config["run_provenance"].update(
        source_commit="1" * 40, normalization_sha256=sha(tmp_path / "context.json")
    )
    restored_context = load_data_context(tmp_path / "context.json")
    checkpoint = dict(
        hyper_parameters=dict(config_tree=checkpoint_config),
        data_context=restored_context.snapshot(),
        inference_binding=checkpoint_binding(checkpoint_config, restored_context),
    )
    assert (
        validate_joint_checkpoint_context(checkpoint, checkpoint_config)[
            "checkpoint_binding"
        ]
        == checkpoint["inference_binding"]
    )
    wrong_checkpoint = copy.deepcopy(checkpoint)
    wrong_checkpoint["data_context"].pop("preprocessing")
    with pytest.raises(ValueError, match="full data context differs"):
        validate_joint_checkpoint_context(wrong_checkpoint, checkpoint_config)
    # Consume genuine fixture context/split plus explicitly synthetic constructor/runtime metadata.
    from scripts.train.libero_native_launch_contract import (
        contract_sha,
        joint_preflight_evidence,
        reference,
    )

    pre = copy.deepcopy(checkpoint_config)
    pre["norm_stats_only"] = False
    pre["trainer"]["accelerator"] = "cpu"
    pre["run_provenance"]["run_kind"] = "preflight"
    config_path = tmp_path / "preflight.yaml"
    OmegaConf.save(OmegaConf.create(pre), config_path)
    constructor_path = tmp_path / "synthetic-constructor.json"
    constructor = dict(
        status="PASS",
        full_model_constructor=True,
        parameter_count=78182614,
        source_commit="1" * 40,
        resolved_config_sha256=sha(config_path),
        model_contract_sha256=contract_sha(pre),
    )
    constructor_path.write_text(json.dumps(constructor))
    runtime_path = tmp_path / "synthetic-runtime.json"
    runtime_path.write_text(
        json.dumps(
            {
                "python": {},
                "installed_distributions": {},
                "core_distribution_versions": {},
                "dependency_inputs": {},
                "torch_runtime": {},
                "source": {"head": "1" * 40, "dirty": False},
            }
        )
    )
    proof = joint_preflight_evidence(
        pre,
        config_path,
        constructor_ref=reference(constructor_path),
        runtime_ref=reference(runtime_path),
        source_head="1" * 40,
    )
    assert proof["launch_ready"] is False
    assert proof["complete_context"]["sha256"] == sha(tmp_path / "context.json")
    constructor["source_commit"] = "0" * 40
    constructor_path.write_text(json.dumps(constructor))
    with pytest.raises(ValueError, match="never relabel cached"):
        joint_preflight_evidence(
            pre,
            config_path,
            constructor_ref=reference(constructor_path),
            runtime_ref=reference(runtime_path),
            source_head="1" * 40,
        )
    with pytest.raises(ValueError, match="fresh absolute"):
        normalize_joint_context(fit)
    fit["norm_stats"]["export_context_path"] = str(tmp_path / "other-context.json")
    result["sources"]["libero_panda"]["episodes"] = 499
    split_path.write_text(json.dumps(result))
    fit["run_provenance"]["split_manifest_sha256"] = sha(split_path)
    with pytest.raises(ValueError, match="split/content binding"):
        normalize_joint_context(fit)
    fit["data"]["train_datasets"]["libero_panda"]["valid_ratio"] = 0.01
    with pytest.raises(ValueError, match="episode split differs"):
        normalize_joint_context(fit)


@pytest.mark.parametrize(
    "action", ["joint-normalize", "joint-preflight", "joint-compose", "joint-prepare"]
)
@pytest.mark.parametrize("step", ["batch", "extern"])
def test_joint_data_cli_rejects_non_numeric_scheduler_step(
    tmp_path, monkeypatch, step, action
):
    import sys

    from scripts.train.libero_native_launch_contract import main

    monkeypatch.setenv("SLURM_JOB_ID", "1234")
    monkeypatch.setenv("SLURM_STEP_ID", step)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "contract",
            "--action",
            action,
            "--repo",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(ValueError, match="scheduled numeric srun"):
        main()
    assert not (tmp_path / "out").exists()


def test_train_only_limits_separate_bodies_and_roundtrip(tmp_path):
    datasets = {}
    for name, width, hand in [("libero_panda", 7, 2), ("libero_panda_sharpa", 28, 22)]:
        path = tmp_path / (name + ".zarr")
        group = replay(path, width, hand)
        r = resolver(path, width == 28)
        train = LiberoDataset._from_resolver(r, mode="train", valid_ratio=0.1)
        valid = LiberoDataset._from_resolver(r, mode="valid", valid_ratio=0.1)
        assert not set(train.datasets) & set(valid.datasets)
        for episode in valid.datasets.values():
            group["data/action"][episode.start : episode.end] = 1000
        datasets[name] = train
    norm = LiberoMultiEmbodimentNormalizer()
    norm.populate_from_datasets(datasets)
    for name, dataset in datasets.items():
        norm.infer_shapes_from_batch(dataset[0])
        norm.infer_norm_from_dataset(dataset, name)
        ident = dataset.resolver.embodiment_id
        assert norm.norm_stats[ident]["actions"]["max"].max() < 1
        dataset.set_norm_stats_from(norm)
        sample = dataset[0]
        raw = next(iter(dataset.datasets.values()))[0]
        restored = norm.unnormalize({"actions": sample["actions"]}, ident)
        torch.testing.assert_close(restored["actions"], raw["actions"])
    assert norm.norm_stats[21]["actions"]["scale"].shape == (7,)
    assert norm.norm_stats[22]["actions"]["scale"].shape == (28,)
    restored = LiberoMultiEmbodimentNormalizer.from_state(norm.to_state())
    assert restored.tokenizer_context() == norm.tokenizer_context()
    valid = LiberoDataset._from_resolver(
        datasets["libero_panda"].resolver, mode="valid"
    )
    with pytest.raises(ValueError, match="training episodes"):
        LiberoMultiEmbodimentNormalizer().infer_norm_from_dataset(valid, "invalid")


def test_complete_joint_context_export_restores_without_refitting(
    tmp_path, monkeypatch
):
    import copy

    from omegaconf import OmegaConf

    from egomimic.rldb.zarr.data_module import ZarrDataModule

    train, valid = {}, {}
    for name, width, hand in (("libero_panda", 7, 2), ("libero_panda_sharpa", 69, 22)):
        path = tmp_path / (name + ".zarr")
        replay(path, width, hand)
        if width == 69:
            (path / "benchmark_manifest.json").write_text(
                json.dumps(
                    dict(
                        complete=True,
                        action_dim=69,
                        action_representation="wrist_hand_keypoints",
                        keypoint_frame="wrist_relative_Quest_XYZ_m",
                        observation_alignment="causal_pre_action_raw_lossless",
                    )
                )
            )
        train[name] = dict(
            _target_="egomimic.rldb.zarr.libero_dataset.LiberoDataset._from_resolver",
            resolver=dict(
                _target_="egomimic.rldb.zarr.libero_dataset.LiberoReplayResolver",
                folder_path=str(path),
                suite="libero_spatial",
                embodiment=name,
                action_dim=width,
                gripper_state_dim=hand,
                action_representation="native"
                if width == 7
                else "wrist_hand_keypoints",
                key_map=dict(
                    _target_="egomimic.rldb.zarr.libero_dataset.keymap",
                    n_obs_steps=2,
                    horizon=32,
                ),
            ),
            mode="train",
            valid_ratio=0.1,
            split_seed=42,
        )
        valid[name] = {**copy.deepcopy(train[name]), "mode": "valid"}

    def module():
        return ZarrDataModule(OmegaConf.create(train), OmegaConf.create(valid), {}, {})

    normalizer = OmegaConf.create(
        dict(
            _target_="egomimic.rldb.zarr.libero_dataset.LiberoMultiEmbodimentNormalizer"
        )
    )
    exported = tmp_path / "complete-context.json"
    context = module().prepare_context(
        mode="normalization",
        normalizer=normalizer,
        normalization=dict(
            norm_mode="libero_train_limits",
            save_cache_dir=str(tmp_path / "stats"),
            export_context_path=str(exported),
        ),
    )
    payload = json.loads(exported.read_text())["data_context"]
    assert payload["normalizer_config"] == OmegaConf.to_container(normalizer)
    assert set(payload["preprocessing"]) == {"21", "22"}
    assert (
        payload["normalizer_state"]["native_embodiment_contexts"]["22"]["action_dim"]
        == 69
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("A complete-context cache must not refit normalization")

    monkeypatch.setattr(
        LiberoMultiEmbodimentNormalizer, "infer_norm_from_dataset", forbidden
    )
    restored = module().prepare_context(
        mode="train",
        normalizer=normalizer,
        normalization=dict(
            norm_mode="libero_train_limits", precomputed_norm_path=str(exported)
        ),
    )
    assert (
        restored.normalizer.tokenizer_context()
        == context.normalizer.tokenizer_context()
    )
    assert restored.fingerprint() == context.fingerprint()
    # A changed export target is a hard collision, not an overwrite or refit.
    exported.write_text('{"different": true}\n')
    with pytest.raises(ValueError, match="Refusing to replace"):
        module().prepare_context(
            mode="normalization",
            normalizer=normalizer,
            restored_state=payload,
            normalization=dict(
                norm_mode="libero_train_limits",
                save_cache_dir=str(tmp_path / "other-stats"),
                export_context_path=str(exported),
            ),
        )
    assert exported.read_text() == '{"different": true}\n'
    assert not list(tmp_path.glob(".context-*"))


def test_original_state_is_identity_and_sharpa_projector_gets_gradient():
    stage = KeyedFeatureProjection(
        selector_key="embodiment",
        input_key=["eef", "quat", "hand", "task"],
        output_key="proprio_condition",
        output_dim=10,
        selector_aliases={21: "panda", 22: "sharpa"},
        projections={
            "panda": {"input_dim": 10, "identity": True},
            "sharpa": {"input_dim": 30, "hidden_dim": 32},
        },
    )
    for ident, hand_width in [(21, 2), (22, 22)]:
        batch = {
            "embodiment": torch.tensor([ident, ident]),
            "eef": torch.randn(2, 2, 3),
            "quat": torch.randn(2, 2, 4),
            "hand": torch.randn(2, 2, hand_width),
            "task": torch.randn(2, 2, 1),
        }
        expected = torch.cat([batch[k] for k in ("eef", "quat", "hand", "task")], -1)
        result = stage(batch)["proprio_condition"]
        assert result.shape == (2, 2, 10)
        if ident == 21:
            assert torch.equal(result, expected)
        else:
            result.square().mean().backward()
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in stage.projections["sharpa"].parameters()
            )
    with pytest.raises(ValueError, match="must match"):
        KeyedFeatureProjection(
            "body",
            "x",
            "y",
            {"panda": {"input_dim": 9, "identity": True}},
            output_dim=10,
        )


def test_native_collection_keeps_upright_rows_quaternion_and_timestamps(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    records = []
    image = np.zeros((2, 2, 2, 3), dtype="uint8")
    image[:, 0, :, 0] = 240
    image[:, 1, :, 2] = 170
    for i, task in enumerate(get_tasks("libero_spatial")):
        filename = f"task-{i}.hdf5"
        with h5py.File(root / filename, "w") as handle:
            handle.attrs["metadata_json"] = json.dumps(
                {"suite": "libero_spatial", "task": task, "embodiment": "sharpa"}
            )
            handle.attrs["automatic_success"] = True
            data = handle.create_group("data")
            data.attrs["rgb_row_transform"] = (
                "vertical_flip_from_raw_opengl_to_original_zarr_upright"
            )
            demo = data.create_group("demo_0")
            demo.attrs["observation_alignment"] = "after_action; raw timeline retained"
            demo.create_dataset("actions", data=np.zeros((2, 28), dtype="float32"))
            demo.create_dataset("action_host_ns", data=[10, 30])
            demo.create_dataset("observation_host_ns", data=[20, 40])
            obs = demo.create_group("obs")
            obs.create_dataset("agentview_rgb", data=image)
            obs.create_dataset("eye_in_hand_rgb", data=image)
            obs.create_dataset("ee_pos", data=np.zeros((2, 3)))
            obs.create_dataset("ee_quat", data=[[0, 0, 0, 1]] * 2)
            obs.create_dataset("ee_ori", data=np.zeros((2, 3)))
            obs.create_dataset("gripper_states", data=np.zeros((2, 22)))
            legacy_view = native_episode_values(demo, task, action_dim=28, hand_dim=22)
            np.testing.assert_array_equal(legacy_view["agentview_rgb"], image[:, ::-1])
        records.append(
            {
                "path": filename,
                "task": task,
                "suite": "libero_spatial",
                "action_dimension": 28,
                "automatic_success": True,
            }
        )
    manifest = {
        "suite": "libero_spatial",
        "embodiment": "custom_Panda_SharpaWaveRight_LIBERO",
        "episodes": records,
        "episode_count": 10,
        "per_task": {r["task"]: 1 for r in records},
    }
    m = root / "manifest.json"
    m.write_text(json.dumps(manifest))
    output = tmp_path / "converted.zarr"
    receipt = convert_collection_manifest(root, output, m)
    arrays = zarr.open_group(str(output), mode="r")["data"]
    np.testing.assert_array_equal(arrays["agentview_rgb"][:2], image)
    np.testing.assert_array_equal(arrays["robot0_eef_quat"][:2], [[0, 0, 0, 1]] * 2)
    np.testing.assert_array_equal(arrays["action_host_ns"][:2], [10, 30])
    np.testing.assert_array_equal(arrays["observation_host_ns"][:2], [20, 40])
    assert receipt["complete"] and not receipt["split_assigned"]
    assert len(receipt["episodes"]) == 10
    manifest["episodes"][0]["path"] = "../outside.hdf5"
    m.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="outside root"):
        convert_collection_manifest(root, tmp_path / "rejected.zarr", m)
    assert not (tmp_path / "rejected.zarr").exists()


def test_keypoint_alignment_rejects_asynchronous_latest_guess(tmp_path):
    from egomimic.benchmarks.libero.convert import aligned_hand_keypoint_actions

    metadata = {
        "hand_calibration": {
            "operator_wrist_position_quest_m": [0, 0, 0],
            "position_world_m": [0, 0, 0],
        },
        "world_from_quest": np.eye(3).tolist(),
        "control_settings": {"translation_gain": 1.0, "vertical_gain": 1.0},
    }
    points = np.arange(63).reshape(21, 3) * 0.001
    packets = [
        {
            "source": "quest_webxr",
            "hand": "right",
            "joint_layout": "mediapipe21.v2",
            "tracking_valid": True,
            "joint_valid": [True] * 21,
            "sequence": i,
            "wrist": {"position_m": [x, 0, 0]},
            "joints_m": (points + x).tolist(),
        }
        for i, x in enumerate((0.01, 0.02))
    ]
    with h5py.File(tmp_path / "record.hdf5", "w") as h:
        h.create_dataset("packet_json", data=[json.dumps(p) for p in packets])
        h.create_dataset("packet_receive_ns", data=[10, 20])
        h.create_dataset("action_host_ns", data=[30, 40])
        h.create_dataset(
            "intent_target_position_world_m", data=[[0.01, 0, 0], [0.02, 0, 0]]
        )
        h.create_dataset("actions", data=np.zeros((2, 28)))
        h.create_dataset("tracking_valid", data=[True, True])
        actions, start, sequence = aligned_hand_keypoint_actions(h, metadata)
        assert actions.shape == (2, 69) and start == 0
        assert sequence.tolist() == [
            0,
            1,
        ]  # Latest-at-action would wrongly choose 1 twice.
        np.testing.assert_allclose(
            actions[0, 6:], (points - points[0]).reshape(63), atol=1e-8
        )
        h["intent_target_position_world_m"][1] = [0.03, 0, 0]
        with pytest.raises(ValueError, match="correspondence at step 1"):
            aligned_hand_keypoint_actions(h, metadata)
        h["intent_target_position_world_m"][1] = [0.02, 0, 0]
        # Recorded IK failure pauses the control after applying a valid-hand
        # command; this is not a missing operator hand target.
        metadata["action_contract"] = {
            "arm_control": {"name": "mink_direct_joint_servo_v4", "mode": "mink"}
        }
        h.create_dataset("control_reason", data=[b"active", b"arm_ik_infeasible"])
        h["tracking_valid"][1] = False
        repaired, _, _ = aligned_hand_keypoint_actions(h, metadata)
        np.testing.assert_array_equal(repaired, actions)
        h["control_reason"][1] = b"tracking_lost"
        with pytest.raises(ValueError, match="hand tracking target"):
            aligned_hand_keypoint_actions(h, metadata)
        h["control_reason"][1] = b"arm_ik_infeasible"
        metadata["action_contract"]["arm_control"]["name"] = "unbound_controller"
        with pytest.raises(ValueError, match="hand tracking target"):
            aligned_hand_keypoint_actions(h, metadata)
        metadata["action_contract"]["arm_control"]["name"] = (
            "mink_direct_joint_servo_v4"
        )
        h.attrs["recenter_events_json"] = "[{}]"
        with pytest.raises(ValueError, match="recenter calibration"):
            aligned_hand_keypoint_actions(h, metadata)


def test_keypoint_resolver_requires_declared_frame_manifest(tmp_path):
    path = tmp_path / "sharpa69.zarr"
    replay(path, 69, 22)
    r = LiberoReplayResolver(
        path,
        keymap(2, 32),
        "libero_spatial",
        action_dim=69,
        gripper_state_dim=22,
        embodiment="libero_panda_sharpa",
        action_representation="wrist_hand_keypoints",
    )
    manifest = path / "benchmark_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "complete": True,
                "action_dim": 69,
                "action_representation": "wrist_hand_keypoints",
                "keypoint_frame": "wrist_relative_Quest_XYZ_m",
                "observation_alignment": "causal_pre_action_raw_lossless",
            }
        )
    )
    assert next(iter(r.resolve().values()))[0]["actions"].shape == (32, 69)
    manifest.write_text(json.dumps({"complete": True, "action_dim": 69}))
    with pytest.raises(ValueError, match="frame manifest"):
        r.resolve()


def test_sharpa_raw_pre_action_views_cannot_leak_post_action_frames(tmp_path):
    from egomimic.benchmarks.libero.convert import causal_collection_episode_values

    with h5py.File(tmp_path / "timeline.hdf5", "w") as h:
        h.create_dataset("actions", data=np.zeros((3, 28)))
        h.create_dataset("observation_host_ns", data=[10, 30, 50, 70])
        h.create_dataset("action_host_ns", data=[20, 40, 60])
        data = h.create_group("data")
        data.attrs["rgb_row_transform"] = (
            "vertical_flip_from_raw_opengl_to_original_zarr_upright"
        )
        rgb = h.create_group("rgb")
        pixels = np.zeros((4, 2, 2, 3), dtype=np.uint8)
        for t in range(4):
            pixels[t, 0, :, 0] = 10 + t
            pixels[t, 1, :, 2] = 100 + t
        rgb.create_dataset("agentview", data=pixels)
        rgb.create_dataset("robot0_eye_in_hand", data=pixels)
        obs = h.create_group("numeric_observations")
        obs.create_dataset("robot0_eef_pos", data=np.arange(12).reshape(4, 3))
        obs.create_dataset("robot0_eef_quat", data=[[0, 0, 0, 1]] * 4)
        obs.create_dataset(
            "robot0_gripper_qpos", data=np.repeat(np.arange(4)[:, None], 22, axis=1)
        )
        meta = {"observation_order": "before_action_and_terminal"}
        result = causal_collection_episode_values(
            h,
            meta,
            get_tasks("libero_spatial")[0],
            np.ones((2, 69), dtype=np.float32),
            1,
            np.array([2, 3]),
        )
        np.testing.assert_array_equal(result["agentview_rgb"], pixels[1:3, ::-1])
        np.testing.assert_array_equal(
            result["robot0_eef_pos"], np.arange(12).reshape(4, 3)[1:3]
        )
        np.testing.assert_array_equal(result["observation_host_ns"], [30, 50])
        np.testing.assert_array_equal(result["action_host_ns"], [40, 60])
        assert np.all(result["observation_host_ns"] < result["action_host_ns"])
        assert result["robot0_gripper_qpos"][:, 0].tolist() == [1, 2]
        h.create_dataset(
            "control_reason", data=[b"operator_stop", b"stale_3d_view", b"active"]
        )
        resumed_target = np.ones((1, 69), dtype=np.float32)
        with pytest.raises(ValueError, match="resume proof"):
            causal_collection_episode_values(
                h,
                meta,
                get_tasks("libero_spatial")[0],
                resumed_target,
                2,
                np.array([3]),
            )
        proof = {
            "status": "PASS_UNIQUE_SOURCE_BACKED_RESUME_CALIBRATION",
            "source_start_step": 2,
            "source_path": h.filename,
            "solutions": [{"recorded_anchor": True}],
            "no_later_gap_discarded": True,
        }
        resumed = causal_collection_episode_values(
            h,
            meta,
            get_tasks("libero_spatial")[0],
            resumed_target,
            2,
            np.array([3]),
            resume_proof=proof,
        )
        np.testing.assert_array_equal(resumed["observation_host_ns"], [50])
        h["control_reason"][0] = b"active"
        with pytest.raises(ValueError, match="discard active"):
            causal_collection_episode_values(
                h,
                meta,
                get_tasks("libero_spatial")[0],
                resumed_target,
                2,
                np.array([3]),
                resume_proof=proof,
            )
        h["observation_host_ns"][1] = 45
        with pytest.raises(ValueError, match="not causal"):
            causal_collection_episode_values(
                h,
                meta,
                get_tasks("libero_spatial")[0],
                np.ones((2, 69), dtype=np.float32),
                1,
                np.array([2, 3]),
            )


def test_routed_dit_half_preserves_three_owner_gradients_and_batch_contract():
    from copy import deepcopy
    from types import SimpleNamespace

    from torch.utils.checkpoint import checkpoint

    from egomimic.utils.libero_dit_half import HalfCheckpointScope, LiberoDiTHalf

    torch.manual_seed(4)
    backbones = {
        name: SimpleNamespace(
            blocks=torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)]),
            gradient_checkpointing=True,
        )
        for name in ("encoder/libero_panda", "encoder/libero_panda_sharpa", "velocity")
    }
    baseline = deepcopy(backbones)
    native = SimpleNamespace(checkpoint=checkpoint)
    scope = HalfCheckpointScope(native, backbones, expected_depth=2)
    inputs = torch.randn(2, 4)

    def evaluate(owners, dispatcher):
        x = inputs.clone().requires_grad_()
        values = []
        for owner in owners.values():
            value = x
            for block in owner.blocks:
                value = dispatcher(block, value, use_reentrant=False)
            values.append(value)
        output = sum(values)
        output.square().sum().backward()
        return output, x.grad

    expected, expected_gradient = evaluate(baseline, checkpoint)
    scope.install()
    try:
        actual, actual_gradient = evaluate(backbones, native.checkpoint)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
        for name in backbones:
            for actual_block, reference_block in zip(
                backbones[name].blocks, baseline[name].blocks
            ):
                torch.testing.assert_close(
                    actual_block.weight.grad,
                    reference_block.weight.grad,
                    rtol=0,
                    atol=0,
                )
        assert all(count > 0 for count in scope.counts.values())
        callback = LiberoDiTHalf({"libero_panda": 16, "libero_panda_sharpa": 16})
        callback.scope = scope
        batch = {
            name: {"actions": torch.zeros(16, 32, width)}
            for name, width in [("libero_panda", 7), ("libero_panda_sharpa", 69)]
        }
        callback.on_train_batch_start(None, None, batch, 0)
        batch["libero_panda_sharpa"]["actions"] = torch.zeros(32, 32, 69)
        with pytest.raises(ValueError, match="per-source batch"):
            callback.on_train_batch_start(None, None, batch, 0)
    finally:
        scope.restore()
    assert native.checkpoint is checkpoint


def test_collection_image_chunk_layout_keeps_pixels_actions_and_temporal_reads_exact(
    tmp_path,
):
    from egomimic.benchmarks.libero.convert import append_replay

    rng = np.random.default_rng(7)
    values = {
        "action": rng.normal(size=(19, 69)).astype(np.float32),
        "agentview_rgb": rng.integers(0, 256, (19, 3, 4, 3), dtype=np.uint8),
        "robot0_eye_in_hand_rgb": rng.integers(0, 256, (19, 3, 4, 3), dtype=np.uint8),
    }
    outputs = []
    for chunk in (1, 8):
        group = zarr.open_group(str(tmp_path / str(chunk)), mode="w", zarr_format=2)
        offset = 0
        for start, end in ((0, 3), (3, 11), (11, 19)):
            offset = append_replay(
                group,
                {k: v[start:end] for k, v in values.items()},
                offset,
                image_chunk_frames=chunk,
            )
        assert offset == 19 and group["agentview_rgb"].chunks[0] == chunk
        for key, expected in values.items():
            np.testing.assert_array_equal(group[key][:], expected)
            np.testing.assert_array_equal(group[key][7:9], expected[7:9])
        outputs.append(group)
    for key in values:
        np.testing.assert_array_equal(outputs[0][key][:], outputs[1][key][:])
    with pytest.raises(ValueError, match="positive frame count"):
        append_replay(outputs[0], values, 19, image_chunk_frames=0)


def test_hand_target_cache_reuses_verified_targets_and_rejects_source_or_cache_drift(
    tmp_path,
):
    import hashlib

    from egomimic.benchmarks.libero.convert import (
        cached_hand_keypoint_actions,
        verified_hand_target_cache,
    )

    source = tmp_path / "episode.hdf5"
    with h5py.File(source, "w") as h:
        h.create_dataset("actions", data=np.zeros((3, 28)))
    records = [{"path": source.name, "task": "task"}]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"episodes": records}))
    cache = tmp_path / "targets.npz"
    targets = np.arange(2 * 69, dtype=np.float32).reshape(2, 69)
    np.savez_compressed(
        cache,
        episode_000000_actions=targets,
        episode_000000_sequence=np.array([10, 11], dtype=np.int64),
    )
    stat = source.stat()
    row = {
        "index": 0,
        **records[0],
        "source_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "source_start_step": 1,
        "source_control_count": 3,
        "target_shape": [2, 69],
        "cache_key": "episode_000000",
    }
    audit = {
        "status": "PASS_ALL_500_EXACT_CAUSAL_HAND_TARGETS",
        "source_episode_count": 1,
        "passed_episode_count": 1,
        "failed_episode_count": 0,
        "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "target_cache": str(cache),
        "target_cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
        "passed": [row],
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit))
    _, rows = verified_hand_target_cache(audit_path, manifest, tmp_path, records)
    with np.load(cache, allow_pickle=False) as arrays, h5py.File(source, "r") as h:
        actual, start, sequence = cached_hand_keypoint_actions(arrays, rows[0], h)
    np.testing.assert_array_equal(actual, targets)
    assert start == 1 and sequence.tolist() == [10, 11]
    with source.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="source identity changed"):
        verified_hand_target_cache(audit_path, manifest, tmp_path, records)
    with cache.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="cache SHA differs"):
        verified_hand_target_cache(audit_path, manifest, tmp_path, records)


def test_joint_energy_verifier_consumes_real_generic_saved_artifact(
    tmp_path, monkeypatch
):
    import copy
    from pathlib import Path
    from types import SimpleNamespace

    import hydra
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    from egomimic.eval.energy_score import energy_score
    from egomimic.eval.planar_action_eval import PlanarActionEval
    from scripts.train.verify_libero_native_action_flow_smoke import (
        validate_joint_energy_artifact,
    )

    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("LIBERO_ORIGINAL_SPATIAL_ROOT", str(tmp_path / "original"))
    monkeypatch.setenv("LIBERO_SHARPA_SPATIAL_ROOT", str(tmp_path / "sharpa"))
    monkeypatch.setenv(
        "LIBERO_DIAGNOSTIC_SEED_BANK",
        str(
            root / "egomimic/hydra_configs/evaluator/energy_score_seed_bank_k32_v1.json"
        ),
    )
    with hydra.initialize_config_dir(
        version_base="1.2", config_dir=str(root / "egomimic/hydra_configs")
    ):
        cfg = hydra.compose(
            config_name="train_zarr_cartesian",
            return_hydra_config=True,
            overrides=[
                "hydra/launcher=basic",
                "+experiment=libero/action_flow_libero_spatial_sharpa_gripper_keypoints_s42",
            ],
        )
        cfg.hydra.runtime.output_dir = str(tmp_path)
        HydraConfig.instance().set_config(cfg)
        cfg.evaluator.seed_bank_path = str(
            root / "egomimic/hydra_configs/evaluator/energy_score_seed_bank_k32_v1.json"
        )
        cfg.evaluator.artifact_root = str(tmp_path / "energy")
        cfg.run_provenance.source_commit = "1" * 40
        cfg.run_provenance.normalization_sha256 = "2" * 64
        cfg.run_provenance.split_manifest_sha256 = "3" * 64
        resolved = OmegaConf.to_container(
            OmegaConf.masked_copy(cfg, [k for k in cfg if k != "hydra"]), resolve=True
        )
    options = copy.deepcopy(resolved["evaluator"])
    options.pop("_target_")
    options["action_flow_diagnostics"] = None
    evaluator = PlanarActionEval(**options)
    evaluator.trainer = SimpleNamespace(
        global_step=8, current_epoch=0, global_rank=0, precision="bf16-mixed"
    )
    samples, scores, batch = {}, {}, {}
    generator = torch.Generator().manual_seed(42)
    for name, identity, width in (
        ("libero_panda", 21, 7),
        ("libero_panda_sharpa", 22, 69),
    ):
        samples[name] = torch.randn(32, 16, 32, width, generator=generator)
        target = torch.randn(16, 32, width, generator=generator)
        batch[name] = dict(embodiment=torch.full((16,), identity), actions=target)
        scores[name] = energy_score(
            samples[name],
            target,
            resolved["evaluator"]["semantic_blocks_by_embodiment"][name],
        )
    evaluator._save_artifact(0, samples, scores, batch)
    path = next((tmp_path / "energy").rglob("*.pt"))
    from tests.test_verify_action_flow_training_smoke import MODULE

    selected, payload = MODULE._step_artifact(
        tmp_path / "energy", label="joint EnergyScore", expected_step=8
    )
    assert selected == path
    with pytest.raises(MODULE.SmokeVerificationError, match="one step-2"):
        MODULE._step_two_artifact(tmp_path / "energy", label="joint EnergyScore")
    result = validate_joint_energy_artifact(payload, resolved)
    assert set(result["sources"]) == {"libero_panda", "libero_panda_sharpa"}
    payload["domains"]["libero_panda_sharpa"]["score_by_condition"][0] += 1
    with pytest.raises(ValueError, match="public math differs"):
        validate_joint_energy_artifact(payload, resolved)

    # Exercise the real immutable diagnostic writer with synthetic captured tensors.
    from egomimic.eval.action_flow_diagnostics import ActionFlowDiagnostics
    from scripts.train.verify_libero_native_action_flow_smoke import (
        validate_joint_diagnostic_artifact,
    )

    diagnostic_seeds = tmp_path / "diagnostic-seeds.json"
    diagnostic_seeds.write_text(
        json.dumps({"seeds": [730001 + 7919 * index for index in range(32)]}, indent=2)
        + "\n"
    )
    resolved["evaluator"]["action_flow_diagnostics"]["noise_seed_bank_path"] = str(
        diagnostic_seeds
    )
    inspector = ActionFlowDiagnostics(resolved["evaluator"]["action_flow_diagnostics"])
    diagnostic = dict(
        schema_version=1,
        metric="ActionFlowValidationDiagnostics",
        global_step=8,
        epoch=0,
        rank=0,
        batch_idx=0,
        precision="bf16-mixed",
        identity=inspector.identity,
        identity_sha256=inspector.identity_sha256,
        provenance=inspector.provenance,
        validation_view=inspector.validation_view,
        noise_seed_bank_sha256=inspector.noise_seed_bank_sha256,
        noise_seed=inspector.noise_seeds[0],
        raw_noise_levels=inspector.noise_levels,
        sampler=dict(name="reverse_euler"),
        sources={},
    )
    for name, width in (("libero_panda", 7), ("libero_panda_sharpa", 69)):
        raw = dict(
            source="validation/" + name,
            schema=inspector.schema,
            noise_seed=inspector.noise_seeds[0],
            num_inference_steps=50,
            returned_samples=8,
        )
        for key, shape in (
            ("target", (8, 32, width)),
            ("condition", (8, 276)),
            ("latent/clean", (8, 16, 16)),
            ("latent/noise", (8, 16, 16)),
            ("latent/generated", (8, 16, 16)),
            ("decoded/reconstruction", (8, 32, width)),
            ("decoded/generated", (8, 32, width)),
            ("latent/trajectory", (51, 8, 16, 16)),
            ("decoded/trajectory", (51, 8, 32, width)),
        ):
            raw[key] = torch.zeros(shape)
        diagnostic["sources"][name] = dict(
            source_id=raw["source"],
            diagnostic=raw,
            computed=dict(clean_reconstruction_mse_by_condition=torch.zeros(8)),
        )
    diagnostic_path = tmp_path / "diagnostics.pt"
    inspector._write_immutable_artifact(diagnostic_path, diagnostic)
    assert validate_joint_diagnostic_artifact(diagnostic_path, diagnostic, resolved)[
        "sha256"
    ]
    sidecar = Path(str(diagnostic_path) + ".sha256")
    sidecar.write_text(sidecar.read_text().replace(inspector.identity_sha256, "0" * 64))
    with pytest.raises(ValueError, match="sidecar differs"):
        validate_joint_diagnostic_artifact(diagnostic_path, diagnostic, resolved)

    # Metadata-only producer fixture: real saved tensor files, synthetic runtime/preflight.
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/epoch-0000-step-000000008.ckpt").write_bytes(b"fixture")
    from scripts.train.libero_native_launch_contract import (
        contract_sha,
        joint_smoke_evidence,
        reference,
    )

    resolved["run_provenance"]["run_kind"] = "smoke"
    resolved["trainer"]["max_steps"] = 8
    artifact_root = tmp_path / "diag-envelope"
    resolved["evaluator"]["action_flow_diagnostics"]["artifact_root"] = str(
        artifact_root
    )
    nested = artifact_root / "epoch-0-step-8/rank-0-batch-0.pt"
    nested.parent.mkdir(parents=True)
    inspector._write_immutable_artifact(nested, diagnostic)
    (tmp_path / ".hydra").mkdir()
    OmegaConf.save(OmegaConf.create(resolved), tmp_path / ".hydra/config.yaml")
    runtime = {
        key: {}
        for key in (
            "python",
            "installed_distributions",
            "core_distribution_versions",
            "dependency_inputs",
            "source",
            "torch_runtime",
        )
    }
    runtime_path = tmp_path / "runtime-lock.json"
    runtime_path.write_text(json.dumps(runtime))
    synthetic_preflight = dict(
        schema="libero-joint-launch-preflight/v1",
        status="PASS",
        profile="libero/action_flow_libero_spatial_sharpa_gripper_keypoints_s42",
        model_contract_sha256=contract_sha(resolved),
        runtime_lock=reference(runtime_path),
        **{
            key: resolved["run_provenance"][key]
            for key in (
                "source_commit",
                "split_manifest_sha256",
                "normalization_sha256",
            )
        },
    )
    preflight_path = tmp_path / "synthetic-preflight.json"
    preflight_path.write_text(json.dumps(synthetic_preflight))
    envelope = joint_smoke_evidence(
        resolved,
        tmp_path,
        preflight_ref=reference(preflight_path),
        runtime_ref=reference(runtime_path),
        shared=MODULE,
    )
    assert envelope["energy_artifact"] == reference(path)
    assert envelope["diagnostic_artifact"] == reference(nested)
    assert (
        "status" not in envelope
    )  # An envelope never claims smoke or launch acceptance.
    resolved["run_provenance"]["run_kind"] = "full"
    with pytest.raises(ValueError, match="eight-update smoke"):
        joint_smoke_evidence(
            resolved,
            tmp_path,
            preflight_ref=reference(preflight_path),
            runtime_ref=reference(runtime_path),
            shared=MODULE,
        )
    resolved["run_provenance"]["run_kind"] = "smoke"
    runtime["source"] = {"wrong": True}
    different_runtime = tmp_path / "wrong-runtime.json"
    different_runtime.write_text(json.dumps(runtime))
    with pytest.raises(AssertionError, match="native runtime differs"):
        joint_smoke_evidence(
            resolved,
            tmp_path,
            preflight_ref=reference(preflight_path),
            runtime_ref=reference(different_runtime),
            shared=MODULE,
        )

    from scripts.train.verify_libero_native_action_flow_smoke import (
        validate_joint_wandb_records,
    )
    from tests.test_verify_action_flow_training_smoke import _history_row

    row = _history_row()
    for key, value in list(row.items()):
        if MODULE.SOURCE_LABEL in key:
            for label in ("libero_panda", "libero_panda_sharpa"):
                row[key.replace(MODULE.SOURCE_LABEL, label)] = value
    row["Train/ActionFlow/Compute/FieldForwardCallsPerStep"] = 2.0
    row["Train/ActionFlow/Compute/FieldSampleEquivalentsPerStep"] = 28.0
    for prefix in (
        "GradientCosine",
        "GradientCosineDefined",
        "GradientIntersectionParameterCount",
    ):
        row[f"Train/ActionFlow/{prefix}/FM__Reconstruction"] = 0.0
    row["Optimizer/LR/AdamW"] = row["Optimizer/LR/Muon"] = 2.5e-8
    for owner in ("encoder/libero_panda", "encoder/libero_panda_sharpa", "velocity"):
        for mode in ("direct", "checkpoint"):
            row["Train/Execution/DiTHalf/" + owner + "/" + mode] = 6.0
    resolved["logger"] = dict(wandb=dict(mode="online", id="synthetic-history-fixture"))
    histories = dict(
        offline={7: row},
        online={7: row},
        online_config=resolved,
        online_state="finished",
        cfg=resolved,
        shared=MODULE,
    )
    assert validate_joint_wandb_records(**histories)["online"]["valid_step"] == 7
    histories["online_state"] = "running"
    with pytest.raises(ValueError, match="finished online"):
        validate_joint_wandb_records(**histories)
    histories["online_state"] = "finished"
    row["Train/Execution/DiTHalf/encoder/libero_panda_sharpa/checkpoint"] = 0.0
    with pytest.raises(ValueError, match="path inactive"):
        validate_joint_wandb_records(**histories)


def test_joint_constructor_explicit_waiver_keeps_cached_evidence_immutable(tmp_path):
    import hashlib

    from scripts.train.libero_native_launch_contract import validate_joint_constructor

    original = tmp_path / "original.json"
    original.write_text(
        json.dumps(
            dict(
                status="PASS_ACTUAL_CPU_MODEL_CONSTRUCTION_NOT_TRAINING_SMOKE",
                total_parameters=78182614,
                private_codecs_per_embodiment=True,
                single_shared_velocity_stage=True,
                all_parameters_finite=True,
            )
        )
    )
    before = original.read_bytes()
    waiver = tmp_path / "waiver.json"
    data = dict(
        schema="libero-joint-constructor-provenance-waiver/v1",
        authorization="explicit_human_override",
        waived_gate="cached_constructor_source_config_binding",
        source_commit="a" * 40,
        human_message="bypass the provenance gate",
        original_constructor=dict(
            path=str(original), sha256=hashlib.sha256(before).hexdigest()
        ),
    )

    def check():
        waiver.write_text(json.dumps(data))
        ref = dict(
            path=str(waiver), sha256=hashlib.sha256(waiver.read_bytes()).hexdigest()
        )
        return validate_joint_constructor(
            ref, source_head="a" * 40, resolved_sha="b" * 64, fingerprint="c" * 64
        )

    assert check() == "HUMAN_WAIVED_SOURCE_CONFIG_BINDING"
    assert original.read_bytes() == before
    data["source_commit"] = "d" * 40
    with pytest.raises(ValueError, match="authorization/source"):
        check()
    data["source_commit"] = "a" * 40
    data["authorization"] = ""
    with pytest.raises(ValueError, match="authorization/source"):
        check()
