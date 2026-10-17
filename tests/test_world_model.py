import copy
import hashlib
import json

import numpy as np
import pytest
import torch

from egomimic.models.world_model import MoTWorldModel
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.stages_world_model import WorldModelStage
from egomimic.rldb.world_model import (
    HomogeneousWorldBatchSampler,
    PreparedWorldDataset,
    collate_world,
    points_to_reference,
    sample_times,
)


def make_batch(pose_dim=7, batch_size=2):
    result = {}
    for name, dim, n in (("image", 8, 6), ("track", 3, 4), ("proprio", pose_dim, 2)):
        time = (
            torch.arange(2)
            .repeat_interleave(n // 2)
            .float()
            .expand(batch_size, -1)
            .clone()
        )
        result[name] = dict(
            value=torch.randn(batch_size, n, dim),
            time=time,
            index=time.clone(),
            valid=torch.ones(batch_size, n, dtype=torch.bool),
            observed=time == 0,
            target_valid=torch.ones(batch_size, n, dtype=torch.bool),
        )
        if name != "proprio":
            result[name].update(
                xy=torch.rand(batch_size, n, 2),
                camera=torch.zeros(batch_size, n, dtype=torch.long),
            )
        if name == "track":
            result[name].update(
                point_id=torch.tensor([0, 1, 0, 1]).expand(batch_size, -1),
                confidence=torch.ones(batch_size, n),
            )
    return result


def make_model(**kwargs):
    return MoTWorldModel(
        {"robot": 7, "human": 11},
        image_dim=8,
        width=24,
        heads=3,
        depth=2,
        max_tracks=8,
        max_tokens=100,
        **kwargs,
    )


def test_pose_gradient_reaches_image_and_track_generators():
    torch.manual_seed(7)
    model = make_model()
    losses = model.training_losses(make_batch(), "robot")
    losses["pose"].backward()
    for module in (model.image_out, model.track_out, model.image_in, model.track_in):
        assert module.weight.grad is not None
        assert module.weight.grad.norm() > 0
    for block in model.blocks:
        assert all(m.weight.grad.norm() > 0 for m in block.qkv)
    assert model.pose_out["human"].weight.grad is None


def test_zero_initialized_adapters_preserve_adapter_free_predictions():
    model = make_model(image_adapter_dim=4).eval()
    baseline = make_model(image_adapter_dim=0).eval()
    baseline.load_state_dict(
        {
            k: v
            for k, v in model.state_dict().items()
            if not k.startswith("image_adapters.")
        },
        strict=True,
    )
    batch = make_batch()
    a = model.sample(batch, "robot", 2, torch.Generator().manual_seed(5))
    b = baseline.sample(batch, "robot", 2, torch.Generator().manual_seed(5))
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    # Old checkpoints must not silently skip the new trainable parameters.
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(baseline.state_dict(), strict=True)


@pytest.mark.parametrize(
    "selected", ["yam", "human", "libero_sharpa", "libero_gripper"]
)
def test_image_adapters_route_and_learn_for_all_four_embodiments(selected):
    torch.manual_seed(14)
    dims = dict(yam=7, human=11, libero_sharpa=9, libero_gripper=5)
    model = MoTWorldModel(
        dims,
        image_dim=8,
        width=24,
        heads=3,
        depth=2,
        max_tracks=8,
        max_tokens=100,
        image_adapter_dim=4,
    )
    untouched = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if name.startswith("image_adapters.") and f".{selected}." not in name
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        # Pose loss alone must reach adapters through generated visual features.
        model.training_losses(make_batch(dims[selected]), selected)["pose"].backward()
        for block in model.image_adapters:
            adapter = block[selected]
            assert adapter.up.weight.grad.norm() > 0
            if step:
                assert adapter.down.weight.grad.norm() > 0
                assert adapter.norm.weight.grad.norm() > 0
            for other in dims.keys() - {selected}:
                assert all(p.grad is None for p in block[other].parameters())
        assert model.blocks[0].qkv[0].weight.grad.norm() > 0
        optimizer.step()
    for name, p in model.named_parameters():
        if name in untouched:
            torch.testing.assert_close(p, untouched[name], rtol=0, atol=0)


def test_adapter_effect_is_embodiment_specific_and_checkpointed():
    model = make_model(image_adapter_dim=4).eval()
    batches = {"robot": make_batch(7), "human": make_batch(11)}

    def predict(embodiment):
        return model.sample(
            batches[embodiment], embodiment, 2, torch.Generator().manual_seed(9)
        )["image"]

    before = {e: predict(e) for e in batches}
    with torch.no_grad():
        for block in model.image_adapters:
            block["robot"].up.weight.normal_(std=0.1)
    assert not torch.equal(before["robot"], predict("robot"))
    torch.testing.assert_close(before["human"], predict("human"), rtol=0, atol=0)
    restored = make_model(image_adapter_dim=4).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(
        predict("robot"),
        restored.sample(batches["robot"], "robot", 2, torch.Generator().manual_seed(9))[
            "image"
        ],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("dimension", [-1, 1.5, True])
def test_invalid_image_adapter_width_is_rejected(dimension):
    with pytest.raises(ValueError, match="image_adapter_dim"):
        make_model(image_adapter_dim=dimension)


@pytest.mark.parametrize("embodiment,dimension", [("robot", 7), ("human", 11)])
def test_actual_optimizer_and_strict_reload(embodiment, dimension):
    torch.manual_seed(10)
    model = make_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.track_out.weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad()
        loss = model.training_losses(make_batch(dimension), embodiment)["total"]
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    assert not torch.equal(before, model.track_out.weight)
    replacement = make_model()
    replacement.load_state_dict(model.state_dict(), strict=True)
    new_optimizer = torch.optim.AdamW(replacement.parameters(), lr=1e-3)
    new_optimizer.load_state_dict(optimizer.state_dict())
    assert len(new_optimizer.state) == len(optimizer.state)
    batch = make_batch(dimension)
    a = model.sample(batch, embodiment, 3, torch.Generator().manual_seed(8))
    b = replacement.sample(batch, embodiment, 3, torch.Generator().manual_seed(8))
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_sampler_does_not_read_withheld_ground_truth():
    model = make_model().eval()
    batch = make_batch()
    changed = copy.deepcopy(batch)
    for stream in changed.values():
        stream["value"][~stream["observed"]] = float("nan")
        stream["target_valid"][:] = False
        if "confidence" in stream:
            stream["confidence"][:] = 0
    a = model.sample(batch, "robot", 2, torch.Generator().manual_seed(4))
    b = model.sample(changed, "robot", 2, torch.Generator().manual_seed(4))
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
        torch.testing.assert_close(
            a[name][batch[name]["observed"]],
            batch[name]["value"][batch[name]["observed"]],
        )


def test_hidden_pose_does_not_condition_first_pass():
    model = make_model()
    batch = make_batch()
    other = copy.deepcopy(batch)
    other["proprio"]["value"][~other["proprio"]["observed"]] += 900
    values = {n: batch[n]["value"] for n in ("image", "track")}
    levels = {n: torch.zeros_like(batch[n]["time"]) for n in values}
    a, b = model(batch, "robot", values, levels), model(other, "robot", values, levels)
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)


def test_padding_invariance():
    model = make_model().eval()
    batch = make_batch()
    padded = copy.deepcopy(batch)
    for stream in padded.values():
        for key, x in list(stream.items()):
            extra = torch.zeros_like(x[:, :1])
            if key == "value":
                extra[:] = float("nan")
            stream[key] = torch.cat((x, extra), 1)
    levels = {n: torch.zeros_like(batch[n]["time"]) for n in ("image", "track")}
    values = {n: batch[n]["value"] for n in levels}
    a = model(batch, "robot", values, levels)
    b = model(
        padded,
        "robot",
        {n: padded[n]["value"] for n in levels},
        {n: torch.zeros_like(padded[n]["time"]) for n in levels},
    )
    for name in a:
        torch.testing.assert_close(a[name], b[name][:, :-1], atol=2e-6, rtol=2e-5)


def test_common_time_shift_invariance_and_discrete_mode():
    model = make_model().eval()
    batch, shifted = make_batch(), None
    shifted = copy.deepcopy(batch)
    for s in shifted.values():
        s["time"] += 3.5
    levels = {n: torch.zeros_like(batch[n]["time"]) for n in ("image", "track")}
    values = {n: batch[n]["value"] for n in levels}
    a, b = (
        model(batch, "robot", values, levels),
        model(shifted, "robot", values, levels),
    )
    for name in a:
        torch.testing.assert_close(a[name], b[name], atol=2e-6, rtol=2e-5)
    discrete = make_model(time_mode="discrete")
    discrete.load_state_dict(model.state_dict())
    for name in a:
        torch.testing.assert_close(
            a[name], discrete(batch, "robot", values, levels)[name]
        )
    del shifted["image"]["index"]
    with pytest.raises(ValueError, match="indices"):
        discrete(shifted, "robot", values, levels)


def test_budget_guard_and_warmup_in_pipeline():
    model = make_model()
    stage = WorldModelStage(model, warmup_steps=3, pose_ramp_steps=2)
    assert [stage.pose_scale(i) for i in range(6)] == [0, 0, 0, 0, 0.5, 1]
    algo = PipelineAlgo([stage], device="cpu")
    raw = {"robot": {"world": make_batch(), "embodiment": "robot", "optimizer_step": 0}}
    predictions = algo.forward_training(raw)
    loss = algo.compute_losses(predictions, raw)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    assert predictions["robot"]["log/world_pose"] == 0
    assert "optimizer_step" not in raw["robot"]["world"]
    model.max_tokens = 2
    with pytest.raises(ValueError, match="budget"):
        model.sample(make_batch(), "robot")


def test_temporal_sampling_and_rigid_frame():
    counts = set()
    rng = np.random.default_rng(2)
    for _ in range(50):
        selected = sample_times(100, rng)
        counts.add(len(selected))
        assert (np.diff(selected) > 0).all()
    assert counts == {2, 3, 4}
    selected = sample_times(100, rng, mode="discrete", stride=3)
    assert (np.diff(selected) == 3).all()
    transform = np.eye(4)
    transform[:3, 3] = [1, 2, 3]
    np.testing.assert_allclose(
        points_to_reference(np.zeros((2, 3)), transform), [[1, 2, 3]] * 2
    )
    transform[0, 0] = -1
    with pytest.raises(ValueError, match="handedness"):
        points_to_reference(np.zeros((2, 3)), transform)


def cache_manifest(tmp_path):
    entries = []
    for embodiment, d in (("robot", 7), ("human", 11)):
        for split in ("train", "val"):
            path = tmp_path / f"{embodiment}-{split}.npz"
            rng = np.random.default_rng(1)
            np.savez(
                path,
                time=np.arange(8, dtype=np.float64) * 0.1 + 1e9,
                image=rng.normal(size=(8, 2, 3, 8)),
                image_xy=rng.random((3, 2)),
                track=rng.normal(size=(8, 2, 3)),
                track_xy=rng.random((2, 2)),
                track_camera=np.zeros(2, dtype=np.int64),
                track_confidence=np.ones((8, 2)),
                pose=rng.normal(size=(8, d)),
                pose_valid=np.ones(8, dtype=bool),
            )
            entries.append(
                dict(
                    embodiment=embodiment,
                    episode_id=path.stem,
                    path=path.name,
                    frames=8,
                    split=split,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    frame="fixed_reference",
                    pose_kind="measured_pose",
                )
            )
    manifest = dict(
        schema="mot-world-prepared-v1",
        dino_identity={
            "model": "test-fixture",
            "revision": "fixture",
            "image_size": [2, 6],
            "patch_grid": [1, 3],
            "feature_layer": "last",
        },
        normalization={
            "scope": "train_only",
            "sha256": "0" * 64,
            "arrays_normalized": True,
        },
        pose_schemas={
            name: {"dimension": dim, "fields": [f"coordinate_{i}" for i in range(dim)]}
            for name, dim in (("robot", 7), ("human", 11))
        },
        track_provenance={
            "interpolated_endpoints": False,
            "method": "synthetic_fixture",
            "reference_frame": "world",
            "calibration_sha256": "0" * 64,
        },
        episodes=entries,
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_prepared_cache_pipeline_and_reject_overlap(tmp_path):
    path = cache_manifest(tmp_path)
    dataset = PreparedWorldDataset(path)
    sampler = HomogeneousWorldBatchSampler(dataset, batch_size=2, shuffle=False)
    for indices in sampler:
        batch = collate_world([dataset[i] for i in indices])
        embodiment, values = next(iter(batch.items()))
        result = make_model().training_losses(values["world"], embodiment)
        assert torch.isfinite(result["total"])
    manifest = json.loads(path.read_text())
    manifest["episodes"].append(manifest["episodes"][0])
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Duplicate"):
        PreparedWorldDataset(path)


def test_reject_interpolated_track_targets(tmp_path):
    path = cache_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["track_provenance"]["interpolated_endpoints"] = True
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="interpolation"):
        PreparedWorldDataset(path)


def test_bf16_mixed_finite_gradients():
    model = make_model()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model.training_losses(make_batch(), "robot")["total"]
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_hydra_both_recipes_and_required_schema():
    from pathlib import Path

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    root = Path(__file__).resolve().parents[1] / "egomimic/hydra_configs"
    for mode in ("continuous", "discrete"):
        with initialize_config_dir(config_dir=str(root), version_base=None):
            cfg = compose(
                config_name="train_zarr_cartesian",
                overrides=[f"+experiment=world_model/{mode}", "hydra/launcher=basic"],
            )
        assert cfg.data.mode == mode
        assert cfg.model.pipeline.stages[0].model.time_mode == mode
        assert cfg.trainer.check_val_every_n_epoch is None
        assert cfg.callbacks.model_checkpoint.every_n_train_steps == 15000
        assert OmegaConf.is_missing(cfg.model.pipeline.stages[0].model, "pose_dims")
        assert cfg.evaluator._target_.endswith("WorldModelEval")


def test_shared_lightning_fit_validate_and_resume(tmp_path):
    from lightning import Trainer
    from omegaconf import OmegaConf

    from egomimic.eval.world_model_eval import WorldModelEval
    from egomimic.pl_utils.pl_model import ModelWrapper
    from egomimic.rldb.world_model_datamodule import WorldModelDataModule

    path = cache_manifest(tmp_path)
    data = WorldModelDataModule(str(path), batch_size=2, max_frames=3)
    context = data.prepare_context(mode="train", normalization=None, normalizer=None)
    with pytest.raises(ValueError, match="mismatch"):
        data.prepare_context(
            mode="train", normalization=None, normalizer=None, restored_state={}
        )
    cfg = OmegaConf.create(
        {
            "model": {
                "pipeline": {
                    "_target_": "egomimic.pipeline.algo.PipelineAlgo",
                    "device": "cpu",
                    "stages": [
                        {
                            "_target_": "egomimic.pipeline.stages_world_model.WorldModelStage",
                            "warmup_steps": 0,
                            "pose_ramp_steps": 0,
                            "sampler_steps": 2,
                            "model": {
                                "_target_": "egomimic.models.world_model.MoTWorldModel",
                                "pose_dims": {"robot": 7, "human": 11},
                                "image_dim": 8,
                                "width": 24,
                                "heads": 3,
                                "depth": 1,
                                "max_tracks": 8,
                            },
                        }
                    ],
                },
                "training_behavior": {
                    "_target_": "egomimic.pl_utils.world_model_behavior.WorldModelTrainingBehavior"
                },
                "optimizer": {
                    "_target_": "torch.optim.AdamW",
                    "_partial_": True,
                    "lr": 0.001,
                },
            }
        }
    )
    wrapper = ModelWrapper(config_tree=cfg, evaluator=WorldModelEval())
    wrapper.data_context = context
    context.bind(wrapper.model, wrapper.evaluator)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=2,
        max_epochs=1,
        limit_val_batches=1,
        val_check_interval=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(wrapper, datamodule=data)
    assert trainer.global_step == 2
    assert any(k.startswith("val/") for k in trainer.callback_metrics)
    checkpoint = tmp_path / "model.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = ModelWrapper(config_tree=cfg, evaluator=WorldModelEval())
    restored.data_context = context
    trainer2 = Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=3,
        max_epochs=2,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
    )
    trainer2.fit(restored, datamodule=data, ckpt_path=str(checkpoint))
    assert trainer2.global_step == 3


def test_invalid_unobserved_track_targets_are_masked():
    model = make_model()
    batch = make_batch()
    batch["track"]["value"][:, -1] = float("nan")
    batch["track"]["target_valid"][:, -1] = False
    batch["track"]["confidence"][:, -1] = 0
    loss = model.training_losses(batch, "robot")["total"]
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_noise_levels_are_per_frame_and_observed_are_clean():
    stream = make_batch()["image"]
    levels = MoTWorldModel.frame_noise_levels(stream)
    assert torch.equal(levels[:, :3], torch.zeros_like(levels[:, :3]))
    assert torch.equal(levels[:, 3], levels[:, 4])
    assert torch.equal(levels[:, 4], levels[:, 5])
    assert ((levels[:, 3:] > 0) & (levels[:, 3:] < 1)).all()
