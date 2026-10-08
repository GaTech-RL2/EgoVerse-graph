import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from egomimic.rldb.zarr.planar_arc_grouped import TokenizePlanarArcGrouped
from egomimic.pipeline.planar_grouped import PlanarArcGroupedNativeDecoder
from egomimic.eval.planar_rollout import PlanarTimedArcExecutionSelector
from egomimic.pipeline.algo import PipelineAlgo

ROOT = Path(__file__).resolve().parents[1]


def codec(group, mode, m=16, horizon=80, dim=4):
    args = dict(translation_groups=group, timing_mode=mode, resampled_vector_length=m)
    return (TokenizePlanarArcGrouped(**args),
            PlanarArcGroupedNativeDecoder(**args, action_horizon=horizon, native_action_dim=dim))


@pytest.mark.parametrize("group", ["xy", "x_y"])
@pytest.mark.parametrize("mode", ["duration", "velocity"])
@pytest.mark.parametrize("m", [16, 56])
def test_independent_linear_streams_and_tensor_batch(group, mode, m):
    t = np.arange(80) / 30
    actions = np.column_stack((10 + 3*t, 20 - 2*t, .1 + .03*t, .1 + .05*t))
    enc, dec = codec(group, mode, m)
    tokens = enc.tokenize(actions)
    assert tokens.shape == (m, 8 if group == "xy" else 9)
    np.testing.assert_allclose(dec(tokens), actions, atol=1e-10)
    batch = dec(torch.from_numpy(np.stack([tokens, tokens])))
    assert batch.shape == (2, 80, 4)
    np.testing.assert_allclose(batch[1], actions, atol=1e-10)


@pytest.mark.parametrize("group", ["xy", "x_y"])
@pytest.mark.parametrize("mode", ["duration", "velocity"])
def test_gripper_clock_is_independent_of_translation_and_rotation(group, mode):
    a = np.zeros((80, 4)); a[:, 3] = np.linspace(.1, .8, 80)
    enc, dec = codec(group, mode)
    before = enc.tokenize(a)
    np.testing.assert_allclose(dec(before), a, atol=1e-12)
    a[:, :3] = np.arange(80)[:, None] * np.array([2, 1, .01])
    after = enc.tokenize(a)
    np.testing.assert_array_equal(before[:, [4, -1]], after[:, [4, -1]])


@pytest.mark.parametrize("group", ["xy", "x_y"])
def test_duration_preserves_gripper_waits_and_final_dwell(group):
    a = np.zeros((31, 4))
    a[:, 3] = np.r_[np.linspace(0, .5, 11), np.repeat(.5, 10), np.linspace(.55, 1, 10)]
    enc, dec = codec(group, "duration", horizon=31)
    np.testing.assert_allclose(dec(enc.tokenize(a)), a, atol=1e-10)
    a[:, 3] = np.r_[np.linspace(0, 1, 16), np.ones(15)]
    np.testing.assert_allclose(dec(enc.tokenize(a)), a, atol=1e-10)


def test_stk_does_not_invent_a_clock_for_zero_distance():
    a = np.zeros((80, 4)); a[40:, 3] = np.linspace(0, 1, 40)
    enc, dec = codec("xy", "velocity")
    result = dec(enc.tokenize(a))
    assert a[20, 3] == 0 and result[20, 3] > .001
    assert result[0, 3] == 0


@pytest.mark.parametrize("mode", ["duration", "velocity"])
def test_rotation_wrap_and_missing_gripper(mode):
    a = np.zeros((80, 3)); a[:, 2] = (3.1 + np.arange(80)*.004 + np.pi) % (2*np.pi) - np.pi
    enc, dec = codec("xy", mode, dim=3)
    token = enc.tokenize(a); result = dec(token)
    error = np.arctan2(np.sin(result[:, 2] - a[:, 2]), np.cos(result[:, 2] - a[:, 2]))
    np.testing.assert_allclose(error, 0, atol=1e-10)
    assert np.all(token[:, 4] == 0)
    assert not dec.waypoint_clocks(token)["gripper"]["active"]


@pytest.mark.parametrize("mode", ["duration", "velocity"])
def test_bad_clock_slows_motion_and_selector_uses_gripper(mode):
    token = np.zeros((16, 8)); token[:, 2] = 1
    token[:, 4] = np.linspace(0, 1, 16)
    _, dec = codec("xy", mode)
    out = dec(token)
    assert out[0, 3] == 0 and out[1, 3] < .002
    token[:, -1] = .1 if mode == "duration" else (1/15)/.1
    selection = PlanarTimedArcExecutionSelector().select(token, dec)
    assert selection["execution_steps"] == 22
    assert selection["clocks"]["gripper"]["active"]


def test_nine_recipes_match_budget_backbone_and_native_decoder():
    names = ["dp"] + [f"{g}_{t}_m{m}" for g in ("xy", "x_y")
                       for t in ("stk", "dur") for m in (16, 56)]
    for name in names:
        with initialize_config_dir(version_base=None, config_dir=str(ROOT / "egomimic/hydra_configs")):
            cfg = compose(config_name="train_zarr_cartesian", overrides=[f"+experiment=pusht/planar_streams_{name}"])
        assert cfg.trainer.max_steps == cfg.model.scheduler.max_steps == 240000
        assert cfg.launch_params.gpus_per_node * cfg.planar.batch_size * 2 == 128
        assert cfg.planar.observation_horizon == 2
        assert cfg.model.pipeline.homogeneous_training is True
        assert cfg.ckpt_path is None
        net = cfg.model.pipeline.stages[3].policy.model
        assert list(net.down_dims) == [512, 1024, 2048]
        assert net.input_dim == cfg.planar.action_dim
        assert cfg.eval_checkpoint.use_ema is True
        for domain, dc in cfg.evaluator.native_decoders.items():
            dec = instantiate(dc)
            assert dec.native_action_dim == (3 if domain.endswith("u_socket") else 4)
            assert dec.action_horizon == (16 if name == "dp" else 80)
        for dc in cfg.data.train_datasets.values():
            assert dc.resolver.split == "train" and dc.mode == "total"
        for dc in cfg.data.valid_datasets.values():
            assert dc.resolver.split == "valid" and dc.mode == "total"


def test_cache_alignment_image_parity_and_pinned_split(tmp_path):
    from egomimic.rldb.zarr.numpy_episode import NumpyEpisodeResolver
    from egomimic.rldb.embodiment.pushshapes import get_planar_keymap, get_planar_paper_transform_list
    from egomimic.pipeline.pushshapes import PlanarCommon5NativeDecoder
    rows = []
    for alignment, offset in (("pre_step", 1), ("post_step", 2)):
        for split in ("train", "valid"):
            eid = alignment + split; folder = tmp_path / eid; folder.mkdir()
            a = np.column_stack((np.arange(12), np.zeros((12, 3)))).astype(float)
            state = np.zeros((12, 6)); state[:, 0] = np.arange(12) + offset - 1
            images = np.arange(12*6*8*3, dtype=np.uint8).reshape(12, 6, 8, 3)
            for key, array in (("actions", a), ("observations.state", state), ("observations.images.front_img_1", images)):
                np.save(folder / (key + ".npy"), array)
            rows.append(dict(episode_id=eid, total_frames=12, embodiment="pushshapes_sim_chain_gripper",
                split=split, source="fixture", observation_alignment=alignment, action_target_offset_obs2=offset))
    body = json.dumps({"episodes": rows}).encode(); (tmp_path / "manifest.json").write_bytes(body)
    options = dict(folder_path=tmp_path, manifest_sha256=hashlib.sha256(body).hexdigest(),
        embodiment="pushshapes_sim_chain_gripper", key_map=get_planar_keymap(action_horizon=4, observation_horizon=2, action_target_offset=1),
        transform_list=get_planar_paper_transform_list(action_horizon=4, action_target_offset=1))
    train = NumpyEpisodeResolver(split="train", **options).resolve()
    valid = NumpyEpisodeResolver(split="valid", **options).resolve()
    assert not set(train) & set(valid)
    for ds in train.values():
        sample = ds[3]
        native = PlanarCommon5NativeDecoder(4, 4)(sample["actions"]).reshape(4, 4)
        assert native[0, 0] == sample["state_agent_obj"][-1, 0]
        np.testing.assert_array_equal(sample["front_img_1"], torch.from_numpy(np.moveaxis(images[3:5], -1, 1)/255).float())
        assert ds[11]["actions"].shape == (4, 5)
    (tmp_path / "manifest.json").write_bytes(body + b" ")
    with pytest.raises(ValueError, match="manifest hash"):
        NumpyEpisodeResolver(split="train", **options).resolve()


def test_homogeneous_mean_and_gradient_match_equal_source_loss():
    x = torch.arange(12., requires_grad=True).reshape(4, 3)
    y = (torch.arange(12.) + 10).reshape(4, 3)
    a = {"x": x, "episode_hash": ["a"]*4}; b = {"x": y, "episode_hash": ["b"]*4}
    w = torch.nn.Parameter(torch.tensor(2.))
    loss_loop = ((w*x).square().mean() + (w*y).square().mean()) / 2
    grad_loop = torch.autograd.grad(loss_loop, w)[0]
    merged = PipelineAlgo._fuse_equal_batches([a, b])
    loss_fused = (w*merged["x"]).square().mean()
    grad_fused = torch.autograd.grad(loss_fused, w)[0]
    torch.testing.assert_close(loss_loop, loss_fused)
    torch.testing.assert_close(grad_loop, grad_fused)
    assert merged["episode_hash"] == ["a"]*4 + ["b"]*4
    assert PipelineAlgo._fuse_equal_batches([a, {"x": y[:2], "episode_hash": ["b"]*2}]) is None
