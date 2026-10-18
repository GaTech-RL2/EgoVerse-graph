"""Offline only: no robot, camera, CAN, or pretrained-network construction."""

import numpy as np
import pytest
import torch

from egomimic.robot.pinned_arc_bbb813a0 import (
    PinnedArcDecoder,
    frozen_modules,
    verify_sources,
)


@pytest.mark.parametrize("mode", ["pw", "logsec"])
def test_all_hold_is_400_frames(mode):
    codec, _ = frozen_modules()
    raw = np.zeros((400, 14))
    raw[:, [0, 7, 6, 13]] = [0.12, 0.23, 0.4, 0.7]
    tok = codec.TokenizeBimanualArcLengthCartesian(
        min_distance_unit=0.42,
        rotation_distance_unit=np.deg2rad(100),
        arc_chunking_mode="multistream",
        resampled_vector_length=100,
        velocity_mode="per_waypoint" if mode == "pw" else "duration",
        velocity_layout="rate_clock" if mode == "pw" else "log_clock6",
        gripper_stream_mode="events",
    )
    tokens = tok.transform({"actions_cartesian": raw.copy()})["actions_cartesian"]
    output = PinnedArcDecoder(mode)(tokens[None])
    assert output.shape == (1, 400, 14)
    np.testing.assert_allclose(output[0], np.repeat(raw[:1], 400, axis=0))


@pytest.mark.parametrize("mode", ["pw", "logsec"])
@pytest.mark.parametrize("stream", range(6))
def test_each_stream_can_end_chunk(mode, stream):
    # Six independent moving streams; fastest needs exactly 29*0.1=2.9 sec.
    tokens = np.zeros((100, 20), dtype=np.float64)
    tokens[:, 0] = np.arange(100) * 0.004
    tokens[:, 7] = np.arange(100) * 0.003
    tokens[:, 3] = np.arange(100) * 0.01
    tokens[:, 10] = np.arange(100) * 0.008
    tokens[:, 6] = np.arange(100) * 0.008
    tokens[:, 13] = np.arange(100) * 0.007
    interval = np.full(6, 0.2)
    interval[stream] = 0.1
    if mode == "pw":
        tokens[:, 14:] = np.array([0.004, 0.003, 0.01, 0.008, 0.008, 0.007]) / interval
    else:
        tokens[:, 14:] = np.log(interval + 0.001)
    decoded = PinnedArcDecoder(mode)(tokens[None])
    assert decoded.shape == (1, 87, 14)  # Not 30 and not ceil(29*exp(seconds)*30).
    np.testing.assert_allclose(decoded[0, 0], tokens[0, :14], atol=1e-12)
    assert decoded[0, :, 6].max() <= tokens[29, 6] + 1e-12


@pytest.mark.parametrize("mode", ["pw", "logsec"])
def test_random_tokenizer_reference_parity(mode):
    codec, source = frozen_modules()
    decoder = PinnedArcDecoder(mode)
    ref = decoder.reference
    tokenizer = codec.TokenizeBimanualArcLengthCartesian(
        min_distance_unit=0.42,
        rotation_distance_unit=np.deg2rad(100),
        arc_chunking_mode="multistream",
        resampled_vector_length=100,
        velocity_mode=ref.velocity_mode,
        velocity_layout=ref.velocity_layout,
        gripper_stream_mode="events",
    )
    rng = np.random.default_rng(42)
    for _ in range(30):
        raw = np.cumsum(rng.normal(size=(400, 14)) * 0.01, axis=0)
        raw[:, 6] = np.repeat(rng.uniform(size=8), 50)
        raw[:, 13] = np.repeat(rng.uniform(size=8), 50)
        tokens = tokenizer.transform({"actions_cartesian": raw})[
            "actions_cartesian"
        ].astype(np.float32)
        seconds = ref._clock_seconds(torch.from_numpy(tokens)).numpy()
        stacked = codec.stack_arc_token(seconds)
        partial = source.arc_execution_prefix(
            stacked, 0.3, ref.velocity_mode, 0.42, "waypoints"
        )
        steps = source.arc_prefix_control_steps(
            stacked,
            0.3,
            ref.velocity_mode,
            1 / 30,
            0.42,
            rotation_distance_unit=np.deg2rad(100),
            arc_chunking_mode="multistream",
            velocity_layout=ref.velocity_layout,
            gripper_stream_mode="events",
        )
        m = len(partial) // 2
        native = np.concatenate(
            (partial[:m], partial[m:, list(codec.SIX_CLOCK_COLUMNS)]), axis=1
        )
        expected = tokenizer.detokenize(
            native,
            action_horizon=steps,
            **({"clocks_in_seconds": True} if mode == "logsec" else {}),
        )
        actual = decoder(torch.from_numpy(tokens[None]))[0]
        np.testing.assert_array_equal(actual, expected)


def test_bad_shape_and_source_pin():
    verify_sources()
    with pytest.raises(ValueError):
        PinnedArcDecoder("pw", training_commit="different")
    with pytest.raises(ValueError):
        PinnedArcDecoder("pw")(np.zeros((1, 100, 18)))
    with pytest.raises(ValueError):
        PinnedArcDecoder("logsec")(np.full((1, 100, 20), np.inf))


def test_source_change_is_rejected(monkeypatch):
    from egomimic.robot import pinned_arc_bbb813a0 as module

    monkeypatch.setitem(module.SOURCE_HASHES, "tokenizer.py", "0" * 64)
    with pytest.raises(ValueError, match="source hash mismatch"):
        PinnedArcDecoder("pw")


def test_no_second_frame_crop():
    from types import SimpleNamespace

    from egomimic.robot.graph_policy import GraphRobotPolicy

    policy = object.__new__(GraphRobotPolicy)
    policy.replan_every = None
    policy.adapter = SimpleNamespace(decoder=PinnedArcDecoder("pw"))
    actions = np.zeros((87, 14))
    assert policy.execution_plan(actions).shape == (87, 14)


def test_rollout_rejects_replan_truncation_with_pinned_binding(monkeypatch):
    from types import SimpleNamespace

    from egomimic.robot import rollout_pinned_arc as entry

    policy = SimpleNamespace(
        adapter=SimpleNamespace(decoder=PinnedArcDecoder("pw")),
        replan_every=30,
        output_shape=None,
    )
    monkeypatch.setattr(entry, "validate_bundle", lambda config: {"mode": "pw"})
    monkeypatch.setattr(entry, "_original_load_policy", lambda config: policy)
    with pytest.raises(ValueError, match="without additional truncation"):
        entry.load_policy({"kind": "graph"})


def test_rollout_rejects_different_control_frequency(monkeypatch):
    from egomimic.robot import rollout_pinned_arc as entry

    monkeypatch.setattr(entry, "_original_validate_config", lambda config: None)
    with pytest.raises(ValueError, match="30 Hz"):
        entry.validate_rollout_config({"frequency": 60})


def test_checkpoint_hash_mismatch_rejected_before_model_load(tmp_path, monkeypatch):
    import json

    from egomimic.robot import rollout_pinned_arc as entry

    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"changed weights")
    checkpoint.with_suffix(".pinned-arc.json").write_text(
        json.dumps(
            {
                "training_commit": entry.TRAINING_COMMIT,
                "files": {"checkpoint": {"name": checkpoint.name, "sha256": "0" * 64}},
            }
        )
    )

    def forbid(config):
        raise AssertionError("Mismatch must be rejected before loading graph")

    monkeypatch.setattr(entry, "_original_load_policy", forbid)
    with pytest.raises(ValueError, match="SHA-256"):
        entry.load_policy({"kind": "graph", "checkpoint": str(checkpoint)})
