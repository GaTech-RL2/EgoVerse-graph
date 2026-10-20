"""Guard action/observation alignment and reuse of the selected LIBERO models."""

import numpy as np
import pytest

from egomimic.benchmarks.ravens.continuous import (
    defer_dependency,
    validate_suction_transitions,
)


def test_integrated_embodiment_ids_are_distinct_and_replay_keeps_ravens(tmp_path):
    import json

    from egomimic.benchmarks.ravens.replay import (
        ACTION_CONTRACT,
        RavensReplayResolver,
    )
    from egomimic.rldb.embodiment.embodiment import EMBODIMENT, get_embodiment_id
    from egomimic.rldb.zarr.libero_dataset import LiberoReplayResolver

    assert get_embodiment_id("libero_panda") == 21
    assert get_embodiment_id("libero_panda_sharpa") == 22
    assert get_embodiment_id("ravens_ur5") == 23
    assert len({member.value for member in EMBODIMENT}) == len(EMBODIMENT.__members__)
    (tmp_path / "benchmark_manifest.json").write_text(
        json.dumps(
            {
                "complete": True,
                "action_contract": ACTION_CONTRACT,
                "task": {"module": "fixture", "class": "Task"},
            }
        )
    )
    resolver = RavensReplayResolver(tmp_path, {}, "fixture", "Task")
    assert resolver.embodiment == "ravens_ur5"
    assert resolver.embodiment_id == 23
    resolver.validate_task_ids(np.array([0]))
    with pytest.raises(ValueError, match="fixed task ID"):
        resolver.validate_task_ids(np.array([1]))
    assert LiberoReplayResolver(tmp_path, {}, "spatial").embodiment_id == 21


@pytest.mark.parametrize("corruption", [None, "checksum", "dataset", "metadata"])
def test_saved_native_normalization_is_exact_and_fails_closed(tmp_path, corruption):
    torch = pytest.importorskip("torch")
    pytest.importorskip(
        "pandas"
    )  # Full data runtime: exercised in the scheduled integration gate.
    import hashlib
    import json
    from types import SimpleNamespace

    from egomimic.benchmarks.ravens.replay import (
        ACTION_CONTRACT,
        SUITE,
        RavensNormalizer,
    )
    from egomimic.rldb.embodiment.embodiment import get_embodiment_id

    emb = get_embodiment_id("ravens_ur5")
    state = {
        "norm_mode": "oat_limits",
        "embodiments": [emb],
        "key_types": {emb: {"actions": "action_keys"}},
        "zarr_keys": {emb: {"actions": "action"}},
        "shapes": {emb: {"actions": (32, 7)}},
        "norm_stats": {
            emb: {
                "actions": {
                    key: np.full(7, value, np.float32)
                    for key, value in [
                        ("min", 0),
                        ("max", 1),
                        ("scale", 2),
                        ("offset", -1),
                    ]
                }
            }
        },
        "benchmark_context": {
            "suite": SUITE,
            "split_seed": 42,
            "val_ratio": 0.1,
            "embodiment": "ravens_ur5",
            "action_contract": ACTION_CONTRACT,
        },
    }
    dataset = tmp_path / "stage" / "replay.zarr"
    dataset.mkdir(parents=True)
    (dataset / "benchmark_manifest.json").write_text('{"fixture":true}')
    (dataset.parent / ".stage-receipt.json").write_text(
        json.dumps({"complete": True, "sha256": "a" * 64})
    )
    names = [f"episode_{i:06d}" for i in range(500)]
    split = {
        "train": names[:450],
        "valid": names[450:],
        "seed": 42,
        "dataset_manifest_sha256": hashlib.sha256(
            (dataset / "benchmark_manifest.json").read_bytes()
        ).hexdigest(),
    }
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    checkpoint = tmp_path / "fixture.ckpt"
    torch.save(
        {"state_dict": {"weight": torch.ones(1)}, "normalizer_state": state}, checkpoint
    )
    saved = tmp_path / "normalization.pt"
    receipt = RavensNormalizer.export_saved_state(
        checkpoint, dataset, split_path, saved
    )
    normalizer = RavensNormalizer(
        state=state, saved_state_path=saved, saved_state_sha256=receipt["sha256"]
    )
    selected = SimpleNamespace(
        resolver=SimpleNamespace(folder_path=dataset),
        datasets=dict.fromkeys(names[:450]),
        split_seed=42,
        val_ratio=0.1,
    )
    if corruption == "checksum":
        normalizer.saved_state_sha256 = "b" * 64
    elif corruption == "dataset":
        (dataset / "benchmark_manifest.json").write_text('{"fixture":false}')
    elif corruption == "metadata":
        normalizer.shapes[emb]["actions"] = (100, 7)
    if corruption:
        with pytest.raises(ValueError, match="mismatch"):
            normalizer.infer_norm_from_dataset(selected, "ravens_ur5")
    else:
        normalizer.infer_norm_from_dataset(selected, "ravens_ur5")
        actual = normalizer.to_state()
        assert actual["benchmark_context"] == state["benchmark_context"]
        for key, value in state["norm_stats"][emb]["actions"].items():
            assert (
                actual["norm_stats"][emb]["actions"][key].tobytes() == value.tobytes()
            )


def test_optional_dependency_loads_actual_module_only_when_used(tmp_path, monkeypatch):
    import importlib
    import sys

    name = "ravens_lazy_dependency_fixture"
    (tmp_path / f"{name}.py").write_text("VALUE = 42\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    proxy = defer_dependency(name)
    try:
        assert "VALUE" not in vars(proxy)
        assert importlib.import_module(name) is proxy
        assert proxy.VALUE == 42
        loaded = sys.modules[name]
        assert loaded is not proxy
        assert proxy.VALUE == loaded.VALUE
        assert sys.modules[name] is loaded
    finally:
        sys.modules.pop(name, None)


def test_suction_recording_requires_decisions_before_state_changes():
    states = np.array([[0.0, 0.0], [1.0, 0.0]], np.float32)
    actions = np.zeros((2, 7), np.float32)
    actions[:, 6] = [1.0, -1.0]
    validate_suction_transitions(states, actions)
    leaked_states = np.array([[1.0, 0.0], [0.0, 0.0]], np.float32)
    with pytest.raises(ValueError, match="pre-action suction"):
        validate_suction_transitions(leaked_states, actions)
