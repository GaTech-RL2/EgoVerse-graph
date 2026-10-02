import json

import numpy as np
import pytest

from egomimic.scripts.data_visualization.arc_tsne import (
    TokenizeBimanualArcLengthCartesian,
    arc_features,
    baseline_features,
    baseline_leaf,
    collect,
    load_data,
    project,
    write_report,
)


def raw_chunk(frame=0):
    t = np.arange(201) / 30
    raw = np.zeros((len(t), 14))
    raw[:, 0] = t * (0.1 + frame * 0.003)
    raw[:, 7] = t * (0.07 + frame * 0.001)
    raw[:, 1] = np.sin(t + frame) * 0.04
    raw[:, 10] = t * (0.03 + frame * 0.001)
    return raw


def codec(**kwargs):
    return TokenizeBimanualArcLengthCartesian(
        min_distance_unit=0.4,
        rotation_distance_unit=(
            0.3 if kwargs.get("velocity_mode") == "per_waypoint" else None
        ),
        resampled_vector_length=28,
        arc_chunking_mode=(
            "multistream" if kwargs.get("velocity_mode") == "per_waypoint" else None
        ),
        **kwargs,
    )


@pytest.mark.parametrize("mode", ["mean", "per_waypoint", "duration"])
def test_clock_removal_and_layout_parity(mode):
    stacked = codec(velocity_mode=mode, velocity_layout="stacked")
    token = stacked.transform({"actions_cartesian": raw_chunk()})["actions_cartesian"]
    geometry = arc_features(token, stacked, False)
    assert geometry.shape == (28 * 14,)
    mutated = token.copy()
    mutated[28:] += 55
    np.testing.assert_array_equal(geometry, arc_features(mutated, stacked, False))
    assert not np.array_equal(
        arc_features(token, stacked), arc_features(mutated, stacked)
    )
    if mode != "mean":
        wide = codec(velocity_mode=mode, velocity_layout="wide")
        wtoken = wide.transform({"actions_cartesian": raw_chunk()})["actions_cartesian"]
        for clock in [False, True]:
            np.testing.assert_allclose(
                arc_features(token, stacked, clock), arc_features(wtoken, wide, clock)
            )


class Leaf:
    def __init__(self):
        self.transform = [codec(velocity_mode="per_waypoint")]
        self.metadata = {"task": "fold </script><script>alert(1)</script>"}

    def __len__(self):
        return 18

    def __getitem__(self, frame):
        data = {"actions_cartesian": raw_chunk(frame), "frame_index": frame}
        for transform in self.transform:
            data = transform.transform(data)
        return data


def test_paired_sampling_and_report(tmp_path):
    leaf = Leaf()
    original = leaf.transform.copy()
    first, records, codecs = collect({"one": leaf}, samples_per_episode=8)
    second, records2, _ = collect({"one": leaf}, samples_per_episode=8)
    assert records == records2
    assert leaf.transform == original
    assert codecs[0]["tokenizer_config"]["min_distance_unit"] == 0.4
    for k in first:
        np.testing.assert_array_equal(first[k], second[k])
    for i, record in enumerate(records):
        np.testing.assert_array_equal(
            first["baseline"][i], raw_chunk(record["frame"])[:100].ravel()
        )
    xy, diag = project(first)
    xy2, _ = project(first)
    assert xy == xy2
    assert all(np.asarray(v).shape == (8, 2) for v in xy.values())
    output = tmp_path / "report"
    write_report(output, first, records, xy, {"projections": diag})
    html = (output / "index.html").read_text()
    assert "fold </script>" not in html
    assert "__REPORT_DATA__" not in html
    assert "https://" not in html
    assert json.loads((output / "report.json").read_text())["records"] == records
    assert set(np.load(output / "features.npz").files) == set(first)
    with pytest.raises(FileExistsError):
        write_report(output, first, records, xy, {})


def test_baseline_padding_and_bad_values():
    raw = raw_chunk()[:2]
    padded = baseline_features(raw, 4).reshape(4, 14)
    np.testing.assert_array_equal(padded[2:], np.repeat(raw[-1:], 2, axis=0))
    with pytest.raises(ValueError, match="identical"):
        project({"bad": np.ones((4, 12))})
    with pytest.raises(ValueError, match="Nonfinite"):
        baseline_features(np.full((3, 14), np.nan))


def test_baseline_window_does_not_inherit_distance_cutoff():
    leaf = Leaf()
    leaf.key_map = {
        "actions": {"horizon": {"distance": 0.4}, "key_type": "action_keys"},
        "obs": {"horizon": 1, "key_type": "proprio_keys"},
    }
    time = baseline_leaf(leaf, leaf.transform, 0, 100)
    assert time.key_map["actions"]["horizon"] == 100
    assert time.key_map["obs"]["horizon"] == 1
    assert leaf.key_map["actions"]["horizon"] == {"distance": 0.4}
    assert time.transform == []
    assert len(leaf.transform) == 1


def test_missing_codec_and_baseline_only():
    leaf = Leaf()
    leaf.transform = []
    with pytest.raises(ValueError, match="requires a data config"):
        collect({"one": leaf})
    values, _, _ = collect({"one": leaf}, representations="baseline")
    assert set(values) == {"baseline"}


def test_fallback_is_rejected_and_transforms_restored():
    class BadLeaf(Leaf):
        def __getitem__(self, frame):
            return super().__getitem__(frame + 1)

    leaf = BadLeaf()
    original = leaf.transform.copy()
    with pytest.raises(ValueError, match="substituted"):
        collect({"one": leaf})
    assert leaf.transform == original


def test_real_zarr_config_pipeline(tmp_path):
    import zarr
    from omegaconf import OmegaConf

    root = tmp_path / "episodes"
    for i in range(4):
        store = zarr.open_group(str(root / f"episode_{i}.zarr"), mode="w")
        raw = raw_chunk(i)
        store.create_array("actions", data=raw)
        store.attrs.update(
            total_frames=len(raw),
            embodiment="yam_bimanual",
            task="test",
            features={"actions": {"dtype": "float64"}},
        )
    target = "egomimic.rldb.zarr."
    config = {
        "train_datasets": {
            "yam": {
                "mode": "train",
                "valid_ratio": 0.05,
                "split_seed": 42,
                "resolver": {
                    "_target_": target + "zarr_dataset_multi.LocalEpisodeResolver",
                    "folder_path": str(root),
                    "key_map": {
                        "actions_cartesian": {"zarr_key": "actions", "horizon": 150}
                    },
                    "transform_list": [
                        {
                            "_target_": target
                            + "arc_length_tokenizer.TokenizeBimanualArcLengthCartesian",
                            "arc_chunking_mode": "multistream",
                            "rotation_distance_unit": 0.3,
                            "velocity_mode": "per_waypoint",
                            "resampled_vector_length": 28,
                        }
                    ],
                },
            }
        }
    }
    config_path = tmp_path / "data.yaml"
    OmegaConf.save(OmegaConf.create(config), config_path)
    leaves, provenance = load_data(config_path, [], None, "train", False)
    assert len(leaves) == 3  # 5% split retains one validation episode.
    assert provenance["config"]["valid_ratio"] == 0.05
    values, records, codecs = collect(leaves, samples_per_episode=3)
    assert len(records) == 9
    assert values["arc_clock"].shape == (9, 28 * 28)
    assert values["arc_no_clock"].shape == (9, 28 * 14)
    assert codecs[0]["arc_chunking_mode"] == "multistream"
    projections, diagnostics = project(values)
    write_report(
        tmp_path / "report",
        values,
        records,
        projections,
        {**provenance, "projections": diagnostics},
    )
    # The same file is valid under a full resolved training-config data section.
    OmegaConf.save(OmegaConf.create({"data": config}), config_path)
    full_leaves, _ = load_data(config_path, [], None, "train", False)
    assert set(full_leaves) == set(leaves)
    config["train_datasets"]["yam"]["resolver"]["_target_"] = (
        target + "zarr_dataset_multi.S3EpisodeResolver"
    )
    OmegaConf.save(OmegaConf.create(config), config_path)
    with pytest.raises(ValueError, match="Non-local resolver refused"):
        load_data(config_path, [], None, "train", False)
