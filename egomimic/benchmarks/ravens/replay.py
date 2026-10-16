"""Ravens replay contract on the existing benchmark replay/split/normalizer path."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from egomimic.rldb.zarr.libero_dataset import LiberoReplayResolver, LiberoNormalizer
from egomimic.rldb.embodiment.embodiment import get_embodiment_id

SUITE = "ravens-continuous-v1"
ACTION_CONTRACT = "world_absolute_xyz_rotvec_suction_20hz"
KEY_ALIASES = {
    "front_rgb": "agentview_rgb", "side_rgb": "robot0_eye_in_hand_rgb",
    "eef_pos": "robot0_eef_pos", "eef_quat": "robot0_eef_quat",
    "suction_state": "robot0_gripper_qpos", "task_uid": "task_uid", "action": "action",
}


class RavensReplayResolver(LiberoReplayResolver):
    embodiment = "ravens_ur5"

    def __init__(self, folder_path, key_map, task_module, task_class, suite=SUITE, decoded_cache=False):
        if suite != SUITE:
            raise ValueError("Unexpected Ravens task contract")
        super().__init__(folder_path, key_map, suite, decoded_cache)
        manifest = json.loads((Path(folder_path) / "benchmark_manifest.json").read_text())
        if not manifest.get("complete") or manifest.get("action_contract") != ACTION_CONTRACT:
            raise ValueError("Incomplete or incompatible continuous-control dataset")
        if manifest.get("task") != {"module": task_module, "class": task_class}:
            raise ValueError("Replay task differs from selected training goal")

    def validate_task_ids(self, task_ids):
        if not np.array_equal(task_ids, [0]):
            raise ValueError("One fixed task ID must cover every randomized scene")


class RavensNormalizer(LiberoNormalizer):
    def __init__(self, saved_state_path=None, saved_state_sha256=None, **kwargs):
        super().__init__(**kwargs)
        if (saved_state_path is None) != (saved_state_sha256 is None):
            raise ValueError("Saved normalization requires path and checksum")
        self.saved_state_path = saved_state_path
        self.saved_state_sha256 = saved_state_sha256

    def infer_norm_from_dataset(self, dataset, dataset_name, **kwargs):
        if self.saved_state_path is not None:
            if kwargs.get("precomputed_norm_path") is not None:
                raise ValueError("Do not combine native saved state and generic norms")
            import inspect
            import torch
            root = Path(dataset.resolver.folder_path)
            raw = Path(self.saved_state_path).read_bytes()
            if hashlib.sha256(raw).hexdigest() != self.saved_state_sha256:
                raise ValueError("Saved Ravens normalization checksum mismatch")
            payload = torch.load(self.saved_state_path, map_location="cpu", weights_only=False)
            stage = json.loads((root.parent / ".stage-receipt.json").read_text())
            if (not stage["complete"] or stage["sha256"] != payload["archive_sha256"] or
                    hashlib.sha256((root / "benchmark_manifest.json").read_bytes()).hexdigest()
                    != payload["dataset_manifest_sha256"]):
                raise ValueError("Saved normalization dataset/archive mismatch")
            implementation = Path(inspect.getfile(LiberoNormalizer)).read_bytes()
            if hashlib.sha256(implementation).hexdigest() != payload["normalizer_implementation_sha256"]:
                raise ValueError("Native normalization implementation changed")
            split = payload["split"]
            state = payload["normalizer_state"]
            context = state["benchmark_context"]
            if (sorted(dataset.datasets) != split["train"] or dataset.split_seed != split["seed"] or
                    dataset.val_ratio != context["val_ratio"] or dataset.split_seed != context["split_seed"] or context["suite"] != SUITE or
                    context["embodiment"] != "ravens_ur5" or context["action_contract"] != ACTION_CONTRACT or
                    payload["normalization_scope"] != "full_corpus_including_validation"):
                raise ValueError("Saved normalization split/contract mismatch")
            if (len(split["train"]) != 450 or len(split["valid"]) != 50 or
                    set(split["train"]) & set(split["valid"]) or
                    sorted(split["train"] + split["valid"]) != [f"episode_{i:06d}" for i in range(500)]):
                raise ValueError("Saved normalization episode coverage mismatch")
            actual = self.to_state()
            for key in ("norm_mode", "embodiments", "key_types", "zarr_keys", "shapes"):
                if actual[key] != state[key]:
                    raise ValueError(f"Saved normalization metadata mismatch: {key}")
            for fields in state["norm_stats"].values():
                for stats in fields.values():
                    if (set(stats) != {"min", "max", "scale", "offset"} or
                            not all(np.isfinite(value).all() for value in stats.values()) or
                            not (np.asarray(stats["scale"]) > 0).all()):
                        raise ValueError("Invalid saved native statistics")
            self._load_state(state)
            self.context = context.copy()
            return
        super().infer_norm_from_dataset(dataset, dataset_name, **kwargs)
        self.context.update(embodiment="ravens_ur5", action_contract=ACTION_CONTRACT)

    @staticmethod
    def export_saved_state(checkpoint, dataset, split_path, output):
        """Preserve fitted native values exactly; never refit or normalize a model."""
        import inspect
        import torch
        from scripts.ice.validate_lightning_checkpoint import (
            finite_scan, _allowed_lightning_callback_sentinel_paths,
        )
        checkpoint, dataset, output = map(Path, (checkpoint, dataset, output))
        if output.exists():
            raise FileExistsError(output)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        allowed = _allowed_lightning_callback_sentinel_paths(payload)
        finite_scan(payload, allowed_nonfinite_paths=allowed)
        stage = json.loads((dataset.parent / ".stage-receipt.json").read_text())
        if not stage["complete"]:
            raise ValueError("Unverified dataset stage")
        split = json.loads(Path(split_path).read_text())
        manifest_sha = hashlib.sha256((dataset / "benchmark_manifest.json").read_bytes()).hexdigest()
        if split["dataset_manifest_sha256"] != manifest_sha:
            raise ValueError("Split and dataset differ")
        torch.save({"normalizer_state": payload["normalizer_state"], "split": split,
                    "archive_sha256": stage["sha256"], "dataset_manifest_sha256": manifest_sha,
                    "normalizer_implementation_sha256": hashlib.sha256(
                        Path(inspect.getfile(LiberoNormalizer)).read_bytes()).hexdigest(),
                    "normalization_scope": "full_corpus_including_validation",
                    "fit_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}, output)
        return {"status": "PASS_EXACT_NATIVE_STATISTICS_EXPORT",
                "path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "allowed_callback_sentinel_count": len(allowed)}

    def tokenizer_context(self):
        stats = self.norm_stats[get_embodiment_id("ravens_ur5")]["actions"]
        return {**{key: value for key, value in self.context.items()
                   if key != "observations_sha256"},
                "action_scale": stats["scale"].tolist(),
                "action_offset": stats["offset"].tolist()}


def convert(input_dirs, output, expected_episodes=500):
    """Keep simulator arrays native; aliases adapt only the model's port names."""
    import zarr

    output = Path(output)
    temporary = output.with_name(output.name + ".incomplete")
    if output.exists() or temporary.exists():
        raise FileExistsError("Never overwrite a replay or an interrupted conversion")
    records = []
    seen_seeds = set()
    common_contract = None
    for directory in map(Path, input_dirs):
        manifest = json.loads((directory / "manifest.json").read_text())
        if not manifest.get("complete"):
            raise ValueError(f"Incomplete collection: {directory}")
        contract = {key: manifest[key] for key in ("upstream_commit", "task", "collector_sha256", "control_hz",
                                                 "action_frame", "simulation_runtime")}
        if common_contract is not None and contract != common_contract:
            raise ValueError("Collection shards differ in task, source or control rate")
        common_contract = contract
        if manifest["control_hz"] != 20:
            raise ValueError("Dataset requires 20 Hz controls")
        for episode in manifest["episodes"]:
            seed = episode["seed"]
            if seed in seen_seeds or episode["reward"] < .999:
                raise ValueError("Duplicate seed or unsuccessful demonstration")
            seen_seeds.add(seed)
            path = directory / episode["file"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != episode["sha256"]:
                raise ValueError(f"Episode bytes changed: {path}")
            records.append((path, episode))
    if len(records) != expected_episodes:
        raise ValueError(f"Expected {expected_episodes} episodes; found {len(records)}")
    records.sort(key=lambda item: item[1]["seed"])
    group = zarr.open_group(str(temporary), mode="w", zarr_version=2)
    data = group.create_group("data")
    total, ends = 0, []
    for path, episode in records:
        with np.load(path, allow_pickle=False) as arrays:
            from egomimic.benchmarks.ravens.continuous import validate_suction_transitions
            validate_suction_transitions(arrays["suction_state"], arrays["action"])
            length = len(arrays["action"])
            if length < 32 or arrays["action"].shape != (length, 7):
                raise ValueError("Demonstration must contain at least 32 seven-dimensional actions")
            for native, alias in KEY_ALIASES.items():
                values = arrays[native]
                if len(values) != length or not np.isfinite(values).all():
                    raise ValueError(f"Invalid {native} in {path}")
                if alias not in data:
                    data.create_dataset(alias, shape=(0, *values.shape[1:]),
                                        chunks=(8 if values.ndim == 4 else 1024, *values.shape[1:]), dtype=values.dtype)
                array = data[alias]
                array.resize((total + length, *values.shape[1:]))
                array[total:total + length] = values
        total += length
        ends.append(total)
    group.create_group("meta").create_dataset("episode_ends", shape=(len(ends),), data=np.asarray(ends, np.int64))
    manifest = {"complete": True, "suite": SUITE, "action_contract": ACTION_CONTRACT,
                **common_contract,
                "policy_camera_contract": "two native fixed cameras; port eye_in_hand is a compatibility alias for side camera",
                "key_aliases": KEY_ALIASES, "frames": total,
                "episodes": [episode for _, episode in records]}
    (temporary / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.rename(output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()
    result = convert(args.input, args.output, args.episodes)
    print(json.dumps({"episodes": len(result["episodes"]), "frames": result["frames"]}))
