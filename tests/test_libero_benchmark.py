import json
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr

from egomimic.benchmarks.libero.catalog import TASK_IDS, TASKS, get_tasks
from egomimic.benchmarks.libero.report import compare_runs, read_run
from egomimic.benchmarks.libero.rollout import rollout_plan, run_rollouts
from egomimic.rldb.zarr.libero_dataset import (
    EMBODIMENT,
    LiberoDataset,
    LiberoNormalizer,
    LiberoReplayResolver,
    keymap,
    validation_mask,
)


def make_replay(path, suite="libero10", episode_length=5):
    tasks = get_tasks(suite)
    ids = np.repeat(
        [TASK_IDS[task][2] for task in tasks for _ in range(2)], episode_length
    )
    n = len(ids)
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    data = group.create_group("data")
    rng = np.random.default_rng(5)
    arrays = {
        "action": rng.uniform(-1, 1, (n, 7)).astype(np.float32),
        "robot0_eef_pos": rng.normal(size=(n, 3)).astype(np.float32),
        "robot0_eef_quat": np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (n, 1)),
        "robot0_gripper_qpos": rng.normal(size=(n, 2)).astype(np.float32),
        "task_uid": ids[:, None],
        "agentview_rgb": rng.integers(0, 256, (n, 128, 128, 3), dtype=np.uint8),
        "robot0_eye_in_hand_rgb": rng.integers(
            0, 256, (n, 128, 128, 3), dtype=np.uint8
        ),
    }
    for key, values in arrays.items():
        data.create_array(key, data=values)
    group.create_group("meta").create_array(
        "episode_ends", data=np.arange(episode_length, n + 1, episode_length)
    )
    return arrays


def dataset_and_normalizer(path, n_obs_steps=2, horizon=8):
    resolver = LiberoReplayResolver(path, keymap(n_obs_steps, horizon), "libero10")
    dataset = LiberoDataset._from_resolver(resolver)
    normalizer = LiberoNormalizer()
    normalizer.populate_from_datasets({"libero_panda": dataset})
    normalizer.infer_shapes_from_batch(dataset[0])
    normalizer.infer_norm_from_dataset(dataset, "libero_panda")
    return dataset, normalizer


def test_catalog_covers_every_upstream_task_and_preserves_uid_order():
    assert {suite: len(tasks) for suite, tasks in TASKS.items()} == {
        "libero_spatial": 10,
        "libero_object": 10,
        "libero_goal": 10,
        "libero_10": 10,
        "libero_90": 90,
    }
    assert len(TASK_IDS) == 130
    assert TASK_IDS[get_tasks("libero10")[0]][2] == 30
    assert TASK_IDS[get_tasks("libero90")[0]][2] == 40
    assert sum(len(rollout_plan(suite)) for suite in TASKS) == 32500
    with pytest.raises(ValueError, match="Unknown"):
        get_tasks("not_a_task")


def test_loader_alignment_padding_split_and_normalizer_match_oat(tmp_path):
    from egomimic.models.oat.model.common.normalizer import LinearNormalizer

    path = tmp_path / "replay.zarr"
    arrays = make_replay(path)
    dataset, normalizer = dataset_and_normalizer(path)
    mask = validation_mask(20)
    expected = np.zeros(20, bool)
    expected[np.random.default_rng(42).choice(20, 2, replace=False)] = True
    np.testing.assert_array_equal(mask, expected)
    valid = LiberoDataset._from_resolver(dataset.resolver, mode="valid")
    assert not set(dataset.datasets) & set(valid.datasets)
    assert set(dataset.datasets) | set(valid.datasets) == {
        f"episode_{i:06d}" for i in range(20)
    }
    leaf = next(iter(dataset.datasets.values()))
    sample = leaf[0]
    np.testing.assert_array_equal(
        sample["robot0_eef_pos"],
        np.repeat(arrays["robot0_eef_pos"][leaf.start : leaf.start + 1], 2, axis=0),
    )
    np.testing.assert_array_equal(sample["actions"][0], arrays["action"][leaf.start])
    np.testing.assert_array_equal(sample["actions"][-1], arrays["action"][leaf.end - 1])
    sample = leaf[4]
    np.testing.assert_array_equal(
        sample["robot0_eef_pos"], arrays["robot0_eef_pos"][leaf.end - 2 : leaf.end]
    )
    dataset.set_norm_stats_from(normalizer)
    assert dataset.norm_stats is normalizer.norm_stats
    raw, normalized = leaf[0], dataset[0]
    source = LinearNormalizer()
    source.fit(
        {"actions" if key == "action" else key: value for key, value in arrays.items()}
    )
    for key in keymap():
        torch.testing.assert_close(
            normalized[key], source[key].normalize(raw[key]), rtol=0, atol=0
        )
    restored = LiberoNormalizer(state=normalizer.to_state())
    for key, value in restored.unnormalize(normalized, EMBODIMENT).items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, raw[key], atol=2e-5, rtol=1e-5)
    restored.assert_tokenizer_context(normalizer.tokenizer_context())


def test_tokenizer_and_policy_normalizers_share_action_stats(tmp_path):
    path = tmp_path / "replay.zarr"
    make_replay(path)
    _, tokenizer = dataset_and_normalizer(path, n_obs_steps=0)
    _, policy = dataset_and_normalizer(path, n_obs_steps=2)
    policy.assert_tokenizer_context(tokenizer.tokenizer_context())
    wrong = tokenizer.tokenizer_context() | {"split_seed": 123}
    with pytest.raises(ValueError, match="differ"):
        policy.assert_tokenizer_context(wrong)


class FakeEnvironment:
    def __init__(self, task):
        self.task = task
        self.closed = False

    def reset(self, seed):
        self.steps = 0
        self.initial_state_sha256 = f"state-{self.task}-{seed}"
        return {"frame": 0}

    def step(self, action):
        self.steps += 1
        return {"frame": self.steps}, self.steps == 3, False

    def close(self):
        self.closed = True


class FakePolicy:
    def reset(self, observation):
        self.seen = [observation["frame"]]

    def observe(self, observation):
        self.seen.append(observation["frame"])

    def predict(self):
        return np.zeros((16, 7))


def test_resume_rollouts_executes_only_missing_trials_and_keeps_existing_bytes(
    tmp_path,
):
    plan = rollout_plan("libero_spatial", trials_per_task=1, repetitions=1)
    metadata = {
        "checkpoint_sha256": "same-checkpoint",
        "suite": "libero_spatial",
        "camera_keys": ("front", "wrist"),
    }
    output = tmp_path / "episodes"

    class InterruptedPolicy(FakePolicy):
        resets = 0

        def reset(self, observation):
            self.resets += 1
            if self.resets == 4:
                raise RuntimeError("simulated preemption")
            super().reset(observation)

    with pytest.raises(RuntimeError, match="preemption"):
        run_rollouts(
            InterruptedPolicy(),
            plan,
            output,
            env_factory=FakeEnvironment,
            metadata=metadata,
        )
    saved = (output / "episodes.jsonl").read_bytes()
    assert len(saved.splitlines()) == 3
    with pytest.raises(ValueError, match="Missing"):
        read_run(output)
    protocol, partial = read_run(output, require_complete=False)
    assert len(partial) == 3
    with pytest.raises(ValueError, match="protocol or checkpoint"):
        run_rollouts(
            FakePolicy(),
            plan,
            output,
            env_factory=FakeEnvironment,
            metadata={**metadata, "checkpoint_sha256": "different"},
            resume=True,
        )
    assert (output / "episodes.jsonl").read_bytes() == saved
    seen = []

    def factory(task):
        seen.append(task)
        return FakeEnvironment(task)

    records = run_rollouts(
        FakePolicy(), plan, output, env_factory=factory, metadata=metadata, resume=True
    )
    assert len(seen) == len(plan) - 3
    assert (output / "episodes.jsonl").read_bytes().startswith(saved)
    _, complete = read_run(output)
    assert len(complete) == len(records) == len(plan)
    seen.clear()
    run_rollouts(
        FakePolicy(), plan, output, env_factory=factory, metadata=metadata, resume=True
    )
    assert not seen


def test_rollout_cli_resumes_interrupted_trials_only_with_explicit_flag(
    tmp_path, monkeypatch
):
    import sys
    from types import SimpleNamespace

    from egomimic.benchmarks.libero import cli, rollout

    class InterruptedPolicy(FakePolicy):
        resets = 0
        interrupted = False

        def reset(self, observation):
            self.resets += 1
            if self.resets == 4 and not self.interrupted:
                self.interrupted = True
                raise RuntimeError("simulated preemption")
            super().reset(observation)

    policy = InterruptedPolicy()
    policy.algo = SimpleNamespace(
        pipeline=SimpleNamespace(stages=[]), nets=torch.nn.Linear(1, 1)
    )
    policy.normalizer = SimpleNamespace(
        tokenizer_context=lambda: {}, context={"observations_sha256": "observations"}
    )
    protocol = {"suite": "libero_spatial", "horizon": 32}
    monkeypatch.setattr(rollout, "load_policy", lambda *a, **kw: (policy, protocol))
    monkeypatch.setattr(cli, "policy_method", lambda *a: "dp_unet")
    monkeypatch.setattr(
        rollout,
        "run_rollouts",
        lambda *a, **kw: run_rollouts(*a, env_factory=FakeEnvironment, **kw),
    )
    checkpoint = tmp_path / "policy.ckpt"
    checkpoint.write_bytes(b"unchanged policy checkpoint")
    output = tmp_path / "rollout"
    command = [
        "libero",
        "rollout",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--trials-per-task",
        "1",
        "--repetitions",
        "1",
        "--video-trials",
        "0",
    ]
    monkeypatch.setattr(sys, "argv", command)
    with pytest.raises(RuntimeError, match="preemption"):
        cli.main()
    saved = (output / "episodes.jsonl").read_bytes()
    assert len(saved.splitlines()) == 3
    with pytest.raises(FileExistsError):
        cli.main()
    monkeypatch.setattr(sys, "argv", command + ["--resume"])
    cli.main()
    assert len(read_run(output)[1]) == 10
    assert (output / "episodes.jsonl").read_bytes().startswith(saved)
    assert policy.resets == 11  # Ten completed trials plus the interrupted trial.
    cli.main()
    assert policy.resets == 11


def test_rollouts_stop_inside_chunk_and_compare_all_expected_records(tmp_path):
    plan = rollout_plan("libero10", trials_per_task=1, repetitions=2)
    assert len(plan) == 20 and len({spec.seed for spec in plan}) == 20
    metadata = {
        "suite": "libero10",
        "horizon": 32,
        "n_obs_steps": 2,
        "n_action_steps": 16,
        "data_context": {"dataset_sha256": "same"},
        "observations_sha256": "same-observations",
        "use_ema": True,
        "checkpoint_sha256": "checkpoint",
    }
    for method in ("arc", "oat"):
        environments = []

        def factory(task):
            env = FakeEnvironment(task)
            environments.append(env)
            return env

        records = run_rollouts(
            FakePolicy(),
            plan,
            tmp_path / method,
            env_factory=factory,
            metadata={**metadata, "method": method},
        )
        assert len(environments) == 10 and all(env.closed for env in environments)
        assert all(record["steps"] == 3 and record["success"] for record in records)
    result = compare_runs(tmp_path / "arc", tmp_path / "oat", require_full=False)
    assert result["paired_success_difference"] == 0
    assert result["arc"]["mean_success_rate"] == 1
    protocol_file = tmp_path / "arc/protocol.json"
    protocol = json.loads(protocol_file.read_text())
    protocol["representation"] = {"mode": "stk"}
    protocol_file.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="representation mode"):
        compare_runs(
            tmp_path / "arc", tmp_path / "oat", require_full=False, arc_mode="dur"
        )
    assert compare_runs(
        tmp_path / "arc", tmp_path / "oat", require_full=False, arc_mode="stk"
    )["arc"]["episodes"] == len(plan)
    with pytest.raises(ValueError, match="Full benchmark"):
        compare_runs(tmp_path / "arc", tmp_path / "oat")
    file = tmp_path / "arc/episodes.jsonl"
    original_records = file.read_text()
    records = [json.loads(line) for line in original_records.splitlines()]
    records[0]["initial_state_sha256"] = "different-state"
    file.write_text("".join(json.dumps(record) + "\n" for record in records))
    with pytest.raises(ValueError, match="Initial states differ"):
        compare_runs(tmp_path / "arc", tmp_path / "oat", require_full=False)
    file.write_text(original_records)
    protocol_file = tmp_path / "arc/protocol.json"
    protocol = json.loads(protocol_file.read_text())
    protocol["observations_sha256"] = "different-cameras"
    protocol_file.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="observations_sha256"):
        compare_runs(tmp_path / "arc", tmp_path / "oat", require_full=False)
    file.write_text(original_records + original_records.splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_run(tmp_path / "arc")
    file.write_text(original_records.splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="Missing"):
        read_run(tmp_path / "arc")


def test_conversion_matches_source_image_orientation_and_quaternion(tmp_path):
    import h5py
    from scipy.spatial.transform import Rotation

    from egomimic.benchmarks.libero.convert import convert_suite

    task = get_tasks("libero10")[0]
    root = tmp_path / "raw"
    root.mkdir()
    with h5py.File(root / f"{task}_demo.hdf5", "w") as handle:
        data = handle.create_group("data")
        data.attrs["bddl_file_name"] = f"/bddl/{task}.bddl"
        data.attrs["problem_info"] = json.dumps({"language_instruction": "test"})
        for index in range(2):
            demo = data.create_group(f"demo_{index}")
            demo["actions"] = np.zeros((4, 7), np.float32)
            obs = demo.create_group("obs")
            pixels = np.zeros((4, 4, 4, 3), np.uint8)
            pixels[:, 0] = 200
            obs["agentview_rgb"] = pixels
            obs["eye_in_hand_rgb"] = pixels
            obs["ee_pos"] = np.zeros((4, 3), np.float32)
            obs["ee_ori"] = np.tile([0, 0, 0.5], (4, 1))
            obs["gripper_states"] = np.zeros((4, 2), np.float32)
    output = tmp_path / "converted.zarr"
    manifest = convert_suite(root, output, task)
    array = zarr.open_group(str(output), mode="r")["data"]
    assert np.all(array["agentview_rgb"][:, -1] == 200)
    np.testing.assert_allclose(
        array["robot0_eef_quat"][0],
        Rotation.from_rotvec([0, 0, 0.5]).as_quat(),
        atol=1e-7,
    )
    assert manifest["complete"] and len(manifest["episodes"]) == 2
    with pytest.raises(FileExistsError):
        convert_suite(root, output, task)


@pytest.mark.parametrize(
    "experiment", ["libero_oattok", "libero_oatpolicy", "libero_arc_policy"]
)
def test_complete_benchmark_recipes_compose(experiment):
    from hydra import compose, initialize_config_dir

    root = Path(__file__).parents[1] / "egomimic/hydra_configs"
    with initialize_config_dir(version_base=None, config_dir=str(root)):
        for suite in TASKS:
            cfg = compose(
                config_name="train_zarr_cartesian",
                overrides=[
                    f"+experiment=oat/{experiment}",
                    f"benchmark.suite={suite}",
                    "benchmark.dataset=/tmp/libero.zarr",
                ],
            )
            assert cfg.benchmark.horizon == 32 and cfg.benchmark.n_action_steps == 16
            assert cfg.trainer.max_epochs == 5001
            assert cfg.model._target_ == "egomimic.pl_utils.pl_model.ModelWrapper"
