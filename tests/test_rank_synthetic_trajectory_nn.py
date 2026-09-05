import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "experiments"
    / "rank_synthetic_trajectory_nn.py"
)


def _matrix(tmp_path, *, variants=("a", "b"), seeds=(42, 43, 44), particles=16):
    rng = np.random.default_rng(7)
    targets = rng.normal(size=(particles, 3)).astype(np.float32)
    dataset = tmp_path / "eval.npz"
    np.savez_compressed(
        dataset,
        split=np.ones(particles, dtype=np.int64),
        target_3d=targets,
    )
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    trajectory_root = tmp_path / "trajectories"
    trajectory_root.mkdir()
    configs = []
    trajectories = {}
    for variant in variants:
        for seed in seeds:
            run_id = f"run-{variant}-{seed}"
            output = tmp_path / "runs" / run_id
            (output / "checkpoints").mkdir(parents=True)
            (output / "summary.json").write_text("{}\n")
            checkpoint = (
                output
                / "checkpoints"
                / "epoch-equivalent-000001-global-step-000010.pt"
            )
            checkpoint.write_bytes(f"checkpoint-{variant}-{seed}".encode())
            config = {
                "variant": variant,
                "seed": seed,
                "evaluation_dataset": str(dataset),
                "output_dir": str(output),
                "max_steps": 10,
                "wandb": {"id": run_id},
            }
            config_path = config_dir / f"{run_id}.json"
            config_path.write_text(json.dumps(config))
            configs.append(str(config_path))
            final = targets if variant == "a" else targets + 5.0
            points = np.stack([np.zeros_like(targets), final])
            trajectory = trajectory_root / f"{run_id}.npz"
            np.savez_compressed(
                trajectory,
                times=np.array([0.0, 1.0], dtype=np.float32),
                points=points,
                target=targets,
            )
            trajectories[run_id] = trajectory
    manifest = {
        "schema_version": 1,
        "variants": list(variants),
        "seeds": list(seeds),
        "datasets": {
            "evaluation": {
                "path": str(dataset),
                "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            }
        },
        "configs": configs,
    }
    manifest_path = config_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path, dataset, trajectory_root, trajectories, targets


def _command(manifest, output_json, output_markdown, particles):
    return [
        sys.executable,
        str(SCRIPT),
        "--manifest",
        str(manifest),
        "--expected-particles",
        str(particles),
        "--distance-chunk-size",
        "5",
        "--output-json",
        str(output_json),
        "--output-markdown",
        str(output_markdown),
    ]


def test_ranks_complete_matrix_and_records_exact_provenance(tmp_path):
    manifest, dataset, root, trajectories, _ = _matrix(tmp_path)
    output_json = tmp_path / "ranking.json"
    output_markdown = tmp_path / "ranking.md"
    subprocess.run(
        _command(manifest, output_json, output_markdown, 16)
        + ["--trajectory-root", str(root)],
        check=True,
    )

    result = json.loads(output_json.read_text())
    assert [row["variant"] for row in result["ranking"]] == ["a", "b"]
    assert result["ranking"][0]["mean"] == 0.0
    assert result["ranking"][0]["sample_std"] == 0.0
    assert [row["seed"] for row in result["ranking"][0]["per_seed"]] == [
        42,
        43,
        44,
    ]
    assert result["comparison"]["particles"] == 16
    assert result["comparison"]["dataset"]["sha256"] == hashlib.sha256(
        dataset.read_bytes()
    ).hexdigest()
    run = next(row for row in result["runs"] if row["run_id"] == "run-a-42")
    assert run["trajectory"]["sha256"] == hashlib.sha256(
        trajectories["run-a-42"].read_bytes()
    ).hexdigest()
    assert run["checkpoint"]["global_step"] == 10
    assert len(run["checkpoint"]["sha256"]) == 64
    markdown = output_markdown.read_text()
    assert "Symmetric NN MSE" in markdown
    assert "42: 0" in markdown


def test_accepts_only_a_complete_set_of_exact_npz_paths(tmp_path):
    manifest, _, _, trajectories, _ = _matrix(
        tmp_path, variants=("a",), seeds=(42, 43), particles=8
    )
    output_json = tmp_path / "ranking.json"
    output_markdown = tmp_path / "ranking.md"
    explicit = []
    for run_id, path in trajectories.items():
        explicit.extend(["--trajectory-npz", f"{run_id}={path}"])
    subprocess.run(
        _command(manifest, output_json, output_markdown, 8) + explicit,
        check=True,
    )
    assert len(json.loads(output_json.read_text())["runs"]) == 2

    incomplete = subprocess.run(
        _command(manifest, output_json, output_markdown, 8)
        + ["--trajectory-npz", f"run-a-42={trajectories['run-a-42']}"],
        text=True,
        capture_output=True,
    )
    assert incomplete.returncode != 0
    assert "explicit trajectory set does not match complete matrix" in incomplete.stderr


def test_rejects_target_or_time_grid_mismatch(tmp_path):
    manifest, _, root, trajectories, targets = _matrix(
        tmp_path, variants=("a",), seeds=(42, 43), particles=8
    )
    mismatched = trajectories["run-a-43"]
    np.savez_compressed(
        mismatched,
        times=np.array([0.0, 0.4, 1.0], dtype=np.float32),
        points=np.stack([targets, targets, targets]),
        target=targets,
    )
    result = subprocess.run(
        _command(manifest, tmp_path / "ranking.json", tmp_path / "ranking.md", 8)
        + ["--trajectory-root", str(root)],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "time grid differs" in result.stderr

    np.savez_compressed(
        mismatched,
        times=np.array([0.0, 1.0], dtype=np.float32),
        points=np.stack([targets, targets]),
        target=targets + 1.0,
    )
    result = subprocess.run(
        _command(manifest, tmp_path / "ranking.json", tmp_path / "ranking.md", 8)
        + ["--trajectory-root", str(root)],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "does not exactly match evaluation dataset" in result.stderr


def test_rejects_run_without_terminal_completion_artifacts(tmp_path):
    manifest, _, root, _, _ = _matrix(
        tmp_path, variants=("a",), seeds=(42,), particles=8
    )
    config = json.loads(next((tmp_path / "config").glob("run-*.json")).read_text())
    (Path(config["output_dir"]) / "summary.json").unlink()
    result = subprocess.run(
        _command(manifest, tmp_path / "ranking.json", tmp_path / "ranking.md", 8)
        + ["--trajectory-root", str(root)],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "run is incomplete because summary.json is missing" in result.stderr
