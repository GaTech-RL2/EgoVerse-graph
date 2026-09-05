import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_builder_emits_frozen_six_variant_three_seed_matrix(tmp_path):
    source = Path(__file__).parents[1]
    output = tmp_path / "config"
    experiment = tmp_path / "experiment"
    training_dataset = tmp_path / "train.npz"
    evaluation_dataset = tmp_path / "eval.npz"
    training_dataset.write_bytes(b"frozen-training-data")
    evaluation_dataset.write_bytes(b"frozen-evaluation-data")
    subprocess.run(
        [
            sys.executable,
            str(source / "scripts/experiments/build_action_flow_affine_configs.py"),
            "--output-dir",
            str(output),
            "--experiment-root",
            str(experiment),
            "--training-dataset",
            str(training_dataset),
            "--evaluation-dataset",
            str(evaluation_dataset),
        ],
        check=True,
    )
    configs = sorted(output.glob("gaussian-torus-*.json"))
    assert len(configs) == 18
    resolved = [json.loads(path.read_text()) for path in configs]
    assert {config["seed"] for config in resolved} == {42, 43, 44}
    assert {config["checkpoint_every"] for config in resolved} == {50_000}
    assert {config["evaluation_particles"] for config in resolved} == {1536}
    assert {config["evaluation_dataset"] for config in resolved} == {
        str(evaluation_dataset)
    }
    action_configs = [
        config for config in resolved if config["architecture"] == "action_adapter_flow"
    ]
    assert len(action_configs) == 15
    assert {config["model"]["latent_dim"] for config in action_configs} == {8}
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["primary_metric"] == "validation_generation_symmetric_nn_mse"
    assert manifest["pass_threshold"] is None
    assert manifest["datasets"]["training"] == {
        "path": str(training_dataset.resolve()),
        "sha256": hashlib.sha256(training_dataset.read_bytes()).hexdigest(),
    }
    assert manifest["datasets"]["evaluation"] == {
        "path": str(evaluation_dataset.resolve()),
        "sha256": hashlib.sha256(evaluation_dataset.read_bytes()).hexdigest(),
    }

    for index, config in enumerate(resolved):
        run = Path(config["output_dir"])
        run.mkdir(parents=True)
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "validation_generation_symmetric_nn_mse": float(index + 1),
                    "validation_generation_energy_distance": float(index + 2),
                }
            )
        )
    summary_json = tmp_path / "summary.json"
    summary_markdown = tmp_path / "summary.md"
    subprocess.run(
        [
            sys.executable,
            str(source / "scripts/experiments/summarize_action_flow_affine.py"),
            "--manifest",
            str(output / "manifest.json"),
            "--output-json",
            str(summary_json),
            "--output-markdown",
            str(summary_markdown),
        ],
        check=True,
    )
    aggregate = json.loads(summary_json.read_text())
    assert len(aggregate["rows"]) == 6
    assert "mean" in aggregate["rows"][0][manifest["primary_metric"]]
    assert "Symmetric NN MSE" in summary_markdown.read_text()


def test_builder_run_suffix_produces_distinct_retry_identities(tmp_path):
    source = Path(__file__).parents[1]
    training_dataset = tmp_path / "train.npz"
    evaluation_dataset = tmp_path / "eval.npz"
    training_dataset.write_bytes(b"train")
    evaluation_dataset.write_bytes(b"eval")
    output = tmp_path / "config"
    subprocess.run(
        [
            sys.executable,
            str(source / "scripts/experiments/build_action_flow_affine_configs.py"),
            "--output-dir",
            str(output),
            "--experiment-root",
            str(tmp_path / "experiment"),
            "--training-dataset",
            str(training_dataset),
            "--evaluation-dataset",
            str(evaluation_dataset),
            "--seeds",
            "42",
            "--run-suffix",
            "smoke-retry1",
        ],
        check=True,
    )
    config = json.loads(
        (output / "gaussian-torus-f-nonlinear-path-seed42-smoke-retry1.json").read_text()
    )
    assert config["wandb"]["id"].endswith("-smoke-retry1")
    assert config["wandb"]["dir"] == str(tmp_path / "experiment" / "wandb")
    assert config["output_dir"].endswith("-smoke-retry1")
