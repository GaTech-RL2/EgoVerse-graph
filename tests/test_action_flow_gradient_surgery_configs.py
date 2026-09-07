import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_builder_emits_single_full_gradient_projection_trial(tmp_path):
    source = Path(__file__).parents[1]
    output = tmp_path / "configs"
    experiment = tmp_path / "experiment"
    training_dataset = tmp_path / "train.npz"
    evaluation_dataset = tmp_path / "eval.npz"
    training_dataset.write_bytes(b"frozen-training-data")
    evaluation_dataset.write_bytes(b"frozen-evaluation-data")

    subprocess.run(
        [
            sys.executable,
            str(
                source
                / "scripts/experiments/build_action_flow_gradient_surgery_configs.py"
            ),
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

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["datasets"]["training"]["sha256"] == hashlib.sha256(
        training_dataset.read_bytes()
    ).hexdigest()
    assert manifest["datasets"]["evaluation"]["sha256"] == hashlib.sha256(
        evaluation_dataset.read_bytes()
    ).hexdigest()
    assert len(manifest["configs"]) == 1
    config = json.loads(Path(manifest["configs"][0]).read_text())
    assert config["clean_gradient_mode"] == "full"
    assert (
        config["encoder_gradient_surgery"]
        == "protect_reconstruction_from_flow"
    )
    assert config["lambda_reconstruction"] == 1.0
    assert config["max_steps"] == 200_000
    assert config["evaluation_particles"] == 2048
