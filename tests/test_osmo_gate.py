"""Exercise the same five-update/save/strict-reload harness before OSMO allocation."""

import hashlib
import io
import json

import pytest
from omegaconf import OmegaConf, open_dict

from scripts.integration import run_gate
from scripts.integration.prepare_inputs import copy_stream, relative_path
from tests.fixtures.synthetic_episodes import write_episode
from tests.test_retained_recipe_steps import small_cpu_model


def test_input_staging_checks_hashes_and_never_replaces_existing_data(tmp_path):
    path = tmp_path / "verified.bin"
    value = b"immutable-input"
    sha = hashlib.sha256(value).hexdigest()
    receipt = copy_stream(
        io.BytesIO(value), path, expected_bytes=len(value), expected_sha256=sha
    )
    assert receipt["sha256"] == sha and path.read_bytes() == value
    with pytest.raises(FileExistsError):
        copy_stream(
            io.BytesIO(value), path, expected_bytes=len(value), expected_sha256=sha
        )
    assert path.read_bytes() == value
    with pytest.raises(ValueError, match="SHA-256"):
        copy_stream(
            io.BytesIO(b"bad"),
            tmp_path / "changed.bin",
            expected_bytes=3,
            expected_sha256=sha,
        )
    assert not (tmp_path / "changed.bin").exists()
    with pytest.raises(ValueError, match="relative"):
        relative_path("../../outside")


def harness_inputs(monkeypatch, tmp_path):
    inputs, output = tmp_path / "inputs", tmp_path / "result"
    output.mkdir()
    data = inputs / "data/eva"
    episodes = []
    for index in range(3):
        write_episode(data, "eva", T=8, H=32, W=32, seed=index)
        episodes.append(
            {
                "source": "eva",
                "split": "train" if index < 2 else "valid",
                "episode_hash": f"eva_{index:02}",
            }
        )
    (inputs / "data-manifest.json").write_text(json.dumps({"episodes": episodes}))
    (inputs / "input-receipt.json").write_text('{"scope":"synthetic CPU fixture"}')
    monkeypatch.setenv("SMOKE_INPUT_ROOT", str(inputs))
    matrix = OmegaConf.load(run_gate.ROOT / "scripts/integration/matrix.yaml")
    matrix.trainer.accelerator = "cpu"
    matrix.trainer.precision = "32-true"
    # Tiny 32px fixtures collapse ResNet's last feature map to 1x1; BatchNorm
    # still needs two samples. Real GPU cases retain their full image sizes.
    matrix.cases["hpt-eva"].batch_size = 2
    matrix_path = tmp_path / "matrix.yaml"
    OmegaConf.save(matrix, matrix_path)
    return inputs, output, matrix_path


@pytest.mark.parametrize("export", [True, False])
def test_wrong_target_frame_rejected_before_network_construction(
    monkeypatch, tmp_path, export
):
    import egomimic.trainHydra as entrypoint

    inputs, output, matrix_path = harness_inputs(monkeypatch, tmp_path)
    cfg, _, _ = run_gate.configured_case(matrix_path, "hpt-eva", inputs, output)
    cfg.inference_config.enabled = export
    for split in ("train_datasets", "valid_datasets"):
        ds = cfg.data[split].eva_bimanual
        with open_dict(ds):
            ds.bounds_check = False
            ds.resolver.transform_list.coord_frame = "eef_frame"

    def forbidden(*args, **kwargs):
        raise AssertionError("Network was constructed before frame preflight")

    monkeypatch.setattr(entrypoint, "_instantiate_model_wrapper", forbidden)
    with pytest.raises(ValueError, match="coord_frame.*camframe.*eef_frame"):
        entrypoint.train(cfg)
    assert not (output / "checkpoints/inference-config.yaml").exists()


def test_training_harness_runs_five_updates_validation_and_bound_reload(
    monkeypatch, tmp_path
):
    inputs, output, matrix_path = harness_inputs(monkeypatch, tmp_path)
    original = run_gate.configured_case

    def configured(*args, **kwargs):
        cfg, _, spec = original(*args, **kwargs)
        cfg = small_cpu_model(cfg, "hpt")
        with open_dict(cfg):
            cfg.model.pipeline.device = "cpu"
            for split in ("train_datasets", "valid_datasets"):
                cfg.data[split].eva_bimanual.bounds_check = False
        saved = OmegaConf.masked_copy(cfg, [key for key in cfg if key != "hydra"])
        saved = OmegaConf.create(OmegaConf.to_container(saved, resolve=True))
        return cfg, saved, spec

    monkeypatch.setattr(run_gate, "configured_case", configured)
    run_gate.training_gate(matrix_path, "hpt-eva", inputs, output)
    receipt = json.loads((output / "training-receipt.json").read_text())
    assert receipt["steps"] == 5 and len(receipt["gradients"]) == 5
    assert receipt["gpu"] is None
    assert (
        receipt["validation"]
        and receipt["initial_parameter_probe"] != receipt["final_parameter_probe"]
    )
    run_gate.verification_gate(output, device="cpu")
    verified = json.loads((output / "inference-receipt.json").read_text())
    assert verified["checkpoint_sha256"] == receipt["checkpoint_sha256"]
    assert [case["executed_shape"][1] for case in verified["cases"]] == [1, 2]
