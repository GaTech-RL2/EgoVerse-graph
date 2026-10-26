"""Local CPU evaluator fixtures; no real sampler, trainer, GPU or data access."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


from egomimic.eval.image_proprio_eval import ImageProprioEval


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def batch_fixture():
    coordinate = (torch.arange(10, dtype=torch.float32) + 0.5) / 5 - 1
    y, x = torch.meshgrid(coordinate, coordinate, indexing="ij")
    image = dict(
        value=torch.zeros(1, 100, 768),
        time=torch.zeros(1, 100),
        index=torch.zeros(1, 100),
        camera=torch.zeros(1, 100, dtype=torch.long),
        xy=torch.stack((x, y), -1).reshape(1, 100, 2),
        valid=torch.ones(1, 100, dtype=torch.bool),
        target_valid=torch.ones(1, 100, dtype=torch.bool),
        observed=torch.zeros(1, 100, dtype=torch.bool),
    )
    metadata = dict(
        episode_id="synthetic-image-0",
        image_size=[160, 160],
        patch_size=16,
        camera_names=["image"],
        frame_indices=[0],
        timestamps_seconds=[0.0],
        time_origin_seconds=0.0,
        normalization="uint8_to_minus_one_one",
        patch_order="row_major_hwc",
    )
    return {"imagenet": dict(
        world={"image": image}, embodiment="imagenet", task_kind=["image_generation"],
        language_features=torch.zeros(1, 1024), pixel_metadata=[metadata],
        validation_batch_index=[17],
    )}


def evaluator_fixture(tmp_path=None, *, objective="image_generation", transform=None):
    batch = batch_fixture()
    logged, outputs, steps = {}, [], []

    def forward_eval(item):
        # A small fake twenty-iteration stochastic output exercises evaluator
        # seeding/lifecycle only. This is not a real DDIM or GPU restore proof.
        current = torch.randn_like(item["imagenet"]["world"]["image"]["value"])
        for level in range(20, 0, -1):
            current = 0.5 * current + 0.125
            steps.append(level)
        prediction = {"image": current}
        if transform is not None:
            prediction = transform(prediction)
        result = {"imagenet": {"image_proprio_prediction": prediction}}
        outputs.append(result)
        return result

    model = SimpleNamespace(
        device=torch.device("cpu"), global_step=7, global_rank=0, current_epoch=1,
        trainer=SimpleNamespace(loggers=[], sanity_checking=False),
        forward_eval=forward_eval, log=lambda key, value, **_: logged.update({key: float(value)}),
    )
    diagnostics = None if tmp_path is None else dict(
        artifact_dir=str(tmp_path), write_video=False, log_wandb=False,
        contact_sheet_frames=1,
    )
    evaluator = ImageProprioEval(objective=objective, pixel_diagnostics=diagnostics)
    evaluator.model = model
    evaluator.on_validation_start()
    return evaluator, batch, model, logged, outputs, steps


def finish(evaluator, batch):
    result = evaluator.on_validation_step(batch, 0)
    evaluator.log_epoch_metrics()
    evaluator.on_validation_end()
    return result


def test_generation_namespace_all_pixel_mse_and_cpu_lifecycle():
    evaluator, batch, model, logged, outputs, steps = evaluator_fixture()
    output = finish(evaluator, batch)
    image = output["imagenet"]["image_proprio_prediction"]["image"]
    assert set(logged) == {"val/imagenet/image_generation/image_normalized_mse"}
    assert logged[next(iter(logged))] == pytest.approx(float(image.square().mean()))
    assert torch.isfinite(image).all() and steps == list(range(20, 0, -1))
    assert evaluator.has_completed_validation(model.global_step)
    assert not evaluator.has_completed_validation(model.global_step + 1)
    assert evaluator.global_validation_batches == 1
    assert evaluator.data_requirements().required_keys == (
        "world", "embodiment", "task_kind", "pixel_metadata", "language_features"
    )


def test_fully_noisy_fixture_rng_is_isolated_and_reference_is_not_conditioning():
    evaluator, batch, _, _, outputs, _ = evaluator_fixture()
    before = torch.random.get_rng_state().clone()
    evaluator.on_validation_step(batch, 0)
    assert torch.equal(before, torch.random.get_rng_state())
    first = outputs[-1]["imagenet"]["image_proprio_prediction"]["image"].clone()
    batch["imagenet"]["world"]["image"]["value"].fill_(0.5)
    evaluator.on_validation_step(batch, 0)
    assert torch.equal(before, torch.random.get_rng_state())
    assert torch.equal(first, outputs[-1]["imagenet"]["image_proprio_prediction"]["image"])


def test_generation_single_camera_png_and_semantic_metadata(tmp_path):
    evaluator, batch, _, logged, _, _ = evaluator_fixture(tmp_path)
    finish(evaluator, batch)
    png = list(tmp_path.rglob("camera-0.png"))
    reports = list(tmp_path.rglob("diagnostic.json"))
    assert len(png) == len(reports) == 1 and not list(tmp_path.rglob("*.mp4"))
    report = json.loads(reports[0].read_text())
    assert report["objective"] == report["task_kind"] == "image_generation"
    assert report["pixel_metadata"]["timestamps_seconds"] == [0.0]
    frame = report["cameras"]["0"]["frames"][0]
    assert frame["observation"] == "query" and frame["frame_index"] == 0
    assert frame["query_pixel_mse_rgb01"] == pytest.approx(
        logged["val/imagenet/image_generation/image_normalized_mse"] / 4
    )


@pytest.mark.parametrize("field,value", [
    ("image_size", [144, 160]), ("patch_size", 8),
    ("timestamps_seconds", [1.0]), ("frame_indices", [1]),
    ("time_origin_seconds", 1.0), ("camera_names", ["a", "b"]),
    ("camera_names", "a"),
    ("normalization", "other"), ("patch_order", "other"),
])
def test_generation_rejects_false_diagnostics_contract(field, value):
    evaluator, batch, model, _, outputs, _ = evaluator_fixture()
    batch["imagenet"]["pixel_metadata"][0][field] = value
    with pytest.raises(ValueError):
        evaluator.on_validation_step(batch, 0)
    assert not outputs and not evaluator.has_completed_validation(model.global_step)


@pytest.mark.parametrize("field,value", [
    ("time", 0.1), ("index", 1), ("camera", 1),
    ("valid", False), ("target_valid", False), ("observed", True),
    ("value", float("nan")), ("value", float("inf")), ("value", 2.0),
])
def test_generation_rejects_context_partial_targets_or_nonfinite_inputs(field, value):
    evaluator, batch, _, _, outputs, _ = evaluator_fixture()
    batch["imagenet"]["world"]["image"][field].reshape(-1)[0] = value
    with pytest.raises(ValueError):
        evaluator.on_validation_step(batch, 0)
    assert not outputs


@pytest.mark.parametrize("case", ["world", "kind", "language_shape", "language_nan", "grid", "patch_count", "missing_metadata"])
def test_generation_requires_exact_image_and_language_contract(case):
    evaluator, batch, _, _, outputs, _ = evaluator_fixture()
    item = batch["imagenet"]
    if case == "world":
        item["world"]["proprio"] = copy.deepcopy(item["world"]["image"])
    elif case == "kind":
        item["task_kind"] = ["forecast"]
    elif case == "language_shape":
        item["language_features"] = torch.zeros(1, 1023)
    elif case == "language_nan":
        item["language_features"][0, 0] = float("nan")
    elif case == "grid":
        item["world"]["image"]["xy"][0, 0] = 0
    elif case == "patch_count":
        item["world"]["image"]["value"] = torch.zeros(1, 99, 768)
    else:
        item["pixel_metadata"] = []
    with pytest.raises(ValueError):
        evaluator.on_validation_step(batch, 0)
    assert not outputs


@pytest.mark.parametrize("case", ["nan", "inf", "extra_stream", "shape", "overflow"])
def test_every_predicted_pixel_is_checked_before_completion(case):
    def transform(prediction):
        if case in ("nan", "inf", "overflow"):
            prediction["image"][0, -1, -1] = {
                "nan": float("nan"), "inf": float("inf"), "overflow": 1e30
            }[case]
        elif case == "extra_stream":
            prediction["proprio"] = torch.zeros(1, 1, 1)
        else:
            prediction["image"] = prediction["image"][:, :-1]
        return prediction

    evaluator, batch, model, _, _, _ = evaluator_fixture(transform=transform)
    with pytest.raises(ValueError):
        evaluator.on_validation_step(batch, 0)
    evaluator.log_epoch_metrics()
    evaluator.on_validation_end()
    assert not evaluator.has_completed_validation(model.global_step)


def test_sanity_validation_never_certifies_completion():
    evaluator, batch, model, _, _, _ = evaluator_fixture()
    model.trainer.sanity_checking = True
    evaluator.on_validation_start()
    finish(evaluator, batch)
    assert not evaluator.has_completed_validation(model.global_step)


def test_artifact_publication_failure_never_certifies_completion(tmp_path, monkeypatch):
    evaluator, batch, model, _, _, _ = evaluator_fixture(tmp_path)
    evaluator.on_validation_step(batch, 0)
    evaluator.log_epoch_metrics()

    def fail(_):
        raise RuntimeError("publication failed")

    monkeypatch.setattr(evaluator.pixel_diagnostics, "log_to_wandb", fail)
    with pytest.raises(RuntimeError, match="publication failed"):
        evaluator.on_validation_end()
    assert not evaluator.has_completed_validation(model.global_step)


def test_empty_generation_validation_never_certifies_completion():
    evaluator, _, model, _, _, _ = evaluator_fixture()
    evaluator.log_epoch_metrics()
    evaluator.on_validation_end()
    assert not evaluator.has_completed_validation(model.global_step)


def test_generation_cannot_enable_multimodal_evaluator_semantics():
    with pytest.raises(ValueError, match="exactly the image"):
        ImageProprioEval(objective="image_generation", active_streams_from_data=True)


def test_legacy_visual_only_forecast_metrics_still_work_without_language_metadata():
    evaluator, batch, model, logged, _, _ = evaluator_fixture(objective="visual_only")
    item = batch["imagenet"]
    item["task_kind"] = ["forecast"]
    del item["pixel_metadata"], item["language_features"]
    finish(evaluator, batch)
    assert set(logged) == {"val/imagenet/forecast/image_normalized_mse"}
    assert evaluator.has_completed_validation(model.global_step)
    assert evaluator.data_requirements().required_keys == ("world", "embodiment", "task_kind")
