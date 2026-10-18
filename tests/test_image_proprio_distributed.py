"""CPU Gloo proves distributed metric weighting and validation completion."""

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

from egomimic.eval.image_proprio_eval import ImageProprioEval
from egomimic.eval.pixel_diagnostics import PixelPredictionDiagnostics
from tests.test_pixel_prediction_diagnostics import batch_fixture, fake_model


def _distributed_metrics_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        logged = {}
        model = SimpleNamespace(
            device=torch.device("cpu"), global_step=4, global_rank=rank,
            trainer=SimpleNamespace(sanity_checking=False),
            log=lambda key, value, **kwargs: logged.update({key: float(value)}),
        )
        evaluator = ImageProprioEval(objective="visual_only")
        evaluator.model = model

        def prepare():
            evaluator.pixel_diagnostics = None
            evaluator.on_validation_start()
            evaluator._validation_batches = (60, 40)[rank]
            # An unweighted average of rank means is wrong: (2 + 6)/2 != 5.
            evaluator.totals = {
                "forecast": (torch.tensor((2.0, 18.0)[rank]), torch.tensor((1.0, 3.0)[rank]))
            }
            if rank == 1:
                evaluator.totals["completion"] = (torch.tensor(10.0), torch.tensor(2.0))

        prepare()
        evaluator.log_epoch_metrics()
        evaluator.on_validation_end()
        assert logged == {"forecast": 5.0, "completion": 5.0}
        assert evaluator.global_validation_batches == 100
        assert evaluator._validation_batches == (60, 40)[rank]
        assert evaluator.has_completed_validation(4)
        # Repeated metric publication must not multiply already-reduced totals.
        evaluator.log_epoch_metrics()
        assert evaluator.global_validation_batches == 100
        assert logged == {"forecast": 5.0, "completion": 5.0}

        prepare()
        evaluator.log_epoch_metrics()

        def publish(_):
            if rank == 0:
                raise RuntimeError("rank-zero artifact publication failed")

        evaluator.pixel_diagnostics = SimpleNamespace(log_to_wandb=publish)
        try:
            evaluator.on_validation_end()
        except RuntimeError:
            assert rank == 0
        assert not evaluator.has_completed_validation(4)

        prepare()
        if rank == 0:
            # A valid global sum must not hide an invalid local contribution.
            evaluator.totals["forecast"] = (torch.tensor(-2.0), torch.tensor(1.0))
        with pytest.raises(ValueError, match="distributed rank"):
            evaluator.log_epoch_metrics()
        evaluator.on_validation_end()
        assert not evaluator.has_completed_validation(4)
        Path(output, f"rank-{rank}.json").write_text(json.dumps({"status": "PASS"}))
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="Gloo unavailable")
def test_distributed_metrics_reduce_sums_and_global_completion(tmp_path):
    mp.spawn(_distributed_metrics_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)),
             nprocs=2, join=True)
    for rank in range(2):
        assert json.loads((tmp_path / f"rank-{rank}.json").read_text()) == {"status": "PASS"}


def test_nonzero_rank_does_not_publish_rgb_artifacts(tmp_path):
    batch, _ = batch_fixture()
    model, prediction, _ = fake_model(batch)
    model.global_rank = 1
    diagnostics = PixelPredictionDiagnostics(artifact_dir=tmp_path, log_wandb=False)
    diagnostics.record(model=model, source="robot", item=batch["robot"],
                       prediction=prediction, objective="visual_only", batch_idx=0, dataloader_idx=0)
    assert diagnostics.written == [] and not list(tmp_path.iterdir())


def test_global_validation_identity_preserves_sampling_noise_across_shards():
    batch, _ = batch_fixture()
    model, prediction, _ = fake_model(batch)
    draws = []

    def forward_eval(_):
        draws.append(torch.rand(4))
        return {"robot": {"image_proprio_prediction": prediction}}

    model.forward_eval = forward_eval
    evaluator = ImageProprioEval(objective="visual_only")
    evaluator.model = model
    evaluator.on_validation_start()
    before = torch.get_rng_state().clone()
    evaluator.on_validation_step(batch, 29)
    batch["robot"]["validation_batch_index"] = [29]
    evaluator.on_validation_step(batch, 4)
    assert torch.equal(draws[0], draws[1])
    assert torch.equal(torch.get_rng_state(), before)
    batch["robot"]["validation_batch_index"] = [30]
    evaluator.on_validation_step(batch, 4)
    assert not torch.equal(draws[0], draws[2])


@pytest.mark.parametrize("invalid", [[], [-1], [True], [1.0], [1, 2]])
def test_invalid_global_validation_identity_is_rejected(invalid):
    batch, _ = batch_fixture()
    batch["robot"]["validation_batch_index"] = invalid
    with pytest.raises(ValueError, match="validation batch identit"):
        ImageProprioEval._sampling_index(batch, 0)
