from types import SimpleNamespace

import pytest
import torch

from egomimic.eval.bimanual_cartesian_eval import BimanualCartesianEval
from egomimic.rldb.zarr import zarr_dataset_multi as data


def test_camera_mse_preserves_source_semantics_at_angle_wrap():
    from hydra import compose, initialize_config_dir

    from egomimic.eval.cartesian_metrics import cartesian_metrics
    from tests.test_retained_recipe_steps import CONFIGS

    target = torch.zeros(1, 2, 14)
    prediction = target.clone()
    target[..., 3] = torch.pi - 0.01
    prediction[..., 3] = -torch.pi + 0.01
    for recipe in ("eval_hpt", "eval_pi"):
        with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
            cfg = compose(
                config_name="train_zarr_cartesian", overrides=[f"evaluator={recipe}"]
            )
        assert cfg.evaluator.camera_mse_wrap is False
        values = cartesian_metrics(
            prediction,
            target,
            distribution=False,
            wrap_angles=cfg.evaluator.camera_mse_wrap,
        )
        torch.testing.assert_close(
            values["paired_mse_avg"], (prediction - target).square().mean()
        )
    wrapped = cartesian_metrics(
        prediction, target, distribution=False, wrap_angles=True
    )
    assert wrapped["paired_mse_avg"].item() == pytest.approx(0.02**2 / 14, rel=1e-4)


def test_vendor_metadata_is_canonicalized_without_editing_episode(monkeypatch):
    metadata = {"embodiment": "MECKA_BIMANUAL", "total_frames": 3}
    reader = SimpleNamespace(metadata=metadata, _collect_keys=lambda: [])
    monkeypatch.setattr(data, "ZarrEpisode", lambda path: reader)
    episode = data.ZarrDataset.__new__(data.ZarrDataset)
    episode.episode_path = "/unused"
    episode.init_episode()
    assert episode.embodiment == "human_bimanual"
    assert metadata["embodiment"] == "MECKA_BIMANUAL"


def test_pose_and_sample_metrics_need_only_graph_prediction_contract(tmp_path):
    class Graph:
        def forward_eval(self, batch):
            return {
                source: {"pred_action": torch.randn_like(row["actions_cartesian"])}
                for source, row in batch.items()
            }

    class Normalizer:
        def unnormalize(self, values, embodiment):
            return {key: value * 2 for key, value in values.items()}

    logged = []
    evaluator = BimanualCartesianEval(
        pose_metrics=True,
        rkl_samples=3,
        viz_every_n_epochs=0,
        group_options={"train_viz": {"rkl_samples": 1, "limit_batches": 1}},
    )
    evaluator.trainer = SimpleNamespace(
        default_root_dir=str(tmp_path),
        current_epoch=0,
        is_global_zero=True,
        lightning_module=SimpleNamespace(
            log_dict=lambda metrics, **kw: logged.append(metrics)
        ),
    )
    evaluator.model = Graph()
    evaluator.bind_data_context(normalizer=Normalizer())
    batch = {
        "opaque-source": {
            "embodiment": torch.tensor([7]),
            "actions_cartesian": torch.zeros(1, 4, 14),
            "observations.state.ee_pose": torch.zeros(1, 14),
        }
    }
    values = evaluator.on_validation_step(batch, 0)
    assert values["Valid/Native_MSE"] == 4 * values["Valid/MSE"]
    assert values["Valid/yam_bimanual_actions_cartesian_sample_diversity_M3"] > 0
    assert all(torch.isfinite(v) for v in values.values())
    evaluator.set_validation_group("train_viz")
    values = evaluator.on_validation_step(batch, 0)
    assert all(k.startswith("Valid_train_viz/") for k in values)
    assert not any("reverse_kl" in k for k in values)
    assert evaluator.on_validation_step(batch, 1) == {}
    assert len(logged) == 2
