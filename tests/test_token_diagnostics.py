"""Provider-scoped attention capture, source identities and reusable archives."""

from functools import wraps
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from egomimic.eval.eval import EvaluationDataRequirements
from egomimic.eval.latent_archive import (
    read_archive,
    rebuild_archives,
    reduce_features,
    write_archive,
)
from egomimic.eval.token_diagnostics import TokenDiagnosticsEval
from egomimic.models.pi05.diagnostics import PI05AttentionProvider, capture_attention


def attention_layers():
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.self_attn.k_proj = torch.nn.Linear(4, 4)
    return torch.nn.ModuleList([layer])


class AttentionBackend(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = torch.nn.Module()
        self.paligemma_with_expert.paligemma = torch.nn.Module()
        self.paligemma_with_expert.paligemma.language_model = torch.nn.Module()
        self.paligemma_with_expert.paligemma.language_model.layers = attention_layers()
        self.paligemma_with_expert.gemma_expert = torch.nn.Module()
        self.paligemma_with_expert.gemma_expert.model = torch.nn.Module()
        self.paligemma_with_expert.gemma_expert.model.layers = attention_layers()

    def embed_prefix(self, images, image_masks, language_tokens, language_masks):
        embeddings = torch.ones(len(language_tokens), 5, 4)
        padding = torch.cat(
            [torch.ones(len(language_tokens), 2, dtype=torch.bool), language_masks], 1
        )
        return embeddings, padding, torch.zeros_like(padding)

    def sample(self):
        prefix, _, _ = self.embed_prefix(
            None,
            None,
            torch.ones(2, 3),
            torch.tensor([[True, False, False], [True, True, False]]),
        )
        language = self.paligemma_with_expert.paligemma.language_model.layers[
            0
        ].self_attn.k_proj(prefix)
        expert = self.paligemma_with_expert.gemma_expert.model.layers[
            0
        ].self_attn.k_proj
        first = expert(torch.ones(2, 2, 4))
        expert(torch.ones(2, 2, 4) * 5)
        return language, first


def test_capture_slices_first_step_masks_and_cleanup():
    backend = AttentionBackend()
    with capture_attention(backend, mask_padding=True) as output:
        language, first = backend.sample()
    torch.testing.assert_close(output["expert_layer_00"]["tokens"], first)
    torch.testing.assert_close(
        output["paligemma_layer_00_img"]["tokens"], language[:, :2]
    )
    assert output["paligemma_layer_00_lang"]["mask"].sum() == 3
    assert "embed_prefix" not in backend.__dict__
    for module in backend.modules():
        assert not module._forward_hooks
    with pytest.raises(RuntimeError, match="sample failure"):
        with capture_attention(backend):
            raise RuntimeError("sample failure")
    assert "embed_prefix" not in backend.__dict__
    assert all(not module._forward_hooks for module in backend.modules())


def test_provider_uses_declared_id_and_normal_graph_forward():
    backend = AttentionBackend()
    seen = []
    stage = SimpleNamespace(backend=True, policy_nets={"policy": backend})

    def lookup(name):
        assert name == "explicit-policy-id"
        return stage

    def forward(batch):
        seen.append(next(iter(batch)))
        backend.sample()
        return {source: {"pred_action": torch.ones(2, 4, 3)} for source in batch}

    graph = SimpleNamespace(
        pipeline=SimpleNamespace(stage_by_id=lookup),
        forward_eval=forward,
        process_batch_for_training=lambda batch: batch,
    )
    provider = PI05AttentionProvider(stage_id="explicit-policy-id")
    result = provider.run(
        graph, {"source_a": {}, "source_b": {}}, already_processed=False
    )
    assert seen == ["source_a", "source_b"]
    assert result["source_b"]["predictions"]["pred_action"].shape == (2, 4, 3)
    assert len(result["source_a"]["activations"]) == 4


def test_capture_bypasses_compiled_sampler_and_restores_it():
    backend = AttentionBackend()

    @wraps(backend.sample)
    def compiled():
        raise AssertionError("Capture must use the original eager sampler")

    backend.sample_actions = compiled
    with capture_attention(backend) as output:
        backend.sample_actions()
    assert output and backend.sample_actions is compiled
    with pytest.raises(RuntimeError, match="inference failed"):
        with capture_attention(backend):
            raise RuntimeError("inference failed")
    assert backend.sample_actions is compiled


def test_prediction_requirements_are_preserved_and_conflicts_rejected():
    prediction = SimpleNamespace(
        data_requirements=lambda: EvaluationDataRequirements(
            ordered=True,
            complete_episodes=True,
            source_fps=20,
            max_episodes=2,
            sample_id_key="episode_hash",
            frame_index_key="frame_index",
        ),
        trainer_overrides=lambda: {"inference_mode": False},
    )
    evaluator = TokenDiagnosticsEval(
        prediction_evaluator=prediction, limit_val_batches=1.0
    )
    assert evaluator.data_requirements() == prediction.data_requirements()
    assert evaluator.trainer_overrides() == {
        "inference_mode": False,
        "limit_val_batches": 1.0,
        "num_sanity_val_steps": 0,
    }
    evaluator.sample_id_key = "different_episode_id"
    with pytest.raises(ValueError, match="sample_id_key"):
        evaluator.data_requirements()


def test_diagnostics_score_the_captured_prediction_through_shared_metrics(tmp_path):
    from egomimic.eval.bimanual_cartesian_eval import BimanualCartesianEval

    prediction_evaluator = BimanualCartesianEval(
        viz_every_n_epochs=0, pose_metrics=False
    )
    evaluator = TokenDiagnosticsEval(
        prediction_evaluator=prediction_evaluator, methods=["pca"], save_plots=False
    )
    logged, calls = [], []
    evaluator.trainer = SimpleNamespace(
        default_root_dir=str(tmp_path),
        current_epoch=0,
        global_step=1,
        global_rank=0,
        is_global_zero=True,
        world_size=1,
        lightning_module=SimpleNamespace(
            log_dict=lambda values, **kw: logged.append(values)
        ),
    )
    evaluator.bind_data_context(
        normalizer=SimpleNamespace(
            unnormalize=lambda values, identity: {k: v * 2 for k, v in values.items()}
        )
    )
    batch = {
        "opaque": {
            "embodiment": torch.tensor([7]),
            "episode_hash": ["episode"],
            "frame_index": torch.tensor([0]),
            "actions_cartesian": torch.zeros(1, 4, 14),
        }
    }

    def run(capability, values):
        calls.append(capability)
        return {
            "opaque": {
                "predictions": {"pred_action": torch.ones(1, 4, 14)},
                "activations": {
                    "layer": {
                        "tokens": torch.eye(4)[None],
                        "mask": torch.ones(1, 4, dtype=torch.bool),
                    }
                },
            }
        }

    evaluator.model = SimpleNamespace(run_diagnostic=run)
    evaluator.on_validation_start()
    metrics = evaluator.on_validation_step(batch, 0)
    evaluator.on_validation_end()
    assert calls == ["token_activations"]
    assert metrics["Valid/MSE"] == 1
    assert metrics["Valid/Native_MSE"] == 4
    assert len(logged) == 1


def test_archives_preserve_frames_balance_sources_and_rebuild(tmp_path):
    evaluator = TokenDiagnosticsEval(
        methods=["pca"], max_tokens_per_source_layer=6, save_plots=True
    )
    evaluator.trainer = SimpleNamespace(
        default_root_dir=str(tmp_path), current_epoch=0, global_step=10, global_rank=1
    )
    batch = {
        name: {
            "episode_hash": [f"{name}-episode"] * 2,
            "frame_index": torch.tensor([4, 17]),
        }
        for name in ["A", "B"]
    }

    def diagnostics(capability, values):
        assert capability == "token_activations"
        return {
            name: {
                "activations": {
                    "layer_00": {
                        "tokens": torch.arange(32).reshape(2, 4, 4).float(),
                        "mask": torch.ones(2, 4, dtype=torch.bool),
                    }
                }
            }
            for name in values
        }

    evaluator.model = SimpleNamespace(run_diagnostic=diagnostics)
    evaluator.on_validation_start()
    evaluator.on_validation_step(batch, 0)
    evaluator.on_validation_step(
        batch, 1
    )  # repeated distributed/combined-loader samples
    evaluator.on_validation_end()
    path = next(evaluator.output_dir.rglob("*.csv"))
    keys, rows = read_archive(path)
    assert keys.shape == (12, 4)
    assert {row["frame_idx"] for row in rows} == {"4", "17"}
    assert sum(row["embodiment"] == "A" for row in rows) == 6
    assert sum(row["embodiment"] == "B" for row in rows) == 6
    assert (
        len(
            {
                tuple(row[key] for key in ("video_hash", "frame_idx", "token_idx"))
                for row in rows
            }
        )
        == 12
    )
    rebuilt = tmp_path / "rebuild"
    rebuild_archives(path.parent, rebuilt, methods=["pca"], seed=0)
    restored, metadata = read_archive(rebuilt / path.name)
    np.testing.assert_array_equal(restored, keys)
    assert metadata == rows
    assert list(rebuilt.glob("*.png"))
    with pytest.raises(FileExistsError):
        rebuild_archives(path.parent, rebuilt, methods=["pca"])
    with pytest.raises(ValueError, match="differ"):
        rebuild_archives(path.parent, path.parent, methods=["pca"])


def test_all_reduction_methods_are_finite_and_reproducible():
    pytest.importorskip("umap")
    features = np.random.default_rng(42).normal(size=(16, 8)).astype(np.float32)
    methods = ["pca", "umap", "pca_umap", "tsne2d", "tsne3d"]
    result, receipt = reduce_features(features, methods=methods, pca_components=5)
    assert set(result) == set(methods)
    assert len(receipt["pca_explained_variance"]) == 5
    for name, coordinates in result.items():
        assert coordinates.shape == (16, 2 if name == "tsne2d" else 3)
        assert np.isfinite(coordinates).all()


def test_old_inline_csv_rebuild_and_bad_row_count(tmp_path):
    source = tmp_path / "old.csv"
    source.write_text(
        "video_hash,embodiment,frame_idx,token_idx,k0,k1\nhash,A,7,0,1.0,2.0\n"
    )
    keys, rows = read_archive(source)
    np.testing.assert_array_equal(keys, [[1, 2]])
    assert rows[0]["frame_idx"] == "7"
    with pytest.raises(ValueError, match="row counts"):
        write_archive(tmp_path / "bad", "layer", keys, [], reductions={}, receipt={})
