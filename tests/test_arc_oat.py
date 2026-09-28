"""Learn OAT codes of ARC supports, retaining the dense action interface."""

from copy import deepcopy

import pytest
import torch

from egomimic.models.oat.checkpoint import validate_input_representation
from egomimic.models.oat.factory import make_tokenizer
from egomimic.pipeline.core import Pipeline
from egomimic.pipeline.stages_libero_arc import LiberoArcStage
from egomimic.pipeline.stages_oat import OATTokenizerStage
from tests.test_libero_benchmark import make_replay
from tests.test_oat_training import config_for


def tokenizer_graph(mode):
    settings = dict(
        arc_mode=mode, num_waypoints=6, horizon=8, velocity_norm_bound=3**0.5
    )
    tokenizer = make_tokenizer(
        action_dim=12,
        horizon=6,
        num_registers=4,
        emb_dim=32,
        head_dim=8,
        encoder_depth=1,
        decoder_depth=1,
    )
    return Pipeline(
        [
            LiberoArcStage(**settings, operation="encode", encode_inference=True),
            OATTokenizerStage(
                tokenizer, action_key="target", prediction_key="pred_arc"
            ),
            LiberoArcStage(**settings, operation="decode"),
        ]
    )


@pytest.mark.parametrize("mode", ["stk", "dur"])
def test_tokenizer_loss_is_on_arc_supports_and_prefixes_decode_dense_actions(mode):
    graph = tokenizer_graph(mode)
    actions = torch.randn(3, 8, 7, requires_grad=True) * 0.15
    # Include stationary motion and a gripper-only transition.
    actions = actions.detach()
    actions[0] = 0
    actions[1, :, :6] = 0
    actions[1, 4:, 6] = -1
    target = graph.stages[0].execute({"actions": actions}, mode="train")["target"]
    assert target.shape == (3, 6, 12)
    torch.manual_seed(42)
    actual = graph.execute({"actions": actions}, mode="train")[
        "loss/oat_reconstruction"
    ]
    torch.manual_seed(42)
    expected = graph.stages[1].tokenizer({"action": target})
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in graph.parameters()
    )
    graph.eval()
    for keep in (1, 2, 4):
        graph.stages[1].use_k_tokens = keep
        result = graph.execute({"actions": actions}, mode="inference")
        assert result["pred_arc"].shape == (3, 6, 12)
        assert result["pred_action"].shape == (3, 8, 7)
        assert torch.isfinite(result["pred_action"]).all()
    representation = validate_input_representation(graph.stages)
    assert representation["mode"] == mode
    assert representation["num_waypoints"] == 6
    # Same tensor dimensions must not allow a different physical representation.
    wrong = deepcopy(representation)
    wrong["mode"] = "dur" if mode == "stk" else "stk"
    with pytest.raises(ValueError, match="input representations differ"):
        validate_input_representation(graph.stages, {"oat_input_representation": wrong})
    with pytest.raises(ValueError, match="input representations differ"):
        validate_input_representation(graph.stages, {})
    graph.stages[-1].codec.max_translation = 0.4
    with pytest.raises(ValueError, match="encoder and decoder representations differ"):
        validate_input_representation(graph.stages)


@pytest.mark.parametrize("mode", ["stk", "dur"])
@pytest.mark.parametrize("precision", ["32-true", "bf16-mixed"])
def test_arc_tokenizer_then_policy_train_resume_and_self_contained_inference(
    tmp_path, mode, precision
):
    from egomimic.benchmarks.libero.rollout import load_policy
    from egomimic.models.oat.factory import load_tokenizer
    from egomimic.trainHydra import train

    replay = tmp_path / "replay.zarr"
    make_replay(replay)
    cfg = config_for("libero_arc_oattok", replay, tmp_path / "tokenizer")
    cfg.benchmark.arc_mode = mode
    cfg.benchmark.arc_waypoints = 6
    cfg.trainer.precision = precision
    spec = cfg.model.pipeline.stages[1].tokenizer
    spec.emb_dim, spec.head_dim = 32, 8
    spec.encoder_depth, spec.decoder_depth, spec.num_registers = 1, 1, 4
    metrics, objects = train(cfg)
    assert objects["trainer"].global_step == 2
    assert torch.isfinite(metrics["Valid/reconst_mse"])
    checkpoint = tmp_path / "tokenizer/checkpoints/last.ckpt"
    tokenizer = load_tokenizer(checkpoint)
    assert tokenizer.decoder.sample_dim == 12
    assert tokenizer._training_input_representation["mode"] == mode
    assert all(not p.requires_grad for p in tokenizer.parameters())
    if precision == "32-true":
        from argparse import Namespace
        from egomimic.benchmarks.libero.cli import reconstruction

        reconstruction_result = reconstruction(
            Namespace(
                checkpoint=checkpoint,
                online=False,
                device="cpu",
                suite="libero10",
                dataset=replay,
                tokens=[1, 2, 4],
                arc_modes=[mode],
                waypoints=[6],
                batch_size=8,
                limit=None,
            )
        )
        assert reconstruction_result["complete_validation"]
        assert set(reconstruction_result["metrics"]) == {
            "arc_only",
            "arc_oat_k1",
            "arc_oat_k2",
            "arc_oat_k4",
        }
        assert (
            reconstruction_result["representation_sizes"]["arc_only"]["float32_scalars"]
            == 6 * 12
        )

    policy_cfg = config_for("libero_arc_oatpolicy", replay, tmp_path / "policy")
    policy_cfg.benchmark.arc_mode = mode
    policy_cfg.benchmark.arc_waypoints = 6
    policy_cfg.benchmark.tokenizer_checkpoint = str(checkpoint)
    policy_cfg.trainer.precision = precision
    policy_spec = policy_cfg.model.pipeline.stages[1].policy
    policy_spec.embed_dim, policy_spec.n_layers, policy_spec.n_heads = 16, 1, 2
    policy_spec.temperature = 0
    _, trained = train(policy_cfg)
    assert trained["trainer"].global_step == 2
    model = trained["model"].model.pipeline.stages[1].policy
    assert model.action_dim == 12
    assert all(p.grad is None for p in model.action_tokenizer.parameters())
    policy_checkpoint = tmp_path / "policy/checkpoints/last.ckpt"
    policy_cfg.ckpt_path = str(policy_checkpoint)
    policy_cfg.trainer.max_epochs = 2
    _, resumed = train(policy_cfg)
    assert resumed["trainer"].global_step == 4
    payload = torch.load(policy_checkpoint, map_location="cpu", weights_only=False)
    assert payload["ema_num_updates"] == 4
    assert payload["oat_input_representation"]["mode"] == mode
    assert payload["oat_tokenizer_config"]["action_dim"] == 12

    # Loading must work after the separately trained tokenizer is unavailable.
    checkpoint.unlink()
    restored, protocol = load_policy(policy_checkpoint, device="cpu", use_k_tokens=2)
    assert protocol["action_representation"] == "arc_oat"
    values = trained["datamodule"].train_datasets["libero_panda"][0]
    values.pop("actions")
    batch = {k: v.unsqueeze(0) for k, v in values.items() if torch.is_tensor(v)}
    result = restored.algo.forward_eval({"libero_panda": batch})["libero_panda"]
    assert result["pred_action"].shape == (1, 8, 7)
    assert result["pred_arc"].shape == (1, 6, 12)
    assert torch.isfinite(result["pred_action"]).all()

    damaged = deepcopy(payload)
    damaged["oat_input_representation"]["max_translation"] = 0.4
    changed = tmp_path / "wrong-representation.ckpt"
    torch.save(damaged, changed)
    with pytest.raises(ValueError, match="input representations differ"):
        load_policy(changed, device="cpu")


@pytest.mark.parametrize("method", ["tokenizer", "oat"])
def test_hybrid_cluster_arguments_preserve_global_training_budget(method, tmp_path):
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    from egomimic.benchmarks.libero.cluster import training_arguments

    args = training_arguments(
        method,
        "libero_spatial",
        tmp_path / "data",
        tmp_path,
        "full",
        5001,
        gpus=4,
        oat_on_arc=True,
    )
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).parents[1] / "egomimic/hydra_configs"),
    ):
        cfg = compose(config_name="train_zarr_cartesian", overrides=args[3:])
    assert cfg.model.benchmark_protocol.action_representation == "arc_oat"
    assert cfg.trainer.max_epochs == 5001
    assert (
        cfg.trainer.devices
        * cfg.benchmark.batch_size
        * cfg.trainer.accumulate_grad_batches
        == 1024
    )
    assert cfg.model.pipeline.stages[1].action_key == "target"


def test_policy_rejects_tokenizer_from_another_arc_clock():
    from egomimic.models.oat.policy.oatpolicy import OATPolicy
    from egomimic.pipeline.stages_oat import OATPolicyStage
    from tests.test_oat_native import TinyObservation

    source = tokenizer_graph("stk")
    tokenizer = source.stages[1].tokenizer
    tokenizer._training_input_representation = validate_input_representation(
        source.stages
    )
    policy = OATPolicy(
        {"action": {"shape": [12]}, "obs": {"state": {"shape": [3], "type": "state"}}},
        TinyObservation(),
        tokenizer,
        4,
        2,
        16,
        1,
        2,
    )
    settings = dict(
        arc_mode="dur", num_waypoints=6, horizon=8, velocity_norm_bound=3**0.5
    )
    stages = [
        LiberoArcStage(**settings),
        OATPolicyStage(policy, action_key="target", prediction_key="pred_arc"),
        LiberoArcStage(**settings, operation="decode"),
    ]
    with pytest.raises(
        ValueError, match="Tokenizer and policy ARC representations differ"
    ):
        validate_input_representation(stages)


@pytest.mark.parametrize(
    "mode,profile,replay",
    [
        ("stk", "stk_2", "arc-five-v2-20260921-stk2-libero-spatial-replay"),
        ("dur", "dur_2", "arc-five-v2-20260921-dur2-libero-spatial-replay"),
    ],
)
def test_workflow_runs_hybrid_with_existing_replay_and_distinct_run_id(
    mode, profile, replay
):
    import json
    from scripts.benchmarks.launch_libero_osmo import arc_oat_workflow

    result = arc_oat_workflow(
        "a" * 40,
        "arc-oat-hybrid-test",
        "libero_spatial",
        arc_mode=mode,
        profile=profile,
        replay_run=replay,
        oat_reference_run="oat-evaluation",
    )
    workflow = result["workflow"]
    env = workflow["tasks"][0]["environment"]
    assert workflow["resources"]["default"]["gpu"] == 4
    assert env["RUN_KIND"] == "arc_oat"
    assert env["EPOCHS"] == "5001"
    assert json.loads(env["ARC_REPLAY_RUNS_JSON"]) == {mode: replay}
    assert env["RESUME_FROM_RUN"] == ""
    with pytest.raises(ValueError, match="5001 epochs"):
        arc_oat_workflow(
            "a" * 40,
            "bad-budget",
            "libero_spatial",
            arc_mode=mode,
            profile=profile,
            replay_run=replay,
            epochs=50,
        )


def test_hybrid_full_checkpoint_gate_rejects_missing_representation_and_partial_budget(
    tmp_path,
):
    from egomimic.benchmarks.libero.arc_oat import (
        input_representation,
        verify_checkpoint,
    )
    from egomimic.benchmarks.libero.arc_sweep import profile_settings

    settings = profile_settings("dur_2", "dur")
    payload = {
        "oat_input_representation": input_representation(settings),
        "benchmark_data_context": {"suite": "libero_spatial"},
        "hyper_parameters": {
            "config_tree": {
                "model": {
                    "benchmark_protocol": {
                        "suite": "libero_spatial",
                        "action_representation": "arc_oat",
                        "horizon": 32,
                        "n_obs_steps": 0,
                        "n_action_steps": 16,
                    }
                }
            }
        },
        "loops": {"fit_loop": {"epoch_progress": {"current": {"completed": 5001}}}},
        "ema_num_updates": 270054,
        "global_step": 270054,
        "ema_state_dict": {"weight": torch.ones(1)},
        "training_budget": {
            "epochs": 5001,
            "global_batch_size": 1024,
            "total_optimizer_steps": 270054,
        },
    }
    path = tmp_path / "tokenizer.ckpt"
    args = dict(
        settings=settings,
        suite="libero_spatial",
        epochs=5001,
        mode="full",
        method="tokenizer",
    )
    torch.save(payload, path)
    assert verify_checkpoint(path, **args)["global_step"] == 270054
    payload["ema_num_updates"] = payload["global_step"] = 270000
    torch.save(payload, path)
    with pytest.raises(ValueError, match="optimizer budget is incomplete"):
        verify_checkpoint(path, **args)
    assert verify_checkpoint(path, complete=False, **args)["global_step"] == 270000
    payload.pop("oat_input_representation")
    torch.save(payload, path)
    with pytest.raises(ValueError, match=r"different ARC\+OAT representation"):
        verify_checkpoint(path, complete=False, **args)


def test_hybrid_comparison_requires_explicit_method_and_still_rejects_state_mismatch(
    tmp_path,
):
    import json
    from egomimic.benchmarks.libero.report import compare_runs
    from egomimic.benchmarks.libero.rollout import rollout_plan, run_rollouts
    from tests.test_libero_benchmark import FakeEnvironment, FakePolicy

    metadata = {
        "suite": "libero_spatial",
        "horizon": 32,
        "n_obs_steps": 2,
        "n_action_steps": 16,
        "data_context": {"dataset_sha256": "same"},
        "observations_sha256": "same",
        "use_ema": True,
    }
    for method in ("arc_oat", "oat"):
        run_rollouts(
            FakePolicy(),
            rollout_plan("libero_spatial", trials_per_task=1, repetitions=1),
            tmp_path / method,
            env_factory=FakeEnvironment,
            metadata={
                **metadata,
                "method": method,
                "representation": {"arc": {"mode": "stk"}},
            },
        )
    with pytest.raises(ValueError, match="requires native"):
        compare_runs(tmp_path / "arc_oat", tmp_path / "oat", require_full=False)
    result = compare_runs(
        tmp_path / "arc_oat",
        tmp_path / "oat",
        require_full=False,
        arc_method="arc_oat",
        arc_mode="stk",
    )
    assert result["paired_success_difference"] == 0
    path = tmp_path / "arc_oat/episodes.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["initial_state_sha256"] = "wrong-state"
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    with pytest.raises(ValueError, match="Initial states differ"):
        compare_runs(
            tmp_path / "arc_oat",
            tmp_path / "oat",
            require_full=False,
            arc_method="arc_oat",
        )
