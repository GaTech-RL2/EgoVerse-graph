"""Scientific isolation, complete-state restoration and matched-evaluation gates."""

import copy
import json
import socket
import struct
import uuid
from pathlib import Path

import numpy as np
import pytest
import torch

from egomimic.experiments.astra_push.artifacts import file_hash, publish_json
from egomimic.experiments.astra_push.campaign import acquire
from egomimic.experiments.astra_push.collect import load_generated
from egomimic.experiments.astra_push.decisions import TaskProposal
from egomimic.experiments.astra_push.diagnostic import (
    finite_difference_fixture,
    preconditioned_direction,
)
from egomimic.experiments.astra_push.learner import (
    read_checkpoint,
    recover_training_blocks,
    restore_rng,
    rng_state,
    save_checkpoint,
)
from egomimic.experiments.astra_push.ledger import Ledger
from egomimic.experiments.astra_push.partitions import check_training_scene, definitions
from egomimic.experiments.astra_push.reporting import paired_cluster_summary
from egomimic.experiments.astra_push.schemas import SceneSpec, TaskSpec, instruction_for
from egomimic.experiments.astra_push.semantics import SuccessEvaluator
from egomimic.experiments.astra_push.transport import (
    MAX_FRAME_BYTES,
    PolicyServer,
    receive_frame,
    send_frame,
    validate_actions,
    validate_observation,
)
from scripts.astra_push.build_workflow import workflow

ROOT = Path(__file__).resolve().parents[1]


def generated():
    return load_generated(
        ROOT
        / "docs/experiments/astra-hpt-libero/evidence/subagent-pilot-v1/generation/commissioning-001"
    )[0]


def test_frozen_partition_pair_semantics_and_family_firewall(tmp_path):
    bank = definitions(
        tmp_path / "bank.json",
        forbidden_scenes=[t.scene for t in generated().templates],
    )
    assert (
        definitions(
            tmp_path / "other.json",
            forbidden_scenes=[t.scene for t in generated().templates],
        )
        == bank
    )
    assert len(bank["templates"]) == 36
    for row in bank["templates"]:
        if row["partition"] == "training-control":
            continue
        scene = SceneSpec.model_validate(row["scene"])
        tasks = [TaskSpec.model_validate(t).validate_scene(scene) for t in row["tasks"]]
        initial = {c.name: np.r_[c.placement.center_xy, 0.82] for c in scene.cubes}
        first, second = [SuccessEvaluator(scene, t, initial) for t in tasks]
        assert first.requested != second.requested
        moved = {k: v.copy() for k, v in initial.items()}
        moved[first.requested][:2] = first.target.center_xy
        rotations = {k: np.eye(3) for k in initial}
        for _ in range(10):
            first.update(moved, rotations)
            second.update(moved, rotations)
        assert first.result()["success"] and not second.result()["success"]
        raw = tasks[0].model_dump(exclude={"schema_version", "scene_hash"})
        with pytest.raises(ValueError, match="canonical language"):
            TaskProposal.model_validate(raw).bind(scene)
        assert tasks[0].instruction != instruction_for(
            scene.stage, tasks[0].referent, tasks[0].destination
        )
    variant = copy.deepcopy(bank["templates"][-1]["scene"])
    variant["scene_id"] = "renamed"
    for i, cube in enumerate(variant["cubes"]):
        cube["name"] = f"renamed_{i}"
        cube["placement"]["center_xy"][0] += 0.003
    for item in variant["targets"] + variant["fixtures"]:
        item["center_xy"][0] += 0.003
    with pytest.raises(ValueError, match="family_conflict"):
        check_training_scene(SceneSpec.model_validate(variant), bank)


def test_policy_transport_roundtrip_identity_allowlist_and_bounds(tmp_path):
    path = Path("/tmp") / f"astra-test-{uuid.uuid4().hex}.sock"
    metadata = {
        "version": 1,
        "kind": "observation",
        "request_id": "one",
        "episode_id": "episode",
        "step": 0,
        "timestamp": 0.0,
        "policy_seed": 17,
        "instruction": "Push the block into the target.",
    }
    arrays = {
        "external_rgb": np.zeros((224, 224, 3), np.uint8),
        "wrist_rgb": np.zeros((224, 224, 3), np.uint8),
        "proprioception": np.zeros(9, np.float32),
    }
    with PolicyServer(path, lambda m, a: np.full((10, 7), 0.25, np.float32), timeout=2):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.connect(str(path))
            send_frame(connection, metadata, arrays)
            reply, values = receive_frame(connection)
            assert (validate_actions(reply, values, "one") == 0.25).all()
            with pytest.raises(ValueError, match="identity"):
                validate_actions(reply, values, "wrong")
    with pytest.raises(ValueError, match="privileged"):
        validate_observation(metadata, {**arrays, "cube_xyz": np.zeros(3)})
    with pytest.raises(ValueError, match="timestamp"):
        validate_observation({**metadata, "timestamp": 0.1}, arrays)
    left, right = socket.socketpair()
    try:
        left.sendall(struct.pack("!II", 3, MAX_FRAME_BYTES + 1))
        with pytest.raises(ValueError, match="bound"):
            receive_frame(right)
    finally:
        left.close()
        right.close()


def test_complete_checkpoint_reproduces_next_update_with_optimizer_buffers_rng(
    tmp_path,
):
    torch.manual_seed(8)
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 3), torch.nn.BatchNorm1d(3), torch.nn.Dropout(0.3)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=4)

    def update():
        optimizer.zero_grad()
        loss = model(torch.randn(4, 3)).square().mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        return float(loss.detach())

    update()
    snapshot = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": rng_state(),
        "phase": "seed",
        "phase_step": 1,
        "total_updates": 1,
        "records": [],
        "normalization": {},
        "parent_checkpoint": None,
        "config_sha256": "a" * 64,
        "source_commit": "b" * 40,
    }
    receipt = save_checkpoint(tmp_path / "state.pt", snapshot)
    expected_loss = update()
    expected = copy.deepcopy(model.state_dict())
    restored = read_checkpoint(receipt)
    model.load_state_dict(restored["model"])
    optimizer.load_state_dict(restored["optimizer"])
    scheduler.load_state_dict(restored["scheduler"])
    restore_rng(restored["rng"])
    assert update() == expected_loss
    assert all(torch.equal(model.state_dict()[k], v) for k, v in expected.items())
    with (tmp_path / "state.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="content hash"):
        read_checkpoint(receipt)


def test_full_collection_interruption_retains_cost_and_quota(tmp_path):
    template = generated().templates[0]
    phase = "seed"
    quotas = {template.template_id: 2}
    ledger = Ledger(tmp_path / "ledger.sqlite", {"test": 1})
    ledger.add_phase(phase, {"attempt": 3}, quotas=quotas)

    class Worker:
        interrupted = False
        calls = []

        def call(self, operation, **arguments):
            self.calls.append(arguments["episode_id"])
            if not self.interrupted:
                self.interrupted = True
                raise KeyboardInterrupt("induced interruption")
            root = Path(arguments["output"])
            root.mkdir()
            episode = root / "episode.hdf5"
            episode.write_bytes(arguments["episode_id"].encode())
            receipt = {
                "accepted": True,
                "failure": None,
                "episode": {"path": str(episode), "sha256": file_hash(episode)},
            }
            publish_json(root / "receipt.json", receipt)
            return receipt

    worker = Worker()
    kwargs = dict(
        phase=phase,
        templates=[template],
        quotas=quotas,
        cap=3,
        ledger=ledger,
        simulator=worker,
    )
    with pytest.raises(KeyboardInterrupt):
        acquire(tmp_path / "data", **kwargs)
    result = acquire(tmp_path / "data", **kwargs)
    assert len(result["accepted"]) == 2 and result["attempts"] == 3
    assert len(worker.calls) == len(set(worker.calls)) == 3
    assert acquire(tmp_path / "data", **kwargs) == result
    assert len(worker.calls) == 3
    ledger.close()


def test_crash_after_checkpoint_publication_recovers_log_without_repeating_update(
    tmp_path,
):
    output = tmp_path / "checkpoints"
    ledger = Ledger(tmp_path / "ledger.sqlite", {"test": 1})
    ledger.add_phase("seed", {"training_block": 2})
    ledger.reserve("block0", "seed", "training_block", {"start": 0, "end": 1})
    block = {
        "operation_id": "block0",
        "start": 0,
        "end": 1,
        "updates": [{"loss": 0.1}],
        "seconds": 0.5,
        "peak_cuda_bytes": 0,
    }
    state = {
        "model": {"weight": torch.tensor([2.0])},
        "records": [],
        "normalization": {},
        "phase": "seed",
        "phase_step": 1,
        "total_updates": 1,
        "parent_checkpoint": None,
        "config_sha256": "a" * 64,
        "source_commit": "b" * 40,
        "training_block": block,
    }
    receipt = save_checkpoint(output / "step-0001-block0.pt", state)
    ledger.close()
    ledger = Ledger(tmp_path / "ledger.sqlite", {"test": 1})
    recover_training_blocks(output, "seed", ledger)
    result = json.loads((output / "block-0000-0001.json").read_text())
    assert result == {**block, "checkpoint": receipt}
    assert ledger.summary() == [
        {"phase": "seed", "kind": "training_block", "status": "succeeded", "count": 1}
    ]
    recover_training_blocks(output, "seed", ledger)
    assert ledger.summary()[0]["count"] == 1
    ledger.close()


def test_paired_cluster_bootstrap_preserves_templates_and_checkpoint_matching():
    def outcomes(success):
        rows = []
        for t in range(16):
            for reset in range(3):
                for side in range(2):
                    rows.append(
                        {
                            "pair_id": f"t{t:02d}_r{reset}",
                            "template_id": f"t{t:02d}",
                            "stage": "S2" if t < 8 else "S3",
                            "instruction_index": side,
                            "initial_state_sha256": f"{t}:{reset}",
                            "policy_seed": reset,
                            "metrics": {"success": success},
                        }
                    )
        return {"episodes": rows}

    results = {
        "warm_start": outcomes(False),
        "U_final": outcomes(False),
        "A_final": outcomes(True),
    }
    summary = paired_cluster_summary(results, resamples=100)
    assert summary["models"]["A_final"]["joint_successes"] == 48
    assert summary["bootstrap"]["clusters"] == 16
    assert summary["differences"]["A_final-minus-U_final"]["cluster_percentile_95"] == [
        1.0,
        1.0,
    ]
    results["A_final"]["episodes"][0]["policy_seed"] = 999
    with pytest.raises(ValueError, match="state/noise"):
        paired_cluster_summary(results, resamples=100)


def test_gradient_preconditioner_uses_actual_adam_steps_and_finite_difference():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01, betas=(0.9, 0.95), eps=1e-7)
    optimizer.state[parameter] = {
        "step": torch.tensor(3000.0),
        "exp_avg": torch.zeros(2),
        "exp_avg_sq": torch.tensor([0.2, 0.3]),
    }
    lookback = {"weight": torch.tensor([1.2, 1.7])}
    direction, steps = preconditioned_direction(
        {"weight": parameter}, optimizer, lookback
    )
    expected = (
        0.01
        / ((torch.tensor([0.2, 0.3]) / (1 - 0.95**3000)).sqrt() + 1e-7)
        * (lookback["weight"] - parameter.detach())
    )
    assert torch.equal(direction["weight"], expected)
    assert steps == {"weight": 3000} and finite_difference_fixture()["passed"]
    del optimizer.state[parameter]["exp_avg_sq"]
    with pytest.raises(ValueError, match="Missing Adam"):
        preconditioned_direction({"weight": parameter}, optimizer, lookback)


def test_full_workflow_is_one_osmo_gpu_without_exhausted_gateway():
    result = workflow(
        "a" * 40, "astra-full-test", provider_probe=False, full_experiment=True
    )["workflow"]
    task = result["tasks"][0]
    assert task["name"] == "learner-curriculum"
    assert result["resources"]["single"]["gpu"] == 1
    assert "astra-reversal-inference-20260924" not in task["credentials"]
    assert task["environment"]["ASTRA_FULL_RUN"] == "1"
