"""Generated supervision provenance, bounded attempts and commissioning isolation."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from egomimic.experiments.astra_push.artifacts import file_hash, publish_json
from egomimic.experiments.astra_push.collect import collect_attempts, load_generated
from egomimic.experiments.astra_push.commissioning import CommissioningProposal
from egomimic.experiments.astra_push.data import (
    EpisodeBalancedReplay,
    EpisodeWindows,
    ProprioceptionStats,
)
from egomimic.experiments.astra_push.episodes import write_episode
from egomimic.experiments.astra_push.ledger import BudgetExhausted, PendingOperation
from egomimic.experiments.astra_push.schemas import canonical_hash
from egomimic.experiments.astra_push.stock_inventory import (
    inventory_file,
    stock_inventory,
)
from egomimic.experiments.astra_push.subagent_exchange import SubagentExchange
from scripts.astra_push.build_workflow import workflow

ARCHIVE = Path(__file__).resolve().parents[1] / (
    "docs/experiments/astra-hpt-libero/evidence/"
    "subagent-pilot-v1/generation/commissioning-001"
)


def proposal():
    return load_generated(ARCHIVE)[0]


def test_archived_actual_astra_response_is_bound_and_novel(tmp_path):
    generated, receipt = load_generated(ARCHIVE)
    assert receipt["agent_id"] == "/root/astra_commissioning_001"
    assert len(generated.templates) == 6
    assert len({t.scene.structural_signature() for t in generated.templates}) == 6
    for path in ARCHIVE.iterdir():
        (tmp_path / path.name).write_bytes(path.read_bytes())
    altered = json.loads((tmp_path / "raw-response.json").read_text())
    altered["templates"][0]["scene"]["scene_id"] = "changed"
    (tmp_path / "raw-response.json").write_text(json.dumps(altered))
    with pytest.raises(ValueError, match="inconsistent"):
        load_generated(tmp_path)


def test_generated_scene_duplicates_and_changed_witnesses_rejected():
    generated = proposal()
    value = generated.model_dump(mode="json")
    value["templates"][1]["scene"] = copy.deepcopy(value["templates"][0]["scene"])
    value["templates"][1]["task"] = copy.deepcopy(value["templates"][0]["task"])
    with pytest.raises(ValueError, match="structural"):
        CommissioningProposal.model_validate(value)
    value = generated.model_dump(mode="json")
    key = generated.templates[0].template_id
    value["templates"][0]["novelty_rationale"] = "Changed after successful witness"
    revised = CommissioningProposal.model_validate(value)
    with pytest.raises(ValueError, match="rewritten"):
        revised.validate_revision(generated, {key})


def test_exchange_archives_invalid_response_without_refunding_or_recalling(tmp_path):
    exchange = SubagentExchange(tmp_path / "archive")
    path = exchange.prepare("first", phase="commissioning", prompt={"task": "fixture"})
    assert path.is_file()
    with pytest.raises(PendingOperation):
        exchange.prepare("first", phase="commissioning", prompt={"task": "fixture"})
    response = tmp_path / "response.json"
    response.write_text('{"templates": [], "templates": []}')
    receipt = exchange.accept(
        "first",
        response,
        agent_id="/root/test",
        validator=CommissioningProposal.model_validate,
    )
    assert receipt["valid"] is False
    assert "Duplicate" in receipt["validation_error"]
    assert file_hash(path.parent / "raw-response.json") == file_hash(response)
    for i in range(2):
        exchange.prepare(
            f"repair{i}", phase="commissioning", prompt={}, repair_of="first"
        )
    with pytest.raises(ValueError, match="two repair"):
        exchange.prepare("repair2", phase="commissioning", prompt={}, repair_of="first")
    for i in range(5):
        exchange.prepare(f"more{i}", phase="commissioning", prompt={})
    with pytest.raises(BudgetExhausted):
        exchange.prepare("over_cap", phase="commissioning", prompt={})
    exchange.close()


def fake_attempt(path, template, seed, attempt_id):
    path.mkdir(parents=True, exist_ok=False)
    episode = path / "episode.hdf5"
    episode.write_bytes(f"fixture {attempt_id} {seed}".encode())
    receipt = {
        "accepted": True,
        "failure": None,
        "executed_steps": 1,
        "episode": {"path": str(episode), "sha256": file_hash(episode)},
    }
    publish_json(path / "receipt.json", receipt)
    return receipt


def test_interrupted_collection_counts_attempt_then_resumes_without_duplicate(tmp_path):
    calls = []

    def interrupt(path, template, seed, attempt_id):
        calls.append(attempt_id)
        path.mkdir(parents=True)
        (path / "started.txt").write_text("External work began")
        raise SystemExit("induced interruption")

    with pytest.raises(SystemExit):
        collect_attempts(tmp_path, proposal(), executor=interrupt)
    result = collect_attempts(tmp_path, proposal(), executor=fake_attempt)
    assert result["complete"] and result["attempts"] == 31
    assert set(result["accepted_by_template"].values()) == {5}
    assert len(result["accepted"]) == 30
    assert (tmp_path / "attempts/commissioning-0001/started.txt").exists()
    assert not (tmp_path / "attempts/commissioning-0001/receipt.json").exists()
    again = collect_attempts(
        tmp_path, proposal(), executor=lambda *args: pytest.fail("duplicate work")
    )
    assert again == result


def test_partial_commissioning_cannot_steal_another_templates_quota(tmp_path):
    result = collect_attempts(
        tmp_path, proposal(), executor=fake_attempt, attempt_cap=3
    )
    assert not result["complete"] and result["attempts"] == 3
    assert sum(result["accepted_by_template"].values()) == 3
    assert max(result["accepted_by_template"].values()) == 1


def commissioning_records(tmp_path):
    records = []
    for scene in range(6):
        for i in range(5):
            name = f"c{scene}_{i}"
            records.append(
                write_episode(
                    tmp_path / f"{name}.hdf5",
                    observations={
                        "external_rgb": np.zeros((1, 224, 224, 3), np.uint8),
                        "wrist_rgb": np.zeros((1, 224, 224, 3), np.uint8),
                        "proprioception": np.full((1, 9), scene + i, np.float32),
                        "timestamps": np.zeros(1),
                    },
                    actions=np.zeros((1, 7), np.float32),
                    instruction="test fixture",
                    provenance={
                        "episode_id": name,
                        "scene_hash": f"{scene:064x}",
                        "task_hash": "1" * 64,
                        "teacher_hash": "2" * 64,
                        "initial_state_hash": canonical_hash(name),
                        "phase": "commissioning",
                        "arm": "common",
                        "round": 0,
                        "stage": f"S{scene // 2 + 1}",
                        "partition": "training",
                        "accepted": True,
                    },
                    audit={},
                )
            )
    return records


def test_thirty_episode_normalizer_and_commissioning_cannot_enter_replay(tmp_path):
    records = commissioning_records(tmp_path)
    stats = ProprioceptionStats.fit_commissioning(records)
    np.testing.assert_allclose(stats.mean, 4.5)
    assert len(stats.source_hashes) == 30
    dataset = EpisodeWindows(records, stats, purpose="commissioning_audit")
    assert dataset[0]["action_valid"].sum() == 1
    with pytest.raises(ValueError, match="imitation"):
        EpisodeBalancedReplay(dataset, seed=17)
    with pytest.raises(ValueError, match="imitation"):
        EpisodeWindows(records, stats)
    records[0]["provenance"]["stage"] = "S3"
    with pytest.raises(ValueError, match="provenance"):
        ProprioceptionStats.fit_commissioning(records)


def test_stock_inventory_preserves_dangling_affordance_limitations(tmp_path):
    path = tmp_path / "stock.bddl"
    path.write_text("""(define (problem example) (:fixtures table_1 - table)
      (:objects bowl_1 - bowl)
      (:regions (unused (:target missing_1))
       (r (:target table_1) (:ranges ((0.0 0.0 0.1 0.1)))))
      (:init (On bowl_1 table_1_r)))""")
    inventory = inventory_file(path)
    assert inventory["asset_multiset"] == {"table": 1, "bowl": 1}
    assert inventory["unsupported_regions"] == [
        {"region": "unused", "target": "missing_1"}
    ]
    result = stock_inventory(tmp_path, [proposal().templates[0].scene])
    assert result["passed"] and len(result["region_inventory_limitations"]) == 1
    (tmp_path / "bad.bddl").write_text("not BDDL")
    assert stock_inventory(tmp_path, [proposal().templates[0].scene])["passed"] is False


def test_subagent_workflow_excludes_gateway_credential():
    archive = "docs/experiments/astra-hpt-libero/evidence/subagent-pilot-v1/generation/commissioning-001"
    spec = workflow(
        "a" * 40, "astra-pilot", provider_probe=False, generation_archive=archive
    )
    task = spec["workflow"]["tasks"][0]
    assert "astra-reversal-inference-20260924" not in task["credentials"]
    assert task["environment"]["ASTRA_GENERATION_ARCHIVE"] == archive
    assert spec["workflow"]["resources"]["single"]["gpu"] == 1
    with pytest.raises(ValueError):
        workflow("a" * 40, "astra-pilot", generation_archive=archive)
