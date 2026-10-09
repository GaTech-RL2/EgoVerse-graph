"""Real Hydra/OmegaConf restoration, no model/data/cluster execution."""

import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from hydra import __version__ as hydra_version
from hydra.conf import HydraConf
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from egomimic.benchmarks.libero.native_launch_profiles import PROFILES
from egomimic.benchmarks.libero.saved_hydra_context import load_saved_native_config


def setup(tmp_path, suite):
    repo = tmp_path / "source"
    repo.mkdir()
    run = tmp_path / "run"
    (run / ".hydra").mkdir(parents=True)
    p = PROFILES[suite]
    receipt = tmp_path / "receipt.json"
    logical_sha256 = "a" * 64
    receipt.write_text(
        json.dumps(
            dict(
                suite=suite,
                replay_path="/replay",
                dataset_logical_sha256=logical_sha256,
            )
        )
    )
    proof = tmp_path / "physical.json"
    proof.write_text("{}")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    binding = dict(
        source_commit="head",
        data_root="/replay",
        dataset_receipt_path=str(receipt),
        dataset_receipt_sha256=sha(receipt),
        physical_proof_path=str(proof),
        physical_proof_sha256=sha(proof),
    )
    config = dict(
        name=p.name,
        benchmark=dict(suite=suite, dataset="${oc.env:" + p.replay_environment + "}"),
        run_provenance=dict(
            source_commit="head",
            dataset_sha256=logical_sha256,
            dataset_content_aggregate_sha256=logical_sha256,
        ),
        norm_stats=dict(native_saved_state_binding=binding),
        data={
            group: {
                "libero_panda": {
                    "resolver": dict(
                        suite="${benchmark.suite}",
                        folder_path="${oc.env:" + p.replay_environment + "}",
                    )
                }
            }
            for group in ("train_datasets", "valid_datasets")
        },
        evaluator=dict(
            energy_seed_bank_path="${hydra:runtime.cwd}/egomimic/hydra_configs/evaluator/energy_score_seed_bank_k32_v1.json"
        ),
    )
    OmegaConf.save(OmegaConf.create(config), run / ".hydra/config.yaml")
    hydra = OmegaConf.structured(HydraConf)
    hydra.runtime.cwd = str(repo)
    hydra.runtime.output_dir = str(run)
    hydra.runtime.version = hydra_version
    saved = OmegaConf.to_container(hydra, resolve=False)
    saved["env"] = {}
    OmegaConf.save(OmegaConf.create({"hydra": saved}), run / ".hydra/hydra.yaml")
    return repo, run, p


@pytest.mark.parametrize("suite", list(PROFILES))
def test_saved_context_preserves_all_fields_and_missing_env(
    tmp_path, monkeypatch, suite
):
    repo, run, p = setup(tmp_path, suite)
    monkeypatch.delenv(p.replay_environment, raising=False)
    with patch("subprocess.check_output", return_value="head\n"):
        cfg = load_saved_native_config(run, repo)
    assert os.environ[p.replay_environment] == "/replay"
    assert cfg.benchmark.dataset == "/replay"
    assert OmegaConf.get_type(HydraConfig.get()) is HydraConf
    assert dict(HydraConfig.get().env) == {}
    saved = OmegaConf.to_container(
        OmegaConf.load(run / ".hydra/hydra.yaml").hydra, resolve=False, enum_to_str=True
    )
    actual = OmegaConf.to_container(HydraConfig.get(), resolve=False, enum_to_str=True)
    assert all(actual[k] == v for k, v in saved.items())


@pytest.mark.parametrize(
    "fault",
    ["conflict", "unknown_env", "cwd", "output", "source", "receipt", "resolver"],
)
def test_context_rejects_identity_drift(tmp_path, monkeypatch, fault):
    repo, run, p = setup(tmp_path, "libero_spatial")
    monkeypatch.delenv(p.replay_environment, raising=False)
    cfg = OmegaConf.load(run / ".hydra/config.yaml")
    saved = OmegaConf.load(run / ".hydra/hydra.yaml")
    if fault == "conflict":
        monkeypatch.setenv(p.replay_environment, "/wrong")
    if fault == "unknown_env":
        cfg.extra = "${oc.env:UNEXPECTED}"
    if fault == "cwd":
        saved.hydra.runtime.cwd = "/wrong"
    if fault == "output":
        saved.hydra.runtime.output_dir = "/wrong"
    if fault == "source":
        cfg.run_provenance.source_commit = "wrong"
    if fault == "receipt":
        Path(cfg.norm_stats.native_saved_state_binding.dataset_receipt_path).write_text(
            "{}"
        )
    if fault == "resolver":
        cfg.data.valid_datasets.libero_panda.resolver.folder_path = "/wrong"
    OmegaConf.save(cfg, run / ".hydra/config.yaml")
    OmegaConf.save(saved, run / ".hydra/hydra.yaml")
    with (
        patch("subprocess.check_output", return_value="head\n"),
        pytest.raises(ValueError),
    ):
        load_saved_native_config(run, repo)
