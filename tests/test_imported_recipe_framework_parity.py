"""Imported recipes retain the exact source7c scientific payload.

Hashes were extracted from immutable7c253e5 YAML after removing only the
framework target. This remains executable in shallow CI without source fetches.
"""

import hashlib
import json
from pathlib import Path

import pytest
import yaml


@pytest.mark.parametrize(
    "name,expected",
    [
        (
            "us_action_flow_bc_latent_fm_sg_200m_adamw",
            "a2007390f6497a04198308190bb259983eee4d75a9beed810926152e5c2f2fb9",
        ),
        (
            "us_action_flow_bc_graph_section",
            "bca57f3f847f55fb4c06e7b53af8ddd22e6d7e9252d288177a18c148e8d91100",
        ),
        (
            "us_action_flow_bc_latent_fm_sg_200m_muon",
            "0d7e4b03b9c5e1321ee009b78a0a89ad5ba9b954402f75833430a3bc5fb0239e",
        ),
        (
            "us_action_flow_bc_latent_fm_sg",
            "4cce3e22ad1a379ec2dfca757631f60d320e903e0a86d2c5cde02def3ab249e3",
        ),
        (
            "us_action_flow_bc_nt16_d8_h512",
            "d909b72505196be3c46d7dfb78b37894e55a9f4f6b812c61af07bb86325a9b00",
        ),
        (
            "cotrain_uc_action_flow_latent_fm_sg_unite_h384d12h12",
            "caba2d2a9432ef36876405a4affa76dc3d32c93ad8728de66cf58d7b9c4fb615",
        ),
    ],
)
def test_framework_port_keeps_every_scientific_field(name, expected):
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load(
        (root / "egomimic/hydra_configs/model/bf" / (name + ".yaml")).read_text()
    )
    assert cfg.pop("_target_") == "egomimic.pl_utils.pl_model.ModelWrapper"
    assert cfg.pop("training_behavior") == {
        "_target_": "egomimic.pl_utils.training_behavior_action_flow.ActionFlowTrainingBehavior"
    }
    assert cfg.pop("diagnostic_provider") == {
        "_target_": "egomimic.eval.pipeline_diagnostics.ActionFlowDiagnosticProvider"
    }
    if name in {
        "us_action_flow_bc_latent_fm_sg_200m_adamw",
        "us_action_flow_bc_latent_fm_sg_200m_muon",
    }:
        # Main adds an explicit non-deployable declaration, not a new adapter.
        assert cfg.pop("inference") == {
            "status": "unsupported",
            "reason": "This simulation recipe has no deployment observation/action adapter. Use its configured simulator evaluator; declare an inference adapter before deployment.",
        }
    actual = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert actual == expected
