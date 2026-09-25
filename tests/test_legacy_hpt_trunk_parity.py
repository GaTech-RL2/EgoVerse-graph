"""The generic initializer retains the source HPT trunk namespace and numerics."""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.stages_hpt import HPTTrunkStage
from scripts.audit_hydra_configs import CONFIGS

FIXTURE = Path(__file__).parent / "fixtures/legacy_hpt_trunk"


@pytest.mark.parametrize("transport", ["local", "huggingface"])
def test_legacy_trunk_loading_outputs_and_gradients(transport, monkeypatch):
    metadata = json.loads((FIXTURE / "metadata.json").read_text())
    assert metadata["source_commit"] == "ec5c903c067bf1b29bbff781bc414c6df2bcf1f2"
    for filename, key in (
        ("trunk.pth", "weights_sha256"),
        ("reference.pt", "reference_sha256"),
    ):
        assert (
            hashlib.sha256((FIXTURE / filename).read_bytes()).hexdigest()
            == metadata[key]
        )
    reference = torch.load(FIXTURE / "reference.pt", weights_only=True)
    with initialize_config_dir(version_base=None, config_dir=str(CONFIGS)):
        cfg = compose(
            config_name="train_zarr_cartesian", overrides=["model=hpt_bc_flow_eva"]
        )
    core = cfg.model.pipeline.stages[1].trunk
    core.embed_dim = core.attn_target.embed_dim = 16
    core.num_blocks = core.attn_target.num_heads = 2
    stage = HPTTrunkStage(instantiate(core), embed_dim=16)
    source = str(FIXTURE / "trunk.pth")
    downloads = []
    if transport == "huggingface":
        # Substitute only transport: tensor loading/validation and the actual
        # source HPT namespace remain the same as the local-file case.
        source = {
            "repo_id": "fixture/hpt",
            "revision": "a" * 40,
            "filename": "trunk.pth",
        }

        def fetch(**kwargs):
            downloads.append(kwargs)
            return str(FIXTURE / "trunk.pth")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", fetch)
    graph = PipelineAlgo(
        [stage],
        device="cpu",
        stage_ids={"trunk": 0},
        initialization=[
            {
                "stage_id": "trunk",
                "module_path": "trunk",
                "source": source,
                "source_prefix": metadata["source_namespace"],
                "sha256": metadata["weights_sha256"],
            }
        ],
    )
    assert graph.initialization_receipts[0]["sha256"] == metadata["weights_sha256"]
    assert downloads == ([source] if transport == "huggingface" else [])
    trunk = graph.pipeline.stage_by_id("trunk").trunk.eval()
    inputs = reference["input"].clone().requires_grad_()
    output, blocks = trunk(inputs, attn_mask=reference["mask"])
    loss = (output * reference["target"]).mean() + 0.1 * output.square().mean()
    loss.backward()
    torch.testing.assert_close(output, reference["output"], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(loss, reference["loss"], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        inputs.grad, reference["input_grad"], rtol=2e-5, atol=2e-6
    )
    for actual, expected in zip(blocks, reference["blocks"], strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert {"trunk." + key for key, _ in trunk.named_parameters()} == set(
        reference["parameter_grad"]
    )
    for name, parameter in trunk.named_parameters():
        torch.testing.assert_close(
            parameter.grad,
            reference["parameter_grad"]["trunk." + name],
            rtol=2e-5,
            atol=2e-6,
        )
