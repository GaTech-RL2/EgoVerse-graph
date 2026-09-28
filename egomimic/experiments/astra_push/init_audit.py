"""Engineering audit of the exact full-size graph; no production training here."""

import argparse
import hashlib
import json
import random
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torchvision  # Import before a meta-device or blocked-network context.
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.experiments.astra_push.artifacts import (
    file_hash,
    named_seed,
    publish_json,
)
from egomimic.models.stems.byte_language import encode_bytes
from egomimic.pipeline.inference_config import build_inference_config

MODEL_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "hydra_configs/model/astra_push/hpt_scratch.yaml"
)


@contextmanager
def random_only():
    """Block remote/cache weight-loading entrances while constructing the graph."""

    def forbidden(*args, **kwargs):
        raise RuntimeError(
            "Random initialization attempted a weight load or network access"
        )

    with ExitStack() as stack:
        for target in (
            "socket.create_connection",
            "socket.socket.connect",
            "torch.load",
            "torch.nn.Module.load_state_dict",
            "torch.hub.load_state_dict_from_url",
            "torchvision.models._api.load_state_dict_from_url",
            "huggingface_hub.hf_hub_download",
            "huggingface_hub.snapshot_download",
            "safetensors.torch.load_file",
            "safetensors.torch.load_model",
            "transformers.PreTrainedModel.from_pretrained",
            "transformers.AutoModel.from_pretrained",
            "transformers.AutoTokenizer.from_pretrained",
        ):
            stack.enter_context(patch(target, forbidden))
        yield


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def construct(seed=17, *, config_path=MODEL_CONFIG):
    cfg = OmegaConf.load(config_path)
    set_seed(named_seed("learner_initialization", seed))
    with random_only():
        graph = instantiate(cfg.pipeline, device="cpu")
    if any(not p.requires_grad for p in graph.nets.parameters()):
        raise ValueError("Every learner parameter must be trainable")
    return graph, cfg


def fixture_batch(device):
    """Explicitly synthetic engineering data. It cannot satisfy real-data W01/W05."""
    set_seed(named_seed("engineering_batch"))
    return {
        "observations.images.external_rgb": torch.rand(
            4, 1, 1, 3, 224, 224, device=device
        )
        * 2
        - 1,
        "observations.images.wrist_rgb": torch.rand(4, 1, 1, 3, 224, 224, device=device)
        * 2
        - 1,
        "observations.state.proprioception": torch.randn(4, 1, 9, device=device),
        "language": encode_bytes(
            [
                "Push the block into the target.",
                "Push the blue block to the left target. Leave the red block in place.",
                "Push the block left of the marker to the right target. Leave the other block in place.",
                "Push the red block to the right target. Leave the blue block in place.",
            ],
            device=device,
        ),
        "embodiment": ["libero_push"] * 4,
        "actions": torch.rand(4, 10, 7, device=device) * 2 - 1,
        "action_valid": torch.arange(10, device=device)[None]
        < torch.tensor([10, 8, 5, 2], device=device)[:, None],
    }


def audit(output, *, device="cuda", config_path=MODEL_CONFIG):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "Full-size engineering update requires an allocated CUDA GPU"
        )
    start = time.monotonic()
    graph, cfg = construct(config_path=config_path)
    first = tensor_hash(graph.nets.state_dict())
    second, _ = construct(config_path=config_path)
    assert (
        tensor_hash(second.nets.state_dict()) == first
    ), "Seed replay changed initialization"
    del second
    other, _ = construct(18, config_path=config_path)
    assert tensor_hash(other.nets.state_dict()) != first, "Seed did not change weights"
    del other
    initial = output / "initial.pt"
    torch.save(
        {
            "model": graph.nets.state_dict(),
            "seed": 17,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        initial,
    )
    artifact = build_inference_config(OmegaConf.create({"model": cfg}))
    assert artifact["status"] == "ready"
    graph.device = torch.device(device)
    graph.nets.to(device).train()
    optimizer = instantiate(cfg.optimizer)(graph.nets.parameters())
    params = dict(graph.nets.named_parameters())
    members = [p for group in optimizer.param_groups for p in group["params"]]
    assert len({id(p) for p in members}) == len(members) == len(params)
    assert {id(p) for p in members} == {id(p) for p in params.values()}
    assert all(p.dtype == torch.float32 for p in members)
    before = {name: p.detach().flatten()[:8].clone() for name, p in params.items()}
    torch.cuda.reset_peak_memory_stats()
    batch = fixture_batch(device)
    torch.cuda.synchronize()
    update_start = time.monotonic()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = graph.forward_training({"libero_push": batch})
        loss = graph.compute_losses(result, {"libero_push": batch})["loss"]
    loss.backward()
    gradients = {
        name: float(p.grad.float().norm()) if p.grad is not None else None
        for name, p in params.items()
    }
    if not all(value is None or np.isfinite(value) for value in gradients.values()):
        raise ValueError("Nonfinite learner gradient")
    components = {
        "external_vision": "stages.0.stems.observations__images__external_rgb",
        "wrist_vision": "stages.0.stems.observations__images__wrist_rgb",
        "proprioception": "stages.0.stems.observations__state__proprioception",
        "language": "stages.0.stems.language",
        "trunk": "stages.1",
        "flow_head": "stages.4",
    }
    participation = {
        label: any(
            prefix in name and value is not None and value > 0
            for name, value in gradients.items()
        )
        for label, prefix in components.items()
    }
    if not all(participation.values()):
        raise ValueError(f"Disconnected trainable component: {participation}")
    torch.nn.utils.clip_grad_norm_(members, 1.0, error_if_nonfinite=True)
    optimizer.step()
    torch.cuda.synchronize()
    update_seconds = time.monotonic() - update_start
    assert all(
        state["exp_avg"].dtype == torch.float32
        and state["exp_avg_sq"].dtype == torch.float32
        for state in optimizer.state.values()
    )
    changed = sum(
        not torch.equal(before[n], p.detach().flatten()[:8]) for n, p in params.items()
    )
    graph.nets.eval()
    observed = {k: v for k, v in batch.items() if k not in {"actions", "action_valid"}}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = graph.forward_eval({"libero_push": observed})["libero_push"][
            "pred_action"
        ]
    assert prediction.shape == (4, 10, 7) and bool(torch.isfinite(prediction).all())
    receipt = {
        "kind": "synthetic_engineering_initialization_update",
        "real_data_gate_passed": False,
        "production_updates": 0,
        "seed": 17,
        "initialization_seed": named_seed("learner_initialization"),
        "initial_state_sha256": first,
        "initial_checkpoint_sha256": file_hash(initial),
        "config_sha256": file_hash(config_path),
        "parameter_count": sum(p.numel() for p in members),
        "parameters": {
            n: {
                "shape": list(p.shape),
                "dtype": str(p.dtype),
                "gradient_norm": gradients[n],
            }
            for n, p in params.items()
        },
        "buffers": {
            n: {"shape": list(b.shape), "dtype": str(b.dtype)}
            for n, b in graph.nets.named_buffers()
        },
        "gradient_components": participation,
        "parameters_changed_in_probe": changed,
        "loss": float(loss.detach()),
        "update_seconds": update_seconds,
        "total_seconds": time.monotonic() - start,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "inference_shape": list(prediction.shape),
        "execution_prefix": 1,
        "inference_contract": artifact,
    }
    publish_json(output / "receipt.json", receipt)
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in (
                    "kind",
                    "parameter_count",
                    "gradient_components",
                    "loss",
                    "update_seconds",
                    "peak_cuda_bytes",
                )
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit(args.output)
