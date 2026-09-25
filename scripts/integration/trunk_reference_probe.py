"""Generate HPT trunk references by executing the pinned legacy implementation."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch


def generate(source_root, output):
    import egomimic
    from egomimic.algo.hpt import HPTModel

    if not Path(egomimic.__file__).resolve().is_relative_to(source_root.resolve()):
        raise RuntimeError("Probe imported a different source checkout")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    if revision != "ec5c903c067bf1b29bbff781bc414c6df2bcf1f2":
        raise ValueError("Select the audited legacy source revision")
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "egomimic"],
        cwd=source_root,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(9226)
    arguments = dict(
        embed_dim=16,
        num_blocks=2,
        num_heads=2,
        observation_horizon=1,
        action_horizon=3,
    )
    policy = HPTModel(**arguments).eval()
    weights = output / "trunk.pth"
    with weights.open("xb") as stream:
        torch.save(policy.trunk.state_dict(), stream)
    restored = HPTModel(**arguments)
    restored.load_trunk(str(weights))
    restored.eval()
    inputs = torch.randn(2, 6, 16, requires_grad=True)
    mask = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
    target = torch.randn(2, 6, 16)
    predicted, blocks = restored.trunk["trunk"](inputs, attn_mask=mask)
    loss = (predicted * target).mean() + 0.1 * predicted.square().mean()
    loss.backward()
    reference = {
        "input": inputs.detach(),
        "mask": mask,
        "target": target,
        "output": predicted.detach(),
        # The legacy core runs sequence-first internally; the graph is batch-first.
        "blocks": [v.detach().transpose(0, 1).contiguous() for v in blocks],
        "loss": loss.detach(),
        "input_grad": inputs.grad,
        "parameter_grad": {
            name: p.grad for name, p in restored.trunk.named_parameters()
        },
    }
    with (output / "reference.pt").open("xb") as stream:
        torch.save(reference, stream)
    metadata = {
        "source_commit": revision,
        "constructor": arguments,
        "seed": 9226,
        "torch_version": torch.__version__,
        "source_namespace": "trunk.",
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(
            (output / "reference.pt").read_bytes()
        ).hexdigest(),
        "keys": list(restored.trunk.state_dict()),
        "scope": "actual legacy HPTModel trunk factory/loader, masked forward, loss and gradients; generated CPU parameters, not pretrained weights or a policy score",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    generate(arguments.source_root, arguments.output)
