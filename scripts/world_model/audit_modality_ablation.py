"""CPU introspection only: prove factorial controls before any training submission."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.pl_utils.joint_smoke import (
    image_parameter_state,
    require,
    weights_digest,
    write_once,
)

CELLS = ("image_only", "image_proprio", "image_action", "image_proprio_action")


def audit(manifest, taskroot):
    config_dir = Path(__file__).resolve().parents[2] / "egomimic/hydra_configs"
    rows = []
    control_reference = None
    for cell in CELLS:
        output = f"{taskroot}/runs/{cell}"
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(
                config_name="train_zarr_cartesian",
                overrides=[
                    f"+experiment=pixel_proprio/ablation/discrete_{cell}",
                    "hydra/launcher=basic",
                    f"data.manifest_path={manifest}",
                    f"paths.output_dir={output}",
                    f"paths.work_dir={taskroot}/source",
                    "hydra.job.chdir=false",
                ],
            )
        # Resolve while nodes are attached, preserving root interpolation context.
        controls = {
            key: OmegaConf.to_container(node, resolve=True)
            for key, node in dict(
                data=cfg.data,
                optimizer=cfg.model.optimizer,
                scheduler=cfg.model.scheduler,
                attention=cfg.world_model_attention,
                runtime=cfg.runtime,
                joint=cfg.joint,
            ).items()
        }
        controls = json.loads(json.dumps(controls).replace(output, "<run-output>"))
        if control_reference is None:
            control_reference = controls
        require(controls == control_reference, "Unpaired science/data/runtime controls")
        torch.manual_seed(42)
        network = instantiate(
            cfg.model.pipeline.stages[cfg.model.pipeline.stage_ids.joint_denoiser].model
        )
        parameters = sum(p.numel() for p in network.parameters())
        trained = sum(p.numel() for p in network.parameters() if p.requires_grad)
        rows.append(
            dict(
                cell=cell,
                active_streams=list(network.streams),
                parameters=parameters,
                trainable=trained,
                expert_parameters=network.expert_parameter_counts(),
                expert_widths=list(network.expert_widths),
                image_weights_sha256=weights_digest(image_parameter_state(network)),
                constructor_rng_sha256=hashlib.sha256(
                    torch.get_rng_state().numpy().tobytes()
                ).hexdigest(),
                run_id=cfg.logger.wandb.id,
                output=output,
                target_steps=cfg.trainer.max_steps,
                denoising_dims=dict(
                    image=network.image_dim,
                    proprio=36 if "proprio" in network.streams else None,
                    action=28 if "action" in network.streams else None,
                ),
                action_vector_dim=7 if "action" in network.streams else None,
                checkpoint_count=5,
                estimated_checkpoint_bytes=parameters * 12,
                checkpoints=OmegaConf.to_container(
                    cfg.callbacks.model_checkpoint, resolve=True
                ),
            )
        )
        del network
        gc.collect()
    require(
        len({row["image_weights_sha256"] for row in rows}) == 1,
        "Common image initialization differs",
    )
    require(
        len({row["constructor_rng_sha256"] for row in rows}) == 1,
        "Constructor consumes different RNG",
    )
    require(
        len({row["run_id"] for row in rows})
        == len({row["output"] for row in rows})
        == 4,
        "Output/identity collision",
    )
    return dict(
        schema="joint-modality-control-introspection-v1",
        status="PASS",
        rows=rows,
        controls=control_reference,
        training_submitted=False,
        gpu_smoke_required=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = audit(args.manifest, args.task_root)
    write_once(args.output, result)
    print(
        json.dumps(
            dict(
                status=result["status"],
                arms=[
                    {
                        k: row[k]
                        for k in (
                            "cell",
                            "parameters",
                            "trainable",
                            "image_weights_sha256",
                        )
                    }
                    for row in result["rows"]
                ],
            )
        )
    )


if __name__ == "__main__":
    main()
