"""Visualize recorded trajectories and annotations without a model or weights."""

from pathlib import Path
from types import SimpleNamespace

import hydra
from omegaconf import OmegaConf

from egomimic.utils.env import load_env


def run_data_tool(config):
    """Programmatic entry point using only declared data and evaluator capabilities."""
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    # Only fields consumed by this tool are part of its reproducible receipt.
    # Unused launcher paths can require Hydra runtime even for Python callers.
    resolved = {
        key: OmegaConf.to_container(config[key], resolve=True)
        for key in ("data", "evaluator")
    }
    resolved.update(output_dir=str(root), split=config.split)
    OmegaConf.save(OmegaConf.create(resolved), root / "resolved.yaml", resolve=True)
    evaluator = hydra.utils.instantiate(config.evaluator)
    data = hydra.utils.instantiate(config.data, _recursive_=False)
    context = data.prepare_visualization(
        split=config.split, requirements=evaluator.data_requirements()
    )
    evaluator.bind_data_context(normalizer=context.normalizer)
    evaluator.trainer = SimpleNamespace(
        current_epoch=0,
        global_step=0,
        global_rank=0,
        is_global_zero=True,
        default_root_dir=str(root),
    )
    evaluator.on_validation_start()
    for index, (group, batch) in enumerate(data.iter_visualization_batches()):
        evaluator.set_validation_group(group)
        evaluator.on_validation_step(batch, index)
    evaluator.on_validation_end()
    return root


def visualize_data(config):
    return run_data_tool(config) / "visualization-receipt.json"


@hydra.main(
    version_base="1.3", config_path="../hydra_configs", config_name="viz_language"
)
def main(config):
    load_env()
    print(visualize_data(config))


if __name__ == "__main__":
    main()
