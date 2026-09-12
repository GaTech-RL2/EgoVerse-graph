"""Load graph artifacts and normalize at the live observation/action boundary."""

import json
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

from egomimic.eval.checkpoint_loading import strict_load_pipeline_checkpoint
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset


def load_normalizer(path):
    payload = json.loads(Path(path).read_text())
    state = payload.get("normalizer_state", payload)
    required = {
        "norm_mode",
        "embodiments",
        "key_types",
        "zarr_keys",
        "shapes",
        "norm_stats",
    }
    if not required <= state.keys():
        raise ValueError(
            "Deployment requires the full normalizer_state, including key types and shapes. "
            "Export a new cache with trainHydra norm_stats_only=true using the training recipe."
        )
    for name in ("key_types", "zarr_keys", "shapes", "norm_stats"):
        state[name] = {int(key): value for key, value in state[name].items()}
    state["embodiments"] = [int(key) for key in state["embodiments"]]
    normalizer = MultiDataset.from_state(state)
    for embodiment in normalizer.embodiments:
        for key, stats in normalizer.norm_stats[embodiment].items():
            shape = tuple(normalizer.key_shape(key, embodiment))
            for values in stats.values():
                tensor = torch.as_tensor(values)
                if not torch.isfinite(tensor).all():
                    raise ValueError(
                        f"Nonfinite normalization statistics: {embodiment}/{key}"
                    )
                torch.broadcast_to(tensor, shape)
    return normalizer


class GraphRobotPolicy:
    def __init__(self, graph, normalizer, adapter):
        self.graph, self.normalizer, self.adapter = graph, normalizer, adapter
        embodiment = adapter.embodiment_id
        if embodiment not in normalizer.embodiments:
            raise ValueError(
                "Deployment embodiment is absent from the training normalizer"
            )
        for key in (adapter.proprio_key, adapter.action_key):
            if key not in normalizer.shapes[embodiment]:
                raise ValueError(
                    f"Deployment key is absent from the training schema: {key}"
                )
            if (
                normalizer.norm_mode != "none"
                and key not in normalizer.norm_stats[embodiment]
            ):
                raise ValueError(
                    f"Deployment key has no normalization statistics: {key}"
                )

    @torch.no_grad()
    def predict(self, samples):
        adapter = self.adapter
        values = adapter.observation(samples)
        values = self.normalizer.normalize(values, adapter.embodiment_id)
        batch = self.graph.process_batch_for_training({"robot": values})
        prediction = self.graph.forward_eval(batch)["robot"]["pred_action"]
        native = self.normalizer.unnormalize(
            {adapter.action_key: prediction}, adapter.embodiment_id
        )[adapter.action_key]
        return adapter.actions(native, samples)


def load_policy(config):
    normalizer = load_normalizer(config.normalizer_path)
    training = OmegaConf.load(config.training_config)
    # This is the saved, fully composed training YAML; data resolvers are never instantiated.
    graph = instantiate(training.model.pipeline, device=str(config.device))
    if not isinstance(graph, PipelineAlgo):
        raise TypeError("Deployment requires a graph PipelineAlgo")
    graph.bind_data_context(normalizer=normalizer)
    checkpoint = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
    strict_load_pipeline_checkpoint(
        graph, checkpoint, use_ema=bool(config.get("use_ema", False))
    )
    graph.nets.eval()
    return GraphRobotPolicy(graph, normalizer, instantiate(config.adapter))
