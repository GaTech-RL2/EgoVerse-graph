"""Bound graph loading and canonical sequence inference without robot hardware."""

from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.eval.checkpoint_loading import strict_load_pipeline_checkpoint
from egomimic.pipeline.action_adapter import CanonicalSequenceAdapter
from egomimic.pipeline.checkpoint_binding import validate_artifact_binding
from egomimic.pipeline.construction import checkpoint_construction
from egomimic.pipeline.core import resolve_homogeneous_scalar
from egomimic.pipeline.inference_config import (
    find_inference_config,
    validate_inference_config,
)
from egomimic.pipeline.inference_controls import configure_profile_controls
from egomimic.pl_utils.data_context import DataContext


def load_bound_graph(
    training,
    *,
    checkpoint_path,
    context_path,
    artifact_path=None,
    device="cpu",
    use_ema=False,
    identity=None,
):
    """Validate all immutable bindings before constructing or binding a model."""
    if type(use_ema) is not bool:
        raise TypeError("use_ema must be a boolean")
    if isinstance(training, (str, Path)):
        training = OmegaConf.load(training)
    artifact_path = artifact_path or find_inference_config(checkpoint_path)
    if artifact_path is None:
        raise ValueError(
            "A checkpoint-bound inference artifact is required; use the training run's export"
        )
    artifact = OmegaConf.to_container(OmegaConf.load(artifact_path), resolve=True)
    declaration = validate_inference_config(artifact, training)
    loader = training.get("data_context_loader")
    if loader is None:
        raise ValueError(
            "Saved training config must declare a complete data_context_loader"
        )
    context = instantiate(loader, path=str(context_path))
    if not isinstance(context, DataContext):
        raise TypeError(
            "data_context_loader must return DataContext, not only normalization statistics"
        )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    validate_artifact_binding(artifact, checkpoint, training, context)
    context.normalizer.validate_inference_schema(
        declaration["compatibility"]["normalizer_schema"], identity=identity
    )
    with checkpoint_construction():
        graph = instantiate(training.model.pipeline, device=str(device))
        runnable, excluded = graph.pipeline.plan(
            declaration["input"]["keys"], mode="inference"
        )
        blocked = [missing for _, missing in excluded if missing != ["<train-only>"]]
        if blocked:
            raise ValueError(
                f"Declared inference inputs leave blocked graph stages: {blocked}"
            )
        outputs = {key for stage in runnable for key in stage.contract("inference")[1]}
        if declaration["native_output"]["key"] not in outputs:
            raise ValueError("Inference graph does not declare its native output")
        context.bind(graph)
    strict_load_pipeline_checkpoint(graph, checkpoint, use_ema=use_ema)
    graph.nets.eval()
    return graph, context, declaration


class InferenceSession:
    """Consume model-facing observations and return canonical tensors in the declared frame.

    Station camera routing, pose conversion, and history buffering belong to the
    input adapter. This session accepts an already assembled observation window;
    it never opens a robot, a dataset, or a camera.
    """

    def __init__(
        self, graph, context, declaration, *, identity, training, normalize_inputs=True
    ):
        if type(normalize_inputs) is not bool:
            raise TypeError("normalize_inputs must be a boolean")
        self.graph, self.context, self.declaration = graph, context, declaration
        self.identity, self.normalize_inputs = identity, normalize_inputs
        schema = declaration["compatibility"]["normalizer_schema"]
        context.normalizer.validate_inference_schema(schema, identity=identity)
        self.action_key = schema["action_key"]
        profile = next(iter(declaration["profiles"].values()))
        decoder = instantiate(profile["adapter"]["decoder"])
        self.adapter = CanonicalSequenceAdapter(
            declaration["output"]["shape"], decoder=decoder
        )
        self._controls = {
            control.name: control
            for control in configure_profile_controls(
                graph, training, declaration["profiles"]
            )
        }
        self.replan_every = None
        for control in self._controls.values():
            if control.target_kind == "policy_attribute":
                if control.attribute_path != "replan_every":
                    raise ValueError(
                        f"Sequence inference does not expose policy setting {control.attribute_path!r}"
                    )
                self.replan_every = control.value

    @classmethod
    def load(cls, training, *, identity, normalize_inputs=True, **paths):
        if isinstance(training, (str, Path)):
            training = OmegaConf.load(training)
        graph, context, declaration = load_bound_graph(
            training, identity=identity, **paths
        )
        return cls(
            graph,
            context,
            declaration,
            identity=identity,
            training=training,
            normalize_inputs=normalize_inputs,
        )

    def inference_controls(self):
        return {name: control.public() for name, control in self._controls.items()}

    def apply_inference_overrides(self, values):
        if not values or set(values) - self._controls.keys():
            raise ValueError("Inference overrides must name declared controls")
        checked = {
            name: self._controls[name].validate(value) for name, value in values.items()
        }
        for name, value in checked.items():
            control = self._controls[name]
            if control.target_kind == "stage_attribute":
                setattr(control.owner, control.attribute, value)
            else:
                setattr(self, control.attribute_path, value)
            control.value = value
        return self.inference_controls()

    @torch.no_grad()
    def predict(self, observations):
        required = set(self.declaration["input"]["keys"])
        if missing := required - observations.keys():
            raise ValueError(f"Missing declared inference inputs: {sorted(missing)}")
        for key, expected in self.declaration["input"].get("constants", {}).items():
            if resolve_homogeneous_scalar(observations[key], label=key) != expected:
                raise ValueError(f"Inference input constant differs at {key}")
        history = self.declaration["input"]["history_length"]
        if history > 1:
            for key in self.declaration["input"].get("history_keys", ()):
                value = observations[key]
                if (
                    not torch.is_tensor(value)
                    or value.ndim < 3
                    or value.shape[1] != history
                ):
                    raise ValueError(
                        f"Inference history key {key!r} requires {history} observations"
                    )
        values = (
            self.context.normalizer.normalize(observations, self.identity)
            if self.normalize_inputs
            else observations
        )
        batch = self.graph.process_batch_for_training({"inference": values})
        native_contract = self.declaration["native_output"]
        result = self.graph.forward_eval(batch)["inference"][native_contract["key"]]
        if result.ndim != 3 or list(result.shape[1:]) != list(native_contract["shape"]):
            raise ValueError("Graph native output differs from its declared shape")
        native = self.context.normalizer.unnormalize(
            {self.action_key: result}, self.identity
        )[self.action_key]
        return self.adapter(native)

    def execution_plan(self, prediction):
        return (
            prediction
            if self.replan_every is None
            else prediction[:, : self.replan_every]
        )
