"""Optional PI0.5 policy as a graph stage with explicit normalization ownership."""

from collections.abc import Mapping

import torch
from hydra.utils import instantiate

from egomimic.pipeline.core import Stage, resolve_homogeneous_scalar
from egomimic.rldb.embodiment.embodiment import get_embodiment


class PI05Stage(Stage):
    """Register the campaign backend before DDP/optimizers, after data shape inference.

    Backend imports and pretrained weights are deferred until data binding.
    Graph outputs use normalized ``pred_action``; the source PI decoder emits
    native actions, so this boundary normalizes that result exactly once.
    """

    def __init__(self, policy, action_key="actions_cartesian", init_weights_ckpt=None):
        super().__init__()
        self.policy_config = policy
        self.action_key = action_key
        self.init_weights_ckpt = init_weights_ckpt
        self.backend = None
        self.normalizer = None
        self.reads_by_mode = {
            "train": (
                "embodiment",
                "base_0_rgb",
                "observations.state.ee_pose",
                action_key,
            ),
            "inference": ("embodiment", "base_0_rgb", "observations.state.ee_pose"),
        }
        self.writes_by_mode = {"train": ("loss/pi05",), "inference": ("pred_action",)}

    def bind_data_context(self, *, normalizer):
        if self.backend is not None:
            if self.normalizer is not normalizer:
                raise RuntimeError(
                    "PI05Stage is already bound to a different normalizer"
                )
            return
        backend = instantiate(self.policy_config, norm_stats=normalizer)
        self.backend = backend
        self.policy_nets = backend.nets
        self.normalizer = normalizer
        if self.init_weights_ckpt:
            self.load_initial_weights(self.init_weights_ckpt)

    def load_initial_weights(self, path):
        """Weights-only initialization from source PI or graph PI checkpoints.

        A PI action expert always has a 32D output, including across human and
        robot finetunes. Require the entire backend to match after translating
        its prefix; missing weights must not silently initialize at random.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        source = checkpoint["state_dict"]
        prefixes = ["nets.", "nets.pipeline.stages.0.policy_nets."]
        expected = self.policy_nets.state_dict()
        candidates = [
            {k[len(prefix) :]: v for k, v in source.items() if k.startswith(prefix)}
            for prefix in prefixes
        ]
        matching = [
            candidate for candidate in candidates if set(candidate) == set(expected)
        ]
        if len(matching) != 1:
            raise ValueError(
                "PI checkpoint does not contain one complete matching policy state"
            )
        self.policy_nets.load_state_dict(matching[0], strict=True)

    def prepare(self, values):
        if self.backend is None:
            raise RuntimeError(
                "Bind PI05Stage to the normalizer before running the graph"
            )
        embodiment = int(
            resolve_homogeneous_scalar(values["embodiment"], label="embodiment")
        )
        label = get_embodiment(embodiment)
        if label is None or label.lower() not in self.backend.domains:
            raise ValueError(f"PI05Stage has no policy for embodiment {embodiment}")
        device = next(self.policy_nets.parameters()).device
        self.backend.device = device
        # Cloning here also leaves torch inference tensors behind: the compiled
        # OpenPI sampler uses mutable capture buffers and needs no_grad tensors.
        row = {
            k: v.clone().to(device) if torch.is_tensor(v) else v
            for k, v in values.items()
        }
        if self.action_key not in row:
            shape = self.normalizer.key_shape(self.action_key, embodiment)
            batch_size = row["base_0_rgb"].shape[0]
            row[self.action_key] = torch.zeros((batch_size, *shape), device=device)
        return embodiment, self.backend.process_batch_for_training({label.lower(): row})

    def execute(self, batch, *, mode):
        if mode == "train":
            embodiment, prepared = self.prepare(batch)
            label = get_embodiment(embodiment).lower()
            predictions = self.backend.forward_training(prepared)
            batch["loss/pi05"] = predictions[f"{label}_loss"]
        else:
            with torch.inference_mode(False), torch.no_grad():
                embodiment, prepared = self.prepare(batch)
                label = get_embodiment(embodiment).lower()
                native = self.backend.forward_eval(prepared)[
                    f"{label}_{self.action_key}"
                ]
                batch["pred_action"] = self.normalizer.normalize(
                    {self.action_key: native}, embodiment
                )[self.action_key]
        return batch
