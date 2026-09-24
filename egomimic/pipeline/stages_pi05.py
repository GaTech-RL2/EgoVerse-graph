"""Optional PI0.5 policy as a graph stage with explicit normalization ownership."""

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

    def __init__(
        self,
        policy,
        action_key="actions_cartesian",
        init_weights_ckpt=None,
        init_weights_sha256=None,
        init_weights_prefix=None,
        image_key="observations.images.front_img_1",
        state_key="observations.state.ee_pose",
    ):
        super().__init__()
        if not policy.get("domains") or not policy.get("ac_keys"):
            raise ValueError(
                "PI policy is an incomplete base fragment; select a concrete recipe declaring domains and ac_keys"
            )
        if init_weights_ckpt and (
            not init_weights_sha256 or init_weights_prefix is None
        ):
            raise ValueError(
                "PI weights initialization requires an exact SHA-256 and source namespace prefix"
            )
        self.policy_config = policy
        self.action_key = action_key
        self.image_key = image_key
        self.init_weights_ckpt = init_weights_ckpt
        self.init_weights_sha256 = init_weights_sha256
        self.init_weights_prefix = init_weights_prefix
        self.initialization_receipts = []
        self.backend = None
        self.normalizer = None
        self.reads_by_mode = {
            "train": (
                "embodiment",
                image_key,
                state_key,
                action_key,
            ),
            "inference": ("embodiment", image_key, state_key),
        }
        self.writes_by_mode = {
            "train": ("loss/pi05",),
            "inference": ("pred_action", "sampled_prompt"),
        }

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
            self.load_initial_weights(
                self.init_weights_ckpt,
                sha256=self.init_weights_sha256,
                source_prefix=self.init_weights_prefix,
            )

    def load_initial_weights(self, path, *, sha256, source_prefix):
        """Strict declared weights-only transfer, never prefix/shape guessing."""
        from egomimic.pipeline.core import Pipeline
        from egomimic.pipeline.initialization import initialize_weights

        self.initialization_receipts = initialize_weights(
            Pipeline([self], stage_ids={"policy": 0}),
            [
                {
                    "source": path,
                    "sha256": sha256,
                    "source_prefix": source_prefix,
                    "state_dict_path": ["state_dict"],
                    "stage_id": "policy",
                    "module_path": "policy_nets",
                }
            ],
        )

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
            batch_size = row[self.image_key].shape[0]
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
                batch["sampled_prompt"] = prepared[embodiment]["sampled_prompt"]
                native = self.backend.forward_eval(prepared)[
                    f"{label}_{self.action_key}"
                ]
                batch["pred_action"] = self.normalizer.normalize(
                    {self.action_key: native}, embodiment
                )[self.action_key]
        return batch
