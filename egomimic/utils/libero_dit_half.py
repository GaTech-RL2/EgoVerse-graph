"""Proposed maintained single-source checkpoint policy. No homogeneous batching.

Only the two resolved native DiT block inventories are intercepted. Decoder
checkpointing is untouched. Current native checkpoint options are forwarded
unchanged on retained blocks; unsupported options fail before direct execution.
"""

from lightning import Callback


def _checkpoint_periods(value, depth):
    periods = {"encoder": 2, "velocity": 2} if value is None else dict(value)
    if set(periods) != {"encoder", "velocity"} or any(
        isinstance(period, bool)
        or not isinstance(period, int)
        or not 2 <= period <= depth
        or depth % period
        for period in periods.values()
    ):
        raise ValueError(
            "checkpoint periods must divide the depth and retain direct and checkpoint blocks"
        )
    return periods


def resolve_backbones(module):
    stages = tuple(module.model.pipeline.stages)
    names = tuple(type(stage).__name__ for stage in stages)
    native = (
        len(stages) == 9
        and names[4] == "ContentEncoderStage"
        and names[6] == "ConditionalVelocityStage"
    )
    oat_topology = (
        "OATObservationStage",
        "ActionTargetBuilder",
        "GaussianLatentNoise",
        "ContentEncoderStage",
        "LatentBridgeStage",
        "ConditionalVelocityStage",
        "ContentDecoderStage",
        "ActionFlowObjectiveStage",
    )
    diffusion_topology = (
        oat_topology[:4]
        + ("LatentDiffusionBridgeStage", "ConditionalEpsilonDDIMStage")
        + oat_topology[6:]
    )
    embodied_topology = (
        "EmbodimentOATObservationStage",
        "EmbodimentActionTargetBuilder",
    ) + oat_topology[2:]
    oat = names in (oat_topology, diffusion_topology, embodied_topology)
    if native:
        encoder, velocity = stages[4], stages[6]
    elif oat:
        encoder, velocity = stages[3], stages[5]
    else:
        raise ValueError(
            "audited native or OAT-observation single-source LIBERO stage topology required"
        )
    if encoder.encoder is not None:
        return {
            "encoder": encoder.encoder.backbone,
            "velocity": velocity.field.backbone,
        }
    return {
        **{f"encoder:{key}": value.backbone for key, value in encoder.encoders.items()},
        "velocity": velocity.field.backbone,
    }


class HalfCheckpointScope:
    def __init__(
        self, native_module, backbones, expected_depth=12, checkpoint_periods=None
    ):
        if (
            "velocity" not in backbones
            or not any(
                key == "encoder" or key.startswith("encoder:") for key in backbones
            )
            or any(
                key != "velocity"
                and key != "encoder"
                and not key.startswith("encoder:")
                for key in backbones
            )
        ):
            raise ValueError("exact two DiT owners required")
        self.native_module = native_module
        self.checkpoint_periods = _checkpoint_periods(
            checkpoint_periods, expected_depth
        )
        self.original = None
        self.inventory = {}
        self.counts = {
            f"{family}/{mode}": 0
            for family in backbones
            for mode in ("direct", "checkpoint")
        }
        for family, backbone in backbones.items():
            blocks = tuple(backbone.blocks)
            if len(blocks) != expected_depth or not backbone.gradient_checkpointing:
                raise ValueError("native DiT depth/checkpoint flag mismatch")
            for index, block in enumerate(blocks):
                if block in self.inventory:
                    raise ValueError("shared or duplicated native block")
                self.inventory[block] = (family, index)

    def install(self):
        if self.original is not None:
            raise ValueError("checkpoint scope already installed")
        self.original = self.native_module.checkpoint

        def wrapped(fn, *args, **kwargs):
            if fn not in self.inventory:
                raise ValueError("unregistered checkpoint owner")
            if kwargs != {"use_reentrant": False}:
                raise ValueError(
                    "changed native checkpoint options: audit contexts/RNG before use"
                )
            family, index = self.inventory[fn]
            mode = (
                "checkpoint"
                if (index + 1) % self.checkpoint_periods[family.split(":", 1)[0]] == 0
                else "direct"
            )
            self.counts[f"{family}/{mode}"] += 1
            if mode == "checkpoint":
                return self.original(fn, *args, **kwargs)
            return fn(*args)

        self.wrapped = wrapped
        self.native_module.checkpoint = wrapped
        return self

    def restore(self):
        if self.original is None:
            return
        if self.native_module.checkpoint is not self.wrapped:
            raise ValueError(
                "checkpoint ownership changed; refuse overwriting another scope"
            )
        self.native_module.checkpoint = self.original
        self.original = None


class LiberoDiTHalf(Callback):
    def __init__(
        self, source_key="libero_panda", source_keys=None, checkpoint_periods=None
    ):
        if not isinstance(source_key, str) or not source_key.strip():
            raise ValueError("a nonempty single-source batch key is required")
        self.source_key = source_key
        self.source_keys = tuple(source_keys or (source_key,))
        if len(set(self.source_keys)) != len(self.source_keys) or any(
            not isinstance(key, str) or not key for key in self.source_keys
        ):
            raise ValueError("source_keys must be unique nonempty source names")
        self.checkpoint_periods = _checkpoint_periods(checkpoint_periods, 12)
        self.scope = None

    def on_fit_start(self, trainer, module):
        if self.scope is not None:
            raise ValueError("callback already active")
        self.scope = HalfCheckpointScope(
            __import__("egomimic.models.unite_dit", fromlist=["checkpoint"]),
            resolve_backbones(module),
            checkpoint_periods=self.checkpoint_periods,
        )
        self.scope.install()

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        if len(batch) != 1 or not set(batch) <= set(self.source_keys):
            raise ValueError(
                "single source from configured homogeneous sources required"
            )
        self.active_source = next(iter(batch))
        self.before = dict(self.scope.counts)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        for name, count in self.scope.counts.items():
            if name.startswith("encoder:") and not name.startswith(
                f"encoder:{self.active_source}/"
            ):
                if count != self.before[name]:
                    raise ValueError("inactive source encoder executed")
                continue
            if count <= self.before[name]:
                raise ValueError(("DiT-half inactive in actual update", name))
            module.log(
                "Train/Execution/DiTHalf/" + name,
                float(count),
                on_step=True,
                on_epoch=False,
            )

    def state_dict(self):
        state = {"policy": "dit-half", "homogeneous": "not_applicable_single_source"}
        if self.source_keys != (self.source_key,):
            state["source_keys"] = list(self.source_keys)
        if self.source_key != "libero_panda":
            state["source_key"] = self.source_key
        if self.checkpoint_periods != {"encoder": 2, "velocity": 2}:
            state["checkpoint_periods"] = dict(self.checkpoint_periods)
        return state

    def load_state_dict(self, state):
        if state != self.state_dict():
            raise ValueError("checkpoint policy identity mismatch")

    def on_fit_end(self, trainer, module):
        if self.scope is not None:
            self.scope.restore()
            self.scope = None

    def on_exception(self, trainer, module, exception):
        self.on_fit_end(trainer, module)
