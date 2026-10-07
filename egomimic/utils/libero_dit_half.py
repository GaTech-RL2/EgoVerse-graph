"""Proposed maintained single-source checkpoint policy. No homogeneous batching.

Only the two resolved native DiT block inventories are intercepted. Decoder
checkpointing is untouched. Current native checkpoint options are forwarded
unchanged on retained blocks; unsupported options fail before direct execution.
"""
from lightning import Callback


def resolve_backbones(module):
    stages = tuple(module.model.pipeline.stages)
    if len(stages) != 9 or type(stages[4]).__name__ != "ContentEncoderStage" or type(stages[6]).__name__ != "ConditionalVelocityStage":
        raise ValueError("exact native single-source LIBERO stage topology required")
    return {"encoder": stages[4].encoder.backbone, "velocity": stages[6].field.backbone}


class HalfCheckpointScope:
    def __init__(self, native_module, backbones, expected_depth=12):
        if set(backbones) != {"encoder", "velocity"}:
            raise ValueError("exact two DiT owners required")
        self.native_module = native_module
        self.original = None
        self.inventory = {}
        self.counts = {f"{family}/{mode}": 0 for family in backbones for mode in ("direct", "checkpoint")}
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
                raise ValueError("changed native checkpoint options: audit contexts/RNG before use")
            family, index = self.inventory[fn]
            mode = "direct" if index % 2 == 0 else "checkpoint"
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
            raise ValueError("checkpoint ownership changed; refuse overwriting another scope")
        self.native_module.checkpoint = self.original
        self.original = None


class LiberoDiTHalf(Callback):
    def __init__(self):
        self.scope = None

    def on_fit_start(self, trainer, module):
        if self.scope is not None:
            raise ValueError("callback already active")
        self.scope = HalfCheckpointScope(__import__("egomimic.models.unite_dit", fromlist=["checkpoint"]), resolve_backbones(module))
        self.scope.install()

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        if set(batch) != {"libero_panda"}:
            raise ValueError("single LIBERO source32 required")
        self.before = dict(self.scope.counts)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        for name, count in self.scope.counts.items():
            if count <= self.before[name]:
                raise ValueError(("DiT-half inactive in actual update", name))
            module.log("Train/Execution/DiTHalf/" + name, float(count), on_step=True, on_epoch=False)

    def state_dict(self):
        return {"policy": "dit-half", "homogeneous": "not_applicable_single_source"}

    def load_state_dict(self, state):
        if state != self.state_dict():
            raise ValueError("checkpoint policy identity mismatch")

    def on_fit_end(self, trainer, module):
        if self.scope is not None:
            self.scope.restore()
            self.scope = None

    def on_exception(self, trainer, module, exception):
        self.on_fit_end(trainer, module)
