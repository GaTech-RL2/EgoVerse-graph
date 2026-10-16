"""Proposed maintained single-source checkpoint policy. No homogeneous batching.

Only the two resolved native DiT block inventories are intercepted. Decoder
checkpointing is untouched. Current native checkpoint options are forwarded
unchanged on retained blocks; unsupported options fail before direct execution.
"""

from lightning import Callback


def resolve_backbones(module):
    stages = tuple(module.model.pipeline.stages)
    names = tuple(type(stage).__name__ for stage in stages)
    native = (
        len(stages) == 9
        and names[4] == "ContentEncoderStage"
        and names[6] == "ConditionalVelocityStage"
    )
    oat = names == (
        "OATObservationStage",
        "ActionTargetBuilder",
        "GaussianLatentNoise",
        "ContentEncoderStage",
        "LatentBridgeStage",
        "ConditionalVelocityStage",
        "ContentDecoderStage",
        "ActionFlowObjectiveStage",
    )
    routed_encoders = [
        stage for stage in stages if type(stage).__name__ == "RoutedContentEncoderStage"
    ]
    shared_fields = [
        stage for stage in stages if type(stage).__name__ == "ConditionalVelocityStage"
    ]
    if len(routed_encoders) == len(shared_fields) == 1:
        encoders = routed_encoders[0].encoder
        if len(encoders) < 2:
            raise ValueError("routed checkpoint policy requires private encoders")
        from egomimic.models.mot_action_codec import MoTActionCodec

        mot = [isinstance(encoder, MoTActionCodec) for encoder in encoders.values()]
        if any(mot) and not all(mot):
            raise ValueError("mixed MoT/native encoder banks are not audited")
        if all(mot):
            shared = next(iter(encoders.values())).shared
            if any(
                encoder.role != "encoder" or encoder.shared is not shared
                for encoder in encoders.values()
            ):
                raise ValueError("MoT encoder routes must own one shared latent bank")
        return {
            **{
                "encoder/" + name: (encoder if all(mot) else encoder.backbone)
                for name, encoder in encoders.items()
            },
            "velocity": shared_fields[0].field.backbone,
        }
    if native:
        encoder, velocity = stages[4], stages[6]
    elif oat:
        encoder, velocity = stages[3], stages[5]
    else:
        raise ValueError(
            "audited native or OAT-observation single-source LIBERO stage topology required"
        )
    return {"encoder": encoder.encoder.backbone, "velocity": velocity.field.backbone}


class HalfCheckpointScope:
    def __init__(self, native_module, backbones, expected_depth=12):
        encoders = {
            name
            for name in backbones
            if name == "encoder" or name.startswith("encoder/")
        }
        if not encoders or set(backbones) != encoders | {"velocity"}:
            raise ValueError(
                "private encoder DiTs plus exactly one shared velocity owner required"
            )
        self.native_module = native_module
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
            raise ValueError(
                "checkpoint ownership changed; refuse overwriting another scope"
            )
        self.native_module.checkpoint = self.original
        self.original = None


class LiberoDiTHalf(Callback):
    def __init__(self, source_batch_sizes=None):
        self.scope = None
        self.source_batch_sizes = dict(source_batch_sizes or {"libero_panda": 32})
        if not self.source_batch_sizes or any(
            type(v) is not int or v <= 0 for v in self.source_batch_sizes.values()
        ):
            raise ValueError("positive declared batch size required for every source")

    def on_fit_start(self, trainer, module):
        if self.scope is not None:
            raise ValueError("callback already active")
        self.scope = HalfCheckpointScope(
            __import__("egomimic.models.unite_dit", fromlist=["checkpoint"]),
            resolve_backbones(module),
        )
        self.scope.install()

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        if set(batch) != set(self.source_batch_sizes):
            raise ValueError("LIBERO source identities differ from checkpoint policy")
        for name, size in self.source_batch_sizes.items():
            if batch[name]["actions"].shape[0] != size:
                raise ValueError(
                    "LIBERO per-source batch differs from checkpoint policy"
                )
        self.before = dict(self.scope.counts)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        for name, count in self.scope.counts.items():
            if count <= self.before[name]:
                raise ValueError(("DiT-half inactive in actual update", name))
            module.log(
                "Train/Execution/DiTHalf/" + name,
                float(count),
                on_step=True,
                on_epoch=False,
            )

    def state_dict(self):
        if self.source_batch_sizes == {"libero_panda": 32}:
            return {"policy": "dit-half", "homogeneous": "not_applicable_single_source"}
        return {"policy": "dit-half", "source_batch_sizes": self.source_batch_sizes}

    def load_state_dict(self, state):
        if state != self.state_dict():
            raise ValueError("checkpoint policy identity mismatch")

    def on_fit_end(self, trainer, module):
        if self.scope is not None:
            self.scope.restore()
            self.scope = None

    def on_exception(self, trainer, module, exception):
        self.on_fit_end(trainer, module)
