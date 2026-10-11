"""Verify typed native DiT-half activation on the selected stationary source.

This callback observes native counters; it replaces no forward/checkpoint/RNG
operation. The historical proportional sampler and all decoder layers remain
unchanged. Inactive private encoders must remain inactive in an update.
"""

from lightning import Callback

from egomimic.pipeline.action_flow_topology import resolve_action_flow_topology


class YamHumanDiTHalf(Callback):
    def __init__(self, velocity_depth=12, encoder_kind="dit", encoder_depth=6):
        self.velocity_depth = int(velocity_depth)
        self.encoder_depth = int(encoder_depth)
        if self.velocity_depth <= 0 or self.velocity_depth % 2:
            raise ValueError("DiT-half requires a positive even velocity depth")
        if encoder_kind not in {"dit", "temporal_unet"}:
            raise ValueError("Unknown action encoder execution contract")
        self.encoder_kind = encoder_kind
        if encoder_kind == "dit" and (
            self.encoder_depth <= 0 or self.encoder_depth % 2
        ):
            raise ValueError("DiT-half requires a positive even encoder depth")

    def on_fit_start(self, trainer, module):
        stages = tuple(module.model.pipeline.stages)
        encoder, field, decoder = resolve_action_flow_topology(module.model)
        extra = [
            stage for stage in stages if type(stage).__name__ == "SharedSpeedCondition"
        ]
        if len(stages) != 9 + len(extra) or len(extra) > 1:
            raise ValueError(
                "native YAM/human graph with optional shared scalar speed required"
            )
        if extra and (
            extra[0].encoding != "scalar"
            or extra[0].output_key != field.inference_condition_key
        ):
            raise ValueError(
                "actual scalar speed condition must reach shared inference field"
            )
        if set(encoder.encoders) != {"yam_bimanual", "human_bimanual"}:
            raise ValueError("native two-source private encoder mapping required")
        if self.encoder_kind == "temporal_unet":
            from egomimic.models.temporal_action_codec import (
                TemporalActionDecoder,
                TemporalActionEncoder,
            )

            if not all(
                isinstance(item, TemporalActionEncoder) and item.gradient_checkpointing
                for item in encoder.encoders.values()
            ):
                raise ValueError("Temporal encoder checkpoint contract mismatch")
            if not all(
                isinstance(item, TemporalActionDecoder) and item.gradient_checkpointing
                for item in decoder.decoders.values()
            ):
                raise ValueError("Temporal decoder checkpoint contract mismatch")
            self.backbones = {}
        else:
            self.backbones = {
                name: item.backbone for name, item in encoder.encoders.items()
            }
        self.backbones["velocity"] = field.field.backbone
        for name, backbone in self.backbones.items():
            if (
                backbone.depth
                != (self.velocity_depth if name == "velocity" else self.encoder_depth)
                or backbone.checkpoint_policy != "dit_half"
                or not backbone.gradient_checkpointing
            ):
                raise ValueError("native DiT-half inventory/policy mismatch")
        blocks = [
            block for backbone in self.backbones.values() for block in backbone.blocks
        ]
        if len({id(block) for block in blocks}) != len(blocks):
            raise ValueError("shared private DiT block inventory")

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        if len(batch) != 1 or next(iter(batch)) not in {
            "yam_bimanual",
            "human_bimanual",
        }:
            raise ValueError("native proportional homogeneous source batch required")
        self.source = next(iter(batch))
        self.before = {
            name: dict(model.checkpoint_policy_counts)
            for name, model in self.backbones.items()
        }

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        for name, backbone in self.backbones.items():
            active = name in {self.source, "velocity"}
            for mode, count in backbone.checkpoint_policy_counts.items():
                delta = count - self.before[name][mode]
                if (active and delta <= 0) or (not active and delta != 0):
                    raise ValueError(
                        (
                            "DiT-half actual activation mismatch",
                            self.source,
                            name,
                            mode,
                            delta,
                        )
                    )
                if active:
                    module.log(
                        "Train/Execution/DiTHalf/" + name + "/" + mode,
                        float(delta),
                        on_step=True,
                        on_epoch=False,
                    )

    def state_dict(self):
        return {
            "policy": "native_dit_half",
            "encoder_depth": self.encoder_depth if self.encoder_kind == "dit" else None,
            "velocity_depth": self.velocity_depth,
            "sampling": "native_proportional_homogeneous",
        }

    def load_state_dict(self, state):
        if state != self.state_dict():
            raise ValueError("checkpoint execution policy identity mismatch")
