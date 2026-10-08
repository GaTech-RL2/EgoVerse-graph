"""Verify typed native DiT-half activation on the selected stationary source.

This callback observes native counters; it replaces no forward/checkpoint/RNG
operation. The historical proportional sampler and all decoder layers remain
unchanged. Inactive private encoders must remain inactive in an update.
"""
from lightning import Callback

class YamHumanDiTHalf(Callback):
    def on_fit_start(self, trainer, module):
        stages = tuple(module.model.pipeline.stages)
        if len(stages) != 9 or type(stages[4]).__name__ != "ContentEncoderStage" or type(stages[6]).__name__ != "ConditionalVelocityStage":
            raise ValueError("native YAM+human stage topology required")
        if set(stages[4].encoders) != {"yam_bimanual", "human_bimanual"}:
            raise ValueError("native two-source private encoder mapping required")
        self.backbones = {name: encoder.backbone for name, encoder in stages[4].encoders.items()}
        self.backbones["velocity"] = stages[6].field.backbone
        for name, backbone in self.backbones.items():
            if backbone.depth != (12 if name == "velocity" else 6) or backbone.checkpoint_policy != "dit_half" or not backbone.gradient_checkpointing:
                raise ValueError("native DiT-half inventory/policy mismatch")
        blocks = [block for backbone in self.backbones.values() for block in backbone.blocks]
        if len({id(block) for block in blocks}) != len(blocks):
            raise ValueError("shared private DiT block inventory")

    def on_train_batch_start(self, trainer, module, batch, batch_idx):
        if len(batch) != 1 or next(iter(batch)) not in {"yam_bimanual", "human_bimanual"}:
            raise ValueError("native proportional homogeneous source batch required")
        self.source = next(iter(batch))
        self.before = {name: dict(model.checkpoint_policy_counts) for name, model in self.backbones.items()}

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        for name, backbone in self.backbones.items():
            active = name in {self.source, "velocity"}
            for mode, count in backbone.checkpoint_policy_counts.items():
                delta = count - self.before[name][mode]
                if (active and delta <= 0) or (not active and delta != 0):
                    raise ValueError(("DiT-half actual activation mismatch", self.source, name, mode, delta))
                if active:
                    module.log("Train/Execution/DiTHalf/" + name + "/" + mode, float(delta), on_step=True, on_epoch=False)

    def state_dict(self):
        return {"policy": "native_dit_half", "encoder_depth":6, "velocity_depth":12, "sampling":"native_proportional_homogeneous"}

    def load_state_dict(self, state):
        if state != self.state_dict():
            raise ValueError("checkpoint execution policy identity mismatch")
