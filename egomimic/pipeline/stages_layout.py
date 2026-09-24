"""Explicit channel layout transforms for configured shared action heads."""

import torch

from egomimic.pipeline.core import Stage, resolve_homogeneous_scalar


class ActionLayoutStage(Stage):
    """Select channels or insert zeros using a YAML-declared index map.

    ``None`` inserts a zero channel. Input widths are validated before mapping;
    neither the runner nor this stage interprets channels as grippers or poses.
    """

    def __init__(
        self, key, layouts, input_widths, selector_key="embodiment", mode="train"
    ):
        super().__init__()
        if mode not in {"train", "inference"}:
            raise ValueError("ActionLayoutStage mode must be train or inference")
        self.train_only, self.inference_only = mode == "train", mode == "inference"
        self.key, self.selector_key = key, selector_key
        self.layouts = {str(k): tuple(v) for k, v in layouts.items()}
        self.input_widths = {str(k): v for k, v in input_widths.items()}
        if self.layouts.keys() != self.input_widths.keys():
            raise ValueError("Every layout needs an explicit input width")
        for name, indices in self.layouts.items():
            width = self.input_widths[name]
            if (
                type(width) is not int
                or width < 1
                or not indices
                or any(
                    index is not None
                    and (type(index) is not int or not 0 <= index < width)
                    for index in indices
                )
            ):
                raise ValueError(f"Invalid action layout for {name!r}")
        self.reads, self.writes = (key, selector_key), (key,)

    def forward(self, batch):
        selector = str(
            resolve_homogeneous_scalar(
                batch[self.selector_key], label=self.selector_key
            )
        )
        if selector not in self.layouts:
            raise ValueError(f"No action layout for selector {selector!r}")
        value = batch[self.key]
        if value.ndim != 3 or value.shape[-1] != self.input_widths[selector]:
            raise ValueError(
                f"{self.key} must have (B, H, {self.input_widths[selector]}) shape"
            )
        batch[self.key] = torch.stack(
            [
                torch.zeros_like(value[..., 0]) if index is None else value[..., index]
                for index in self.layouts[selector]
            ],
            dim=-1,
        )
        return batch
