"""Scoped full VisualCore native-row graph, test-only and crop-contract guarded."""
from contextlib import contextmanager
from types import MethodType
import torch


@contextmanager
def native_visual_rows(visual, rows=32):
    assert isinstance(rows, int) and rows > 0
    # Only the resolved no-resize VisualCore contract is currently supported.
    assert visual.resize_to is None and visual.crop_aug
    native = visual.forward
    forward_saved = ("forward" in vars(visual), vars(visual).get("forward"))
    crop_saved = ("_crop" in vars(visual), vars(visual).get("_crop"))
    stats = {"calls": 0, "grouped_calls": 0, "native_chunks": 0}

    def restore_crop():
        if crop_saved[0]:
            visual._crop = crop_saved[1]
        elif "_crop" in vars(visual):
            del visual._crop

    def forward(self, x):
        if not self.training:
            return native(x)
        leading = x.shape[:-3]
        flat = x.reshape(-1, *x.shape[-3:])
        assert flat.shape[0] > 0 and flat.shape[0] % rows == 0
        stats["calls"] += 1
        stats["native_chunks"] += flat.shape[0] // rows
        if flat.shape[0] == rows:
            return native(x)
        stats["grouped_calls"] += 1
        # Existing source-major replay _crop consumes the exact two-source
        # native draws once, returns concatenated unchanged cropped pixels.
        cropped = self._crop(flat)
        assert cropped.shape[0] == flat.shape[0]
        outputs = []
        try:
            for raw_part, crop_part in zip(flat.split(rows), cropped.split(rows)):
                def cached_crop(owner, images, _part=crop_part):
                    assert images.shape[0] == rows
                    return _part
                self._crop = MethodType(cached_crop, self)
                # Preserve the COMPLETE native backbone/pool/head graph per
                # source; do not interleave shared-parameter layer reductions.
                outputs.append(native(raw_part))
        finally:
            restore_crop()
        return torch.cat(outputs, dim=0).reshape(*leading, self.embed_dim)

    visual.forward = MethodType(forward, visual)
    try:
        yield stats
    finally:
        restore_crop()
        if forward_saved[0]:
            visual.forward = forward_saved[1]
        elif "forward" in vars(visual):
            del visual.forward
