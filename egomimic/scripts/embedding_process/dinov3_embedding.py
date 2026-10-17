"""
DINOv3 image-embedding ZarrKeyTransform.

Each input key must be a JPEG-encoded image array (as produced by ZarrWriter).
For each input key, produces a per-frame patch-token embedding of shape
``(T, N, D)`` float32 under the correspondingly-positioned output key,
where ``N`` is the number of patch tokens (CLS and any register tokens are
dropped) and ``D`` is the model hidden dim.
"""

from __future__ import annotations

import copy
import logging

import numpy as np
import simplejpeg
import torch
import zarr
from PIL import Image

from egomimic.scripts.embedding_process.zarr_key_transform import ZarrKeyTransform

logger = logging.getLogger(__name__)


class DINOv3ImageEmbedding(ZarrKeyTransform):
    """
    Per-frame DINOv3 image embeddings.

    Default output naming swaps the leading dotted segment of the input key
    for ``"dino"`` (e.g. ``images.front_1`` -> ``dino.front_1``). Pass
    ``output_keys`` explicitly to override.
    """

    DEFAULT_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    OUTPUT_PREFIX = "dino"

    def __init__(
        self,
        input_keys: list[str],
        output_keys: list[str] | None = None,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = 64,
        overwrite: bool = False,
        device: str | torch.device = "cuda",
        chunk_timesteps: int = 100,
        dtype: torch.dtype = torch.float16,
        revision: str | None = None,
        image_side: int | None = None,
    ):
        if output_keys is not None and len(input_keys) != len(output_keys):
            raise ValueError(
                "DINOv3ImageEmbedding requires len(input_keys) == len(output_keys); "
                f"got {len(input_keys)} vs {len(output_keys)}"
            )
        super().__init__(
            input_keys=input_keys,
            output_keys=output_keys,
            batch_size=batch_size,
            overwrite=overwrite,
            device=device,
            chunk_timesteps=chunk_timesteps,
        )
        self.model_name = model_name
        self.dtype = dtype
        if image_side is not None and (
            isinstance(image_side, bool)
            or not isinstance(image_side, int)
            or image_side < 1
        ):
            raise ValueError("image_side must be a positive integer")
        self.revision = revision
        self.image_side = image_side
        self.last_preprocessing = None
        self.output_metadata = {}
        self.processor = None
        self.model = None

    def setup(self) -> None:
        from transformers import AutoImageProcessor, AutoModel

        logger.info("Loading DINOv3 model: %s", self.model_name)
        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name, revision=self.revision
        )
        self.model = AutoModel.from_pretrained(
            self.model_name, revision=self.revision, torch_dtype=self.dtype
        )
        self.model.to(self.device).eval()
        self.model.requires_grad_(False)

    def _prepare(self, images):
        """Explicit full-FOV letterboxing; default processor behavior stays opt-in legacy."""
        geometry = []
        prepared = []
        for image in images:
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("DINO images must be uint8 HWC RGB")
            h, w = image.shape[:2]
            if self.image_side is None:
                prepared.append(image)
                geometry.append({"original_hw": [h, w]})
                continue
            side = self.image_side
            scale = side / max(h, w)
            rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
            top, left = (side - rh) // 2, (side - rw) // 2
            mean = np.asarray(self.processor.image_mean)
            canvas = np.empty((side, side, 3), dtype=np.uint8)
            canvas[:] = np.rint(mean * 255).astype(np.uint8)
            canvas[top : top + rh, left : left + rw] = np.asarray(
                Image.fromarray(image).resize((rw, rh), Image.Resampling.BICUBIC)
            )
            prepared.append(canvas)
            geometry.append(
                {
                    "original_hw": [h, w],
                    "resized_hw": [rh, rw],
                    "padding_top_left": [top, left],
                }
            )
        kwargs = (
            {}
            if self.image_side is None
            else {"do_resize": False, "do_center_crop": False}
        )
        inputs = self.processor(images=prepared, return_tensors="pt", **kwargs)
        if self.image_side is not None and tuple(inputs["pixel_values"].shape[-2:]) != (
            self.image_side,
            self.image_side,
        ):
            raise ValueError("DINO processor changed explicit letterbox resolution")
        return inputs, geometry

    @torch.no_grad()
    def _embed_batch(self, images: list[np.ndarray]) -> np.ndarray:
        inputs, geometry = self._prepare(images)
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
        h, w = pixel_values.shape[-2:]
        patch = self.model.config.patch_size
        ph, pw = (patch, patch) if isinstance(patch, int) else patch
        if h % ph or w % pw:
            raise ValueError(
                "Processed image dimensions must be divisible by DINO patch size"
            )
        grid = [h // ph, w // pw]
        num_patches = grid[0] * grid[1]
        outputs = self.model(pixel_values=pixel_values)
        # last_hidden_state ends with the patch tokens; CLS + any register
        # tokens sit at the start. Slice the trailing num_patches tokens to
        # get just the patch grid in row-major order.
        prefix = 1 + int(getattr(self.model.config, "num_register_tokens", 0))
        if outputs.last_hidden_state.shape[1] != prefix + num_patches:
            raise ValueError(
                "DINO token count disagrees with actual processed patch grid"
            )
        patches = outputs.last_hidden_state[:, prefix:]
        self.last_preprocessing = {
            "model": self.model_name,
            "requested_revision": self.revision,
            "resolved_revision": getattr(self.model.config, "_commit_hash", None),
            "processed_hw": [h, w],
            "patch_grid": grid,
            "patch_hw": [ph, pw],
            "feature_layer": "last_hidden_state_patch_tokens",
            "ordering": "row_major",
            "precision": str(self.dtype),
            "policy": "full_fov_letterbox_bicubic"
            if self.image_side
            else "processor_default",
            "image_mean": list(self.processor.image_mean),
            "image_std": list(self.processor.image_std),
            "frames": geometry,
        }
        return patches.float().cpu().numpy()

    def _write_numeric(self, store, key, arr):
        super()._write_numeric(store, key, arr)
        store[key].attrs["dinov3_preprocessing"] = self.output_metadata[key]

    def compute(self, store: zarr.Group) -> dict[str, np.ndarray]:
        outputs: dict[str, np.ndarray] = {}
        self.output_metadata = {}
        for in_key, out_key in zip(self.input_keys, self.output_keys):
            arr = store[in_key]
            n = arr.shape[0]
            total_frames = int(store.attrs.get("total_frames", n))
            n = min(n, total_frames)
            if n < 1:
                raise ValueError("DINO embedding requires at least one frame")

            chunks: list[np.ndarray] = []
            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                jpeg_batch = arr[start:end]
                images = [
                    simplejpeg.decode_jpeg(bytes(b), colorspace="RGB")
                    for b in jpeg_batch
                ]
                chunks.append(self._embed_batch(images))
                current = copy.deepcopy(self.last_preprocessing)
                frames = current.pop("frames")
                if any(frame != frames[0] for frame in frames):
                    raise ValueError(
                        "DINO frame geometry must be constant within an episode camera"
                    )
                current["geometry"] = frames[0]
                if (
                    out_key in self.output_metadata
                    and self.output_metadata[out_key] != current
                ):
                    raise ValueError(
                        "DINO preprocessing geometry changed within an output key"
                    )
                self.output_metadata[out_key] = current
            outputs[out_key] = np.concatenate(chunks, axis=0).astype(np.float32)
        return outputs
