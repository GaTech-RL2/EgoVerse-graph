"""Manifest-bound ImageFolder RGB data for the native joint image expert.

No class-ID embedding, vector placeholders, learned normalization or trainer is
introduced here. Every image is one camera at time zero. The complete joint
model may retain inactive robot/human parameter banks for the ImageNet source.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageOps
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from egomimic.pl_utils.data_context import DataContext, state_fingerprint
from egomimic.rldb.pixel_language import (
    QWEN_MODEL_ID, QWEN_REVISION, collate_pixel_language,
    load_reusable_language_features,
)
from egomimic.rldb.pixel_proprio import (
    DistributedPixelTrainingBatchSampler, DistributedPixelValidationBatchSampler,
    PixelValidationIndex, patchify_rgb,
)
from egomimic.rldb.world_model import HomogeneousWorldBatchSampler


DATA_SCHEMA = "imagenet-imagefolder-manifest-v1"
CLASS_SCHEMA = "imagenet-class-language-manifest-v1"
CONTEXT_KIND = "imagenet-imagefolder-context-v1"
PREPROCESSING = {
    "schema": "imagenet-native-rgb160-p16-v1",
    "decode": "Pillow_exif_transpose_convert_RGB",
    "resize": "shorter_side_160_round_half_up_bicubic",
    "crop": "center_160_integer_floor_origin",
    "augmentation": "none",
    "image_size": [160, 160], "patch_size": 16,
    "patch_order": "row_major_hwc", "normalization": "uint8_to_minus_one_one",
    "frames": 1, "cameras": 1, "camera_id": 0, "timestamp_seconds": 0.0,
}
_HEX = re.compile(r"[0-9a-f]{64}")
_WNID = re.compile(r"n[0-9]{8}")
_EXTENSIONS = {".jpeg", ".jpg", ".png"}


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate manifest key: " + key)
        value[key] = item
    return value


def _read_json(path, digest, *, maximum=512 * 1024 * 1024):
    if not isinstance(digest, str) or _HEX.fullmatch(digest) is None:
        raise ValueError("An exact lowercase SHA256 is required")
    path = Path(path)
    if path.stat().st_size > maximum:
        raise ValueError("Manifest exceeds its declared metadata bound")
    raw = path.read_bytes()
    if _sha(raw) != digest:
        raise ValueError("Manifest SHA256 mismatch")
    def reject(value):
        raise ValueError("Nonfinite manifest number: " + value)
    value = json.loads(raw, object_pairs_hook=_unique, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError("A manifest must be a JSON object")
    return value


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def _feature_sha(tensor):
    return _sha(tensor.detach().cpu().numpy().astype("<f4", copy=False).tobytes())


def load_class_language(path, digest, feature_cache_path, feature_cache_sha256,
                        *, expected_classes=1000):
    """Bind WNID -> exact class-name text -> cached frozen Qwen vector hash."""
    _positive(expected_classes, "expected_classes")
    value = _read_json(path, digest, maximum=8 * 1024 * 1024)
    classes = value.get("classes")
    if (value.get("schema") != CLASS_SCHEMA
            or value.get("dataset") != "ILSVRC2012_ImageNet1k"
            or value.get("class_count") != expected_classes
            or not isinstance(classes, dict) or len(classes) != expected_classes
            or any(_WNID.fullmatch(wnid) is None for wnid in classes)
            or value.get("feature_cache_sha256") != feature_cache_sha256
            or value.get("feature_hash_encoding") != "sha256_float32_le_1024"):
        raise ValueError("Invalid exact ImageNet class/language manifest")
    if Path(feature_cache_path).stat().st_size > 64 * 1024 * 1024:
        raise ValueError("Class vector cache exceeds the bounded metadata contract")
    cached = load_reusable_language_features(feature_cache_path, feature_cache_sha256)
    encoder = cached.context_identity()["encoder"]
    required = dict(model_id=QWEN_MODEL_ID, revision=QWEN_REVISION,
                    tokenizer_revision=QWEN_REVISION, pooling="last_nonpadding_token",
                    dimension=1024, normalization="l2", frozen=True,
                    dtype="float32", max_length=128)
    if (any(encoder.get(key) != item for key, item in required.items())
            or _HEX.fullmatch(str(encoder.get("weights_sha256"))) is None
            or value.get("encoder") != encoder
            or value.get("instruction_catalog_sha256") !=
               cached.context_identity().get("source_catalog_sha256")
            or set(cached.features) != set(classes)):
        raise ValueError("ImageNet requires the unchanged exact frozen Qwen contract")
    for wnid, item in classes.items():
        if (not isinstance(item, dict) or set(item) != {"class_name", "feature_sha256"}
                or not isinstance(item["class_name"], str) or not item["class_name"].strip()
                or item["class_name"] != cached.instructions[wnid]
                or item["feature_sha256"] != _feature_sha(cached.features[wnid])):
            raise ValueError("Class text/vector identity mismatch: " + wnid)
    return value, cached


def build_class_language_manifest(class_names_path, class_names_sha256,
                                 feature_cache_path, feature_cache_sha256,
                                 output_path, *, expected_classes=1000):
    """Bind an explicit WNID/name catalog to an existing native Qwen cache.

    The catalog is the maintained pixel-instruction-catalog-v1 accepted by
    encode_instruction_catalog(). This function neither encodes nor downloads.
    """
    catalog = _read_json(class_names_path, class_names_sha256, maximum=8 * 1024 * 1024)
    tasks = catalog.get("tasks")
    if (catalog.get("schema") != "pixel-instruction-catalog-v1"
            or catalog.get("data_admitted") is not False
            or catalog.get("clock_verified") is not False
            or not isinstance(tasks, dict) or len(tasks) != expected_classes
            or any(_WNID.fullmatch(wnid) is None or not isinstance(text, str)
                   or not text.strip() for wnid, text in tasks.items())):
        raise ValueError("Expected an explicit ImageNet WNID/class-name text catalog")
    cached = load_reusable_language_features(feature_cache_path, feature_cache_sha256)
    if (cached.instructions != tasks
            or cached.context_identity().get("source_catalog_sha256") != class_names_sha256):
        raise ValueError("Class feature cache is not bound to this exact catalog")
    value = dict(schema=CLASS_SCHEMA, dataset="ILSVRC2012_ImageNet1k",
                 class_count=expected_classes, instruction_catalog_sha256=class_names_sha256,
                 feature_cache_sha256=feature_cache_sha256,
                 feature_hash_encoding="sha256_float32_le_1024",
                 encoder=cached.context_identity()["encoder"],
                 classes={wnid: dict(class_name=tasks[wnid],
                                     feature_sha256=_feature_sha(cached.features[wnid]))
                          for wnid in sorted(tasks)})
    output = Path(output_path)
    with output.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write("\n")
    digest = _sha(output.read_bytes())
    load_class_language(output, digest, feature_cache_path, feature_cache_sha256,
                        expected_classes=expected_classes)
    return dict(path=str(output), sha256=digest, classes=expected_classes)


def _relative_image_path(path, split, wnid):
    if not isinstance(path, str):
        raise ValueError("ImageFolder path must be a relative string")
    relative = Path(path)
    if (relative.as_posix() != path or relative.is_absolute() or len(relative.parts) != 3
            or relative.parts[:2] != (split, wnid)
            or any(part in ("", ".", "..") for part in relative.parts)
            or relative.suffix.lower() not in _EXTENSIONS):
        raise ValueError("Expected split/WNID/filename ImageFolder path")
    return relative


def build_imagefolder_manifest(data_root, class_language_path, class_language_sha256,
                               feature_cache_path, feature_cache_sha256, output_path,
                               *, expected_classes=1000):
    """Inventory/hash actual ImageFolder files; run only in allocated preparation."""
    classes, _ = load_class_language(class_language_path, class_language_sha256,
                                     feature_cache_path, feature_cache_sha256,
                                     expected_classes=expected_classes)
    root = Path(data_root).resolve(strict=True)
    splits = {}
    for split in ("train", "val"):
        directory = root / split
        if directory.is_symlink():
            raise ValueError("Resolve source split aliases explicitly before image manifest preparation")
        entries = list(directory.iterdir())
        if any(not path.is_dir() or path.is_symlink() for path in entries):
            raise ValueError("Unclassified root sidecars require explicit preparation provenance; no silent filtering")
        actual = {path.name for path in entries}
        if actual != set(classes["classes"]):
            raise ValueError("Both splits must contain every exact declared WNID")
        records = []
        for wnid in sorted(actual):
            folder = directory / wnid
            if folder.is_symlink():
                raise ValueError("ImageFolder class directory cannot be a symlink")
            files = sorted(folder.iterdir())
            if not files:
                raise ValueError("Empty ImageNet class: " + wnid)
            for path in files:
                if path.is_symlink() or not path.is_file() or path.suffix.lower() not in _EXTENSIONS:
                    raise ValueError("Unexpected or linked ImageFolder member: " + str(path))
                relative = path.relative_to(root).as_posix()
                _relative_image_path(relative, split, wnid)
                raw = path.read_bytes()
                if not raw:
                    raise ValueError("Empty image file")
                records.append(dict(path=relative, wnid=wnid, bytes=len(raw), sha256=_sha(raw)))
        splits[split] = dict(count=len(records), records_sha256=state_fingerprint(records),
                             images=records)
    value = dict(schema=DATA_SCHEMA, dataset="ILSVRC2012_ImageNet1k",
                 format="ImageFolder/train_and_val_WNID_directories",
                 class_count=expected_classes, class_language_sha256=class_language_sha256,
                 feature_cache_sha256=feature_cache_sha256,
                 preprocessing=copy.deepcopy(PREPROCESSING), splits=splits)
    output = Path(output_path)
    with output.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write("\n")
    return dict(path=str(output), sha256=_sha(output.read_bytes()),
                counts={key: item["count"] for key, item in splits.items()})


def _validate_manifest(value, classes, class_sha, cache_sha, expected_classes):
    if (value.get("schema") != DATA_SCHEMA or value.get("dataset") != "ILSVRC2012_ImageNet1k"
            or type(value.get("class_count")) is not int or value.get("class_count") != expected_classes
            or value.get("format") != "ImageFolder/train_and_val_WNID_directories"
            or value.get("class_language_sha256") != class_sha
            or value.get("feature_cache_sha256") != cache_sha
            or value.get("preprocessing") != PREPROCESSING
            or set(value.get("splits", {})) != {"train", "val"}):
        raise ValueError("ImageNet manifest/config identity mismatch")
    seen = set()
    for split, part in value["splits"].items():
        rows = part.get("images")
        if (not isinstance(rows, list) or not rows or type(part.get("count")) is not int or part.get("count") != len(rows)
                or part.get("records_sha256") != state_fingerprint(rows)):
            raise ValueError("Invalid immutable ImageNet split inventory")
        wnids = set()
        for row in rows:
            if (not isinstance(row, dict) or set(row) != {"path", "wnid", "bytes", "sha256"}
                    or row["wnid"] not in classes["classes"]
                    or type(row["bytes"]) is not int or row["bytes"] < 1
                    or _HEX.fullmatch(str(row["sha256"])) is None):
                raise ValueError("Invalid ImageNet source-file record")
            _relative_image_path(row["path"], split, row["wnid"])
            if row["path"] in seen:
                raise ValueError("Duplicate ImageNet source path")
            seen.add(row["path"]); wnids.add(row["wnid"])
        if wnids != set(classes["classes"]):
            raise ValueError("ImageNet split misses a declared class")
        if [row["path"] for row in rows] != sorted(row["path"] for row in rows):
            raise ValueError("ImageFolder manifest must retain canonical class/file order")


def build_validation_anchor_manifest(manifest_path, manifest_sha256, output_path,
                                     *, expected_classes=1000, anchors_per_class=1):
    """Freeze one sorted official-validation key per class; never select TRAIN."""
    _positive(expected_classes, "expected_classes")
    if type(anchors_per_class) is not int or anchors_per_class != 1:
        raise ValueError("This explicit diagnostic selects exactly one official-val image per class")
    source = _read_json(manifest_path, manifest_sha256)
    rows = source["splits"]["val"]["images"]
    groups = {}
    for row in rows:
        groups.setdefault(row["wnid"], []).append(row)
    if len(groups) != expected_classes:
        raise ValueError("Validation anchors require every exact class")
    anchors = [min(group, key=lambda row: row["path"]) for _, group in sorted(groups.items())]
    value = dict(schema="imagenet-validation-anchors-v1", dataset="ILSVRC2012_ImageNet1k",
                 source_manifest_sha256=manifest_sha256,
                 official_val_records_sha256=source["splits"]["val"]["records_sha256"],
                 class_language_sha256=source["class_language_sha256"],
                 selector="first_sorted_official_val_image_per_wnid", anchors_per_class=1,
                 class_count=expected_classes, count=len(anchors),
                 records_sha256=state_fingerprint(anchors), images=anchors)
    output = Path(output_path)
    with output.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write("\n")
    digest = _sha(output.read_bytes())
    _validate_anchors(value, source, manifest_sha256, expected_classes)
    return dict(path=str(output), sha256=digest, count=len(anchors))


def _validate_anchors(value, manifest, manifest_sha, expected_classes):
    required = {"schema", "dataset", "source_manifest_sha256", "official_val_records_sha256",
                "class_language_sha256", "selector", "anchors_per_class", "class_count",
                "count", "records_sha256", "images"}
    by_class = {}
    for row in manifest["splits"]["val"]["images"]:
        by_class.setdefault(row["wnid"], []).append(row)
    exact = [min(rows, key=lambda row: row["path"]) for _, rows in sorted(by_class.items())]
    if (not isinstance(value, dict) or set(value) != required
            or value.get("schema") != "imagenet-validation-anchors-v1"
            or value.get("dataset") != manifest["dataset"]
            or value.get("source_manifest_sha256") != manifest_sha
            or value.get("official_val_records_sha256") != manifest["splits"]["val"]["records_sha256"]
            or value.get("class_language_sha256") != manifest["class_language_sha256"]
            or value.get("selector") != "first_sorted_official_val_image_per_wnid"
            or type(value.get("anchors_per_class")) is not int or value["anchors_per_class"] != 1
            or type(value.get("class_count")) is not int or value["class_count"] != expected_classes
            or type(value.get("count")) is not int or value["count"] != expected_classes
            or len(exact) != expected_classes or value.get("images") != exact
            or value.get("records_sha256") != state_fingerprint(exact)):
        raise ValueError("Validation anchors must be the exact frozen official-val class subset")
    return exact


def _decode_rgb160(raw):
    with Image.open(io.BytesIO(raw)) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        width, height = image.size
        if min(width, height) < 1:
            raise ValueError("Image has empty geometry")
        scale = 160 / min(width, height)
        size = (max(160, int(width * scale + 0.5)),
                max(160, int(height * scale + 0.5)))
        image = image.resize(size, resample=Image.Resampling.BICUBIC)
        left, top = (size[0] - 160) // 2, (size[1] - 160) // 2
        return np.array(image.crop((left, top, left + 160, top + 160)), np.uint8)


class ImageNetImageFolderDataset(Dataset):
    """One independent image, genuine class-name language, and no vector streams."""
    def __init__(self, manifest, data_root, classes, cached_features, *, split, selected_records=None):
        if split not in ("train", "val"):
            raise ValueError("Unknown ImageNet split")
        self.manifest, self.split = manifest, split
        self.root = Path(data_root).resolve(strict=True)
        self.records = (manifest["splits"][split]["images"] if selected_records is None else selected_records)
        self.classes, self.language = classes, cached_features
        # This is a single sampler bucket, not a fabricated temporal episode.
        self.episodes = [dict(embodiment="imagenet", kind="independent_image_sampler_bucket")]
        self.sampling_episodes, self.counts = self.episodes, [len(self.records)]
        yy, xx = np.meshgrid((np.arange(10) + 0.5) / 10 * 2 - 1,
                             (np.arange(10) + 0.5) / 10 * 2 - 1, indexing="ij")
        self.xy = torch.from_numpy(np.stack((xx, yy), -1).reshape(100, 2).astype(np.float32))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        cursor = validation_index = None
        if isinstance(index, PixelValidationIndex):
            validation_index, index = index.batch_index, index.index
        elif isinstance(index, tuple):
            if len(index) != 3 or any(type(x) is not int or x < 0 for x in index):
                raise ValueError("Expected deterministic image index/cursor/global-slot draw")
            index, cursor, _ = index
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError("Invalid image index")
        row = self.records[index]
        relative = _relative_image_path(row["path"], self.split, row["wnid"])
        path = self.root / relative
        if any((self.root / Path(*relative.parts[:n])).is_symlink()
               for n in range(1, len(relative.parts) + 1)):
            raise ValueError("Linked ImageNet member is not an admitted source")
        path.resolve(strict=True).relative_to(self.root)
        raw = path.read_bytes()
        if len(raw) != row["bytes"] or _sha(raw) != row["sha256"]:
            raise ValueError("ImageNet file changed after immutable manifest preparation")
        rgb = _decode_rgb160(raw)
        patches = patchify_rgb(rgb[None, None], 16).reshape(100, 768)
        image = dict(value=torch.from_numpy(patches), time=torch.zeros(100),
                     index=torch.zeros(100), valid=torch.ones(100, dtype=torch.bool),
                     target_valid=torch.ones(100, dtype=torch.bool),
                     observed=torch.zeros(100, dtype=torch.bool),
                     xy=self.xy.clone(), camera=torch.zeros(100, dtype=torch.long))
        metadata = dict(sample_kind="independent_image", sample_id=row["path"],
                        episode_id="imagenet/" + row["path"], camera_names=["image"],
                        frame_indices=[0], timestamps_seconds=[0.0], time_origin_seconds=0.0,
                        image_size=[160, 160], patch_size=16, chunk_size=1,
                        normalization="uint8_to_minus_one_one", patch_order="row_major_hwc",
                        source_sha256=row["sha256"], wnid=row["wnid"],
                        class_name=self.classes["classes"][row["wnid"]]["class_name"],
                        class_feature_sha256=self.classes["classes"][row["wnid"]]["feature_sha256"])
        sample = dict(embodiment="imagenet", task_kind="image_generation",
                      world={"image": image}, pixel_metadata=metadata,
                      **self.language.sample(row["wnid"]))
        if cursor is not None:
            sample["sampling_cursor"] = cursor + 1
        if validation_index is not None:
            sample["validation_batch_index"] = validation_index
        return sample


def collate_imagenet_images(samples):
    if not samples or any(sample.get("embodiment") != "imagenet"
                          or sample.get("task_kind") != "image_generation"
                          or set(sample.get("world", {})) != {"image"} for sample in samples):
        raise ValueError("Expected homogeneous independent ImageNet images only")
    fields = set(samples[0]["world"]["image"])
    if any(set(sample["world"]["image"]) != fields for sample in samples):
        raise ValueError("Inconsistent image stream metadata")
    stream = {key: torch.stack([sample["world"]["image"][key] for sample in samples])
              for key in fields}
    item = dict(embodiment="imagenet", world={"image": stream},
                task_kind=["image_generation"] * len(samples),
                pixel_metadata=[sample["pixel_metadata"] for sample in samples],
                **collate_pixel_language(samples))
    for key in ("sampling_cursor", "validation_batch_index"):
        values = [sample.get(key) for sample in samples]
        if any(value is not None for value in values):
            if any(type(value) is not int or value < 0 for value in values) or len(set(values)) != 1:
                raise ValueError("A batch must share one exact image draw/validation identity")
            item[key] = values[0] if key == "sampling_cursor" else values
    return {"imagenet": item}


def load_data_context(path):
    """Restore adapter metadata without opening either raw image split."""
    path = Path(path)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Saved data context exceeds its bounded metadata contract")
    def reject(value):
        raise ValueError("Nonfinite saved data context: " + value)
    payload = json.loads(path.read_text(), object_pairs_hook=_unique, parse_constant=reject)
    state = payload.get("data_context", payload)
    _validate_context_state(state)
    return DataContext(None, {"world": "image", "language_features": [1024]},
                       ("imagenet",), copy.deepcopy(state))


def _validate_context_state(state):
    required = {"kind", "manifest_sha256", "dataset", "class_count", "split_provenance",
                "preprocessing", "decoder_version", "class_language", "sampling", "validation_anchors"}
    if (not isinstance(state, dict) or set(state) != required
            or state.get("kind") != CONTEXT_KIND
            or state.get("dataset") != "ILSVRC2012_ImageNet1k"
            or state.get("preprocessing") != PREPROCESSING
            or _HEX.fullmatch(str(state.get("manifest_sha256"))) is None):
        raise ValueError("A full immutable ImageNet data context is required")
    count = _positive(state["class_count"], "saved class_count")
    language = state["class_language"]
    if (not isinstance(language, dict)
            or set(language) != {"manifest_sha256", "feature_cache_sha256", "encoder", "classes"}
            or any(_HEX.fullmatch(str(language.get(key))) is None
                   for key in ("manifest_sha256", "feature_cache_sha256"))
            or not isinstance(language["classes"], dict) or len(language["classes"]) != count):
        raise ValueError("Saved ImageNet language/class binding is incomplete")
    for wnid, entry in language["classes"].items():
        if (_WNID.fullmatch(wnid) is None or not isinstance(entry, dict)
                or set(entry) != {"class_name", "feature_sha256"}
                or not isinstance(entry["class_name"], str) or not entry["class_name"].strip()
                or _HEX.fullmatch(str(entry["feature_sha256"])) is None):
            raise ValueError("Invalid saved class text/vector identity")
    encoder = language["encoder"]
    expected = dict(model_id=QWEN_MODEL_ID, revision=QWEN_REVISION,
                    tokenizer_revision=QWEN_REVISION, pooling="last_nonpadding_token",
                    dimension=1024, normalization="l2", frozen=True, dtype="float32", max_length=128)
    if (not isinstance(encoder, dict) or any(encoder.get(key) != value for key, value in expected.items())
            or _HEX.fullmatch(str(encoder.get("weights_sha256"))) is None):
        raise ValueError("Invalid saved frozen Qwen encoder identity")
    decoder = state["decoder_version"]
    if (not isinstance(decoder, dict) or set(decoder) != {"library", "version"}
            or decoder["library"] != "Pillow" or not isinstance(decoder["version"], str)
            or not decoder["version"].strip()):
        raise ValueError("Saved image decoder version is required")
    splits = state["split_provenance"]
    if not isinstance(splits, dict) or set(splits) != {"train", "val"}:
        raise ValueError("Both saved split identities are required")
    for split in splits.values():
        if (not isinstance(split, dict) or set(split) != {"count", "records_sha256"}
                or type(split["count"]) is not int or split["count"] < 1
                or _HEX.fullmatch(str(split["records_sha256"])) is None):
            raise ValueError("Invalid saved ImageNet split identity")
    anchors = state["validation_anchors"]
    if (not isinstance(anchors, dict)
            or set(anchors) != {"manifest_sha256", "count", "records_sha256", "selector", "anchors_per_class"}
            or any(_HEX.fullmatch(str(anchors.get(key))) is None for key in ("manifest_sha256", "records_sha256"))
            or type(anchors["count"]) is not int or anchors["count"] != count
            or type(anchors["anchors_per_class"]) is not int or anchors["anchors_per_class"] != 1
            or anchors["selector"] != "first_sorted_official_val_image_per_wnid"):
        raise ValueError("Saved explicit official-val diagnostic anchor identity is required")
    sampling = state["sampling"]
    keys = {"schema", "seed", "training", "validation", "world_size", "per_rank_batch_size",
            "global_batch_size", "validation_batch_size"}
    if (not isinstance(sampling, dict) or set(sampling) != keys
            or sampling["schema"] != "imagenet-global-image-draw-v1"
            or sampling["training"] != "uniform_images_with_replacement_global_rank_slots"
            or sampling["validation"] != "frozen_class_balanced_anchors_ordered_no_padding_no_dropping"
            or type(sampling["seed"]) is not int or not 0 <= sampling["seed"] < 2**32):
        raise ValueError("Invalid saved ImageNet sampling contract")
    for key in ("world_size", "per_rank_batch_size", "global_batch_size", "validation_batch_size"):
        _positive(sampling[key], key)
    if sampling["global_batch_size"] != sampling["world_size"] * sampling["per_rank_batch_size"]:
        raise ValueError("Saved global ImageNet batch is inconsistent")


class ImageNetImageFolderDataModule(LightningDataModule):
    def __init__(self, manifest_path, manifest_sha256, data_root,
                 class_language_path, class_language_sha256,
                 language_features_path, language_features_sha256,
                 validation_anchor_path, validation_anchor_sha256,
                 expected_classes=1000, expected_train_images=1281167, expected_val_images=50000,
                 batch_size=8, validation_batch_size=1,
                 distributed_world_size=4, seed=42, num_workers=2,
                 pin_memory=True, persistent_workers=True, prefetch_factor=2):
        super().__init__()
        for name in ("expected_classes", "expected_train_images", "expected_val_images", "batch_size", "validation_batch_size", "distributed_world_size"):
            _positive(locals()[name], name)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("seed must be uint32")
        if type(num_workers) is not int or num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        if any(type(value) is not bool for value in (pin_memory, persistent_workers)):
            raise ValueError("Loader memory flags must be explicit booleans")
        if persistent_workers and num_workers == 0:
            raise ValueError("Persistent workers require actual workers")
        if prefetch_factor is not None and (num_workers == 0 or type(prefetch_factor) is not int or prefetch_factor < 1):
            raise ValueError("Prefetching requires workers and a positive factor")
        self.manifest_path, self.manifest_sha256, self.data_root = manifest_path, manifest_sha256, data_root
        self.class_language_path, self.class_language_sha256 = class_language_path, class_language_sha256
        self.language_features_path, self.language_features_sha256 = language_features_path, language_features_sha256
        self.validation_anchor_path, self.validation_anchor_sha256 = validation_anchor_path, validation_anchor_sha256
        self.expected_counts = {"train": expected_train_images, "val": expected_val_images}
        self.expected_classes, self.batch_size = expected_classes, batch_size
        self.validation_batch_size, self.distributed_world_size = validation_batch_size, distributed_world_size
        self.seed, self.num_workers = seed, num_workers
        self.pin_memory, self.persistent_workers, self.prefetch_factor = pin_memory, persistent_workers, prefetch_factor
        self.datasets, self.train_sampler, self.consumed_batches = {}, None, 0
        # The maintained WorldModelTrainingBehavior acknowledges consumed draws
        # only for this explicit capability; prefetched draws never advance it.
        self.checkpointable_sampling = True
        self.context = None

    def prepare_context(self, *, mode, normalization, normalizer, restored_state=None):
        if mode not in ("train", "eval") or normalizer is not None or normalization not in (None, "prepared"):
            raise ValueError("ImageNet uses fixed RGB scaling; no normalizer fit or temporal data path")
        if mode == "eval" and restored_state is None:
            raise ValueError("Standalone image evaluation requires its saved complete data context")
        classes, features = load_class_language(
            self.class_language_path, self.class_language_sha256,
            self.language_features_path, self.language_features_sha256,
            expected_classes=self.expected_classes)
        manifest = _read_json(self.manifest_path, self.manifest_sha256)
        _validate_manifest(manifest, classes, self.class_language_sha256,
                           self.language_features_sha256, self.expected_classes)
        if any(manifest["splits"][split]["count"] != count for split, count in self.expected_counts.items()):
            raise ValueError("Full ImageNet official split image counts differ from the explicit configuration")
        anchors = _read_json(self.validation_anchor_path, self.validation_anchor_sha256, maximum=8 * 1024 * 1024)
        anchor_rows = _validate_anchors(anchors, manifest, self.manifest_sha256, self.expected_classes)
        state = dict(kind=CONTEXT_KIND, manifest_sha256=self.manifest_sha256,
                     dataset=manifest["dataset"], class_count=self.expected_classes,
                     split_provenance={split: {key: value[key] for key in ("count", "records_sha256")}
                                       for split, value in manifest["splits"].items()},
                     validation_anchors=dict(manifest_sha256=self.validation_anchor_sha256,
                                             count=anchors["count"], records_sha256=anchors["records_sha256"],
                                             selector=anchors["selector"], anchors_per_class=1),
                     preprocessing=copy.deepcopy(PREPROCESSING),
                     decoder_version=dict(library="Pillow", version=PIL.__version__),
                     class_language=dict(manifest_sha256=self.class_language_sha256,
                                         feature_cache_sha256=self.language_features_sha256,
                                         encoder=classes["encoder"], classes=classes["classes"]),
                     sampling=dict(schema="imagenet-global-image-draw-v1", seed=self.seed,
                                   training="uniform_images_with_replacement_global_rank_slots",
                                   validation="frozen_class_balanced_anchors_ordered_no_padding_no_dropping",
                                   world_size=self.distributed_world_size,
                                   per_rank_batch_size=self.batch_size,
                                   global_batch_size=self.batch_size * self.distributed_world_size,
                                   validation_batch_size=self.validation_batch_size))
        _validate_context_state(state)
        if restored_state is not None and restored_state != state:
            raise ValueError("ImageNet checkpoint/data context mismatch")
        self.datasets = {split: ImageNetImageFolderDataset(manifest, self.data_root, classes, features, split=split,
                                      selected_records=anchor_rows if split == "val" else None)
                         for split in (("val",) if mode == "eval" else ("train", "val"))}
        self.context = DataContext(None, {"world": "image", "language_features": [1024]}, ("imagenet",), state)
        return self.context

    def setup(self, stage=None):
        if self.context is None:
            raise ValueError("Prepare the canonical ImageNet data context before constructing loaders")

    def configure_evaluation(self, requirements):
        if requirements.complete_episodes or set(requirements.required_keys) - {
                "world", "embodiment", "task_kind", "pixel_metadata", "language_features",
                "language_instruction", "task_id"}:
            raise ValueError("Unsupported independent-image evaluation requirements")

    def frame_counts(self):
        return [(split, "imagenet", len(dataset)) for split, dataset in self.datasets.items()]

    def _loader(self, split):
        self.setup()
        trainer = getattr(self, "trainer", None)
        rank = 0 if trainer is None else trainer.global_rank
        if trainer is not None and trainer.world_size != self.distributed_world_size:
            raise ValueError("Actual trainer topology differs from the bound ImageNet source")
        dataset = self.datasets[split]
        if split == "train":
            sampler = DistributedPixelTrainingBatchSampler(dataset, self.batch_size, self.seed,
                       world_size=self.distributed_world_size, rank=rank)
            sampler.consumed_batches = self.consumed_batches
            self.train_sampler = sampler
        else:
            base = HomogeneousWorldBatchSampler(dataset, self.validation_batch_size, self.seed, shuffle=False)
            sampler = DistributedPixelValidationBatchSampler(base,
                       world_size=self.distributed_world_size if trainer is not None else 1,
                       rank=rank)
        return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_imagenet_images,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers, prefetch_factor=self.prefetch_factor,
                          generator=torch.Generator().manual_seed(self.seed))

    def train_dataloader(self):
        return self._loader("train")

    def val_dataloader(self):
        return self._loader("val")

    def acknowledge_batch(self, cursor):
        if type(cursor) is not int or cursor != self.consumed_batches + 1:
            raise ValueError("Nonconsecutive ImageNet source draw acknowledgement")
        self.consumed_batches = cursor
        if self.train_sampler is not None:
            self.train_sampler.consumed_batches = cursor

    def state_dict(self):
        if self.context is None:
            raise ValueError("Checkpointed image sampling requires prepared data identity")
        return dict(schema="imagenet-source-draw-state-v1", consumed_batches=self.consumed_batches,
                    data_context_sha256=self.context.fingerprint())

    def load_state_dict(self, state_dict):
        expected = self.state_dict()
        if any(state_dict.get(key) != expected[key] for key in ("schema", "data_context_sha256")):
            raise ValueError("ImageNet sampling restore identity mismatch")
        cursor = state_dict.get("consumed_batches")
        if type(cursor) is not int or cursor < 0:
            raise ValueError("Invalid ImageNet consumed source-draw cursor")
        self.consumed_batches = cursor
        if self.train_sampler is not None:
            self.train_sampler.consumed_batches = cursor
