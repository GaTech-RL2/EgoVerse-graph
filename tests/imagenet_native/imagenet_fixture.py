"""Two-WNID synthetic local images and native-schema text-vector metadata.

The deterministic unit vectors and encoder weight digest are synthetic test
inputs, not real Qwen output or a certification of ImageNet admission.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from egomimic.rldb.imagenet_imagefolder import (
    build_class_language_manifest, build_imagefolder_manifest,
    build_validation_anchor_manifest,
)
from egomimic.rldb.pixel_language import (
    LANGUAGE_VECTOR_CACHE_SCHEMA, QWEN_MODEL_ID, QWEN_REVISION,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put_json(path, value, *, allow_nan=False):
    path = Path(path)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=allow_nan) + "\n")
    return sha(path)


def make_fixture(tmp_path, *, train_per_class=6, val_per_class=2):
    """Return exact file bindings plus ``datamodule_kwargs`` for local tests.

    Use ImageNetImageFolderDataModule(**fixture['datamodule_kwargs']). The
    default topology is two ranks, two training images per rank, validation
    batch size one and no workers. Callers may explicitly override topology.
    """
    root = Path(tmp_path).resolve()
    root.mkdir(parents=True, exist_ok=True)
    data_root = root / "imagefolder"
    names = {"n00000001": "synthetic red object", "n00000002": "synthetic blue object"}
    for split, count in (("train", train_per_class), ("val", val_per_class)):
        for class_index, wnid in enumerate(names):
            directory = data_root / split / wnid
            directory.mkdir(parents=True)
            for index in range(count):
                y, x = np.indices((180 + 4 * index, 240 + 3 * class_index))
                offset = 17 * class_index + 29 * index + (7 if split == "val" else 0)
                rgb = np.stack(((x + offset) % 256, (y + offset) % 256,
                                (x + y + offset) % 256), -1).astype(np.uint8)
                Image.fromarray(rgb).save(directory / f"{wnid}_{index:03d}.png")

    catalog_path = root / "class-catalog.json"
    catalog_sha = put_json(catalog_path, dict(
        schema="pixel-instruction-catalog-v1", data_admitted=False,
        clock_verified=False, tasks=names,
    ))
    rng = np.random.default_rng(20261026)
    vectors = {}
    for wnid in names:
        vector = rng.normal(size=1024).astype(np.float32)
        vector /= np.linalg.norm(vector.astype(np.float64))
        vectors[wnid] = vector.tolist()
    encoder = dict(
        model_id=QWEN_MODEL_ID, revision=QWEN_REVISION,
        tokenizer_revision=QWEN_REVISION, pooling="last_nonpadding_token",
        dimension=1024, normalization="l2", frozen=True, dtype="float32", max_length=128,
        weights_sha256=hashlib.sha256(b"synthetic fixture weights, no actual encoder").hexdigest(),
    )
    cache_path = root / "language-cache.json"
    cache_sha = put_json(cache_path, dict(
        schema=LANGUAGE_VECTOR_CACHE_SCHEMA, source_catalog_sha256=catalog_sha,
        data_admitted=False, clock_verified=False, encoder=encoder,
        tasks={wnid: dict(instruction=text, features=vectors[wnid]) for wnid, text in names.items()},
    ))
    classes_path = root / "class-language.json"
    classes = build_class_language_manifest(
        catalog_path, catalog_sha, cache_path, cache_sha, classes_path,
        expected_classes=2,
    )
    manifest_path = root / "manifest.json"
    manifest = build_imagefolder_manifest(
        data_root, classes_path, classes["sha256"], cache_path, cache_sha,
        manifest_path, expected_classes=2,
    )
    anchor_path = root / "validation-anchors.json"
    anchors = build_validation_anchor_manifest(
        manifest_path, manifest["sha256"], anchor_path, expected_classes=2,
    )
    kwargs = dict(
        manifest_path=str(manifest_path), manifest_sha256=manifest["sha256"],
        data_root=str(data_root), class_language_path=str(classes_path),
        class_language_sha256=classes["sha256"], language_features_path=str(cache_path),
        language_features_sha256=cache_sha, expected_classes=2,
        expected_train_images=2 * train_per_class, expected_val_images=2 * val_per_class,
        validation_anchor_path=str(anchor_path), validation_anchor_sha256=anchors["sha256"],
        batch_size=2, validation_batch_size=1, distributed_world_size=2,
        seed=42, num_workers=0, pin_memory=False, persistent_workers=False,
        prefetch_factor=None,
    )
    return dict(
        root=root, data_root=data_root, manifest_path=manifest_path,
        class_catalog_path=catalog_path, class_catalog_sha256=catalog_sha,
        class_language_path=classes_path, class_language_sha256=classes["sha256"],
        language_features_path=cache_path, language_features_sha256=cache_sha,
        manifest_sha256=manifest["sha256"], names=names, vectors=vectors,
        validation_anchor_path=anchor_path, validation_anchor_sha256=anchors["sha256"],
        datamodule_kwargs=kwargs,
    )
