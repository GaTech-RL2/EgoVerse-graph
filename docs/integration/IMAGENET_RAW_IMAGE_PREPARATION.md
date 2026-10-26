# Raw ImageNet inventory and complete decode candidate

Local implementation only. No production ImageNet image, Qwen encoder/cache,
checkpoint, native allocation, deployment or remote job was certified here.
The sealed image-adapter-candidate-v1 is unchanged.

## Public API and CLI

Publish `source/egomimic/rldb/imagenet_preparation.py` as
`egomimic/rldb/imagenet_preparation.py` and
`source/scripts/data/verify_imagenet_images.py` as
`scripts/data/verify_imagenet_images.py`. The module imports the exact sealed
adapter's `_decode_rgb160`, hard-pins its module SHA256
`b43103a338b1cb732c1db6d3c07516699c162a9517f2212cc648948d9d93d737`, and
records Pillow version plus the complete existing preprocessing contract.
No Qwen cache or model is required by `inventory-decode`. Importing the shared
adapter requires its existing CPU Python dependencies; it does not construct or
load a model. Run only in a genuinely allocated CPU producer through the separately
reviewed source/runtime/native/host guards.

```
python scripts/data/verify_imagenet_images.py inventory-decode \
  --train-root /coc/dataset/ImageNet/train \
  --val-root /coc/dataset/ImageNet/val \
  --train-sidecars ACTUAL_SIDECAR_RECIPE.json \
  --train-sidecars-sha256 ACTUAL_RECIPE_SHA256 \
  --output FRESH_RAW_INVENTORY.json
```

CLI counts are fixed to 1,000 WNIDs in BOTH splits, 1,281,167 TRAIN and 50,000
official VAL images. API expected-count overrides exist for explicit synthetic
fixture tests, not to relabel an incomplete production inventory. Both roots must
be physical absolute sibling directories named `train` and `val`. Root aliases,
class/file symlinks, unknown sidecars, unexpected extensions and empty/corrupt
images fail. Every accepted image is opened once, raw-size/hashed and fully
decoded with the unchanged native resize/crop; each canonical record retains
`path, wnid, bytes, sha256, rgb160_sha256`. No image/cache writes occur.

The sidecar receipt is an object with exactly
`schema: imagenet-train-sidecar-exclusions-v1` and `exclusions: [...]`.
An ordinary exclusion has exactly `name, bytes, sha256`. Only
`tar_extract_script.sh~` (0 bytes) and `tar_extract_script.sh` (108 bytes) are
eligible, and their actual size/hash must match. An explicit empty list is legal
only when no sidecar is present. No optional directory filtering occurs.

For the specifically reported unreadable 108-byte nonconsumed TRAIN script,
the alternative reviewed recipe is exactly:

```
{"name":"tar_extract_script.sh","bytes":108,
 "status":"UNREADABLE_EXCLUDED","uid":ACTUAL_UID,
 "mode":"ACTUAL_FOUR_OCTAL_DIGITS","mtime_seconds":ACTUAL_FLOOR_MTIME}
```

The producer actually attempts its no-follow read. Only `PermissionError` with
errno13 is accepted, and before/after regular-file identity/metadata must match
the explicit uid/mode/size/mtime recipe. A readable file, another errno, metadata
change, another name or a symlink fails. The raw report retains the actual read
attempt outcome and makes NO SHA256 claim for this excluded nonimage sidecar.
No chmod, deletion or permission changes are performed.

`verify_images(...)` defaults to 4 threads and at most 16 submitted futures;
canonical ordered consumption avoids eager executor.map over 1.33 million rows.
Directory identity/mtime is rechecked after the complete scan. The portable
content digest excludes physical roots and timestamps; physical roots, excluded
sidecar provenance, concurrency and external native/host requirements remain
separate report metadata. Later normal dataset reads still verify raw file SHA.

## Immutable outputs and failure

An exclusive `.intent.json` consumes the namespace before inventory work. An
exclusive fsynced `.progress.jsonl` records bounded progress and failure. A
failure writes `.failure.json`; no retry or cleanup is automatic. A completed
fsynced `.payload` is atomically hardlinked no-clobber to the requested report,
and both names are retained. A failure receipt invalidates even a final report
published before a later I/O failure. Existing intent/output/journal/failure/
payload names reject. External host/native success must also be bound before
any downstream admission; `PASS_IMAGE_BYTES_AND_DECODE_ONLY` is explicitly
`data_admitted: false` and never an allocation or training readiness certificate.

## Later class-language join

```
python scripts/data/verify_imagenet_images.py join-language \
  --raw-report ACTUAL_RAW.json --raw-report-sha256 ACTUAL_RAW_SHA256 \
  --class-language ACTUAL_CLASSES.json --class-language-sha256 ACTUAL_CLASSES_SHA256 \
  --language-features ACTUAL_QWEN.json --language-features-sha256 ACTUAL_QWEN_SHA256 \
  --output FRESH_MANIFEST.json
```

`join_raw_inventory(...)` revalidates the actual raw-report SHA/schema/content
digests/provenance and exact actual class-name/Qwen bindings, then projects image
rows to the SEALED adapter's exact `imagenet-imagefolder-manifest-v1` DATA_SCHEMA
without adding unrecognized fields. It does not stat/read/hash/decode any source
image or reopen the physical dataset roots. It returns a separate join receipt
retaining raw-report SHA and class/Qwen/manifest bindings, with
`images_rehashed_or_decoded: 0` and `data_admitted: false`.

Raw producer, join and eventual training runtime must use the same pinned Pillow
version and exact adapter module; a cross-cluster decoder-version mismatch fails
closed and must not be relabelled. Actual allocated CPU byte/decode producer
source/claim/host/native proof, language encoder proof, data staging, ImageNet
specific smoke/full admission and actual 4-rank GPU smoke remain external gates.

## Local verification

43 focused synthetic CPU tests passed, including unreadable-sidecar rejection
cases, bounded submission, alias/corrupt/count failures, retained provenance and
join with all image read/decode calls forbidden. The tests use the sealed
`tests/imagenet_native/imagenet_fixture.py`; synthetic vectors are never claimed
as actual Qwen output. Python 3.10 grammar is checked separately. See the dated
validation receipt and publication path map for exact bytes.
