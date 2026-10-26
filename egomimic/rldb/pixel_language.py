"""Verified, immutable task-language features for prepared pixel datasets."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

LANGUAGE_FEATURE_DIMENSION = 1024
LANGUAGE_FEATURE_SCHEMA = "pixel-task-language-features-v1"
LANGUAGE_VECTOR_CACHE_SCHEMA = "pixel-language-vector-cache-v1"


def validate_language_feature_options(path, sha256):
    """Keep the opt-in explicit: neither an unbound file nor a digest alone."""
    if (path is None) != (sha256 is None):
        raise ValueError(
            "language_features_path and language_features_sha256 require both or neither"
        )
    if path is not None and (
        not isinstance(path, (str, Path))
        or not str(path).strip()
        or not isinstance(sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", sha256) is None
    ):
        raise ValueError("Language features require a path and exact lowercase SHA256")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate language sidecar JSON key: {key}")
        result[key] = value
    return result


def _nonfinite_json(value):
    raise ValueError(f"Nonfinite language sidecar JSON value: {value}")


@dataclass(frozen=True)
class PixelTaskLanguageFeatures:
    """Loaded features plus the complete small identity used by checkpoints."""

    features: dict[str, torch.Tensor]
    instructions: dict[str, str]
    identity: dict

    def context_identity(self):
        return copy.deepcopy(self.identity)

    def tensor(self, task_id):
        """Return a private FP32 feature for a real bound task."""
        return self.features[task_id].clone()

    def sample(self, task_id):
        # A consumer must not be able to mutate the shared feature table.
        return dict(
            language_features=self.tensor(task_id),
            language_instruction=self.instructions[task_id],
            task_id=task_id,
        )


def _load_language_features(path, sha256, manifest=None, manifest_sha256=None, *, vector_cache=False):
    """Bind every task in both splits to finite, unit-normalized frozen features.

    The sidecar never rewrites the prepared manifest or caches. Encoder metadata
    is retained in full, including any additional generation provenance.
    """
    validate_language_feature_options(path, sha256)
    if path is None:
        return None
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("Language features SHA256 mismatch")
    sidecar = json.loads(
        raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite_json
    )
    if (
        not isinstance(sidecar, dict)
        or sidecar.get("schema") != (LANGUAGE_VECTOR_CACHE_SCHEMA if vector_cache else LANGUAGE_FEATURE_SCHEMA)
    ):
        raise ValueError("Unsupported pixel task-language feature schema")
    if vector_cache:
        if (sidecar.get("data_admitted") is not False or
                re.fullmatch(r"[0-9a-f]{64}", str(sidecar.get("source_catalog_sha256"))) is None):
            raise ValueError("Vector cache requires an exact catalog identity and cannot admit data")
    elif sidecar.get("prepared_manifest_sha256") != manifest_sha256:
        raise ValueError("Language features prepared manifest SHA256 mismatch")
    encoder = sidecar.get("encoder")
    if (
        not isinstance(encoder, dict)
        or any(
            not isinstance(encoder.get(key), str) or not encoder[key].strip()
            for key in ("model_id", "revision", "tokenizer_revision", "pooling")
        )
        or type(encoder.get("dimension")) is not int
        or encoder["dimension"] != LANGUAGE_FEATURE_DIMENSION
        or encoder.get("frozen") is not True
        or encoder.get("normalization") != "l2"
    ):
        raise ValueError(
            "Language features require exact frozen encoder provenance and 1024D L2 normalization"
        )
    episodes = ([dict(task_id=tid) for tid in sidecar.get("tasks", {})]
                if vector_cache else manifest.get("episodes", []))
    if not episodes or any(
        not isinstance(ep.get("task_id"), str) or not ep["task_id"].strip()
        for ep in episodes
    ):
        raise ValueError("Every prepared episode requires an explicit language task_id")
    tasks = sidecar.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != {ep["task_id"] for ep in episodes}:
        raise ValueError(
            "Language feature tasks must exactly cover all prepared episodes in both splits"
        )
    features, instructions = {}, {}
    for task_id, item in tasks.items():
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("instruction"), str)
            or not item["instruction"].strip()
        ):
            raise ValueError(f"Task {task_id} requires a nonempty language instruction")
        values = item.get("features")
        if (
            not isinstance(values, list)
            or len(values) != LANGUAGE_FEATURE_DIMENSION
            or any(type(value) not in (int, float) for value in values)
        ):
            raise ValueError(
                f"Task {task_id} requires exactly 1024 numeric language features"
            )
        try:
            array = np.asarray(values, dtype=np.float64)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(
                f"Task {task_id} language features must be finite numbers"
            ) from error
        if not np.isfinite(array).all() or not np.isclose(
            np.linalg.norm(array), 1.0, rtol=0, atol=1e-4
        ):
            raise ValueError(
                f"Task {task_id} language features must be finite and unit-normalized"
            )
        features[task_id] = torch.from_numpy(array.astype(np.float32))
        instructions[task_id] = item["instruction"]
    return PixelTaskLanguageFeatures(
        features=features,
        instructions=instructions,
        identity=dict(
            schema=sidecar["schema"],
            sha256=sha256,
            source_catalog_sha256=sidecar.get("source_catalog_sha256"),
            prepared_manifest_sha256=manifest_sha256,
            encoder=copy.deepcopy(encoder),
            tasks=copy.deepcopy(instructions),
        ),
    )


def load_pixel_task_language_features(path, sha256, manifest, manifest_sha256):
    """Admit only exact prepared-manifest-bound language sidecars."""
    return _load_language_features(path, sha256, manifest, manifest_sha256)


def load_reusable_language_features(path, sha256):
    """Read an SHA-bound text/encoder cache without admitting dataset rows.

    Frozen text vectors depend on exact text and encoder identity, not on the
    physical episode split. Final dataset sidecars are rebound separately.
    """
    validate_language_feature_options(path, sha256)
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("Reusable language cache SHA256 mismatch")
    value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite_json)
    if value.get("schema") == LANGUAGE_VECTOR_CACHE_SCHEMA:
        return _load_language_features(path, sha256, vector_cache=True)
    bound = value.get("prepared_manifest_sha256")
    if re.fullmatch(r"[0-9a-f]{64}", str(bound)) is None:
        raise ValueError("Reusable sidecar has no valid original manifest identity")
    manifest = dict(episodes=[dict(task_id=tid) for tid in value.get("tasks", {})])
    return load_pixel_task_language_features(path, sha256, manifest, bound)


def collate_pixel_language(samples):
    """Language is an all-or-none batch contract, with explicit provenance."""
    keys = {"language_features", "language_instruction", "task_id"}
    if not any(keys.intersection(sample) for sample in samples):
        return {}
    if any(not keys.issubset(sample) for sample in samples):
        raise ValueError(
            "Language features and provenance must be present for every sample"
        )
    for sample in samples:
        features = sample["language_features"]
        if (
            not isinstance(features, torch.Tensor)
            or features.dtype != torch.float32
            or features.shape != (LANGUAGE_FEATURE_DIMENSION,)
            or not torch.isfinite(features).all()
            or any(
                not isinstance(sample[key], str) or not sample[key].strip()
                for key in ("language_instruction", "task_id")
            )
        ):
            raise ValueError("Invalid per-sample language features or provenance")
    return dict(
        language_features=torch.stack(
            [sample["language_features"] for sample in samples]
        ),
        language_instruction=[sample["language_instruction"] for sample in samples],
        task_id=[sample["task_id"] for sample in samples],
    )


QWEN_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
QWEN_REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"


def encode_subtask_language(manifest_path, index_path, index_sha256, output_path, *, cache_dir, device="cpu", reuse_features_path=None, reuse_features_sha256=None):
    """Encode reviewed subtask text using the unchanged native Qwen pooler.

    Exact revisions and weights are cached once. Existing verified outputs take
    the fast path without loading the model or redownloading its weights.
    """
    from egomimic.rldb.pixel_subtasks import load_subtask_manifest
    manifest_raw = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_raw)
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    units = load_subtask_manifest(index_path, index_sha256, manifest, manifest_sha)
    instructions = {}
    for spans in units.values():
        for span in spans:
            tid, text = span["task_id"], span["instruction"]
            if tid in instructions and instructions[tid] != text:
                raise ValueError("Task ID aliases different subtask instructions")
            instructions[tid] = text
    return _encode_instructions(instructions, output_path,
        binding=dict(schema=LANGUAGE_FEATURE_SCHEMA, prepared_manifest_sha256=manifest_sha,
                     subtask_index_sha256=index_sha256),
        cache_dir=cache_dir, device=device, reuse_features_path=reuse_features_path,
        reuse_features_sha256=reuse_features_sha256)


def encode_instruction_catalog(catalog_path, catalog_sha256, output_path, *, cache_dir,
                               device="cpu", reuse_features_path=None, reuse_features_sha256=None):
    """Precompute exact text vectors; this never admits a training window."""
    validate_language_feature_options(catalog_path, catalog_sha256)
    raw=Path(catalog_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != catalog_sha256:
        raise ValueError("Instruction catalog SHA256 mismatch")
    catalog=json.loads(raw,object_pairs_hook=_unique_object,parse_constant=_nonfinite_json)
    tasks=catalog.get("tasks")
    if (catalog.get("schema") != "pixel-instruction-catalog-v1" or
            catalog.get("data_admitted") is not False or catalog.get("clock_verified") is not False or
            not isinstance(tasks,dict) or not tasks or
            any(not isinstance(k,str) or not k.strip() or not isinstance(v,str) or not v.strip()
                for k,v in tasks.items())):
        raise ValueError("Expected a non-admitting exact instruction catalog")
    return _encode_instructions(tasks,output_path,
        binding=dict(schema=LANGUAGE_VECTOR_CACHE_SCHEMA,source_catalog_sha256=catalog_sha256,
                     data_admitted=False,clock_verified=False),
        cache_dir=cache_dir,device=device,reuse_features_path=reuse_features_path,
        reuse_features_sha256=reuse_features_sha256)


def _encode_instructions(instructions, output_path, *, binding, cache_dir, device,
                         reuse_features_path=None, reuse_features_sha256=None):
    import importlib.metadata
    from huggingface_hub import snapshot_download
    from egomimic.scripts.embedding_process.qwen3_embedding import Qwen3TextEmbedding
    language_manifest = dict(episodes=[dict(task_id=tid) for tid in instructions])
    def validate_output(digest):
        if binding["schema"] == LANGUAGE_VECTOR_CACHE_SCHEMA:
            cached = _load_language_features(output, digest, vector_cache=True)
            if cached.identity["source_catalog_sha256"] != binding["source_catalog_sha256"]:
                raise ValueError("Existing vector cache has another catalog identity")
            return cached
        return load_pixel_task_language_features(output, digest, language_manifest, binding["prepared_manifest_sha256"])
    output = Path(output_path)
    if output.exists():
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        cached = validate_output(digest)
        if cached.instructions != instructions:
            raise ValueError("Existing language features differ from reviewed instructions")
        if cached.context_identity()["encoder"]["revision"] != QWEN_REVISION:
            raise ValueError("Existing encoder output has a different revision")
        return dict(status="VERIFIED_EXISTING",path=str(output),sha256=digest,tasks=len(instructions))
    reused = {}
    reuse_encoder = None
    if reuse_features_path is not None or reuse_features_sha256 is not None:
        validate_language_feature_options(reuse_features_path,reuse_features_sha256)
        cached=load_reusable_language_features(reuse_features_path,reuse_features_sha256)
        reuse_encoder=cached.context_identity()['encoder']
        expected = dict(model_id=QWEN_MODEL_ID, revision=QWEN_REVISION,
                        tokenizer_revision=QWEN_REVISION, pooling="last_nonpadding_token",
                        dimension=1024, frozen=True, normalization="l2", dtype="float32", max_length=128)
        if any(reuse_encoder.get(k) != v for k,v in expected.items()):
            raise ValueError('Reuse requires the same exact native encoder contract')
        reused={tid:cached.features[tid].numpy() for tid,text in instructions.items()
                if cached.instructions.get(tid)==text}
    missing_ids=sorted(set(instructions)-set(reused))
    snapshot = Path(snapshot_download(repo_id=QWEN_MODEL_ID,revision=QWEN_REVISION,
        cache_dir=cache_dir,local_files_only=not bool(missing_ids),allow_patterns=["config.json","tokenizer*","special_tokens_map.json",
                                            "vocab.json","merges.txt","model*.safetensors"]))
    weights = sorted(snapshot.glob("*.safetensors"))
    if not weights:
        raise ValueError("Pinned Qwen snapshot has no safetensors weights")
    proof_path = Path(cache_dir)/"qwen-weights-proof.json"
    weight_identities=[]
    if proof_path.exists():
        proof=json.loads(proof_path.read_text())
        if proof.get("revision") != QWEN_REVISION:
            raise ValueError("Cached Qwen proof revision mismatch")
        weight_identities=proof["weights"]
        if [(x["name"],x["bytes"]) for x in weight_identities] != [(x.name,x.stat().st_size) for x in weights]:
            raise ValueError("Cached Qwen weight identity mismatch")
    else:
        for weight in weights:
            digest=hashlib.sha256()
            with weight.open("rb") as handle:
                for chunk in iter(lambda:handle.read(8*1024*1024),b""):
                    digest.update(chunk)
            weight_identities.append(dict(name=weight.name,bytes=weight.stat().st_size,sha256=digest.hexdigest()))
        with proof_path.open("x") as handle:
            json.dump(dict(revision=QWEN_REVISION,weights=weight_identities),handle,indent=2)
    weights_digest = (weight_identities[0]["sha256"] if len(weight_identities)==1 else
                      hashlib.sha256(json.dumps(weight_identities,sort_keys=True).encode()).hexdigest())
    if reuse_encoder is not None and reuse_encoder.get("weights_sha256") != weights_digest:
        raise ValueError("Reusable language vectors use different exact encoder weights")
    task_ids=sorted(instructions)
    if missing_ids:
        embedder=Qwen3TextEmbedding(input_keys=["annotations"],model_name=str(snapshot),
            batch_size=4,max_length=128,device=device,dtype=torch.float32)
        embedder.setup()
        embedder.model.requires_grad_(False)
        new_vectors=embedder._embed_texts([instructions[tid] for tid in missing_ids])
    else:
        new_vectors=np.empty((0,1024),np.float32)
    by_id=dict(reused);by_id.update(zip(missing_ids,new_vectors))
    vectors=np.stack([by_id[tid] for tid in task_ids])
    if vectors.shape != (len(task_ids),1024) or not np.isfinite(vectors).all():
        raise ValueError("Native Qwen embedding output shape/finiteness failure")
    value=dict(**binding,
        feature_reuse=dict(source_sha256=reuse_features_sha256,reused_tasks=len(reused),new_tasks=len(missing_ids)),
        encoder=dict(model_id=QWEN_MODEL_ID,revision=QWEN_REVISION,tokenizer_revision=QWEN_REVISION,
            pooling="last_nonpadding_token",dimension=1024,frozen=True,normalization="l2",
            dtype="float32",max_length=128,weights_sha256=weights_digest,weights=weight_identities,
            torch_version=torch.__version__,transformers_version=importlib.metadata.version("transformers")),
        tasks={tid:dict(instruction=instructions[tid],features=vec.tolist()) for tid,vec in zip(task_ids,vectors)})
    with output.open("x") as handle:
        json.dump(value,handle,indent=2)
        handle.write("\n")
    digest=hashlib.sha256(output.read_bytes()).hexdigest()
    validate_output(digest)
    return dict(status="VERIFIED_NEW",path=str(output),sha256=digest,tasks=len(task_ids))


def main():
    import argparse
    parser=argparse.ArgumentParser(description="Native immutable subtask language features")
    parser.add_argument("--prepared-manifest")
    parser.add_argument("--instruction-catalog")
    parser.add_argument("--catalog-sha256")
    parser.add_argument("--subtask-index")
    parser.add_argument("--subtask-sha256")
    parser.add_argument("--output",required=True)
    parser.add_argument("--encoder-cache",required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--reuse-features")
    parser.add_argument("--reuse-features-sha256")
    args=parser.parse_args()
    if args.instruction_catalog:
        if args.prepared_manifest or args.subtask_index or args.subtask_sha256 or not args.catalog_sha256:
            parser.error("Catalog mode requires --catalog-sha256 and excludes prepared-data arguments")
        print(json.dumps(encode_instruction_catalog(args.instruction_catalog,args.catalog_sha256,
            args.output,cache_dir=args.encoder_cache,device=args.device,
            reuse_features_path=args.reuse_features,reuse_features_sha256=args.reuse_features_sha256)))
        return
    if not all((args.prepared_manifest,args.subtask_index,args.subtask_sha256)) or args.catalog_sha256:
        parser.error("Prepared mode requires manifest, subtask index and SHA")
    print(json.dumps(encode_subtask_language(args.prepared_manifest,args.subtask_index,
        args.subtask_sha256,args.output,cache_dir=args.encoder_cache,device=args.device,reuse_features_path=args.reuse_features,reuse_features_sha256=args.reuse_features_sha256)))


if __name__ == "__main__":
    main()
