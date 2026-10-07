"""Isolated native-state binding candidate; no imports of corpus/model code.

Call from LiberoActionFlowNormalizer's maintained saved-state branch only after
source/module and cached dataset/split receipt identities are independently
validated. Preserve rejection of generic precomputed_norm_path. Binding must
replace fitting, never fall through to inference after a verification failure.
"""
import copy
import hashlib
import json
import math
from pathlib import Path


def load_saved_native_state(path, *, expected_file_sha256, expected_context,
                            expected_shapes, expected_key_types, expected_zarr_keys):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_file_sha256:
        raise ValueError("saved native normalization file identity mismatch")
    payload = json.loads(raw)
    state = payload["normalizer_state"]
    if state["benchmark_context"] != expected_context:
        raise ValueError("saved native normalization context mismatch")
    if expected_context.get("normalization_scope") != "training_episodes_only":
        raise ValueError("native saved state requires training-only scope")
    if expected_context.get("val_ratio") != 0.01:
        raise ValueError("native saved state requires 1% episode split")
    if state["norm_mode"] != "oat_limits" or state["embodiments"] != [21]:
        raise ValueError("native LIBERO normalization boundary mismatch")
    for name, expected in (("shapes", expected_shapes), ("key_types", expected_key_types),
                           ("zarr_keys", expected_zarr_keys)):
        if state[name] != expected:
            raise ValueError("saved native " + name + " mismatch")
    stats = state["norm_stats"]
    if stats != payload["stats"]:
        raise ValueError("duplicate normalization payload statistics disagree")
    required = {k for k in expected_shapes["21"] if k != "task_uid"}
    if set(stats) != {"21"} or set(stats["21"]) != required:
        raise ValueError("saved native statistic keys mismatch")
    for key, fields in stats["21"].items():
        if set(fields) != {"min", "max", "scale", "offset"}:
            raise ValueError("saved native statistic fields mismatch")
        width = expected_shapes["21"][key][-1]
        for field, values in fields.items():
            if len(values) != width or not all(math.isfinite(v) for v in values):
                raise ValueError("saved native statistic width/finiteness mismatch")
        if any(v <= 0 for v in fields["scale"]) or any(a > b for a,b in zip(fields["min"], fields["max"])):
            raise ValueError("saved native statistic limits invalid")
    result = copy.deepcopy(state)
    for name in ("key_types", "zarr_keys", "shapes", "norm_stats"):
        result[name] = {int(k): v for k,v in result[name].items()}
    return result

def _checked_json(path, expected_sha256):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("native saved-state receipt identity mismatch")
    return json.loads(raw)


def validate_receipt_contract(binding, provenance, dataset, *, source_head):
    required = {"path", "file_sha256", "normalizer_module_sha256", "source_commit",
                "dataset_receipt_path", "dataset_receipt_sha256", "split_receipt_path",
                "split_receipt_sha256", "data_root", "normalizer_target",
                "physical_proof_path", "physical_proof_sha256"}
    if set(binding) != required:
        raise ValueError("native saved-state binding schema mismatch")
    target = "egomimic.rldb.zarr.libero_action_flow.LiberoActionFlowNormalizer"
    if binding["normalizer_target"] != target:
        raise ValueError("native saved-state normalizer target mismatch")
    if source_head != binding["source_commit"] or provenance["source_commit"] != source_head:
        raise ValueError("native saved-state source identity mismatch")
    if binding["file_sha256"] != provenance["normalization_sha256"]:
        raise ValueError("native saved-state normalization provenance mismatch")
    if binding["split_receipt_sha256"] != provenance["split_manifest_sha256"]:
        raise ValueError("native saved-state split provenance mismatch")
    physical = _checked_json(binding["physical_proof_path"],binding["physical_proof_sha256"])
    required_proof={"schema_version", "status", "split_manifest_sha256", "dataset_receipt_sha256", "data_root", "episode_ends_sha256", "episode_count", "frames", "zero_frame_overlap", "complete_frame_coverage", "actual_selected_train_ids", "train_ranges", "valid_ranges"}
    if set(physical)!=required_proof or physical["schema_version"]!=1 or physical["status"]!="PASS":
        raise ValueError("native physical frame proof missing/nonpassing")
    if physical["split_manifest_sha256"]!=binding["split_receipt_sha256"] or physical["dataset_receipt_sha256"]!=binding["dataset_receipt_sha256"]:
        raise ValueError("native physical frame proof identity mismatch")
    if not physical["zero_frame_overlap"] or not physical["complete_frame_coverage"]:
        raise ValueError("native physical frame proof incomplete/overlapping")
    receipt = _checked_json(binding["dataset_receipt_path"],binding["dataset_receipt_sha256"])
    split = _checked_json(binding["split_receipt_path"],binding["split_receipt_sha256"])
    root = str(Path(dataset.resolver.folder_path).resolve())
    if root != str(Path(binding["data_root"]).resolve()) or root != str(Path(receipt["replay_path"]).resolve()):
        raise ValueError("native saved-state dataset path mismatch")
    if dataset.resolver.suite != receipt["suite"]:
        raise ValueError("native saved-state dataset mode/suite mismatch")
    if dataset.split_seed != receipt["split_seed"] or dataset.val_ratio != receipt["valid_ratio"]:
        raise ValueError("native saved-state dataset split mismatch")
    if receipt["dataset_logical_sha256"] != provenance["dataset_sha256"]:
        raise ValueError("native saved-state dataset provenance mismatch")
    train = receipt["train_episode_indices"]; valid = receipt["valid_episode_indices"]
    if set(dataset.datasets) != {f"episode_{index:06d}" for index in train}:
        raise ValueError("native saved-state actual selected training episodes mismatch")
    if physical["data_root"]!=root or physical["episode_count"]!=receipt["episodes"] or physical["frames"]!=receipt["frames"] or physical["actual_selected_train_ids"]!=[f"episode_{i:06d}" for i in train]:
        raise ValueError("native physical frame proof dataset mismatch")
    if len(set(train)) != len(train) or len(set(valid)) != len(valid) or set(train)&set(valid):
        raise ValueError("native saved-state split overlap/duplicates")
    if sorted(train+valid) != list(range(receipt["episodes"])):
        raise ValueError("native saved-state incomplete split coverage")
    required_split = {"schema_version", "artifact_kind", "historical_split_manifest_sha256",
        "serialization_replacement_requires_parent_review", "dataset_receipt_sha256", "dataset_logical_sha256",
        "replay_path", "suite", "episodes", "split_seed", "valid_ratio", "recipe",
        "train_episode_indices", "valid_episode_indices", "train_episode_ids", "valid_episode_ids", "physical_frame_range_proof"}
    if set(split) != required_split or split["schema_version"] != 1 or split["artifact_kind"] != "libero_native_episode_split":
        raise ValueError("native split manifest schema mismatch")
    if split["historical_split_manifest_sha256"] != provenance["historical_split_manifest_sha256"]:
        raise ValueError("historical split provenance lost")
    if split["dataset_receipt_sha256"] != binding["dataset_receipt_sha256"] or split["dataset_logical_sha256"] != receipt["dataset_logical_sha256"]:
        raise ValueError("native split dataset receipt mismatch")
    if str(Path(split["replay_path"]).resolve()) != root or split["suite"] != receipt["suite"] or split["episodes"] != receipt["episodes"]:
        raise ValueError("native split dataset identity mismatch")
    if split["train_episode_ids"] != [f"episode_{i:06d}" for i in train] or split["valid_episode_ids"] != [f"episode_{i:06d}" for i in valid]:
        raise ValueError("native split virtual episode identities mismatch")
    if split["train_episode_indices"] != train or split["valid_episode_indices"] != valid:
        raise ValueError("native saved-state split receipt identities mismatch")
    if split["split_seed"] != dataset.split_seed or split["valid_ratio"] != dataset.val_ratio:
        raise ValueError("native saved-state split receipt contract mismatch")
    if (provenance["train_episode_count"],provenance["valid_episode_count"],provenance["union_episode_count"]) != (len(train),len(valid),receipt["episodes"]):
        raise ValueError("native saved-state episode count provenance mismatch")
    return dict(dataset_sha256=receipt["dataset_logical_sha256"],suite=receipt["suite"],
                split_seed=dataset.split_seed,val_ratio=dataset.val_ratio,
                normalization_scope="training_episodes_only",training_episode_count=len(train),
                validation_episode_count=len(valid))


def bind_saved_native_state(*, norm_stats, dataset, dataset_name, binding, run_provenance):
    import inspect
    import subprocess
    from egomimic.rldb.zarr.libero_action_flow import LiberoActionFlowNormalizer
    if type(norm_stats) is not LiberoActionFlowNormalizer or dataset_name != "libero_panda":
        raise ValueError("native saved-state exact normalizer/source boundary mismatch")
    module = Path(inspect.getfile(LiberoActionFlowNormalizer)).resolve()
    repo = module.parents[3]
    head = subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    dirty = subprocess.check_output(["git","-C",str(repo),"status","--porcelain","--untracked-files=all"],text=True)
    if dirty:
        raise ValueError("native saved-state requires clean immutable source")
    if hashlib.sha256(module.read_bytes()).hexdigest() != binding["normalizer_module_sha256"]:
        raise ValueError("native saved-state normalizer module identity mismatch")
    context=validate_receipt_contract(binding,run_provenance,dataset,source_head=head)
    physical=_checked_json(binding["physical_proof_path"],binding["physical_proof_sha256"])
    if make_physical_split_proof(dataset,binding)!=physical:
        raise ValueError("native physical proof changed at runtime")
    # Derive metadata from actual populated datasets and post-transform batch shapes.
    from egomimic.rldb.zarr.libero_dataset import validation_mask
    receipt=_checked_json(binding["dataset_receipt_path"],binding["dataset_receipt_sha256"])
    valid=validation_mask(receipt["episodes"],dataset.val_ratio,dataset.split_seed)
    if valid.nonzero()[0].tolist()!=receipt["valid_episode_indices"] or (~valid).nonzero()[0].tolist()!=receipt["train_episode_indices"]:
        raise ValueError("native saved-state actual split recipe mismatch")
    metadata=norm_stats.to_state()
    def maps(name):
        return {str(k):{a:list(v) if name=="shapes" else v for a,v in values.items()} for k,values in metadata[name].items()}
    state=load_saved_native_state(binding["path"],expected_file_sha256=binding["file_sha256"],
        expected_context=context,expected_shapes=maps("shapes"),expected_key_types=maps("key_types"),
        expected_zarr_keys=maps("zarr_keys"))
    return LiberoActionFlowNormalizer(state=state,norm_mode=norm_stats.norm_mode)


def make_physical_split_proof(dataset, binding):
    """Attach to scheduled canonical schema/integration CPU job; read only meta ends."""
    import numpy as np
    import zarr
    from egomimic.rldb.zarr.libero_dataset import validation_mask
    r=_checked_json(binding["dataset_receipt_path"],binding["dataset_receipt_sha256"])
    s=_checked_json(binding["split_receipt_path"],binding["split_receipt_sha256"])
    root=str(Path(dataset.resolver.folder_path).resolve())
    if root!=str(Path(r["replay_path"]).resolve()) or root!=str(Path(s["replay_path"]).resolve()):
        raise ValueError("physical proof data root mismatch")
    ends=np.asarray(zarr.open_group(root,mode="r")["meta/episode_ends"][:],dtype=np.int64)
    if ends.ndim!=1 or len(ends)!=r["episodes"] or int(ends[-1])!=r["frames"] or np.any(np.diff(np.r_[0,ends])<=0):
        raise ValueError("physical proof invalid episode ends")
    mask=validation_mask(len(ends),dataset.val_ratio,dataset.split_seed)
    if mask.nonzero()[0].tolist()!=r["valid_episode_indices"] or (~mask).nonzero()[0].tolist()!=r["train_episode_indices"]:
        raise ValueError("physical proof native recipe mismatch")
    if sorted(dataset.datasets)!=s["train_episode_ids"]:
        raise ValueError("physical proof actual selected episode mismatch")
    starts=np.r_[0,ends[:-1]]
    ranges=lambda ids:[[int(i),int(starts[i]),int(ends[i])] for i in ids]
    return dict(schema_version=1,status="PASS",split_manifest_sha256=binding["split_receipt_sha256"],dataset_receipt_sha256=binding["dataset_receipt_sha256"],data_root=root,episode_ends_sha256=hashlib.sha256(ends.tobytes()).hexdigest(),episode_count=len(ends),frames=int(ends[-1]),zero_frame_overlap=True,complete_frame_coverage=True,actual_selected_train_ids=sorted(dataset.datasets),train_ranges=ranges(r["train_episode_indices"]),valid_ranges=ranges(r["valid_episode_indices"]))
