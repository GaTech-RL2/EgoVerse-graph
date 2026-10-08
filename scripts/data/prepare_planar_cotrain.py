"""Freeze mixed-alignment Planar corpora and decode images once, on CPU.

The input specification selects source collections. No source object is changed.
Downloads use inventory ETags; the resulting cache and manifest are checksummed.
"""
from __future__ import annotations

import concurrent.futures as cf
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
import zipfile

import boto3
from botocore.config import Config
import numpy as np
import simplejpeg
import zarr


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def infer_observation_alignment(actions, state, attrs, *, speed, fps):
    """Disambiguate legacy imports using informative free-motion transitions.

    Contact frames need not fit the ideal kinematic update and are ignored.
    A pre-step row predicts s[t+1] with a[t]; a post-step row uses a[t+1].
    Equal-command predictions carry no timing information and are excluded.
    """
    init = attrs["episode_init"]
    init = json.loads(init) if isinstance(init, str) else init
    maximum_step = speed / fps

    def predict(start, target):
        delta = target - start
        length = np.linalg.norm(delta, axis=-1, keepdims=True)
        return start + delta * np.minimum(1.0, maximum_step / np.maximum(length, 1e-12))

    pre = predict(state[:-1, :2], actions[:-1, :2])
    post = predict(state[:-1, :2], actions[1:, :2])
    informative = np.max(np.abs(pre - post), axis=-1) > 1e-3
    pre_fits = int(np.sum(informative & (np.max(np.abs(pre - state[1:, :2]), axis=-1) < 1e-5)))
    post_fits = int(np.sum(informative & (np.max(np.abs(post - state[1:, :2]), axis=-1) < 1e-5)))
    reset_error = float(np.max(np.abs(state[0, :2] - np.asarray(init["agent_pos"]))))
    declared = attrs.get("observation_alignment")
    evidence = {"declared": declared, "informative_transitions": int(informative.sum()),
        "pre_step_exact_matches": pre_fits, "post_step_exact_matches": post_fits,
        "reset_xy_error": reset_error, "speed": speed, "fps": fps}
    pre_supported = pre_fits >= 3 and pre_fits >= 10 * post_fits
    post_supported = post_fits >= 3 and post_fits >= 10 * pre_fits
    if declared == "pre_step" and (pre_supported or (reset_error < 1e-5 and post_fits < 3)):
        return "pre_step", evidence
    if declared == "post_step" and post_supported:
        return "post_step", evidence
    if declared is None:
        if pre_supported:
            return "pre_step", evidence
        if post_supported:
            return "post_step", evidence
    raise ValueError("Ambiguous or contradictory observation alignment: " + json.dumps(evidence))


def main(spec):
    root = Path(spec["work_dir"])
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "cache"
    cache.mkdir(exist_ok=True)
    bucket, prefix = spec["bucket"], spec["output_prefix"]
    client = boto3.client("s3", endpoint_url=os.environ["R2_ENDPOINT_URL"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"], region_name="auto",
        config=Config(max_pool_connections=128, retries={"max_attempts": 8}))

    def save(name, value):
        raw = json_bytes(value)
        (root / name).write_bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()
        client.put_object(Bucket=bucket, Key=prefix + name, Body=raw,
                          Metadata={"sha256": sha})
        return {"key": prefix + name, "sha256": sha, "bytes": len(raw)}

    if client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1).get("KeyCount"):
        raise FileExistsError("Output prefix already exists; select a fresh attempt")
    save("STARTED.json", {"time": time.time(), "spec": spec,
                           "script_sha256": digest(__file__)})
    try:
        original = spec["original_archive"]
        archive = root / "original.zip"
        client.download_file(bucket, original["key"], str(archive))
        assert digest(archive) == original["sha256"], "Original archive changed"
        with zipfile.ZipFile(archive) as stream:
            for name in stream.namelist():
                assert not Path(name).is_absolute() and ".." not in Path(name).parts
            stream.extractall(root / "original")
        raw = client.get_object(Bucket=bucket, Key=spec["original_manifest"]["key"])["Body"].read()
        assert hashlib.sha256(raw).hexdigest() == spec["original_manifest"]["sha256"]
        old = json.loads(raw)["episodes"]
        rows = []
        present = set()
        for e in old:
            source = ("usocket" if e["folder"] == "usocket" else
                      "chain_obstacle" if e["obstacle_level"] else "chain_clean")
            rows.append({"source": source, "source_prefix": e["source_prefix"],
                         "path": str(root / "original" / e["folder"] / (e["episode_id"] + ".zarr")),
                         "observation_alignment": e["observation_alignment"],
                         "original_content_sha256": e["sha256"]})
            present.add(e["source_prefix"])
        assert Counter(r["source"] for r in rows) == {"usocket": 2999, "chain_clean": 3000, "chain_obstacle": 1920}
        inventory = []
        for source, collection in spec["additional_sources"].items():
            episodes = {}
            for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=collection["prefix"]):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if ".zarr/" not in key:
                        continue
                    ep = key.split(".zarr/", 1)[0] + ".zarr/"
                    episodes.setdefault(ep, [])
                    if ep not in present:
                        row = {"key": key, "etag": obj["ETag"], "bytes": obj["Size"],
                               "episode_prefix": ep, "source": source}
                        episodes[ep].append(row)
            assert len(episodes) == collection["expected_episodes"], (source, len(episodes))
            for ep, objects in sorted(episodes.items()):
                if ep in present:
                    continue
                local_id = source + "-" + hashlib.sha256(ep.encode()).hexdigest()[:20]
                dest = root / "additional" / (local_id + ".zarr")
                rows.append({"source": source, "source_prefix": ep, "path": str(dest),
                             "observation_alignment": collection["observation_alignment"]})
                for obj in objects:
                    obj["local"] = str(dest / obj["key"][len(ep):])
                    inventory.append(obj)
        assert Counter(r["source"] for r in rows) == spec["expected_counts"]
        save("inventory.json", inventory)
        print("INVENTORY_PINNED", len(rows), len(inventory), flush=True)

        def download(obj):
            path = Path(obj["local"])
            path.parent.mkdir(parents=True, exist_ok=True)
            body = client.get_object(Bucket=bucket, Key=obj["key"], IfMatch=obj["etag"])["Body"]
            h = hashlib.sha256()
            with path.open("wb") as f:
                for part in iter(lambda: body.read(1024 * 1024), b""):
                    f.write(part); h.update(part)
            body.close()
            assert path.stat().st_size == obj["bytes"]
            return {**obj, "sha256": h.hexdigest()}

        downloads = []
        with cf.ThreadPoolExecutor(max_workers=64) as pool:
            for n, result in enumerate(pool.map(download, inventory), 1):
                downloads.append(result)
                if n % 5000 == 0:
                    print("DOWNLOADED", n, len(inventory), flush=True)
                    save("PROGRESS.json", {"phase": "download", "done": n, "total": len(inventory)})
        receipt = save("download-receipts.json", downloads)

        def decode(row):
            group = zarr.open_group(row["path"], mode="r")
            attrs = dict(group.attrs)
            frames = int(attrs["total_frames"])
            assert frames > 1 and int(attrs["fps"]) == spec["fps"]
            actions = np.asarray(group["actions"][:frames])
            state = np.asarray(group["observations.state"][:frames])
            assert actions.shape == (frames, 3 if row["source"] == "usocket" else 4)
            assert state.shape == (frames, 6)
            assert np.isfinite(actions).all() and np.isfinite(state).all()
            init = attrs["episode_init"]
            init = json.loads(init) if isinstance(init, str) else init
            inferred = row["observation_alignment"] == "infer"
            if inferred:
                alignment, alignment_evidence = infer_observation_alignment(actions, state, attrs,
                    speed=spec["kinematic_pusher_speed"], fps=spec["fps"])
            else:
                alignment = attrs.get("observation_alignment", row["observation_alignment"])
                assert alignment == row["observation_alignment"], (row["source_prefix"], alignment)
                alignment_evidence = {"source": "previously audited collection", "declared": attrs.get("observation_alignment")}
            if alignment == "pre_step" and not inferred:
                error = float(np.max(np.abs(state[0, :2] - np.asarray(init["agent_pos"]))))
                assert error < 1e-5, (row["source_prefix"], "pre-step reset mismatch", error)
            if alignment == "post_step":
                cmd = np.asarray(group["observations.pusher_cmd_pose"][:frames])
                assert np.max(np.abs(cmd - actions[:, :3])) < 1e-8
            eid = row["source"] + "-" + hashlib.sha256(row["source_prefix"].encode()).hexdigest()[:20]
            output = cache / eid
            output.mkdir(exist_ok=True)
            arrays = {"actions": actions, "observations.state": state}
            for name, array in arrays.items():
                np.save(output / (name + ".npy"), array, allow_pickle=False)
            jpeg = group["observations.images.front_img_1"][:frames]
            images = np.stack([simplejpeg.decode_jpeg(bytes(v), colorspace="RGB") for v in jpeg])
            assert images.shape == (frames, 96, 96, 3)
            np.save(output / "observations.images.front_img_1.npy", images, allow_pickle=False)
            metadata = {"source": row["source"], "source_prefix": row["source_prefix"], "episode_id": eid,
                "total_frames": frames, "fps": spec["fps"], "observation_alignment": alignment,
                "action_target_offset_obs2": 1 if alignment == "pre_step" else 2,
                "alignment_evidence": alignment_evidence,
                "embodiment": "pushshapes_sim_u_socket" if row["source"] == "usocket" else "pushshapes_sim_chain_gripper",
                "obstacle_level": int(init.get("obstacle_level", 0)), "episode_init": init,
                "generation_source": init.get("generation", {}).get("source_episode"),
                "actions_sha256": hashlib.sha256(actions.astype("<f8").tobytes()).hexdigest(),
                "original_content_sha256": row.get("original_content_sha256"),
                "arrays": {p.stem: {"sha256": digest(p), "bytes": p.stat().st_size} for p in output.glob("*.npy")}}
            initial = {k: init.get(k) for k in ("agent_pos", "agent_angle", "object_pose", "goal_pose", "obstacles")}
            metadata["initial_state_sha256"] = hashlib.sha256(json_bytes(initial)).hexdigest()
            (output / "metadata.json").write_bytes(json_bytes(metadata))
            return metadata

        recovered = {}
        if spec.get("resume_cache"):
            resume = spec["resume_cache"]
            previous = json.loads(client.get_object(Bucket=bucket, Key=resume["prefix"] + "STARTED.json")["Body"].read())
            assert previous["script_sha256"] == resume["script_sha256"]
            assert previous["spec"]["original_archive"] == spec["original_archive"]
            objects = [o for page in client.get_paginator("list_objects_v2").paginate(
                Bucket=bucket, Prefix=resume["prefix"] + "shards/") for o in page.get("Contents", [])]

            def recover(obj):
                index = int(Path(obj["Key"]).name.removeprefix("cache-").removesuffix(".tar.gz"))
                expected = rows[index*128:(index+1)*128]
                # Re-audit every newer gen episode; only the previously proven
                # clean/original-obstacle sources can reuse cached alignment.
                if not expected or any(e["source"] == "chain_gen" for e in expected):
                    return None
                path = root / ("recover-" + Path(obj["Key"]).name)
                client.download_file(bucket, obj["Key"], str(path))
                sha = Path(obj["Key"]).parent.name
                assert digest(path) == sha
                block = []
                with tarfile.open(path, "r|gz") as tar:
                    for member in tar:
                        if member.name.endswith("/metadata.json"):
                            block.append(json.load(tar.extractfile(member)))
                path.unlink()
                assert {e["source_prefix"] for e in block} == {e["source_prefix"] for e in expected}
                shard = {"key": obj["Key"], "sha256": sha, "bytes": obj["Size"],
                         "episodes": [e["episode_id"] for e in block]}
                return index, block, shard

            with cf.ThreadPoolExecutor(max_workers=8) as pool:
                for result in pool.map(recover, objects):
                    if result is not None:
                        index, block, shard = result
                        recovered[index] = (block, shard)
            print("RECOVERED_VERIFIED_CACHE_SHARDS", len(recovered), flush=True)
        episodes, shards = [], []
        for start in range(0, len(rows), 128):
            if start // 128 in recovered:
                block, shard = recovered[start // 128]
                episodes.extend(block); shards.append(shard)
                continue
            with cf.ThreadPoolExecutor(max_workers=8) as pool:
                block = list(pool.map(decode, rows[start:start + 128]))
            path = root / f"cache-{start // 128:04d}.tar.gz"
            with tarfile.open(path, "w:gz", compresslevel=1) as tar:
                for e in block:
                    tar.add(cache / e["episode_id"], arcname=e["episode_id"])
            sha = digest(path)
            key = prefix + "shards/" + sha + "/" + path.name
            client.upload_file(str(path), bucket, key, ExtraArgs={"Metadata": {"sha256": sha}})
            shard = {"key": key, "sha256": sha, "bytes": path.stat().st_size,
                     "episodes": [e["episode_id"] for e in block]}
            shards.append(shard)
            save(f"CACHE_SHARD_{start // 128:04d}.json", {"episodes": block, "archive": shard})
            episodes.extend(block)
            path.unlink()
            for e in block:
                shutil.rmtree(cache / e["episode_id"])
            print("CACHE_SHARD", len(episodes), len(rows), flush=True)
            save("PROGRESS.json", {"phase": "decode", "done": len(episodes), "total": len(rows)})

        # Keep byte-identical trajectories, shared initial states, and known
        # generation families in one split, even across collection boundaries.
        parent = list(range(len(episodes)))
        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]; i = parent[i]
            return i
        keys = {}
        for i, e in enumerate(episodes):
            for tag in ("actions_sha256", "initial_state_sha256"):
                key = (e["embodiment"], tag, e[tag])
                if key in keys:
                    parent[find(i)] = find(keys[key])
                else:
                    keys[key] = i
        by_name = {}
        for i, e in enumerate(episodes):
            name = Path(e["source_prefix"].rstrip("/")).stem
            by_name.setdefault((e["embodiment"], name), []).append(i)
        for i, e in enumerate(episodes):
            if e["generation_source"]:
                name = Path(str(e["generation_source"])).stem
                for j in by_name.get((e["embodiment"], name), []):
                    parent[find(i)] = find(j)
        families = {}
        for i, e in enumerate(episodes):
            families.setdefault(find(i), []).append(e)
        for family in families.values():
            identity = min(e["source_prefix"] for e in family)
            h = hashlib.sha256((str(spec["seed"]) + ":" + identity).encode()).hexdigest()
            split = "valid" if int(h[:8], 16) / 2**32 < spec["valid_ratio"] else "train"
            for e in family:
                e.update(split=split, family_sha256=h)
        counts = dict(Counter(e["source"] for e in episodes))
        split_counts = dict(Counter(e["embodiment"] + "/" + e["split"] for e in episodes))
        assert all(split_counts.get(d + "/" + s, 0) > 0 for d in
                   ("pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper") for s in ("train", "valid"))
        manifest = save("manifest.json", {"version": 1, "episodes": episodes, "shards": shards,
            "spec": spec, "source_downloads": receipt, "counts": counts, "split_counts": split_counts,
            "split_rule": "hash families connected by actions, initial states or known generation parent; seed 42"})
        save("READY.json", {"status": "PASS", "manifest": manifest, "counts": counts,
                            "split_counts": split_counts, "episodes": len(episodes), "shards": len(shards)})
        print("DATA_READY", len(episodes), flush=True)
    except BaseException as e:
        save("FAILED.json", {"error_type": type(e).__name__, "error": str(e), "time": time.time()})
        raise


if __name__ == "__main__":
    main(json.loads(Path(sys.argv[1]).read_text()))
