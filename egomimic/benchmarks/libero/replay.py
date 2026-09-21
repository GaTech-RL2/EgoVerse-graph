"""Calibrate ARC R/D/M by replaying demonstrations through LIBERO physics.

This is a codec experiment, not a trained-policy benchmark. Each candidate
starts from the same saved XML/state as raw actions and runs open loop, without
state injection or corrective actions. Chunk cadence matches GraphPolicy.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import h5py
import numpy as np
import yaml

from egomimic.benchmarks.libero.catalog import TASKS
from egomimic.benchmarks.libero.cluster import (
    DATA_REPO,
    DATA_REVISION,
    ArtifactUploader,
    configure_simulator,
    digest,
    download,
    write_json,
)
from egomimic.rldb.zarr.libero_arc import LiberoArcCodec


def candidate_id(candidate):
    def label(value):
        return "full" if value is None else f"{value:g}"

    return (
        f"R{label(candidate['max_rotation_degrees'])}"
        f"_D{label(candidate['max_translation'])}_M{candidate['num_waypoints']}"
    )


def candidates_from_spec(spec):
    candidates = {}
    for rotation, distance, waypoints in itertools.product(
        spec["rotation_degrees"], spec["translation_metres"], spec["waypoints"]
    ):
        candidate = {
            "max_rotation_degrees": rotation,
            "max_translation": distance,
            "num_waypoints": waypoints,
        }
        if waypoints % 4:
            raise ValueError("Policy UNet requires M to be a multiple of four")
        LiberoArcCodec(**candidate, horizon=spec["horizon"])
        candidates[candidate_id(candidate)] = candidate
    return candidates


def validate_spec(spec):
    splits = [
        spec[name + "_demos"] for name in ("calibration", "selection", "confirmation")
    ]
    if any(not rows or len(set(rows)) != len(rows) for rows in splits):
        raise ValueError("Replay splits must be nonempty without duplicate demos")
    if any(set(a) & set(b) for a, b in itertools.combinations(splits, 2)):
        raise ValueError("Replay calibration/selection/confirmation demos overlap")
    if any(not isinstance(i, int) or i < 0 for rows in splits for i in rows):
        raise ValueError("Demo IDs must be nonnegative integers")
    if not 1 <= spec["execute_steps"] <= spec["horizon"] or spec["workers"] < 1:
        raise ValueError("Invalid execution horizon or worker count")
    if spec.get("max_tasks_per_worker", 4) < 1:
        raise ValueError("Worker recycling interval must be positive")
    for key in (
        "minimum_execution_coverage",
        "minimum_raw_success",
        "minimum_retention",
        "maximum_success_rate_drop",
    ):
        if not 0 <= spec[key] <= 1:
            raise ValueError(f"Invalid replay threshold {key}")
    if spec.get("action_dtype") != "float32":
        raise ValueError("Replay must use the training dataset's float32 commands")
    if spec.get("selection_objective", "tokens_then_success") not in {
        "tokens_then_success",
        "success_then_tokens",
    }:
        raise ValueError("Unknown replay selection objective")
    if (
        spec.get("allow_reference_gap", False)
        and spec.get("selection_objective") != "success_then_tokens"
    ):
        raise ValueError(
            "Accepting a reference gap requires prioritizing replay success"
        )
    for key in ("max_dense_action_mse", "max_raw_repeat_state_error"):
        if not np.isfinite(spec[key]) or spec[key] < 0:
            raise ValueError(f"Invalid numerical control threshold {key}")
    candidates_from_spec(spec)


def reconstruct_episode(actions, candidate, spec):
    """Encode future commands, execute only the configured prefix, advance once.

    No cycling shortened chunks faster than the policy, compensating truncation,
    or looking at simulator observations. Episode-end padding matches the dataset.
    """
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions):
        raise ValueError("Expected a nonempty (T,7) demonstration")
    codec = LiberoArcCodec(**candidate, horizon=spec["horizon"], dt=spec["dt"])
    decoded = np.empty_like(actions)
    covered, windows, durations = 0.0, 0, []
    for start in range(0, len(actions), spec["execute_steps"]):
        window = actions[start : start + codec.horizon]
        window = np.pad(window, ((0, codec.horizon - len(window)), (0, 0)), mode="edge")
        tokens = codec.encode(window)
        duration = float(tokens[:-1, 10].sum(dtype=np.float64))
        count = min(spec["execute_steps"], len(actions) - start)
        decoded[start : start + count] = codec.decode(tokens)[:count]
        covered += min(count, duration / codec.dt)
        durations.append(duration)
        windows += 1
    error = decoded - actions
    return decoded, {
        "action_mse": float(np.mean(error**2)),
        "translation_command_mse": float(np.mean(error[:, :3] ** 2)),
        "rotation_command_mse": float(np.mean(error[:, 3:6] ** 2)),
        "gripper_mismatch": float(
            np.mean(np.sign(decoded[:, 6]) != np.sign(actions[:, 6]))
        ),
        "execution_coverage": float(min(1.0, covered / len(actions))),
        "min_duration_seconds": min(durations),
        "waypoints_per_executed_action": windows * codec.num_waypoints / len(actions),
        "action_sha256": hashlib.sha256(decoded.tobytes()).hexdigest(),
    }


def stage_raw_dataset(root, suite, evidence):
    """Use official HDF5s: the released OAT Zarr has no saved simulator states."""
    url = f"https://huggingface.co/api/datasets/{DATA_REPO}/revision/{DATA_REVISION}?blobs=true"
    with urllib.request.urlopen(url, timeout=120) as response:
        metadata = json.load(response)
    if metadata["sha"] != DATA_REVISION:
        raise ValueError("Dataset revision differs")
    sources = {row["rfilename"]: row for row in metadata["siblings"]}
    records = [sources[f"{suite}/{task}_demo.hdf5"] for task in TASKS[suite]]
    cache = os.environ.get("LIBERO_RAW_CACHE") or None
    raw = Path(cache) if cache else Path(root) / "data/raw"

    def fetch(row):
        path = raw / row["rfilename"]
        if cache:
            # A cache is read-only and must match the official LFS receipt.
            # Never repair or overwrite another run's data in this path.
            if not path.is_file():
                raise FileNotFoundError(path)
            if digest(path) != row["lfs"]["sha256"]:
                raise ValueError(f"Cached demonstration checksum differs: {path}")
            return
        url = f"https://huggingface.co/datasets/{DATA_REPO}/resolve/{DATA_REVISION}/{row['rfilename']}"
        for attempt in range(3):
            try:
                download(url, path, row["lfs"]["sha256"])
                return
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(2**attempt)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, records))
    write_json(
        evidence / "data-source.json",
        {
            "repo": DATA_REPO,
            "revision": DATA_REVISION,
            "files": records,
            "verified_read_only_cache": cache,
        },
    )
    return raw / suite


def rebase_demo_xml(xml, robosuite_root, assets_root):
    """Relocate saved asset paths, preserving every model and physics parameter."""
    tree = ET.fromstring(xml)
    for element in tree.iter():
        old = element.get("file")
        if old is None:
            continue
        parts = Path(old).parts
        if "robosuite" in parts:
            index = max(i for i, part in enumerate(parts) if part == "robosuite")
            new = Path(robosuite_root).joinpath(*parts[index + 1 :])
        elif "assets" in parts:
            index = max(i for i, part in enumerate(parts) if part == "assets")
            new = Path(assets_root).joinpath(*parts[index + 1 :])
        else:
            raise ValueError(f"Unrecognized demonstration asset: {old}")
        if not new.is_file():
            raise FileNotFoundError(new)
        element.set("file", str(new))
    return ET.tostring(tree, encoding="unicode")


def demo_task_definition(bddl, xml):
    """Bind the goal to the object version actually saved in this demonstration.

    Some official LIBERO-90 recordings predate the salad-dressing asset rename.
    Keep their entire saved physics model intact, including its geometry, and
    change only the BDDL object identifiers/type used to look up that model.
    """
    bodies = {body.get("name") for body in ET.fromstring(xml).iter("body")}
    aliases = {}
    if (
        "salad_dressing_1_main" in bodies
        and "new_salad_dressing_1_main" not in bodies
        and re.search(r"\bnew_salad_dressing_1\b", bddl)
    ):
        aliases = {
            "new_salad_dressing_1": "salad_dressing_1",
            "new_salad_dressing": "salad_dressing",
        }
        for original, recorded in aliases.items():
            bddl = re.sub(r"\b" + re.escape(original) + r"\b", recorded, bddl)
    return bddl, aliases


@contextmanager
def replay_environment(bddl_path, xml, dt):
    from libero.libero.envs.env_wrapper import ControlEnv

    original = Path(bddl_path).read_text()
    definition, aliases = demo_task_definition(original, xml)
    metadata = {
        "original_bddl_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "replay_bddl_sha256": hashlib.sha256(definition.encode()).hexdigest(),
        "recorded_object_aliases": aliases,
    }
    with tempfile.TemporaryDirectory(prefix="libero-replay-bddl-") as temporary:
        path = Path(temporary) / Path(bddl_path).name
        path.write_text(definition)
        env = ControlEnv(
            bddl_file_name=str(path),
            use_camera_obs=False,
            has_renderer=False,
            has_offscreen_renderer=False,
            control_freq=round(1 / dt),
            ignore_done=True,
        )
        try:
            yield env, metadata
        finally:
            env.close()


def read_demo(path, demo_id):
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        demo = data[f"demo_{demo_id}"]
        source_actions = demo["actions"][:]
        return {
            # Both the released OAT replay and native HDF5 conversion train on
            # float32. Even this cast can change brittle contact outcomes.
            "actions": source_actions.astype(np.float32),
            "source_actions": source_actions,
            "states": demo["states"][:],
            "xml": demo.attrs["model_file"],
            "bddl": Path(data.attrs["bddl_file_name"]).name,
        }


def offline_job(job):
    path, demo_id, candidates, spec = job
    demo = read_demo(path, demo_id)
    return {
        "task": Path(path).stem.removesuffix("_demo"),
        "demo": demo_id,
        "length": len(demo["actions"]),
        "candidates": {
            key: reconstruct_episode(demo["actions"], candidate, spec)[1]
            for key, candidate in candidates.items()
        },
    }


def replay_job(job):
    """Each worker owns its simulator; only initialization injects saved state."""
    path, demo_id, candidates, spec, suite = job
    import robosuite
    from libero.libero import get_libero_path

    demo = read_demo(path, demo_id)
    task = Path(path).stem.removesuffix("_demo")
    xml = rebase_demo_xml(
        demo["xml"], Path(robosuite.__file__).parent, get_libero_path("assets")
    )
    result = {
        "task": task,
        "demo": demo_id,
        "length": len(demo["actions"]),
        "initial_state_sha256": hashlib.sha256(demo["states"][0].tobytes()).hexdigest(),
        "original_actions_sha256": hashlib.sha256(
            demo["source_actions"].tobytes()
        ).hexdigest(),
        "training_actions_sha256": hashlib.sha256(
            demo["actions"].tobytes()
        ).hexdigest(),
        "action_dtype": str(demo["actions"].dtype),
        "source_action_dtype": str(demo["source_actions"].dtype),
        "model_xml_sha256": hashlib.sha256(demo["xml"].encode()).hexdigest(),
        "candidates": {},
    }
    baseline_positions, baseline_states, cache = None, None, {}
    with replay_environment(
        Path(get_libero_path("bddl_files")) / suite / demo["bddl"], xml, spec["dt"]
    ) as (env, task_definition):
        result.update(task_definition)
        controller = env.robots[0].controller
        if not np.allclose(controller.output_max[:6], [0.05] * 3 + [0.5] * 3):
            raise RuntimeError("Unexpected OSC scales")
        if not getattr(controller, "use_delta", True):
            raise RuntimeError("Expected delta OSC controller")
        for key, candidate in candidates.items():
            if candidate is None:
                actions = (
                    demo["source_actions"]
                    if key == "source_precision_raw"
                    else demo["actions"]
                )
                metrics = {
                    "action_mse": float(
                        np.mean(
                            (
                                actions.astype(np.float64)
                                - demo["actions"].astype(np.float64)
                            )
                            ** 2
                        )
                    )
                }
            else:
                actions, metrics = reconstruct_episode(demo["actions"], candidate, spec)
            action_hash = hashlib.sha256(actions.tobytes()).hexdigest()
            # A repeatability control must actually rerun the simulator.
            if key != "raw_repeat" and action_hash in cache:
                previous = cache[action_hash]
                result["candidates"][key] = {
                    **result["candidates"][previous],
                    **metrics,
                    "identical_to": previous,
                }
                continue
            started = time.monotonic()
            # No settling steps after restoring the recorded initial state.
            env.seed(spec["seed"] + demo_id)
            env.reset()
            env.reset_from_xml_string(xml)
            env.sim.reset()
            env.set_init_state(demo["states"][0])
            initialized = env.get_sim_state()
            if not np.array_equal(initialized, demo["states"][0]):
                raise RuntimeError("Simulator did not restore the exact saved state")
            positions, states, first_success = [], [], None
            for step, action in enumerate(actions):
                observation, _, _, _ = env.step(action)
                positions.append(np.asarray(observation["robot0_eef_pos"]).copy())
                states.append(env.get_sim_state().copy())
                if env.check_success() and first_success is None:
                    first_success = step + 1
            positions, states = np.asarray(positions), np.asarray(states)
            if key == "raw":
                baseline_positions, baseline_states = positions, states
            metrics.update(
                {
                    "success": first_success is not None,
                    "final_success": bool(env.check_success()),
                    "first_success_step": first_success,
                    "wall_seconds": time.monotonic() - started,
                    "state_l2_vs_recorded_mean": float(
                        np.linalg.norm(states[:-1] - demo["states"][1:], axis=1).mean()
                    ),
                    "eef_rmse_vs_raw_metres": float(
                        np.sqrt(np.mean((positions - baseline_positions) ** 2))
                    ),
                    "state_l2_vs_raw_mean": float(
                        np.linalg.norm(states - baseline_states, axis=1).mean()
                    ),
                }
            )
            result["candidates"][key] = metrics
            cache[action_hash] = key
    return result


def run_jobs(pool, function, jobs, evidence, phase):
    rows = []
    for row in pool.imap_unordered(function, jobs, chunksize=1):
        rows.append(row)
        write_json(evidence / phase / f"{row['task']}-demo-{row['demo']}.json", row)
        print(
            json.dumps(
                {
                    "phase": phase,
                    "completed": len(rows),
                    "total": len(jobs),
                    "task": row["task"],
                    "demo": row["demo"],
                }
            ),
            flush=True,
        )
        write_json(
            evidence / "status.json",
            {
                "state": phase,
                "completed": len(rows),
                "total": len(jobs),
            },
        )
    return sorted(rows, key=lambda row: (row["task"], row["demo"]))


def summarize(rows, candidates):
    summary = {}
    raw_successes = sum(row["candidates"]["raw"]["success"] for row in rows)
    for key in candidates:
        values = [row["candidates"][key] for row in rows]
        lost = sum(
            row["candidates"]["raw"]["success"]
            and not row["candidates"][key]["success"]
            for row in rows
        )
        summary[key] = {
            "episodes": len(rows),
            "successes": sum(v["success"] for v in values),
            "raw_successes": raw_successes,
            "lost_raw_successes": lost,
            "retention": (raw_successes - lost) / raw_successes if raw_successes else 0,
            "success_rate": float(np.mean([v["success"] for v in values])),
            "action_mse": float(np.mean([v.get("action_mse", 0) for v in values])),
            "eef_rmse_vs_raw_metres": float(
                np.mean([v["eef_rmse_vs_raw_metres"] for v in values])
            ),
            "max_action_mse": max(v.get("action_mse", 0) for v in values),
            "max_state_l2_vs_raw_mean": max(
                v.get("state_l2_vs_raw_mean", 0) for v in values
            ),
            "gained_successes": sum(v["success"] for v in values)
            - (raw_successes - lost),
        }
    return summary


def rank_candidates(summary, candidates, spec):
    eligible = [
        key
        for key, value in summary.items()
        if key in candidates
        and value["retention"] >= spec["minimum_retention"]
        and value["success_rate"]
        >= value["raw_successes"] / value["episodes"]
        - spec["maximum_success_rate_drop"]
    ]
    success_first = spec.get("selection_objective") == "success_then_tokens"
    return sorted(
        eligible,
        key=lambda key: (
            (
                -summary[key]["successes"]
                if success_first
                else candidates[key]["num_waypoints"]
            ),
            (
                candidates[key]["num_waypoints"]
                if success_first
                else -summary[key]["successes"]
            ),
            summary[key]["action_mse"],
            key,
        ),
    )


def validate_controls(summary, spec):
    if (
        summary["raw"]["success_rate"] <= 0
        or summary["raw"]["success_rate"] < spec["minimum_raw_success"]
    ):
        raise RuntimeError(
            "Raw demonstration replay success is too low; investigate reset/simulator before tuning"
        )
    if (
        summary.get("raw_repeat", {}).get("max_state_l2_vs_raw_mean", float("inf"))
        > spec["max_raw_repeat_state_error"]
    ):
        raise RuntimeError(
            "Raw simulator replay is not repeatable from the saved reset"
        )
    # Dynamics can amplify roundoff: validate the dense codec numerically and
    # report its task outcomes, rather than pretending float32 is lossless.
    if (
        summary["dense"].get("max_action_mse", float("inf"))
        > spec["max_dense_action_mse"]
    ):
        raise RuntimeError(
            "Dense ARC command reconstruction exceeds numerical tolerance"
        )


def load_calibration_parent(client, parent, suite, spec, evidence):
    """Reuse completed calibration, never earlier confirmation outcomes."""
    import importlib.metadata

    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,62}", parent):
        raise ValueError("Invalid calibration parent run ID")
    prefix = f"experiments/arc-oat-20260919/{parent}/"
    receipts = {}

    def read(name):
        body = client.get_object(Bucket="rldb", Key=prefix + name)["Body"].read()
        receipts[name] = hashlib.sha256(body).hexdigest()
        return json.loads(body)

    runtime, original_spec, environment, data = (
        read("runtime.json"),
        read("spec.json"),
        read("environment-audit.json"),
        read("data-source.json"),
    )
    if runtime["suite"] != suite or data["revision"] != DATA_REVISION:
        raise ValueError("Calibration parent suite or data revision differs")
    for key in ("horizon", "execute_steps", "dt", "action_dtype", "calibration_demos"):
        if original_spec[key] != spec[key]:
            raise ValueError(f"Calibration parent differs in {key}")
    if environment["codec_sha256"] != digest(
        Path(__file__).parents[2] / "rldb/zarr/libero_arc.py"
    ):
        raise ValueError("Calibration parent uses a different codec")
    for name in ("numpy", "scipy", "mujoco", "robosuite", "libero"):
        if environment["versions"][name] != importlib.metadata.version(name):
            raise ValueError(f"Calibration parent uses a different {name} version")
    screen, summary = read("screen.json"), read("calibration.json")
    validate_controls(summary, spec)
    count = len(TASKS[suite]) * len(spec["calibration_demos"])
    if any(value["episodes"] != count for value in summary.values()):
        raise ValueError("Calibration parent is incomplete")
    allowed = candidates_from_spec(spec)
    eligible = {
        key: value
        for key, value in screen["candidates"].items()
        if key in allowed
        and value == allowed[key]
        and screen["execution_coverage"][key] >= spec["minimum_execution_coverage"]
        and key in summary
    }
    if not eligible:
        raise ValueError("Calibration parent has no matching evaluated candidates")
    write_json(
        evidence / "calibration-parent.json",
        {
            "run_id": parent,
            "runtime": runtime,
            "spec": original_spec,
            "environment": environment,
            "artifact_sha256": receipts,
            "reused_only_calibration": True,
        },
    )
    write_json(
        evidence / "screen.json",
        {**screen, "eligible_for_simulator_replay": list(eligible)},
    )
    write_json(evidence / "calibration.json", summary)
    return eligible, summary


def calibrate(root, suite, spec, evidence, *, calibration_parent=None, client=None):
    # Simulator configuration needs a local data directory even with raw files
    # read directly from a verified external cache.
    (Path(root) / "data").mkdir(parents=True, exist_ok=True)
    raw = stage_raw_dataset(root, suite, evidence)
    configure_simulator(root)
    from egomimic.benchmarks.libero.rollout import verify_libero_installation

    verify_libero_installation()
    candidates = candidates_from_spec(spec)
    controls = {
        "raw": None,
        "raw_repeat": None,
        "source_precision_raw": None,
        "dense": {
            "num_waypoints": spec["horizon"] + 1,
            "max_translation": None,
            "max_rotation_degrees": None,
        },
    }
    previous = {
        "num_waypoints": 16,
        "max_translation": None,
        "max_rotation_degrees": None,
    }

    def jobs(demos, candidates, *, offline=False):
        return [
            (str(raw / f"{task}_demo.hdf5"), demo, candidates, spec)
            + (() if offline else (suite,))
            for task in TASKS[suite]
            for demo in demos
        ]

    # MuJoCo/model allocations can accumulate across many environment resets.
    # Bound each process's lifetime, including long selection/confirmation runs.
    with multiprocessing.get_context("spawn").Pool(
        processes=spec["workers"],
        maxtasksperchild=spec.get("max_tasks_per_worker", 4),
    ) as pool:
        if calibration_parent:
            eligible, summary = load_calibration_parent(
                client, calibration_parent, suite, spec, evidence
            )
            return select_and_confirm(
                pool,
                jobs,
                candidates,
                eligible,
                summary,
                spec,
                evidence,
                suite,
                controls,
                previous,
            )
        baseline = run_jobs(
            pool,
            replay_job,
            jobs(spec["calibration_demos"], controls),
            evidence,
            "controls",
        )
        control_summary = summarize(baseline, controls)
        write_json(evidence / "controls.json", control_summary)
        validate_controls(control_summary, spec)
        offline = run_jobs(
            pool,
            offline_job,
            jobs(spec["calibration_demos"], candidates, offline=True),
            evidence,
            "offline",
        )
        coverage = {
            key: sum(
                row["length"] * row["candidates"][key]["execution_coverage"]
                for row in offline
            )
            / sum(row["length"] for row in offline)
            for key in candidates
        }
        eligible = {
            key: value
            for key, value in candidates.items()
            if coverage[key] >= spec["minimum_execution_coverage"]
        }
        write_json(
            evidence / "screen.json",
            {
                "candidates": candidates,
                "execution_coverage": coverage,
                "eligible_for_simulator_replay": list(eligible),
                "pruned_reason": "Insufficient coverage at the unchanged policy execution cadence",
            },
        )
        calibration = run_jobs(
            pool,
            replay_job,
            jobs(spec["calibration_demos"], {**controls, **eligible}),
            evidence,
            "calibration",
        )
        summary = summarize(calibration, {**controls, **eligible})
        write_json(evidence / "calibration.json", summary)
        validate_controls(summary, spec)
        return select_and_confirm(
            pool,
            jobs,
            candidates,
            eligible,
            summary,
            spec,
            evidence,
            suite,
            controls,
            previous,
        )


def select_and_confirm(
    pool, jobs, candidates, eligible, summary, spec, evidence, suite, controls, previous
):
    ranked = rank_candidates(summary, eligible, spec)
    # Compare the best R/D at each M on the independent selection split.
    shortlist = {}
    for key in ranked:
        if candidates[key]["num_waypoints"] not in {
            value["num_waypoints"] for value in shortlist.values()
        }:
            shortlist[key] = candidates[key]
    if not shortlist:
        raise RuntimeError(
            "No tested candidate matches the raw calibration success criterion"
        )
    selection = run_jobs(
        pool,
        replay_job,
        jobs(spec["selection_demos"], {**controls, **shortlist}),
        evidence,
        "selection",
    )
    summary = summarize(selection, {**controls, **shortlist})
    write_json(evidence / "selection.json", summary)
    validate_controls(summary, spec)
    ranked = rank_candidates(summary, shortlist, spec)
    if not ranked:
        raise RuntimeError(
            "No shortlisted candidate matches the raw selection success criterion"
        )
    selected = ranked[0]
    frozen = {
        "candidate_id": selected,
        "codec": candidates[selected],
        "suite": suite,
        "selected_before_confirmation": True,
        "spec_sha256": digest(evidence / "spec.json"),
        "action_dtype": spec["action_dtype"],
        "objective": spec.get("selection_objective", "tokens_then_success"),
    }
    write_json(evidence / "selected-before-confirmation.json", frozen)
    confirmation_candidates = {
        **controls,
        "previous_M16": previous,
        selected: candidates[selected],
    }
    confirmation = run_jobs(
        pool,
        replay_job,
        jobs(spec["confirmation_demos"], confirmation_candidates),
        evidence,
        "confirmation",
    )
    summary = summarize(confirmation, confirmation_candidates)
    write_json(evidence / "confirmation.json", summary)
    validate_controls(summary, spec)
    confirmed = selected in rank_candidates(
        summary,
        {selected: candidates[selected]},
        {**spec, "maximum_success_rate_drop": 0.0},
    )
    result = {
        **frozen,
        "confirmed": confirmed,
        "confirmation_complete": True,
        "matches_raw_success": summary[selected]["successes"]
        >= summary["raw"]["successes"],
        "selection_objective": spec.get("selection_objective", "tokens_then_success"),
        "confirmation": summary,
        "calibration_candidates": len(candidates),
        "replayed_candidates": len(eligible),
        "task_count": len(TASKS[suite]),
        "codec_only_not_policy_scores": True,
        "best_tested_not_global_optimum": True,
        "training_overrides": [
            f"benchmark.arc_waypoints={candidates[selected]['num_waypoints']}",
            f"benchmark.arc_max_translation={candidates[selected]['max_translation'] if candidates[selected]['max_translation'] is not None else 'null'}",
            f"benchmark.arc_max_rotation_degrees={candidates[selected]['max_rotation_degrees'] if candidates[selected]['max_rotation_degrees'] is not None else 'null'}",
        ],
    }
    write_json(evidence / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--suite", choices=TASKS, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--calibration-parent",
        default=os.environ.get("REPLAY_CALIBRATION_PARENT") or None,
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path(__file__).parents[2]
        / "hydra_configs/benchmark/libero_arc_replay.yaml",
    )
    args = parser.parse_args()
    spec = yaml.safe_load(args.spec.read_text())
    validate_spec(spec)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if commit != os.environ["SOURCE_COMMIT"]:
        raise RuntimeError("Unexpected source revision")
    evidence = args.root / "evidence"
    evidence.mkdir(parents=True, exist_ok=False)
    uploader = ArtifactUploader(evidence, args.run_id)
    write_json(
        evidence / "runtime.json",
        {
            "source_commit": commit,
            "suite": args.suite,
            "python": sys.version,
            "kind": "demonstration_replay",
            "artifact_prefix": "s3://rldb/" + uploader.prefix,
        },
    )
    write_json(evidence / "spec.json", spec)
    write_json(evidence / "status.json", {"state": "STAGING_RAW_DEMONSTRATIONS"})
    uploader.thread.start()
    try:
        result = calibrate(
            args.root,
            args.suite,
            spec,
            evidence,
            calibration_parent=args.calibration_parent,
            client=uploader.client,
        )
        write_json(
            evidence / "status.json",
            {
                "state": "CONFIRMED"
                if result["confirmed"]
                else (
                    "REPLAY_EVALUATED"
                    if spec.get("allow_reference_gap", False)
                    else "NOT_CONFIRMED"
                )
            },
        )
    except Exception as error:
        write_json(evidence / "status.json", {"state": "FAILED", "error": str(error)})
        raise
    finally:
        uploader.stop.set()
        uploader.thread.join()
        uploader.upload(final=True)


if __name__ == "__main__":
    main()
