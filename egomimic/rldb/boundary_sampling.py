"""Opt-in bounded annotation-uncertainty sampling of recorded native frames.

No annotation is rewritten and no neighbor instruction equivalence is asserted.
The legacy sampler supplies K and advances its original RNG before this helper;
the component/pair draw uses a separate checkpointed cursor/slot namespace.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
import math
import numpy as np

POLICY = {
    "schema": "annotation-boundary-uncertainty-sampling-v1",
    "policy": "annotation_boundary_uncertainty",
    "boundary_probability": 0.5,
    "maximum_radius_seconds": 0.25,
    "duration_fraction": 0.1,
    "rng_namespace": 0x424E4459,
}
SCHEMA = "annotation-boundary-uncertainty-realization-v1"


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def validate_boundary_sampling(value):
    if value is None:
        return None
    _need(isinstance(value, Mapping) and dict(value) == POLICY,
          "Require the explicit versioned annotation_boundary_uncertainty policy")
    _need(type(value["boundary_probability"]) is float
          and type(value["maximum_radius_seconds"]) is float
          and type(value["duration_fraction"]) is float
          and type(value["rng_namespace"]) is int,
          "Boundary policy requires exact typed parameters")
    return copy.deepcopy(POLICY)


def _clock(full_time, episode):
    times = np.asarray(full_time, dtype=np.float64)
    _need(times.shape == (episode["frames"],) and np.isfinite(times).all()
          and (np.diff(times) > 0).all(), "Boundary sampling needs real increasing episode timestamps")
    return times


def _snapshot(unit):
    return {key: copy.deepcopy(unit[key]) for key in
            ("start_idx", "end_idx", "task_id", "instruction", "annotation_provenance")}


def sample_boundary_pair(legacy_chunks, *, full_time, episode, unit, all_units,
                         seed, draw, policy):
    """Return one synchronized continuous-frame selection and honest provenance.

    The caller invokes its unmodified legacy sampler first, even in the boundary
    branch. Therefore K and the following origin/role RNG draw order are exact.
    A requested but unrealizable boundary pair falls back to that same sample.
    """
    policy = validate_boundary_sampling(policy)
    _need(policy is not None, "Boundary helper requires explicit opt-in")
    _need(isinstance(draw, tuple) and len(draw) == 2
          and all(type(value) is int and value >= 0 for value in draw),
          "Boundary sampling requires checkpointed cursor/global-slot keys")
    _need(type(seed) is int and seed >= 0, "Boundary sampling requires nonnegative seed")
    times = _clock(full_time, episode)
    a, b = unit["start_idx"], unit["end_idx"]
    _need(type(a) is int and type(b) is int and 0 <= a < b <= len(times), "Invalid original half-open unit")
    chunks = np.asarray(legacy_chunks)
    _need(chunks.ndim == 2 and chunks.shape[1] == 1 and chunks.dtype.kind in "iu"
          and 2 <= len(chunks) <= min(64, b-a)
          and (np.diff(chunks[:, 0]) > 0).all()
          and ((a <= chunks[:, 0]) & (chunks[:, 0] < b)).all(),
          "Boundary recipe requires original distinct continuous K2..64 sample")
    segments = [list(segment) for segment in episode.get("continuity_segments", [[0, len(times)]])
                if len(segment) == 2 and segment[0] <= a < b <= segment[1]]
    _need(len(segments) == 1, "Original subtask needs one unambiguous recording-continuity segment")
    lo, hi = segments[0]
    _need(type(lo) is int and type(hi) is int and 0 <= lo <= a < b <= hi <= len(times),
          "Invalid physical continuity bounds")
    if "continuity_segment" in unit:
        _need(list(unit["continuity_segment"]) == [lo, hi], "Subtask continuity identity differs")
    relative = times - times[0]
    duration = float(times[b-1] - times[a])
    radius = min(policy["maximum_radius_seconds"], policy["duration_fraction"] * duration)
    _need(duration > 0 and radius > 0, "No recorded physical duration for boundary jitter")
    end_index = b if b < hi else b-1
    kind = "first_excluded_recorded_frame" if b < hi else "last_included_recorded_frame_at_recording_end"
    centers = relative[[a, end_index]]
    eligible = np.arange(lo, hi, dtype=np.int64)
    bands = [eligible[np.abs(relative[eligible] - center) <= radius] for center in centers]
    left = np.repeat(bands[0], len(bands[1]))
    right = np.tile(bands[1], len(bands[0]))
    pair_ok = left < right
    left, right = left[pair_ok], right[pair_ok]
    rng = np.random.default_rng(np.random.SeedSequence([seed, *draw, policy["rng_namespace"]]))
    requested = bool(rng.random() < policy["boundary_probability"])
    reason = None
    pair = None
    selected = chunks[:, 0].copy()
    if requested:
        if len(bands[0]) < 2:
            reason = "fewer_than_two_recorded_start_band_candidates"
        elif len(bands[1]) < 2:
            reason = "fewer_than_two_recorded_end_band_candidates"
        elif len(left) < 2:
            reason = "fewer_than_two_distinct_ordered_boundary_pairs"
        else:
            choice = int(rng.integers(len(left)))
            pair = [int(left[choice]), int(right[choice])]
            remainder = np.setdiff1d(np.arange(a, b, dtype=np.int64), pair)
            extra = rng.choice(remainder, len(chunks)-2, replace=False)
            selected = np.sort(np.r_[pair, extra]).astype(np.int64)
    component = "boundary_pair" if pair is not None else ("uniform_fallback" if requested else "legacy_uniform")
    admitted_lo = min(a, *(int(band[0]) for band in bands if len(band)))
    admitted_hi = max(b, *(int(band[-1])+1 for band in bands if len(band)))
    neighbors = []
    for other in all_units:
        if other is unit or (other["start_idx"], other["end_idx"], other["task_id"]) == (a, b, unit["task_id"]):
            continue
        if other["start_idx"] < admitted_hi and admitted_lo < other["end_idx"]:
            neighbors.append({key: copy.deepcopy(other[key]) for key in
                              ("start_idx", "end_idx", "task_id", "event_ids") if key in other})
    metadata = {
        "schema": SCHEMA, "policy": policy,
        "instruction_policy": "original_instruction_with_bounded_annotation_uncertainty",
        "assert_neighbor_instruction_equivalence": False,
        "original_subtask": _snapshot(unit),
        "original_event_ids": copy.deepcopy(unit.get("event_ids", unit["annotation_provenance"].get("event_ids", []))),
        "original_bounds": [a, b], "admitted_bounds": [admitted_lo, admitted_hi],
        "episode_id": episode["episode_id"], "episode_frames": len(times),
        "source_identity": {key: copy.deepcopy(episode[key]) for key in
                            ("embodiment", "rgb_path", "rgb_sha256", "path", "sha256") if key in episode},
        "continuity_segment": [lo, hi],
        "continuity_first_last_timestamps_seconds": [float(times[lo]), float(times[hi-1])],
        "duration_basis": "recorded_last_minus_first", "duration_seconds": duration,
        "original_first_timestamp_seconds": float(times[a]),
        "original_last_timestamp_seconds": float(times[b-1]),
        "center_indices": [a, end_index], "center_kind": ["first_included_recorded_frame", kind],
        "centers_seconds": [float(times[a]), float(times[end_index])],
        "radius_seconds": radius,
        "band_candidate_indices": [band.tolist() for band in bands],
        "band_candidate_timestamps_seconds": [times[band].tolist() for band in bands],
        "admissible_ordered_pair_count": int(len(left)),
        "component": component, "requested_boundary": requested,
        "pair_indices": pair, "fallback_reason": reason,
        "rng_identity": {"seed": seed, "cursor": draw[0], "global_slot": draw[1], "namespace": policy["rng_namespace"]},
        "count": len(selected), "selected_frame_indices": selected.tolist(),
        "selected_timestamps_seconds": times[selected].tolist(),
        "selected_outside_original": selected[(selected < a) | (selected >= b)].tolist(),
        "pair_offsets_frames": None if pair is None else [pair[0]-a, pair[1]-end_index],
        "pair_offsets_seconds": None if pair is None else [float(times[pair[0]]-times[a]), float(times[pair[1]]-times[end_index])],
        "physical_elapsed_seconds": float(times[selected[-1]]-times[selected[0]]),
        "truncation": {
            "start_band_left": bool(float(times[a])-radius < float(times[lo])),
            "start_band_right": bool(float(times[a])+radius > float(times[hi-1])),
            "end_band_left": bool(float(times[end_index])-radius < float(times[lo])),
            "end_band_right": bool(float(times[end_index])+radius > float(times[hi-1])),
            "terminal_center_truncated": bool(b == hi),
        },
        "mixture_counters": {"requested_boundary": int(requested), "realized_boundary": int(pair is not None),
                             "fallback": int(requested and pair is None), "legacy_uniform": int(not requested)},
        "neighbor_units_if_available": neighbors,
    }
    return selected[:, None], metadata


def validate_boundary_record(record, unit, indices):
    """Strict opt-in smoke provenance, without weakening the legacy branch."""
    packet = record["boundary_sampling"]
    _need(packet.get("schema") == SCHEMA and validate_boundary_sampling(packet.get("policy")) is not None,
          "Missing explicit boundary-uncertainty provenance")
    _need(packet.get("instruction_policy") == "original_instruction_with_bounded_annotation_uncertainty"
          and packet.get("assert_neighbor_instruction_equivalence") is False,
          "Boundary uncertainty may not assert exact neighbor label truth")
    a, b = unit["start_idx"], unit["end_idx"]
    _need(packet["original_subtask"] == _snapshot(unit) and packet["original_bounds"] == [a, b]
          and packet["episode_id"] == record["episode_id"], "Original annotation identity was rewritten")
    selected = packet["selected_frame_indices"]
    times = packet["selected_timestamps_seconds"]
    _need(record["frame_indices"] == selected and record["timestamps_seconds"] == times
          and len(selected) == packet["count"] and 2 <= len(selected) <= min(64, b-a)
          and all(type(i) is int for i in selected) and all(x < y for x, y in zip(selected, selected[1:]))
          and len(times) == len(selected) and all(math.isfinite(t) for t in times)
          and all(x < y for x, y in zip(times, times[1:])), "Invalid actual boundary selection/clock")
    lo, hi = packet["continuity_segment"]
    admitted_lo, admitted_hi = packet["admitted_bounds"]
    _need(0 <= lo <= admitted_lo <= a < b <= admitted_hi <= hi <= packet["episode_frames"]
          and indices and all(admitted_lo <= i < admitted_hi for i in indices),
          "Boundary selection crossed actual recording/continuity bounds")
    _need(set(indices) == set(selected), "Boundary modalities are not synchronized to one selection")
    duration = packet["original_last_timestamp_seconds"] - packet["original_first_timestamp_seconds"]
    radius = min(0.25, 0.1*duration)
    _need(packet["duration_basis"] == "recorded_last_minus_first" and duration > 0
          and math.isclose(packet["duration_seconds"], duration, rel_tol=1e-12, abs_tol=1e-9)
          and math.isclose(packet["radius_seconds"], radius, rel_tol=1e-12, abs_tol=1e-9),
          "Boundary radius differs from physical original span")
    end_index = b if b < hi else b-1
    expected_kind = "first_excluded_recorded_frame" if b < hi else "last_included_recorded_frame_at_recording_end"
    _need(packet["center_indices"] == [a, end_index]
          and packet["center_kind"] == ["first_included_recorded_frame", expected_kind]
          and packet["centers_seconds"][0] == packet["original_first_timestamp_seconds"],
          "Invalid half-open boundary center semantics")
    _need(len(packet["centers_seconds"]) == 2
          and all(math.isfinite(v) for v in packet["centers_seconds"])
          and (packet["centers_seconds"][1] == packet["original_last_timestamp_seconds"] if b == hi
               else packet["centers_seconds"][1] > packet["original_last_timestamp_seconds"]),
          "End center is not the recorded half-open or terminal center")
    bands = packet["band_candidate_indices"]
    band_times = packet["band_candidate_timestamps_seconds"]
    _need(len(bands) == len(band_times) == 2, "Missing recorded boundary bands")
    for band, clock, center_index, center in zip(bands, band_times, packet["center_indices"], packet["centers_seconds"]):
        _need(len(band) == len(clock) and band and center_index in band
              and all(type(v) is int and lo <= v < hi for v in band)
              and all(x < y for x,y in zip(band,band[1:]))
              and all(math.isfinite(v) for v in clock)
              and all(x < y for x,y in zip(clock,clock[1:]))
              and clock[band.index(center_index)] == center,
              "Boundary candidate indices/timestamps are not actual ordered continuous records")
        for index, value in zip(band,clock):
            tolerance = abs(float(np.spacing(max(abs(value),abs(center),1.0))))*4
            _need(abs(value-center) <= radius+tolerance, "Candidate band exceeds physical radius")
            if index in selected:
                _need(times[selected.index(index)] == value, "Pair clock differs from selected native frame")
    pair_count = sum(i < j for i in bands[0] for j in bands[1])
    _need(packet["admissible_ordered_pair_count"] == pair_count
          and [min(a,bands[0][0],bands[1][0]), max(b,bands[0][-1]+1,bands[1][-1]+1)] == packet["admitted_bounds"],
          "Admitted bounds or pair count differs from actual candidate bands")
    outside = [i for i in selected if i < a or i >= b]
    _need(packet["selected_outside_original"] == outside, "Outward frames lost uncertainty provenance")
    identity = packet["rng_identity"]
    _need(set(identity) == {"seed", "cursor", "global_slot", "namespace"}
          and all(type(identity[k]) is int and identity[k] >= 0 for k in ("seed", "cursor", "global_slot"))
          and type(identity["namespace"]) is int and identity["namespace"] == POLICY["rng_namespace"],
          "Invalid checkpointed boundary RNG identity")
    rng = np.random.default_rng(np.random.SeedSequence([identity["seed"],identity["cursor"],identity["global_slot"],identity["namespace"]]))
    requested = bool(rng.random() < POLICY["boundary_probability"])
    _need(type(packet["requested_boundary"]) is bool and packet["requested_boundary"] == requested,
          "Requested mixture does not match checkpointed component draw")
    component = packet["component"]
    pair = packet["pair_indices"]
    if component == "boundary_pair":
        _need(packet["requested_boundary"] is True and packet["fallback_reason"] is None
              and isinstance(pair, list) and len(pair) == 2 and pair[0] < pair[1]
              and all(i in selected for i in pair) and all(i in pair for i in outside)
              and all(len(band) >= 2 for band in packet["band_candidate_indices"])
              and packet["admissible_ordered_pair_count"] >= 2, "Boundary branch is not a genuine random pair")
        pairs = [(i,j) for i in bands[0] for j in bands[1] if i < j]
        expected_pair = list(pairs[int(rng.integers(len(pairs)))])
        remainder = np.setdiff1d(np.arange(a,b,dtype=np.int64),expected_pair)
        expected_selected = np.sort(np.r_[expected_pair,rng.choice(remainder,len(selected)-2,replace=False)]).tolist()
        _need(pair == expected_pair and selected == expected_selected,
              "Boundary pair/remaining frames differ from deterministic cursor/slot draw")
        for index, center, band in zip(pair, packet["centers_seconds"], packet["band_candidate_indices"]):
            value = times[selected.index(index)]
            tolerance = abs(float(np.spacing(max(abs(value), abs(center), 1.0)))) * 4
            _need(index in band and abs(value-center) <= radius+tolerance,
                  "Outward pair exceeds bounded physical-time jitter")
    else:
        _need(component in ("legacy_uniform", "uniform_fallback") and pair is None and not outside,
              "Uniform component may not contain outward frames")
        _need((component == "uniform_fallback") == packet["requested_boundary"], "Misreported requested/realized mixture")
        _need((packet["fallback_reason"] is not None) == (component == "uniform_fallback"), "Missing fallback reason")
        if component == "uniform_fallback":
            reason = ("fewer_than_two_recorded_start_band_candidates" if len(bands[0]) < 2 else
                      "fewer_than_two_recorded_end_band_candidates" if len(bands[1]) < 2 else
                      "fewer_than_two_distinct_ordered_boundary_pairs" if pair_count < 2 else None)
            _need(reason is not None and packet["fallback_reason"] == reason, "Fallback does not match actual insufficient candidates")
    _need(type(packet["requested_boundary"]) is bool, "Requested component must be an actual boolean")
    if pair is not None:
        actual_offsets = [pair[0]-a,pair[1]-end_index]
        time_offsets = [times[selected.index(pair[0])]-packet["centers_seconds"][0],
                        times[selected.index(pair[1])]-packet["centers_seconds"][1]]
        _need(packet["pair_offsets_frames"] == actual_offsets and packet["pair_offsets_seconds"] == time_offsets,
              "Boundary offsets differ from actual selected pair")
    else:
        _need(packet["pair_offsets_frames"] is None and packet["pair_offsets_seconds"] is None,
              "Uniform selection invented a boundary pair")
    _need(packet["physical_elapsed_seconds"] == times[-1]-times[0], "Physical span differs from recorded selection")
    first,last = packet["continuity_first_last_timestamps_seconds"]
    centers = packet["centers_seconds"]
    expected_truncation = {
        "start_band_left": centers[0]-radius < first,
        "start_band_right": centers[0]+radius > last,
        "end_band_left": centers[1]-radius < first,
        "end_band_right": centers[1]+radius > last,
        "terminal_center_truncated": b == hi,
    }
    _need(packet["truncation"] == expected_truncation, "Recording truncation provenance differs")
    expected_counters = {"requested_boundary": int(packet["requested_boundary"]),
                         "realized_boundary": int(component == "boundary_pair"),
                         "fallback": int(component == "uniform_fallback"),
                         "legacy_uniform": int(component == "legacy_uniform")}
    _need(packet["mixture_counters"] == expected_counters, "Misreported boundary mixture counters")
    return packet
