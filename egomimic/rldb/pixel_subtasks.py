"""Half-open, language-conditioned subtask units for native pixel world models.

Physical episodes retain their original split/normalization identity. This index
only restricts sampling; it never rewrites image, state, action or source Zarr.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def normalize_scale_events(document, *, fps, context_labels=()):
    """Preserve original Scale object names; reject mistakes/unknown actions.

    Raw Scale clip time is the annotated video's clock, not the human robot
    clock. The caller must separately verify video-row/native-frame alignment
    before an episode can enter a training index.
    """
    import math
    from fractions import Fraction
    if document.get("time_unit") != "microseconds" or document.get("time_offset") != 0:
        raise ValueError("Require explicit microsecond video clock with zero offset")
    if not isinstance(fps,(int,float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Positive verified native video frame rate required")
    rate=Fraction(str(fps))
    result=[]
    for annotation in document.get("annotations",[]):
        if annotation.get("type") != "text_annotation":
            continue
        label=annotation.get("label","")
        hand=("left" if "left" in label.lower() else "right" if "right" in label.lower() else "both" if "both" in label.lower() else None)
        role="context" if label in context_labels else "hand_action"
        for clip in annotation.get("clips",[]):
            attributes={}
            for attribute in clip.get("attributes",[]):
                attributes.setdefault(attribute.get("name"),[]).append(attribute)
            def single(name):
                options=attributes.get(name,[])
                if len(options)!=1 or options[0].get("static") is not True:
                    return None
                values=options[0].get("values",[])
                return values[0] if len(values)==1 and isinstance(values[0],str) else None
            action,mistake=single("Action"),single("Mistake")
            text=clip.get("text","")
            instruction=((action or "")+" "+(text if isinstance(text,str) else "")).strip()
            reasons=[]
            if role == "context": reasons.append("non_hand_context_annotation")
            if mistake != "No": reasons.append("source_mistake_or_unknown")
            if not action or not action.strip(): reasons.append("missing_or_ambiguous_action")
            if not hand: reasons.append("missing_hand")
            if not isinstance(text,str) or not text.strip(): reasons.append("empty_source_text")
            start,duration=clip.get("timestamp"),clip.get("duration")
            if type(start) is not int or type(duration) is not int or start<0 or duration<=0:
                raise ValueError("Malformed source clip time")
            result.append(dict(event_id=annotation["id"]+":"+clip["id"],hand=hand,
                start_idx=math.ceil(Fraction(start,1000000)*rate),end_idx=math.ceil(Fraction(start+duration,1000000)*rate),
                instruction=instruction,approved=not reasons,review_reasons=reasons,
                source_label=label,source_action=action,source_mistake=mistake,event_role=role))
    return result


def atomic_subtasks(events, frames):
    """Combine simultaneous hand instructions over disjoint atomic intervals.

    Callers supply reviewed, frame-aligned source events. Unapproved events,
    invalid bounds, and contradictory same-hand overlaps are quarantined with
    reasons. No clipping, synonym replacement or fabricated labels occurs.
    """
    if type(frames) is not int or frames < 1:
        raise ValueError("Require positive physical episode frame count")
    valid, rejected, blocked = [], [], []
    seen = set()
    for event in events:
        reason = None
        if not isinstance(event, dict):
            raise ValueError("Subtask events must be objects")
        a, b = event.get("start_idx"), event.get("end_idx")
        if event.get("event_role") == "context":
            if event.get("approved") is True:
                raise ValueError("Context annotations cannot be approved hand targets")
            rejected.append(dict(event=event,reason="non_hand_context_annotation"))
            continue
        if event.get("approved") is not True:
            reason = "annotation_not_approved"
        elif (type(a) is not int or type(b) is not int or not 0 <= a < b <= frames):
            reason = "invalid_or_out_of_bounds_half_open_interval"
        elif (event.get("hand") not in ("left", "right", "both")
              or not isinstance(event.get("instruction"), str)
              or not event["instruction"].strip()
              or not isinstance(event.get("event_id"), str)
              or not event["event_id"]):
            reason = "missing_hand_instruction_or_source_event_id"
        elif event["event_id"] in seen:
            raise ValueError("Duplicate source event ID")
        if reason:
            rejected.append(dict(event=event, reason=reason))
            if type(a) is int and type(b) is int and a < b:
                blocked.append(event)
        else:
            seen.add(event["event_id"])
            valid.append(event)
    boundaries = sorted({x for e in valid + blocked
                         for x in (e["start_idx"], e["end_idx"]) if 0 <= x <= frames})
    units = []
    for a, b in zip(boundaries, boundaries[1:]):
        active = [e for e in valid if e["start_idx"] <= a and b <= e["end_idx"]]
        if not active:
            continue
        uncertain = [e for e in blocked if e["start_idx"] < b and a < e["end_idx"]]
        if uncertain:
            rejected.append(dict(start_idx=a, end_idx=b,
                                 event_ids=sorted(e.get("event_id", "unknown") for e in uncertain),
                                 reason="unapproved_concurrent_instruction"))
            continue
        per_hand = {h: {e["instruction"] for e in active if e["hand"] == h}
                    for h in ("left", "right", "both")}
        if any(len(v) > 1 for v in per_hand.values()) or (per_hand["both"] and
                (per_hand["left"] or per_hand["right"])):
            rejected.append(dict(start_idx=a, end_idx=b,
                                 event_ids=sorted(e["event_id"] for e in active),
                                 reason="conflicting_concurrent_hand_instructions"))
            continue
        instruction = " ".join(f"{h.capitalize()} hand: {next(iter(per_hand[h]))}"
                               for h in ("left", "right", "both") if per_hand[h])
        event_ids = sorted(e["event_id"] for e in active)
        if units and units[-1]["end_idx"] == a and units[-1]["event_ids"] == event_ids:
            units[-1]["end_idx"] = b
        else:
            units.append(dict(start_idx=a, end_idx=b, instruction=instruction,
                              event_ids=event_ids))
    for unit in units:
        unit["task_id"] = "subtask-" + hashlib.sha256(unit["instruction"].encode()).hexdigest()
    return units, rejected


def load_subtask_manifest(path, sha256, manifest, manifest_sha256):
    """Load an immutable reviewed index bound to exact prepared episode bytes.

    A generated annotation is not admitted just because a file exists: every
    unit requires explicit quality/name and source-clock alignment admission.
    """
    if (path is None) != (sha256 is None):
        raise ValueError("Subtask path and SHA256 require both or neither")
    if path is None:
        return None
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("Subtask manifest SHA256 mismatch")
    index = json.loads(raw)
    if (index.get("schema") != "pixel-subtask-index-v1"
            or index.get("prepared_manifest_sha256") != manifest_sha256
            or index.get("interval_semantics") != "half_open"
            or index.get("concurrent_policy") != "combine_active_hands_atomic_intervals"):
        raise ValueError("Subtask index contract/manifest identity mismatch")
    expected = {(e["embodiment"], e["episode_id"]): e for e in manifest["episodes"]}
    found = {}
    for record in index.get("episodes", []):
        key = record.get("embodiment"), record.get("episode_id")
        if key not in expected or key in found:
            raise ValueError("Unknown or duplicate physical episode in subtask index")
        ep = expected[key]
        if record.get("split") != ep["split"] or record.get("frames") != ep["frames"]:
            raise ValueError("Subtask index may not change physical episode split/length")
        units, previous = record.get("subtasks"), 0
        if not isinstance(units, list) or not units:
            raise ValueError("Every selected episode needs reviewed subtask intervals")
        for unit in units:
            a, b = unit.get("start_idx"), unit.get("end_idx")
            if (type(a) is not int or type(b) is not int
                    or not previous <= a < b <= ep["frames"]):
                raise ValueError("Require disjoint in-bounds half-open subtasks")
            previous = b
            segments = ep.get("continuity_segments", [[0, ep["frames"]]])
            if not any(lo <= a < b <= hi for lo,hi in segments):
                raise ValueError("Subtask crosses a native recording discontinuity")
            if (unit.get("quality_and_names_verified") is not True
                    or unit.get("source_clock_alignment_verified") is not True
                    or not isinstance(unit.get("task_id"), str) or not unit["task_id"]
                    or not isinstance(unit.get("instruction"), str) or not unit["instruction"].strip()
                    or not unit.get("annotation_provenance")):
                raise ValueError("Subtask lacks annotation, naming or clock admission")
        found[key] = units
    if set(found) != set(expected):
        raise ValueError("Subtask index must cover the selected prepared corpus exactly")
    return found


def build_subtask_index(prepared_manifest_path, events_path):
    """Compile reviewed frame-aligned sidecars without touching physical data."""
    manifest_raw = Path(prepared_manifest_path).read_bytes()
    event_raw = Path(events_path).read_bytes()
    manifest, source = json.loads(manifest_raw), json.loads(event_raw)
    if source.get("schema") != "pixel-subtask-events-v1":
        raise ValueError("Require pixel-subtask-events-v1 normalized source events")
    records = {}
    for record in source.get("episodes", []):
        key = record.get("embodiment"), record.get("episode_id")
        if key in records:
            raise ValueError("Duplicate normalized event episode")
        records[key] = record
    index = dict(schema="pixel-subtask-index-v1",
                 prepared_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
                 annotation_events_sha256=hashlib.sha256(event_raw).hexdigest(),
                 interval_semantics="half_open",
                 concurrent_policy="combine_active_hands_atomic_intervals", episodes=[])
    quarantine = []
    for ep in manifest["episodes"]:
        key = ep["embodiment"], ep["episode_id"]
        record = records.get(key)
        if not record or record.get("source_clock_alignment_verified") is not True:
            quarantine.append(dict(episode_id=ep["episode_id"],reason="missing_verified_source_clock"))
            continue
        if record.get("frames") != ep["frames"] or record.get("split") != ep["split"]:
            raise ValueError("Event source may not alter episode frame count or split")
        units, rejected = atomic_subtasks(record.get("events", []), ep["frames"])
        quarantine.extend(dict(episode_id=ep["episode_id"], **r) for r in rejected)
        segments = ep.get("continuity_segments", [[0, ep["frames"]]])
        bounded=[]
        for unit in units:
            for lo,hi in segments:
                a,b=max(unit["start_idx"],lo),min(unit["end_idx"],hi)
                if a<b:
                    bounded.append(dict(unit,start_idx=a,end_idx=b,continuity_segment=[lo,hi]))
        units=bounded
        for unit in units:
            unit.update(quality_and_names_verified=True,
                        source_clock_alignment_verified=True,
                        annotation_provenance=dict(events_sha256=index["annotation_events_sha256"],
                            event_ids=unit["event_ids"],source=record.get("annotation_provenance", {})))
        if not units:
            quarantine.append(dict(episode_id=ep["episode_id"], reason="no_approved_subtasks"))
            continue
        index["episodes"].append(dict(embodiment=ep["embodiment"],episode_id=ep["episode_id"],
                                    split=ep["split"],frames=ep["frames"],subtasks=units))
    return index, dict(schema="pixel-subtask-quarantine-v1", rejected=quarantine,
                      selected_episodes=len(manifest["episodes"]),
                      indexed_episodes=len(index["episodes"]),
                      complete=len(index["episodes"]) == len(manifest["episodes"]))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-manifest", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quarantine", required=True)
    args = parser.parse_args()
    index, report = build_subtask_index(args.prepared_manifest, args.events)
    for destination, value in ((args.output,index),(args.quarantine,report)):
        with Path(destination).open("x") as handle:
            json.dump(value,handle,indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
