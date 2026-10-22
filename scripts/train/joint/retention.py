"""Read the maintained mirror's strict receipts for remotely retained milestones.

Local pruning is still owned by ice_checkpoint_mirror. This verifier does not
copy or delete files and never substitutes a filename for remote SHA evidence.
"""

import json
import hashlib
from pathlib import Path
import re
import shutil
import subprocess

from scripts.train.joint.contract import bound, need


def mirror_row(request):
    policy = request["archive_policy"]
    manifest = bound(policy["manifest"])
    rows = [r for r in manifest["runs"] if r["id"] == request["run_id"]]
    need(len(rows) == 1, "Mirror manifest does not identify this run uniquely")
    row = rows[0]
    need(Path(row["local_root"]) == Path(request["output"])
         and row["checkpoint_glob"] == "checkpoints/*.ckpt"
         and row["retain_local"] == 2, "Mirror retention policy changed")
    bound(policy["validator"],False)
    need(row["validator"] == policy["validator"]["path"], "Mirror validator changed")
    return row


def check_storage(request, completion):
    """Reduced local reserve requires an actual strict zero-error mirror proof."""
    smoke = bound(request["smoke"]["request"])
    size = (Path(smoke["output"])/completion["ranks"][0]["checkpoint_name"]).stat().st_size
    stops = request.get("allocation_boundaries", [request["target_steps"]])
    need(stops == sorted(set(stops)) and stops[-1] == request["target_steps"]
         and all(type(s) is int and s > 0 and s % 20000 == 0 for s in stops)
         and request.get("allocation_stop_step", request["target_steps"]) in stops,
         "Unbound allocation/checkpoint storage plan")
    count = request["target_steps"]//20000
    policy = request.get("archive_policy")
    if policy is not None:
        # The probe binds the exact already-GPU-verified checkpoint and native
        # mirror receipts; a running mirror job or empty cycle cannot pass.
        probe = policy["probe"]
        state, pressure = bound(probe["state"]), bound(probe["pressure"])
        native = dict(s.split("=",1) for s in bound(probe["native"],False).split() if "=" in s)
        owner = (request.get("allocation") or {}).get("user","paphiwetsa3")
        need(native["JobId"] == str(probe["job_id"]) and native["JobState"] == "COMPLETED"
             and native["ExitCode"] == "0:0" and native["UserId"].startswith(owner+"(")
             and pressure["cycle_errors"] == 0,
             "Mirror probe did not complete a zero-error native cycle")
        matches = [v for v in state["files"].values() if v.get("sha256") == completion["checkpoint_sha256"]]
        need(len(matches) == 1 and matches[0]["remote_verified"] is True
             and matches[0]["global_step"] == matches[0]["metadata"]["global_step"] == 50
             and matches[0]["metadata"]["strict_model_reload"] is True
             and matches[0]["metadata"]["strict_model_parameter_count"] == request["parameters"]
             and pressure["updated_at_unix"] >= matches[0]["verified_at_unix"],
             "Mirror probe lacks the exact strict model/checkpoint archive")
        mirror_row(request)
        bound(policy["ssh"],False)
        need(policy["ssh"]["sha256"] == probe["ssh_sha256"]
             and policy["remote_host"] == probe["remote_host"]
             and policy["validator"]["sha256"] == probe["validator_sha256"],
             "Archive SSH route or strict validator changed")
        from scripts.ice.ice_checkpoint_mirror import remote_command

        remote_command(policy["ssh"]["path"],policy["remote_host"],["true"])
        # Two verified recovery copies remain local while the next immutable
        # save is written and archived. Never spend that in-flight save space
        # on boundary recoveries or artifact headroom.
        count = 3
    required = (count+len(stops)-1)*size + 1000000000
    need(type(request["reserved_checkpoint_bytes"]) is int
         and request["reserved_checkpoint_bytes"] >= required,
         "Reservation does not retain every scheduled and boundary checkpoint")
    output = Path(request["output"])
    existing = output
    while not existing.exists():
        existing = existing.parent
    retained = sum(p.stat().st_size for pattern in ("checkpoints/*.ckpt","allocation-evidence/allocation-recovery*.ckpt")
                   for p in output.glob(pattern) if p.is_file() and not p.is_symlink())
    remaining = max(0,request["reserved_checkpoint_bytes"]-retained)
    need(shutil.disk_usage(existing).free > remaining,
         "Insufficient filesystem storage")
    allocation = request.get("allocation") or {}
    if allocation.get("cluster") == "ice-slurm":
        from scripts.train.joint.recovery_check import quota_free_bytes

        quota = request["quota"]
        need(quota["user"] == allocation["user"] and output.is_relative_to(quota["mount"]),
             "Quota does not identify this output owner")
        text = subprocess.check_output(["lfs","quota","-u",quota["user"],quota["mount"]],text=True)
        need(quota_free_bytes(text,quota["mount"]) > remaining,
             "Insufficient live user quota for retained training")
    return required


def archived_milestones(request, missing):
    if not missing:
        return {}, None
    policy = request.get("archive_policy")
    need(policy is not None, "Missing scheduled local milestones and no archive policy")
    row = mirror_row(request)
    bound(policy["ssh"], False)
    directory = Path(request["output"])/"checkpoints"
    state_path = Path(policy["state"])
    state_bytes = state_path.read_bytes()
    state = json.loads(state_bytes)
    found, verified = {}, []
    for local_name, entry in state["files"].items():
        if entry.get("run_id") != request["run_id"]:
            continue
        local = Path(local_name)
        need(local.parent == directory, "Mirror checkpoint escapes this run")
        match = re.search(r"([0-9]{9})\.ckpt$", local.name)
        need(match is not None and local.name != "last.ckpt", "Unqualified archived checkpoint name")
        step = int(match[1])
        if step not in missing:
            continue
        metadata = entry["metadata"]
        digest = entry["sha256"]
        remote = entry["remote_path"]
        need(step not in found and entry["global_step"] == metadata["global_step"] == step
             and entry["remote_verified"] is True
             and metadata["strict_model_reload"] is True
             and metadata["strict_model_parameter_count"] == request["parameters"]
             and metadata["run_id"] == request["run_id"]
             and re.fullmatch(r"[0-9a-f]{64}", digest)
             and remote == row["remote_dir"].rstrip("/")+"/"+local.stem+"."+digest+".ckpt",
             "Missing strict checkpoint identity or immutable archive receipt")
        from scripts.ice.ice_checkpoint_mirror import remote_sha

        need(remote_sha(policy["ssh"]["path"], policy["remote_host"], remote) == digest,
             "Archived milestone failed current remote SHA verification")
        found[step] = Path(remote)
        verified.append(dict(global_step=step, sha256=digest, remote_path=remote,
                             strict_model_parameter_count=metadata["strict_model_parameter_count"]))
    need(set(found) == set(missing), "Scheduled checkpoints are absent from the verified archive")
    return found, dict(state_path=str(state_path), state_sha256=hashlib.sha256(state_bytes).hexdigest(),
                       verified_steps=sorted(found), verified_checkpoints=verified,
                       manifest_sha256=policy["manifest"]["sha256"])
