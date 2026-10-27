"""Exact original-afterany terminal provenance; no submission or ML imports.

Only the three already accepted original full allocations are supported. A
replacement needs a new reviewed profile; an arbitrary path cannot be adopted.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess

from scripts.train.joint.contract import bound, need

from scripts.train.joint.qk_recovery_profile import RUN_ID as QK_RUN_ID, PROFILE as QK_PROFILE

ORIGINAL_OBSERVERS = {'mot-elmo-pickplace-continuous-large314m-boundary-robot-human-camera-subtask-s42-300k-20261026-v1': {'capture_script_sha256': 'be8ca57b2e5c09503ec312c9d84c91451d932d4d9a168c484f22439ccf521b57',
                                                                                                      'files_sha256': 'fbce43e2cf1ad9c090563bc7e5492c654f00db7e84890ef83d039e66cc9aea5f',
                                                                                                      'has_smoke_identity': True,
                                                                                                      'node': 'gpu03',
                                                                                                      'observer_id': '412801',
                                                                                                      'observer_name': 'elmo-native-full-large314m-boundary-v2',
                                                                                                      'parent_id': '412800',
                                                                                                      'parent_name': 'elmo-large314m-continuous-boundary-full-v2',
                                                                                                      'request_path': '/workspace/users/paphiwetsa3/tasks/elmo-large-continuous-20261026-v1/boundary-full-preparation-v2/ops/requests/elmo-large314m-continuous-boundary-full-v2.json',
                                                                                                      'request_sha256': '210e6f73c8a1a396716fab2f1e7e17f3b9d0663b099971eb05807181f9f00684',
                                                                                                      'root': '/workspace/users/paphiwetsa3/tasks/elmo-large-continuous-20261026-v1/boundary-full-preparation-v2',
                                                                                                      'submitter_sha256': '9e1722e56a117f8b42297e7fa3c82ef973a3435dd25bff51b69b09557fd68ddb'},
 'mot-elmo-pickplace-continuous-large314m-fused-robot-human-camera-subtask-s42-300k-20261026-v1': {'capture_script_sha256': '87b011c6bd1f5d71c0758a75fb6ad9e05a807b7e5af33eba28264240b6071930',
                                                                                                   'files_sha256': '3a82e42346138a97a78fbca4201cc89b9635fe8987132620ceace54965386b58',
                                                                                                   'has_smoke_identity': True,
                                                                                                   'node': 'gpu11',
                                                                                                   'observer_id': '412799',
                                                                                                   'observer_name': 'elmo-native-full-large314m-fused-v2',
                                                                                                   'parent_id': '412798',
                                                                                                   'parent_name': 'elmo-large314m-continuous-fused-full-v2',
                                                                                                   'request_path': '/workspace/users/paphiwetsa3/tasks/elmo-large-continuous-20261026-v1/fused-full-preparation-v2/ops/requests/elmo-large314m-continuous-fused-full-v2.json',
                                                                                                   'request_sha256': 'e34190bfef155942fe216a1e5e81317cfa31b65d4c98ccb025a579c874fd3876',
                                                                                                   'root': '/workspace/users/paphiwetsa3/tasks/elmo-large-continuous-20261026-v1/fused-full-preparation-v2',
                                                                                                   'submitter_sha256': 'bede26ca7e2678c874e8442c5e5f6de01fdc58f38ee97c61d0f9f2c4c2a49dea'},
 'mot-elmo-pickplace-continuous-large314m-plain-robot-human-camera-subtask-s42-300k-20261026-v1': {'capture_script_sha256': '5eeb85de75821627054787e0046972913442f2b2e7a751b681f2e71f81da21d4',
                                                                                                   'files_sha256': 'e2af24ae9daa20defd6a75235cb191101f4bc1abfe1bcf374b117659b2b1328d',
                                                                                                   'has_smoke_identity': False,
                                                                                                   'node': 'gpu11',
                                                                                                   'observer_id': '412790',
                                                                                                   'observer_name': 'elmo-native-full-large314m-v1',
                                                                                                   'parent_id': '412789',
                                                                                                   'parent_name': 'elmo-large314m-continuous-plain-full-v1',
                                                                                                   'request_path': '/workspace/users/paphiwetsa3/tasks/elmo-large-continuous-20261026-v1/baseline-full-preparation-v1/ops/requests/elmo-large314m-continuous-plain-full-v1.json',
                                                                                                   'request_sha256': '936e9acd30fa517f6f38311c0bcfd2b88544da8d476bd41125c4008de2845073',
                                                                                                   'root': '/workspace/users/paphiwetsa3/tasks/elmo-large-continuous-20261026-v1/baseline-full-preparation-v1',
                                                                                                   'submitter_sha256': '94c077facd7836fb55708199dab988fb5902db5348ce449daa861a888d95901f'}}
ORIGINAL_OBSERVERS[QK_RUN_ID] = QK_PROFILE


PURGED_STDERR = "slurm_load_jobs error: Invalid job id specified"


def read_binding(binding, json_value=True):
    path = Path(binding["path"])
    need(path.is_absolute() and path.is_file() and not path.is_symlink()
         and path.resolve(strict=True) == path, "Observer proof must be a physical regular file")
    need(path.stat().st_size <= 4 * 1024 * 1024, "Observer metadata exceeds4MiB pre-read bound")
    return bound(binding, json_value)


def query(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=25, check=False)
    need(len(result.stdout.encode()) <= 1048576 and len(result.stderr.encode()) <= 65536,
         "Native observation exceeded bound")
    return dict(schema="native-elmo-terminal-query-v1", argv=argv, returncode=result.returncode,
                stdout=result.stdout, stderr=result.stderr, observed_at_utc=datetime.now(timezone.utc).isoformat())


def query_record(record, argv):
    need(set(record) == {"schema", "argv", "returncode", "stdout", "stderr", "observed_at_utc"}
         and record["schema"] == "native-elmo-terminal-query-v1" and record["argv"] == argv
         and type(record["returncode"]) is int and isinstance(record["stdout"], str)
         and isinstance(record["stderr"], str), "Wrong actual native query observation")
    observed = datetime.fromisoformat(record["observed_at_utc"])
    need(observed.tzinfo is not None and observed <= datetime.now(timezone.utc), "Native query time missing/future")


def validate_original(q, reader=read_binding):
    from scripts.train.joint.native_interruption import fields, native_identity, TERMINAL

    parent = reader(q["parent_request"])
    if parent.get("run_id") == QK_RUN_ID:
        from scripts.train.joint.qk_original_observer import validate
        return validate(q, reader)
    profile = ORIGINAL_OBSERVERS.get(parent.get("run_id"))
    need(profile is not None, "No reviewed original observer for this identity")
    root = Path(profile["root"]); pid, oid = profile["parent_id"], profile["observer_id"]
    request_sha = profile["request_sha256"]
    need(q["parent_request"] == dict(path=profile["request_path"], sha256=request_sha)
         and str(q["parent_job_id"]) == pid and parent["job_name"] == profile["parent_name"],
         "Observer does not belong to the exact original production request")
    proof = q["original_observer"]
    claim = root / "claims" / request_sha
    expected = dict(files_manifest=root/"FILES.sha256", capture_script=root/"capture-native.sbatch",
                    guard=root/"guard_full.py", submitter=root/"submit_full_once.py",
                    parent_script=root/"full.sbatch",
                    prepared=claim/"PREPARED.json", parent_id=claim/"PARENT_JOB_ID",
                    observer_id=claim/"CAPTURE_JOB_ID", pending_native=claim/"CAPTURE_PENDING_NATIVE.txt",
                    running_admission=claim/"RUNNING_ADMISSION.json", terminal_native=root/("native-"+pid+".txt"),
                    query_exit=root/("native-"+pid+".exit"), query_stderr=root/("native-"+pid+".stderr"))
    if profile["has_smoke_identity"]:
        expected["smoke_identity"] = root / "smoke-identity.json"
    need(set(proof) == set(expected) | {"schema"} and proof["schema"] == "native-elmo-original-observer-v1",
         "Complete original observer provenance required")
    for key, path in expected.items():
        value = proof[key]
        need(set(value) == {"path", "sha256"} and value["path"] == str(path)
             and re.fullmatch(r"[0-9a-f]{64}", value["sha256"]), "Unreviewed original observer path/hash: "+key)
    need(proof["files_manifest"]["sha256"] == profile["files_sha256"]
         and proof["capture_script"]["sha256"] == profile["capture_script_sha256"]
         and proof["submitter"]["sha256"] == profile["submitter_sha256"]
         and proof["running_admission"] == q["parent_running_admission"], "Original immutable code/claim pins differ")
    raw = {key: reader(value, False) for key, value in proof.items() if key != "schema"}
    manifest = {}
    for line in raw["files_manifest"].splitlines():
        m = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        need(m and m[2] not in manifest and not Path(m[2]).is_absolute() and ".." not in Path(m[2]).parts,
             "Unsafe/ambiguous original FILES manifest")
        manifest[m[2]] = m[1]
    for key in ("capture_script", "guard", "submitter", "parent_script", *(["smoke_identity"] if profile["has_smoke_identity"] else [])):
        need(manifest.get(str(expected[key].relative_to(root))) == proof[key]["sha256"],
             "Observer dependency not bound by original manifest: "+key)
    need(manifest.get(str(Path(profile["request_path"]).relative_to(root))) == request_sha,
         "Original scientific request absent from immutable manifest")
    prepared = json.loads(raw["prepared"])
    need(prepared["request_sha256"] == request_sha and prepared["manifest_sha256"] == profile["files_sha256"]
         and raw["parent_id"].strip() == pid and raw["observer_id"].strip() == oid,
         "Original accepted claim does not bind this parent/observer")
    running = json.loads(raw["running_admission"])
    need(running["status"] == "ACTUAL_BOUND_FULL_NATIVE_ALLOCATION_PASS"
         and running["job_id"] == pid and running["request_sha256"] == request_sha,
         "Actual original RUNNING claim missing")
    native = q["parent_native"]
    need(native["JobId"] == pid and native["JobName"] == profile["parent_name"]
         and native["UserId"] == "paphiwetsa3(41003)" and native["Account"] == "loaner"
         and native["QOS"] == "loaner-normal" and native["Partition"] == "rl2"
         and native["NodeList"] == native["ReqNodeList"] == profile["node"]
         and native["NumCPUs"] == "32" and native["MinMemoryNode"] in {"192G", "196608M", "196608"}
         and native["TimeLimit"] == "7-00:00:00" and native["Requeue"] == native["Restarts"] == "0"
         and native["Command"] == str(root/"full.sbatch") and native["TresPerNode"] == "gres/gpu:h100:4",
         "Original full native owner/resources/Command differ")
    actual_running = native_identity(running["native"], native, "RUNNING")
    pending = fields(raw["pending_native"])
    required = dict(JobId=oid, JobName=profile["observer_name"], UserId=native["UserId"], Account="loaner",
                    QOS="loaner-normal", Partition="rl2", ReqNodeList=profile["node"], NumTasks="1",
                    NumCPUs="2", TimeLimit="00:10:00", Requeue="0", Restarts="0", JobState="PENDING",
                    Command=str(root/"capture-native.sbatch"), WorkDir=str(root))
    need(all(pending.get(k) == v for k, v in required.items()) and pending.get("NumNodes") in {"1", "1-1"}
         and pending.get("MinMemoryNode") in {"1G", "1024M", "1024"}
         and re.fullmatch(r"afterany:"+pid+r"\((?:unfulfilled|unsatisfied)\)", pending.get("Dependency", "")),
         "Original observer accepted dependency/native identity differs")
    need(pending.get("StdOut") == str(root/"logs"/(profile["observer_name"]+"-"+oid+".out"))
         and pending.get("StdErr") == str(root/"logs"/(profile["observer_name"]+"-"+oid+".err")),
         "Original observer output namespace differs")
    tres = pending.get("ReqTRES", "").split(",")
    need("cpu=2" in tres and "node=1" in tres and any(x in tres for x in ("mem=1G", "mem=1024M")),
         "Original observer CPU TRES differ")
    for key, value in pending.items():
        if key in {"ReqTRES", "AllocTRES"}:
            need(not any("gpu" in v.lower() for v in value.split(",")), "Observer has GPU allocation")
        if key.lower().startswith(("gres", "job_gres", "tresper")):
            need(value.lower() in {"", "(null)", "none", "n/a", "0"} or key == "TresPerTask" and value == "cpu:2",
                 "Observer has GPU/GRES resources")
    need(raw["query_exit"].strip() == "0" and not raw["query_stderr"].strip(),
         "Original observer parent-query did not succeed cleanly")
    terminal_fields = fields(raw["terminal_native"])
    need(terminal_fields.get("JobState") in TERMINAL, "Original observer did not capture terminal parent")
    terminal = native_identity(raw["terminal_native"], native, terminal_fields["JobState"])
    start = datetime.fromisoformat(terminal["StartTime"]); end = datetime.fromisoformat(terminal["EndTime"])
    need(start <= end and terminal["StartTime"] == actual_running["StartTime"], "Terminal interval/original allocation differs")
    return raw["terminal_native"], dict(schema="native-elmo-original-observer-validation-v1", original_parent_id=pid,
        original_observer_id=oid, observer_own_terminal="UNKNOWN_NOT_REQUIRED", capture_query_exit=0,
        scientific_parent_state=terminal["JobState"], scientific_parent_exit=terminal["ExitCode"], proofs=proof)


def resolve_terminal(q, record, reader=read_binding):
    from scripts.train.joint.native_interruption import fields, native_identity, TERMINAL

    argv = ["/usr/local/bin/scontrol", "show", "job", str(q["parent_job_id"]), "-o", "-dd"]
    query_record(record, argv)
    observer = validate_original(q, reader) if q.get("original_observer") is not None else None
    if record["returncode"] == 0:
        need(not record["stderr"].strip(), "Live parent query returned diagnostics")
        raw = record["stdout"]; current = fields(raw)
        need(current.get("JobState") in TERMINAL, "Live parent is not terminal; observer cannot override it")
        native_identity(raw, q["parent_native"], current["JobState"])
        if observer:
            old = fields(observer[0])
            need(all(current.get(k) == old.get(k) for k in ("JobState", "ExitCode", "StartTime", "EndTime")),
                 "Live and captured parent terminal records contradict")
        return raw, dict(source="LIVE_NATIVE_QUERY", original_observer=None if observer is None else observer[1])
    need(record["returncode"] == 1 and not record["stdout"].strip() and record["stderr"].strip() == PURGED_STDERR,
         "Native query failed for a reason other than the exact reviewed purged-ID response")
    need(observer is not None, "Purged parent requires genuine original observer provenance")
    return observer[0], dict(source="ORIGINAL_AFTERANY_CAPTURE_LIVE_ID_PURGED", original_observer=observer[1])


def step_argv(user):
    need(user == "paphiwetsa3", "Exact original owner step scope required")
    return ["/usr/local/bin/squeue", "--steps", "--noheader", "--user", user, "--format=%i|%T"]


def strict_step_rows(raw):
    """Parse complete owner inventory; malformed success is not release proof."""
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = [v.strip() for v in line.split("|")]
        need(len(row) == 2 and re.fullmatch(r"[0-9]+(?:_[0-9]+)?(?:\.[A-Za-z0-9_-]+)?", row[0])
             and row[1] in {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED", "STOPPED", "RESIZING",
                            "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED"},
             "Ambiguous complete owned step inventory")
    return raw


def validate_steps(record, parent):
    from scripts.train.joint.native_interruption import check_steps

    query_record(record, step_argv("paphiwetsa3"))
    need(record["returncode"] == 0 and not record["stderr"].strip(), "Complete owned step query failed")
    strict_step_rows(record["stdout"])
    check_steps(record["stdout"], parent)
    return record["stdout"]
