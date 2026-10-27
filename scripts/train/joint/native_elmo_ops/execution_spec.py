"""Immutable native Elmo phase submission and late actual-job request binding.

Standard library only; no source/checkpoint loads, training, or implicit retries.
The hash-bound host guard owns current scientific/infrastructure admission.
"""
import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone


RELEASE_OBSERVER_REQUIRED = True

SERVICE_COMMIT = "abbb00499096faabf7035fde23bf4b1eba984377"  # Verified upstream service publication.
SERVICE_SHA = "adf45524de3c2e6dba186572e7151d48bfca0b3adc8e6cf9defd686b47744c87"
SERVICE_FILE_COUNT = 35
QK_SERVICE_COMMIT = "5fad76e080d04c82346e3363041edafad815dcb5"  # Actual published PR325 head/export.
QK_SERVICE_SHA = "3350d6279e60acd18e19c00f4625784468f398fdf0f2fce2b1438e2bd30284ef"
QK_SERVICE_FILE_COUNT = 54

def service_pins(spec):
    if spec.get('arm') != 'qk': return SERVICE_COMMIT, SERVICE_SHA, SERVICE_FILE_COUNT
    need(isinstance(QK_SERVICE_COMMIT, str) and re.fullmatch('[0-9a-f]{40}', QK_SERVICE_COMMIT)
         and isinstance(QK_SERVICE_SHA, str) and re.fullmatch('[0-9a-f]{64}', QK_SERVICE_SHA)
         and type(QK_SERVICE_FILE_COUNT) is int and QK_SERVICE_FILE_COUNT >= 38,
         'QK interruption service is UNPUBLISHED/UNBOUND')
    return QK_SERVICE_COMMIT, QK_SERVICE_SHA, QK_SERVICE_FILE_COUNT

from qk_recovery_profile import (PARENT_ID as QK_PARENT_ID, NODE as QK_NODE, REQUEST_SHA as QK_REQUEST_SHA,
    TASK_ROOT as QK_TASK_ROOT, profile_run_id, supported_parameters)

PINS = {
    "plain": ("412789", "gpu11", "936e9acd30fa517f6f38311c0bcfd2b88544da8d476bd41125c4008de2845073"),
    "fused": ("412798", "gpu11", "e34190bfef155942fe216a1e5e81317cfa31b65d4c98ccb025a579c874fd3876"),
    "boundary": ("412800", "gpu03", "210e6f73c8a1a396716fab2f1e7e17f3b9d0663b099971eb05807181f9f00684"),
}
PINS['qk'] = (QK_PARENT_ID, QK_NODE, QK_REQUEST_SHA)
MODES = {
    "capture": ("native-elmo-interruption-capture-v1", "2", "1G", "00:10:00", "", "native_interruption"),
    "inspect": ("native-elmo-checkpoint-inspection-v1", "4", "24G", "00:20:00", "", "checkpoint_inspection"),
    "diagnostic": ("native-elmo-production-recovery-request-v1", "32", "192G", "01:00:00", "gres/gpu:h100:4", "production_recovery"),
    "finalize": ("native-elmo-interruption-finalizer-v1", "4", "24G", "00:20:00", "", "interruption_admission"),
    "full": ("joint-training-v2", "32", "192G", "7-00:00:00", "gres/gpu:h100:4", "run"),
}
NATIVE_KEYS = {"JobName", "UserId", "Account", "QOS", "Partition", "ReqNodeList", "NodeList",
               "NumCPUs", "MinMemoryNode", "TimeLimit", "Requeue", "Restarts", "Command", "TresPerNode"}

_LOADED_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()



def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    p = Path(path)
    need(p.stat().st_size <= 4 * 1024 * 1024, "Operations metadata exceeds 4 MiB")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def absolute(path):
    p = Path(path)
    need(p.is_absolute() and ".." not in p.parts and p.resolve() == p, "Noncanonical/symlink path")
    return p


def bound(binding):
    p = absolute(binding["path"])
    need(p.is_file() and not p.is_symlink() and sha(p) == binding["sha256"], "Bound file missing or changed: " + str(p))
    return p


def read_json(path):
    def pairs(items):
        need(len(items) == len(dict(items)), "Duplicate JSON keys")
        return dict(items)
    need(Path(path).stat().st_size <= 4 * 1024 * 1024, "Operations JSON exceeds 4 MiB")
    return json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + x)))


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write_once(path, value):
    p = absolute(path)
    payload = value if isinstance(value, bytes) else encoded(value)
    with p.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(p.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    need(p.read_bytes() == payload, "Immutable write readback mismatch")


def mkdir_once(path):
    p = absolute(path)
    p.mkdir()
    fd = os.open(p.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def manifest(root, binding):
    root = absolute(root)
    path = bound(binding)
    need(path == root / "FILES.sha256", "Manifest outside exact root")
    names = set()
    lines = path.read_text().splitlines()
    need(0 < len(lines) <= 128, "Unbounded operations/service manifest")
    for line in lines:
        digest, name = line.split("  ", 1)
        need(re.fullmatch(r"[a-f0-9]{64}", digest) and name not in names
             and not Path(name).is_absolute() and ".." not in Path(name).parts, "Invalid manifest entry")
        bound(dict(path=str(root / name), sha256=digest))
        names.add(name)
    need(names, "Empty manifest")
    return names


def fields(raw):
    pairs = re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9:/_-]*)=(\S+)", raw)
    need(len(pairs) == len(dict(pairs)), "Ambiguous native record")
    return dict(pairs)


def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    need(len(result.stdout.encode()) <= 1048576 and len(result.stderr.encode()) <= 65536, "Native query exceeded bounds")
    need(result.returncode == 0 and not result.stderr.strip(), "Native query failed: " + result.stderr[-1500:])
    return result.stdout


def paths(spec, spec_sha, job=None, request_sha=None):
    root = Path(spec["root"])
    return dict(spec_claim=root / "execution-claims" / spec_sha,
                request=root / "requests" / (spec_sha + "-" + str(job) + ".json"),
                request_root=root / ("scientific-claims" if spec["mode"] in {"diagnostic", "full"} else "producer-claims"),
                request_claim=root / ("scientific-claims" if spec["mode"] in {"diagnostic", "full"} else "producer-claims") / str(request_sha))


def read_spec(spec_path, spec_sha, mode, manifest_sha):
    need(isinstance(SERVICE_COMMIT, str) and re.fullmatch(r"[0-9a-f]{40}", SERVICE_COMMIT),
         "Adapter service is unpublished; explicit reviewed publication binding required")
    need(re.fullmatch(r"[a-f0-9]{64}", spec_sha) and re.fullmatch(r"[a-f0-9]{64}", manifest_sha), "Invalid hashes")
    p = bound(dict(path=str(spec_path), sha256=spec_sha))
    spec = read_json(p)
    need(spec["schema"] == "native-elmo-execution-spec-v1" and spec["mode"] == mode and mode in MODES,
         "Wrong execution schema/mode")
    service_commit, service_sha, service_count = service_pins(spec)
    root = absolute(spec["root"])
    if spec["arm"] == "qk":
        need(str(root).startswith(QK_TASK_ROOT + "/"), "QK execution escaped exact campaign")
    need(p.parent == root / "execution" and spec["arm"] in PINS, "Spec root/arm differs")
    claim_roots = spec["execution_claim_roots"]
    need(isinstance(claim_roots, list) and claim_roots
         and len({r["path"] for r in claim_roots}) == len(claim_roots)
         and str(root / "execution-claims") in {r["path"] for r in claim_roots},
         "Complete unique same-identity execution claim roots required")
    for item in claim_roots:
        need(absolute(item["path"]).is_dir() and absolute(item["terminal_directory"]).is_dir(),
             "Execution claim/terminal directory unavailable")
    need(spec["ops_manifest"] == dict(path=str(root / "FILES.sha256"), sha256=manifest_sha), "Ops manifest binding differs")
    files = manifest(root, spec["ops_manifest"])
    need({"execution_spec.py", "allocated.sbatch", "host_guard.py", "wrapper_common.py", "phase_observer.py", "observe-phase.sbatch", "qk_recovery_profile.py", "host_hashlib_compat.py"} <= files
         and all(Path(name).parts[0] not in {"execution", "templates", "execution-claims", "requests", "scientific-claims", "producer-claims", "status", "logs", "observer-specs", "observer-claims", "controller-specs"} for name in files),
         "Ops manifest missing executable guards or includes circular runtime artifacts")
    need(sha(root / "execution_spec.py") == sha(__file__), "Execution implementation drift")
    observer_path = absolute(spec["release_observer_path"])
    need(observer_path.parent == root / "observer-specs" and observer_path.suffix == ".json",
         "Every execution must predeclare its immutable observer path")
    parent_id, node, parent_sha = PINS[spec["arm"]]
    need(str(spec["original_parent_job_id"]) == parent_id and spec["parent_request"]["sha256"] == parent_sha,
         "Only the exact three original production parents are supported")
    parent = read_json(bound(spec["parent_request"]))
    run_id = profile_run_id(spec["arm"])
    need(parent["run_id"] == run_id and parent["kind"] == "native-subtask-world-model-v1"
         and parent["phase"] == "full" and parent["target_steps"] == 300000
         and parent["parameters"] == supported_parameters(parent) and parent["world_size"] == 4, "Parent scientific identity differs")
    service = spec["operational_service"]
    need(service["manifest_sha256"] == service_sha, "Wrong published recovery service")
    service_names = manifest(service["root"], dict(path=service["manifest"], sha256=service_sha))
    need(len(service_names) == service_count, "Incomplete adapter recovery service")
    template_path = bound(spec["request_template"])
    need(template_path.parent == root / "templates", "Template outside immutable template directory")
    template = read_json(template_path)
    need(template["schema"] == MODES[mode][0] and template["operational_service"] == service, "Request phase/service differs")
    parent_binding = template.get("parent_request") if mode != "full" else template["interruption"]["parent_request"]
    need(parent_binding == spec["parent_request"], "Template belongs to another production parent")
    native = spec["native_contract"]
    _, cpus, memory, limit, gres, _ = MODES[mode]
    need(set(native) == NATIVE_KEYS and all(isinstance(v, str) for v in native.values()), "ID-free complete native contract required")
    need(native["UserId"] == "paphiwetsa3(41003)" and native["Account"] == "loaner"
         and native["Partition"] == "rl2" and native["QOS"] == "loaner-normal"
         and native["NodeList"] == native["ReqNodeList"] == node
         and native["NumCPUs"] == cpus and native["MinMemoryNode"] == memory
         and native["TimeLimit"] == limit and native["TresPerNode"] == gres
         and native["Requeue"] == native["Restarts"] == "0"
         and native["Command"] == str(root / "allocated.sbatch"), "Native phase resources/owner/node changed")
    prefix = "elmo-native-recovery-" if mode not in {"diagnostic", "full"} else "elmo-large314m-continuous-"
    expected_prefix = prefix + spec["arm"] + "-" + mode + "-"
    need(native["JobName"].startswith(expected_prefix)
         and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", native["JobName"]), "Unsafe or out-of-scope job name")
    if mode == "full":
        need("producer_native" not in template and template["job_name"] == native["JobName"], "Full identity cannot gain producer fields")
        service_module(spec, "interruption_admission").compatible(parent, template)
    else:
        need(template["producer_native"] == native, "Template must omit only the future producer JobId")
        if mode != "capture":
            for key in ("source", "source_revision", "source_manifest", "source_manifest_sha256",
                        "data_manifest", "data_manifest_sha256", "lambda_runtime", "data_cache"):
                need(template.get(key) == parent.get(key), "Producer changed parent binding: " + key)
    if mode == "capture":
        need(str(template["parent_job_id"]) == parent_id and template["parent_native"]["JobId"] == parent_id,
             "Capture parent ID differs")
    return spec, template


def submission_argv(spec, spec_path, spec_sha, manifest_sha):
    n, root = spec["native_contract"], Path(spec["root"])
    return ["/usr/local/bin/sbatch", "--parsable", "--hold", "--no-requeue", "--export=NONE",
            "--account=" + n["Account"], "--qos=" + n["QOS"], "--partition=" + n["Partition"],
            "--job-name=" + n["JobName"], "--nodelist=" + n["ReqNodeList"], "--nodes=1", "--ntasks=1",
            "--cpus-per-task=" + n["NumCPUs"], "--mem=" + n["MinMemoryNode"], "--time=" + n["TimeLimit"],
            "--gres=" + ("gpu:h100:4" if n["TresPerNode"] else "none"),
            "--comment=elmo-execution:" + spec_sha, "--chdir=" + str(root),
            "--output=" + str(root / "logs/%x-%j.out"), "--error=" + str(root / "logs/%x-%j.err"),
            str(root / "allocated.sbatch"), str(root), spec["mode"], str(spec_path), spec_sha, manifest_sha]


def held_native(raw, spec, spec_sha, job):
    f, n = fields(raw), spec["native_contract"]
    need(f.get("JobId") == job and f.get("JobState") == "PENDING" and f.get("Reason") == "JobHeldUser",
         "Exact held native acceptance required")
    need(all(f.get(k, "") == v for k, v in n.items() if k != "NodeList")
         and f.get("NodeList", "") in {"", "(null)", n["NodeList"]}
         and f.get("NumNodes") in {"1", "1-1"} and f.get("NumTasks") == "1"
         and f.get("Comment") == "elmo-execution:" + spec_sha
         and f.get("SubmitTime") not in {None, "Unknown", "None"}, "Held native spec/argv/resources differ")
    need(f.get("AllocTRES", "") in {"", "(null)"} and f.get("GRES", "") in {"", "(null)"}
         and f.get("StartTime") in {None, "Unknown", "N/A"},
         "Held job already has an allocation/start")
    requested = f.get("ReqTRES", "").split(",")
    memory = n["MinMemoryNode"]
    need("cpu=" + n["NumCPUs"] in requested and "node=1" in requested
         and bool(set(requested) & {"mem=" + memory, "mem=" + str(int(memory[:-1]) * 1024) + "M"}),
         "Held requested CPU/memory/node TRES differ")
    gpu = {part for part in requested if "gpu" in part.lower()}
    need(gpu <= ({"gres/gpu=4", "gres/gpu:h100=4"} if n["TresPerNode"] else set()),
         "Held requested GPU TRES conflict")
    # Pending allocations may not yet expose JOB_GRES. If present it must
    # agree; absence never substitutes for eventual RUNNING GPU-index proof.
    job_gres = f.get("JOB_GRES", "")
    if n["TresPerNode"]:
        need("JOB_GRES" not in f or job_gres == "gpu:h100:4", "Held JOB_GRES conflicts with four H100s")
    else:
        need(job_gres in {"", "(null)"}, "CPU held allocation has JOB_GRES")
    for key, value in f.items():
        if key in {"ReqTRES", "AllocTRES", "TresPerNode", "JOB_GRES", "GRES"}:
            continue
        if key.lower().startswith("tresper") or "gpu" in key.lower():
            # One-task CPU allocation metadata is legitimate; per-task GPU
            # requests or any other GPU-bearing auxiliary field are not part
            # of the exact spec, even if the primary TRES fields look right.
            need("gpu" not in value.lower() and "gpu" not in key.lower(),
                 "Unexpected held auxiliary GPU resource field")
            if key == "TresPerTask":
                need(value == "cpu:" + n["NumCPUs"], "Held per-task CPU resources differ")
    return f


def materialized(spec, template, job):
    if spec["mode"] == "full":
        return bound(spec["request_template"]).read_bytes()
    q = copy.deepcopy(template)
    q["producer_native"]["JobId"] = job
    return encoded(q)


def bind_accepted(spec_path, spec_sha, mode, manifest_sha, job, raw):
    """Separate operation after exact accepted native ID; never submits/releases."""
    spec, template = read_spec(spec_path, spec_sha, mode, manifest_sha)
    need(str(job).isdecimal(), "Actual accepted job ID required")
    job = str(job)
    held_native(raw, spec, spec_sha, job)
    p = paths(spec, spec_sha, job)
    prepared = read_json(p["spec_claim"] / "PREPARED.json")
    argv = submission_argv(spec, spec_path, spec_sha, manifest_sha)
    need(prepared["spec_sha256"] == spec_sha and prepared["argv"] == argv, "Submission argv/spec claim differs")
    need((p["spec_claim"] / "PARENT_JOB_ID").read_text().strip() == job, "Accepted ID not preserved in original spec claim")
    need((p["spec_claim"] / "sbatch.stdout").read_text().strip() == job
         and (p["spec_claim"] / "sbatch.exit").read_text().strip() == "0",
         "Binding requires the preserved unambiguous accepted sbatch outcome")
    payload = materialized(spec, template, job)
    request_sha = hashlib.sha256(payload).hexdigest()
    p = paths(spec, spec_sha, job, request_sha)
    write_once(p["request"], payload)
    mkdir_once(p["request_claim"])
    write_once(p["request_claim"] / "PREPARED.json", dict(request_sha256=request_sha, spec_sha256=spec_sha,
               spec_path=str(spec_path), actual_request_path=str(p["request"]), argv=argv,
               original_parent_job_id=str(spec["original_parent_job_id"])))
    write_once(p["request_claim"] / "PARENT_JOB_ID", (job + "\n").encode())
    write_once(p["spec_claim"] / "HELD_NATIVE.txt", raw.encode())
    acceptance = dict(schema="native-elmo-held-execution-binding-v1", spec_sha256=spec_sha,
                      ops_manifest_sha256=manifest_sha, job_id=job, argv=argv,
                      request=dict(path=str(p["request"]), sha256=request_sha),
                      request_claim=str(p["request_claim"]), held_native=dict(path=str(p["spec_claim"] / "HELD_NATIVE.txt"),
                      sha256=sha(p["spec_claim"] / "HELD_NATIVE.txt")))
    write_once(p["spec_claim"] / "acceptance.json", acceptance)
    return acceptance


def read_binding(spec_path, spec_sha, mode, manifest_sha):
    spec, template = read_spec(spec_path, spec_sha, mode, manifest_sha)
    sc = paths(spec, spec_sha)["spec_claim"]
    a = read_json(sc / "acceptance.json")
    job = a["job_id"]
    need(a["schema"] == "native-elmo-held-execution-binding-v1" and str(job).isdecimal()
         and a["spec_sha256"] == spec_sha and a["ops_manifest_sha256"] == manifest_sha
         and a["argv"] == submission_argv(spec, spec_path, spec_sha, manifest_sha), "Acceptance spec/argv differs")
    need(a["held_native"]["path"] == str(sc / "HELD_NATIVE.txt"), "Held source escaped spec claim")
    held = held_native(bound(a["held_native"]).read_text(), spec, spec_sha, job)
    request_path = bound(a["request"])
    p = paths(spec, spec_sha, job, a["request"]["sha256"])
    need(request_path == p["request"] and a["request_claim"] == str(p["request_claim"])
         and request_path.read_bytes() == materialized(spec, template, job), "Actual request differs from sole authorized late binding")
    for directory in (sc, p["request_claim"]):
        need(absolute(directory).is_dir() and (directory / "PARENT_JOB_ID").read_text().strip() == job,
             "Spec/request accepted job mismatch")
    prepared = read_json(p["request_claim"] / "PREPARED.json")
    need(prepared["request_sha256"] == a["request"]["sha256"] and prepared["spec_sha256"] == spec_sha
         and prepared["argv"] == a["argv"] and prepared["actual_request_path"] == str(request_path)
         and prepared["spec_path"] == str(spec_path)
         and prepared["original_parent_job_id"] == str(spec["original_parent_job_id"]), "Canonical request-SHA claim mismatch")
    original = read_json(sc / "PREPARED.json")
    need(original["spec_sha256"] == spec_sha and original["argv"] == a["argv"]
         and original["spec_path"] == str(spec_path) and original["arm"] == spec["arm"]
         and original["parent_request"] == spec["parent_request"], "Original submission intent changed")
    need((sc / "sbatch.stdout").read_text().strip() == job and (sc / "sbatch.exit").read_text().strip() == "0",
         "Original submission acceptance became unresolved")
    return spec, request_path, read_json(request_path), job, p["request_claim"], held


def service_module(spec, name):
    """Activate only the verified published service before importing its stdlib guards."""
    service_commit, service_sha, service_count = service_pins(spec)
    need(isinstance(service_commit, str) and re.fullmatch(r"[0-9a-f]{40}", service_commit),
         "Adapter service is unpublished; explicit reviewed publication binding required")
    need(name in {"native_interruption", "interruption_admission", "original_observer", "phase_observer_proof"},
         "Unapproved pre-Torch service import")
    binding = spec["operational_service"]
    need(binding["manifest_sha256"] == service_sha, "Wrong published recovery service")
    root = absolute(binding["root"])
    names = manifest(root, dict(path=binding["manifest"], sha256=service_sha))
    need(len(names) == service_count, "Incomplete adapter recovery service")
    from wrapper_common import install_host_hash_compat
    install_host_hash_compat(spec)
    path = root / "scripts/train/joint/service_bootstrap.py"
    import_spec = importlib.util.spec_from_file_location("_elmo_bound_bootstrap", path)
    module = importlib.util.module_from_spec(import_spec)
    import_spec.loader.exec_module(module)
    module.activate(dict(operational_service=spec["operational_service"]))
    return importlib.import_module("scripts.train.joint." + name)


def load_execution(spec_path, spec_sha, mode, manifest_sha):
    """Read-only guard, safe before Torch; authenticates native in its own environment."""
    spec, request_path, q, job, claim, held = read_binding(spec_path, spec_sha, mode, manifest_sha)
    expected = dict(spec["native_contract"], JobId=job)
    native = service_module(spec, "native_interruption")
    native.process_identity(expected, cpu=mode not in {"diagnostic", "full"})
    raw = command(["/usr/local/bin/scontrol", "show", "job", job, "-o", "-dd"])
    f = native.native_identity(raw, expected, "RUNNING")
    need(f.get("SubmitTime") == held["SubmitTime"] and f.get("Comment") == "elmo-execution:" + spec_sha,
         "RUNNING job is not the held execution")
    release = read_json(paths(spec, spec_sha)["spec_claim"] / "RELEASE_ATTEMPT.json")
    need(release["job_id"] == job and release["spec_sha256"] == spec_sha
         and release["request_sha256"] == sha(request_path), "Preserved release attempt differs")
    return spec, request_path, q, job, claim


def record_running_admission(spec_path, spec_sha, mode, manifest_sha):
    spec, path, q, job, claim = load_execution(spec_path, spec_sha, mode, manifest_sha)
    raw = command(["/usr/local/bin/scontrol", "show", "job", job, "-o", "-dd"])
    service_module(spec, "native_interruption").native_identity(raw, dict(spec["native_contract"], JobId=job), "RUNNING")
    record = dict(job_id=job, request_sha256=sha(path), spec_sha256=spec_sha, native=raw)
    write_once(claim / "RUNNING_ADMISSION.json", record)
    return record


def execution_claim_census(spec, *, current_job=None, current_request_sha=None, allow_held=False):
    """All registered spec ledgers, including claims without accepted/materialized IDs.

    Never equate an absent/partial acceptance with no job. Current held admission
    is separate from the RUNNING-only scientific gate and preserves raw state.
    """
    need((current_job is None) == (current_request_sha is None), "Incomplete current execution binding")
    native = service_module(spec, "native_interruption")
    observer = service_module(spec, "original_observer")
    step_query = observer.query(observer.step_argv("paphiwetsa3"))
    observer.query_record(step_query, observer.step_argv("paphiwetsa3"))
    need(step_query["returncode"] == 0 and not step_query["stderr"].strip(), "Complete owned step query failed")
    observer.strict_step_rows(step_query["stdout"])
    evidence, seen = [], set()
    for entry in spec["execution_claim_roots"]:
        directory = absolute(entry["path"])
        for claim in sorted(directory.iterdir()):
            need(claim.is_dir() and not claim.is_symlink() and re.fullmatch(r"[0-9a-f]{64}", claim.name),
                 "Unknown execution ledger entry requires inspection")
            prepared = read_json(claim / "PREPARED.json")
            need(prepared["spec_sha256"] == claim.name and prepared["arm"] == spec["arm"]
                 and prepared["parent_request"] == spec["parent_request"], "Execution root includes another identity")
            previous_spec_path = bound(dict(path=prepared["spec_path"], sha256=claim.name))
            previous = read_json(previous_spec_path)
            need(previous["arm"] == spec["arm"] and previous["parent_request"] == spec["parent_request"]
                 and paths(previous, claim.name)["spec_claim"] == claim,
                 "Historical execution claim escaped its actual spec root")
            need((claim / "acceptance.json").is_file(), "Unresolved submission or partial actual-ID binding; inspect, never retry")
            args = (str(previous_spec_path), claim.name, previous["mode"], prepared["ops_manifest_sha256"])
            old, path, q, job, request_claim, held = read_binding(*args)
            need(job not in seen, "Duplicate actual job across execution claims")
            seen.add(job)
            expected = dict(old["native_contract"], JobId=job)
            if job == current_job:
                need(Path(old["root"]) == Path(spec["root"]) and sha(path) == current_request_sha,
                     "Current job belongs to another execution/request")
                raw = command(["/usr/local/bin/scontrol", "show", "job", job, "-o", "-dd"])
                if allow_held:
                    f = held_native(raw, old, claim.name, job)
                    service_module(spec, "phase_observer_proof").check_owned_steps(step_query["stdout"], job)
                else:
                    f = native.native_identity(raw, expected, "RUNNING")
                need(f.get("SubmitTime") == held["SubmitTime"] and f.get("Comment") == "elmo-execution:" + claim.name,
                     "Current native identity differs from accepted execution")
                evidence.append(dict(job_id=job, spec_sha256=claim.name, current=True, actual_state=f["JobState"]))
                continue
            captured, raw = observed_terminal(*args, expected_job=job, expected_request_sha=sha(path),
                terminal_directory=entry["terminal_directory"])
            service_module(spec, "phase_observer_proof").check_owned_steps(step_query["stdout"], job)
            evidence.append(dict(job_id=job, spec_sha256=claim.name, observer_spec=captured["observer_spec"],
                terminal_sha256=captured["producer_native"]["sha256"], terminal_state=captured["producer_terminal_state"]))
    if current_job is not None:
        need(current_job in seen, "Current job absent from complete execution-claim census")
    return evidence


def operations_module(spec, name):
    """Load fixed operations code only after the caller's entire immutable closure."""
    need(name in {"host_guard", "wrapper_common", "phase_observer"}, "Unapproved operations module")
    root = absolute(spec["root"])
    if spec["arm"] == "qk":
        need(str(root).startswith(QK_TASK_ROOT + "/"), "QK execution escaped exact campaign")
    names = manifest(root, spec["ops_manifest"])
    need("execution_spec.py" in names and sha(root / "execution_spec.py") == sha(__file__),
         "Operations caller implementation drift")
    need(name + ".py" in names, "Operations module absent from immutable manifest")
    if name == "host_guard":
        operations_module(spec, "wrapper_common")
    location = root / (name + ".py")
    existing = sys.modules.get(name)
    if existing is not None:
        need(getattr(existing, "__file__", None) == str(location)
             and getattr(existing, "_LOADED_IMPLEMENTATION_SHA256", None) == sha(location)
             and getattr(existing, "_verified_ops_manifest_sha256", None) in {None, spec["ops_manifest"]["sha256"]},
             "Foreign or unverified operations module already imported")
        existing._verified_ops_manifest_sha256 = spec["ops_manifest"]["sha256"]
        existing._verified_ops_member_sha256 = sha(location)
        return existing
    loader = importlib.util.spec_from_file_location(name, location)
    module = importlib.util.module_from_spec(loader)
    module._verified_ops_manifest_sha256 = spec["ops_manifest"]["sha256"]
    module._verified_ops_member_sha256 = sha(location)
    sys.modules[name] = module
    try:
        loader.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def observed_terminal(spec_path, spec_sha, mode, manifest_sha, *, expected_job,
                      expected_request_sha, terminal_directory):
    """Observer final commit is required even when an isolated raw terminal exists."""
    spec, path, q, job, claim, held = read_binding(spec_path, spec_sha, mode, manifest_sha)
    need(job == expected_job and sha(path) == expected_request_sha,
         "Observer terminal belongs to another accepted request")
    observer_path = absolute(spec["release_observer_path"])
    need(observer_path.parent == Path(spec["root"]) / "observer-specs", "Observer escaped execution root")
    result = operations_module(spec, "phase_observer").verify_captured(str(observer_path), sha(observer_path), manifest_sha)
    need(result["schema"] == "native-elmo-phase-observer-result-v1"
         and result["status"] == "GENUINE_PRODUCER_TERMINAL_CAPTURED"
         and result["producer_execution"] == dict(path=str(spec_path), sha256=spec_sha, mode=mode, ops_manifest_sha256=manifest_sha)
         and result["producer_request"] == dict(path=str(path), sha256=expected_request_sha)
         and result["producer_job_id"] == job and result["producer_phase"] == mode
         and result["observer_spec"] == dict(path=str(observer_path), sha256=sha(observer_path)),
         "Observer final commit identity differs from accepted execution")
    terminal = absolute(terminal_directory) / ("native-" + job + ".txt")
    need(result["producer_native"]["path"] == str(terminal), "Terminal escaped registered claim evidence directory")
    raw = bound(result["producer_native"]).read_text()
    native = service_module(spec, "native_interruption")
    state = native.fields(raw).get("JobState")
    need(state in native.TERMINAL and state == result["producer_terminal_state"], "Observer terminal state differs")
    f = native.native_identity(raw, dict(spec["native_contract"], JobId=job), state)
    need(f.get("SubmitTime") == held["SubmitTime"] and f.get("Comment") == "elmo-execution:" + spec_sha,
         "Terminal identity differs from actual held execution")
    return result, raw


def request_claim_terminal(spec, claim, prepared, job, terminal_directory):
    """Follow a canonical request-SHA claim back to its immutable execution spec."""
    old_path = bound(dict(path=prepared["spec_path"], sha256=prepared["spec_sha256"]))
    old = read_json(old_path)
    need(old["arm"] == spec["arm"] and old["parent_request"] == spec["parent_request"],
         "Prior scientific claim belongs to another identity")
    need(paths(old, prepared["spec_sha256"], job, claim.name)["request_claim"] == claim,
         "Prior request claim escaped actual execution root")
    return observed_terminal(str(old_path), prepared["spec_sha256"], old["mode"], old["ops_manifest"]["sha256"],
        expected_job=job, expected_request_sha=claim.name, terminal_directory=terminal_directory)


def host_preflight(spec, template, *, current_job=None, current_request_sha=None):
    operations_module(spec, "host_guard").admit_before_submission(
        spec, template, current_job=current_job, current_request_sha=current_request_sha)


def submit_held(spec_path, spec_sha, mode, manifest_sha):
    """One sbatch maximum. Any existing claim/unknown outcome stops this operation."""
    spec, template = read_spec(spec_path, spec_sha, mode, manifest_sha)
    host_preflight(spec, template)
    sc = paths(spec, spec_sha)["spec_claim"]
    mkdir_once(sc)  # exclusive durable claim before the sole native side effect
    argv = submission_argv(spec, spec_path, spec_sha, manifest_sha)
    write_once(sc / "PREPARED.json", dict(spec_sha256=spec_sha, ops_manifest_sha256=manifest_sha,
               spec_path=str(spec_path), arm=spec["arm"], parent_request=spec["parent_request"],
               argv=argv, original_parent_job_id=str(spec["original_parent_job_id"]),
               created_at=datetime.now(timezone.utc).isoformat()))
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        write_once(sc / "SUBMISSION_UNKNOWN.json", dict(error=type(error).__name__, inspect_required=True))
        raise RuntimeError("Submission outcome unknown; inspect exact existing spec claim/native; never retry") from error
    write_once(sc / "sbatch.stdout", result.stdout.encode())
    write_once(sc / "sbatch.stderr", result.stderr.encode())
    write_once(sc / "sbatch.exit", (str(result.returncode) + "\n").encode())
    job = result.stdout.strip()
    need(result.returncode == 0 and job.isdecimal(), "Submission outcome unresolved; inspect existing claim/native; never retry")
    write_once(sc / "PARENT_JOB_ID", (job + "\n").encode())
    raw = command(["/usr/local/bin/scontrol", "show", "job", job, "-o", "-dd"])
    return bind_accepted(spec_path, spec_sha, mode, manifest_sha, job, raw)


def require_release_observer(spec, spec_sha, manifest_sha, job, request_sha):
    """Fresh actual observer dependency is mandatory for every producer release."""
    observer_path = absolute(spec["release_observer_path"])
    binding = {"path": str(observer_path), "sha256": sha(observer_path)}
    observer = read_json(bound(binding))
    need(observer["producer_execution"] == dict(path=read_json(paths(spec, spec_sha)["spec_claim"] / "PREPARED.json")["spec_path"],
         sha256=spec_sha, mode=spec["mode"], ops_manifest_sha256=manifest_sha),
         "Observer spec does not bind this exact producer execution")
    need(observer_path.parent == Path(spec["root"]) / "observer-specs",
         "Release observer spec escaped exact operations root")
    module = operations_module(spec, "phase_observer")
    ready = module.verify_ready_for_release(str(observer_path), binding["sha256"], manifest_sha)
    need(ready["schema"] == "native-elmo-phase-observer-ready-v1"
         and ready["observer_spec_sha256"] == binding["sha256"]
         and ready["production_completion_proven"] is False
         and ready["producer_job_id"] == job and ready["producer_request_sha256"] == request_sha
         and str(ready["observer_job_id"]).isdecimal(), "Observer readiness belongs to another producer")
    return ready


def release_bound(spec_path, spec_sha, mode, manifest_sha):
    """Separate release only after immutable late binding; an unknown release is never repeated."""
    spec, path, q, job, claim, held = read_binding(spec_path, spec_sha, mode, manifest_sha)
    host_preflight(spec, q, current_job=job, current_request_sha=sha(path))
    raw = command(["/usr/local/bin/scontrol", "show", "job", job, "-o", "-dd"])
    current = held_native(raw, spec, spec_sha, job)
    need(current["SubmitTime"] == held["SubmitTime"], "Accepted held identity changed")
    observer = require_release_observer(spec, spec_sha, manifest_sha, job, sha(path))
    sc = paths(spec, spec_sha)["spec_claim"]
    write_once(sc / "RELEASE_HELD_NATIVE.txt", raw.encode())
    write_once(sc / "RELEASE_ATTEMPT.json", dict(job_id=job, spec_sha256=spec_sha,
               request_sha256=sha(path), held_native_sha256=hashlib.sha256(raw.encode()).hexdigest(),
               observer_readiness=observer))
    result = subprocess.run(["/usr/local/bin/scontrol", "release", job], capture_output=True, text=True, timeout=30)
    write_once(sc / "RELEASE_RESULT.json", dict(job_id=job, returncode=result.returncode,
               stdout=result.stdout, stderr=result.stderr, completion_proven=False))
    need(result.returncode == 0, "Release unresolved; inspect exact native state; never retry release")
    return dict(job_id=job, request_sha256=sha(path), status="RELEASE_ACCEPTED_NOT_RUNNING_OR_COMPLETED")


def inspect_existing(spec_path, spec_sha, mode, manifest_sha):
    """Read only. Report exact queue candidates; do not infer an ID or repair a claim."""
    spec, _ = read_spec(spec_path, spec_sha, mode, manifest_sha)
    sc = paths(spec, spec_sha)["spec_claim"]
    queue = command(["/usr/local/bin/squeue", "--noheader", "--user", "paphiwetsa3", "--format=%i|%j|%T|%k"])
    rows = []
    for line in queue.splitlines():
        row = line.split("|")
        need(len(row) == 4, "Ambiguous owner queue")
        if row[1] == spec["native_contract"]["JobName"] or row[3] == "elmo-execution:" + spec_sha:
            rows.append(row)
    return dict(spec_claim=str(sc), claim_exists=sc.exists(), native_candidates=rows,
                automatic_retry_authorized=False, binding_exists=(sc / "acceptance.json").is_file())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=["plan", "submit-held", "release-bound", "inspect-existing"])
    parser.add_argument("spec_path"); parser.add_argument("spec_sha"); parser.add_argument("mode", choices=MODES)
    parser.add_argument("manifest_sha")
    args = parser.parse_args()
    values = (args.spec_path, args.spec_sha, args.mode, args.manifest_sha)
    if args.operation == "plan":
        spec, _ = read_spec(*values)
        result = dict(argv=submission_argv(spec, args.spec_path, args.spec_sha, args.manifest_sha), submitted=False)
    else:
        result = {"submit-held": submit_held, "release-bound": release_bound, "inspect-existing": inspect_existing}[args.operation](*values)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
