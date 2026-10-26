"""Portable synthetic execution fixture; no deployment request is read."""
import copy, hashlib, json, shutil, subprocess
from pathlib import Path

e=HERE=SERVICE=None

def put(p, obj):
    p.write_bytes(e.encoded(obj))

def fixture(tmp_path, mode="inspect", arm="plain"):
    root = tmp_path / (mode + "-" + arm)
    root.mkdir()
    for name in ("execution", "templates", "requests", "execution-claims", "scientific-claims", "producer-claims", "status", "logs"):
        (root / name).mkdir()
    shutil.copyfile(HERE / "execution_spec.py", root / "execution_spec.py")
    (root / "allocated.sbatch").write_text("#!/bin/bash\nexit 99\n")
    (root / "host_guard.py").write_text("def admit_before_submission(spec, template, **kwargs):\n    raise RuntimeError('test guard never executes')\n")
    (root / "FILES.sha256").write_text("".join(e.sha(root / name) + "  " + name + "\n" for name in ("allocated.sbatch", "execution_spec.py", "host_guard.py")))
    parent_id, node, parent_sha = e.PINS[arm]
    run_id="mot-elmo-pickplace-continuous-large314m-"+arm+"-robot-human-camera-subtask-s42-300k-20261026-v1"
    parent=dict(schema="joint-training-v2",phase="full",kind="native-subtask-world-model-v1",run_id=run_id,
        job_name="synthetic-original-"+arm,target_steps=300000,parameters=314187064,world_size=4,
        output=str(root/"original-output"),local_logs=str(root/"original-logs"))
    for key in ("source","source_revision","source_manifest","source_manifest_sha256","data_manifest","data_manifest_sha256","lambda_runtime","data_cache"):
        parent[key]="synthetic-"+key
    parent_path=root/"synthetic-parent.json";put(parent_path,parent)
    parent_sha=e.sha(parent_path)
    e.PINS=dict(e.PINS, **{arm:(parent_id,node,parent_sha)})
    _, cpus, mem, limit, gres, _ = e.MODES[mode]
    n = dict(JobName=("elmo-native-recovery-" if mode not in {"diagnostic", "full"} else "elmo-large314m-continuous-") + arm + "-" + mode + "-test",
             UserId="paphiwetsa3(41003)", Account="loaner", QOS="loaner-normal", Partition="rl2", ReqNodeList=node,
             NodeList=node, NumCPUs=cpus, MinMemoryNode=mem, TimeLimit=limit, Requeue="0", Restarts="0",
             Command=str(root / "allocated.sbatch"), TresPerNode=gres)
    service = dict(root=str(SERVICE), manifest=str(SERVICE / "FILES.sha256"), manifest_sha256=e.SERVICE_SHA)
    pb = dict(path=str(parent_path), sha256=parent_sha)
    if mode == "full":
        q = copy.deepcopy(parent)
        q.update(job_name=n["JobName"], output=str(root / "production-output"), local_logs=str(root / "production-logs"),
                 operational_service=service, interruption=dict(parent_request=pb))
    else:
        q = dict(schema=e.MODES[mode][0], parent_request=pb, operational_service=service, producer_native=n,
                 output=str(root / "phase-output"), allocated_prerequisites_path=str(root / "status/prerequisites.json"))
        for key in ("source", "source_revision", "source_manifest", "source_manifest_sha256", "data_manifest", "data_manifest_sha256", "lambda_runtime", "data_cache"):
            q[key] = parent.get(key)
        if mode == "capture":
            q.update(parent_job_id=parent_id, parent_native=dict(JobId=parent_id))
    template = root / "templates/phase.json"
    put(template, q)
    spec = dict(schema="native-elmo-execution-spec-v1", root=str(root), mode=mode, arm=arm,
                original_parent_job_id=parent_id, parent_request=pb, request_template=dict(path=str(template), sha256=e.sha(template)),
                native_contract=n, operational_service=service,
                execution_claim_roots=[dict(path=str(root / "execution-claims"),terminal_directory=str(root / "status"))],
                ops_manifest=dict(path=str(root / "FILES.sha256"), sha256=e.sha(root / "FILES.sha256")))
    sp = root / "execution/phase.json"
    put(sp, spec)
    return dict(spec=spec, template=q, path=sp, args=(str(sp), e.sha(sp), mode, e.sha(root / "FILES.sha256")), root=root)

def native(f, state="PENDING", job="900001", **changes):
    n = dict(f["spec"]["native_contract"], JobId=job, JobState=state, Reason="JobHeldUser", NumNodes="1-1",
             NumTasks="1", SubmitTime="2026-10-27T01:00:00", StartTime="Unknown", NodeList="", AllocTRES="(null)",
             ReqTRES="cpu=" + f["spec"]["native_contract"]["NumCPUs"] + ",mem=" + f["spec"]["native_contract"]["MinMemoryNode"] + ",node=1,billing=32",
             Comment="elmo-execution:" + f["args"][1])
    if state == "RUNNING":
        n.update(Reason="None", NumNodes="1", NodeList=f["spec"]["native_contract"]["NodeList"],
                 AllocTRES=n["ReqTRES"], StartTime="2026-10-27T01:00:02")
        if n["TresPerNode"]:
            n.update(JOB_GRES="gpu:h100:4", GRES="gpu:h100:4(IDX:0-3)")
    n.update(changes)
    return " ".join(k + "=" + v for k, v in n.items()) + "\n"

def accepted(f, monkeypatch, job="900001"):
    monkeypatch.setattr(e, "host_preflight", lambda *a, **kw: None)
    monkeypatch.setattr(e.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, job+"\n", ""))
    monkeypatch.setattr(e, "command", lambda argv: native(f,job=job))
    return e.submit_held(*f["args"])
