"""Synthetic raw-CPU envelope fixtures, never actual producer/data certification.

No image, class vector, source archive or wheel is read or produced. The cache
native verifier is an explicitly pending external interface in this test scope.
All native records below are synthetic P-style strings, never submitted.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.train.joint import imagenet


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(value if isinstance(value,str) else json.dumps(value,sort_keys=True)+"\n")


def native(q,request_sha,job="123",state="PENDING",observer=False):
    # Same native field shapes as P/test_preparation.py, including the actual
    # Slurm pending NumNodes range and CPU-only TresPerTask representation.
    r=q["capture_resources"] if observer else q["resources"]
    cpus=r["cpus"];mem=str(r["memory_mib"]//1024)+"G";version=q["root"].rsplit("-",1)[1]
    fields=dict(JobId=job,JobName=("elmo-native-imagenet-cpu-" if observer else
        "elmo-imagenet-cpu-preparation-")+version,UserId="paphiwetsa3(3346154)",
        Account=q["account"],Partition=q["partition"],QOS=q["qos"],ReqNodeList=q["node"],
        NumNodes="1-1" if state=="PENDING" else "1",NumTasks="1",NumCPUs=str(cpus),
        MinMemoryNode=mem,TimeLimit=r["time_limit"],Requeue="0",Restarts="0",JobState=state,
        Command=str(Path(q["root"])/("capture.sbatch" if observer else "producer.sbatch")),
        WorkDir=q["root"],Comment=("imagenet-cpu-observer:"+request_sha+":123" if observer else
            "imagenet-cpu:"+request_sha),Reason="Dependency" if observer else "JobHeldUser",
        Dependency="afterany:123(unfulfilled)" if observer else "(null)",
        NodeList=q["node"] if state!="PENDING" else "(null)",SubmitTime="2026-10-26T22:59:00",
        StartTime="2026-10-26T23:00:00" if state!="PENDING" else "Unknown",
        EndTime="2026-10-26T23:30:00" if state=="COMPLETED" else "Unknown",ExitCode="0:0",
        ReqTRES=f"cpu={cpus},mem={mem},node=1,billing={cpus}",
        AllocTRES=f"cpu={cpus},mem={mem},node=1,billing={cpus}" if state!="PENDING" else "",
        TresPerTask="cpu="+str(cpus))
    return " ".join(key+"="+value for key,value in fields.items())


class SyntheticRawChain:
    """A local fixture graph with real small-file hashes and fake native strings."""

    def __init__(self,tmp_path,version="v2"):
        self.root=tmp_path/"SYNTHETIC_ONLY_evidence";self.root.mkdir()
        self.original=Path("/coc/flash7/paphiwetsa3/tasks/"
            "elmo-qknorm-swiglu-rmsnorm-imagenet-20261026-v1/imagenet-cpu-"+version)
        self.source=tmp_path/"SYNTHETIC_ONLY_training_source";self.source.mkdir()
        required=("egomimic/rldb/imagenet_imagefolder.py","egomimic/rldb/imagenet_preparation.py",
            "egomimic/rldb/pixel_language.py","egomimic/rldb/pixel_proprio.py",
            "egomimic/scripts/embedding_process/qwen3_embedding.py")
        components={}
        for name in required:
            write(self.source/name,"# SYNTHETIC TEST COMPONENT, NEVER IMPORTED: "+name+"\n")
            components[name]=digest(self.source/name)
        write(self.root/"inputs/source.files.sha256","".join(
            value+"  "+name+"\n" for name,value in components.items()))
        for name in ("catalog","publication-receipt","qwen-expected","sidecars"):
            write(self.root/("inputs/"+name+".json"),{"synthetic_fixture_only":True})
        self.blocked={self.root/"inputs/source.tar.gz",self.root/"inputs/pillow-fixture.whl",
            self.root/"output/raw-inventory.json",self.root/"output/qwen-features.json",
            self.root/"output/imagefolder-manifest.json"}
        # These payload paths intentionally do not exist. The gateway must
        # authenticate their declared producer bindings without hashing them.
        self.q=dict(schema="imagenet-allocated-cpu-request-v1",state="BOUND_CPU_ONLY",
            root=str(self.original),request_id="SYNTHETIC-ONLY-raw-chain-"+version,uid=3346154,
            account="rl2-lab",partition="rl2-lab",qos="short",node="sky1",
            train_root="/coc/dataset/ImageNet/train",val_root="/coc/dataset/ImageNet/val",
            expected_counts=dict(classes=1000,train=1281167,val=50000),
            resources=dict(cpus=4,gpus=0,memory_mib=24576,time_limit="04:00:00"),
            capture_resources=dict(cpus=2,gpus=0,memory_mib=1024,time_limit="00:10:00"),
            native={name:"/opt/slurm/Ubuntu-24.04/current/bin/"+name for name in ("sbatch","scontrol","squeue")},
            runtime=dict(pillow_version="13.0.0"),training_authorized_by_this_packet=False,
            source=dict(revision="e9bff72c82618e74175886d16983f3cf90780fb0",entries=5,archive=self.unread("inputs/source.tar.gz","a"),
                manifest=self.bind("inputs/source.files.sha256"),publication=self.bind("inputs/publication-receipt.json")),
            pillow_wheel=self.unread("inputs/pillow-fixture.whl","b"),
            catalog=self.bind("inputs/catalog.json"),qwen_expected=self.bind("inputs/qwen-expected.json"),
            sidecars=self.bind("inputs/sidecars.json"),fixture_only_never_execute=True)
        write(self.root/"request.json",self.q);self.qsha=digest(self.root/"request.json")
        code_names=("request.json","guard.py","producer.sbatch","capture.sbatch","run_allocated.py","observe.py","submit_once.py")
        for name in code_names[1:]:write(self.root/name,"SYNTHETIC TEST COMPONENT NEVER EXECUTED\n")
        files={name:digest(self.root/name) for name in code_names}
        for path in sorted((self.root/"inputs").iterdir()):files[str(path.relative_to(self.root))]=digest(path)
        files.update({"inputs/source.tar.gz":"a"*64,"inputs/pillow-fixture.whl":"b"*64})
        write(self.root/"FILES.sha256","".join(value+"  "+name+"\n" for name,value in files.items()))
        self.fsha=digest(self.root/"FILES.sha256")
        self.claim=self.root/"claims"/self.qsha
        write(self.claim/"PREPARED.json",dict(request_sha256=self.qsha,files_manifest_sha256=self.fsha,
            parent_argv=imagenet._cpu_argv(self.q,self.qsha,self.fsha),fixture_only=True))
        for label,job,observer in (("parent","123",False),("capture","124",True)):
            write(self.claim/(label+"-sbatch.argv.json"),imagenet._cpu_argv(self.q,self.qsha,self.fsha,
                parent="123" if observer else None))
            write(self.claim/(label+"-sbatch.stdout"),job+"\n");write(self.claim/(label+"-sbatch.exit"),"0\n")
            write(self.claim/("CAPTURE_JOB_ID" if observer else "PARENT_JOB_ID"),job+"\n")
        for name,job,observer in (("HELD_NATIVE.txt","123",False),("CAPTURE_PENDING_NATIVE.txt","124",True),
            ("RELEASE_HELD_NATIVE.txt","123",False),("RELEASE_CAPTURE_PENDING_NATIVE.txt","124",True)):
            write(self.claim/name,native(self.q,self.qsha,job=job,observer=observer))
        write(self.claim/"RELEASE_ATTEMPT.json",dict(parent="123",capture="124",request_sha256=self.qsha,
            at_utc="2026-10-26T22:59:01+00:00",fixture_only=True))
        write(self.root/"request-claims"/self.q["request_id"]/"REQUEST_SHA256",self.qsha+"\n")
        for name in ("root-review.json","deployment-readback.json"):
            write(self.root/name,dict(status="PASS",root=str(self.original),request_sha256=self.qsha,
                files_manifest_sha256=self.fsha,fixture_only=True))
        self.result={}
        names=dict(class_manifest="classes.json",join="join-receipt.json",anchors="validation-anchors.json",
            runtime="runtime.json",qwen_fresh_inputs="qwen-fresh-inputs.json",source_stage="source-stage.json")
        for key,name in names.items():
            write(self.root/"output"/name,dict(synthetic_fixture_only=True));self.result[key]=self.bind("output/"+name)
        self.result.update(raw_inventory=self.unread("output/raw-inventory.json","c"),
            features=self.unread("output/qwen-features.json","d"),manifest=self.unread("output/imagefolder-manifest.json","e"))
        write(self.root/"output/native-running.txt",native(self.q,self.qsha,state="RUNNING"))
        self.result.update(native_running=self.bind("output/native-running.txt"),schema="imagenet-raw-cpu-producer-result-v1",
            status="PASS_DATA_PREPARATION_ONLY",job_id="123",request_sha256=self.qsha,files_manifest_sha256=self.fsha,
            source_revision=self.q["source"]["revision"],resources=self.q["resources"],uid=3346154,
            counts=dict(train=1281167,val=50000),classes=1000,training_admitted=False,fixture_only=True)
        write(self.root/"host-exit-123.txt","0\n")
        write(self.root/"capture-running-124.txt",native(self.q,self.qsha,job="124",state="RUNNING",observer=True))
        queries=[]
        for kind,name,content in (("job","native-123.txt",native(self.q,self.qsha,state="COMPLETED")),
            ("steps","native-123.steps.txt","No job steps in the system\n")):
            write(self.root/name,content);write(self.root/(name+".stderr"),"")
            queries.append(dict(kind=kind,exit_code=0,argv=[self.q["native"]["scontrol"],"show","job","123","-o","-dd"]
                if kind=="job" else [self.q["native"]["scontrol"],"show","step","-o"],
                stdout=self.bind(name),stderr=self.bind(name+".stderr")))
        self.capture=dict(schema="imagenet-cpu-native-capture-v1",parent_job_id="123",observer_job_id="124",
            request_sha256=self.qsha,files_manifest_sha256=self.fsha,observer_running=self.bind("capture-running-124.txt"),
            queries=queries,fixture_only=True)
        self.completion=dict(schema="imagenet-raw-cpu-completion-v1",status="PASS",job_id="123",
            job_name="elmo-imagenet-cpu-preparation-"+version,request_sha256=self.qsha,
            files_manifest_sha256=self.fsha,resources=self.q["resources"],node="sky1",uid=3346154,
            counts=dict(train=1281167,val=50000),training_admitted=False,source_revision=self.q["source"]["revision"],
            native_start="2026-10-26T23:00:00",native_end="2026-10-26T23:30:00",fixture_only=True)
        self.request=dict(source=str(self.source),data_cache="/fixture-only/pending-cache",
            image_generation_contract=dict(schema="native-imagenet-dataset-v1",preparation_components=components,
                cache_native={"fixture_only_external_pending":True},cache_readback={"fixture_only_external_pending":True}))
        self.refresh()

    def unread(self,name,char):
        return dict(path=str(self.original/name),sha256=char*64)

    def bind(self,name):
        return dict(path=str(self.original/name),sha256=digest(self.root/name))

    def actual(self,binding):
        return self.root/Path(binding["path"]).relative_to(self.original)

    def refresh_binding(self,binding):
        path=self.actual(binding)
        if path not in self.blocked and path.is_file():binding["sha256"]=digest(path)

    def refresh(self):
        # Rehash synthetic altered leaf records so negative tests exercise the
        # semantic chain rather than merely detecting stale file hashes.
        for value in self.result.values():
            if isinstance(value,dict) and set(value)=={"path","sha256"}:self.refresh_binding(value)
        self.refresh_binding(self.capture["observer_running"])
        for query in self.capture["queries"]:
            for key in ("stdout","stderr"):self.refresh_binding(query[key])
        write(self.root/"output/producer-result.json",self.result)
        write(self.root/"native-123.capture.json",self.capture)
        self.completion.update(producer_result=self.bind("output/producer-result.json"),host_exit=self.bind("host-exit-123.txt"),
            native_capture=self.bind("native-123.capture.json"),native_terminal=copy.deepcopy(self.capture["queries"][0]["stdout"]),
            raw_inventory=self.result["raw_inventory"])
        write(self.root/"completion.json",self.completion)
        contract=self.request["image_generation_contract"]
        contract["raw_preparation"]=dict(original_root=str(self.original),evidence_root=str(self.root),
            request=dict(path=str(self.root/"request.json"),sha256=self.qsha),
            completion=dict(path=str(self.root/"completion.json"),sha256=digest(self.root/"completion.json")),
            files_manifest_sha256=self.fsha)
        for target,key in (("manifest","manifest"),("class_language","class_manifest"),
            ("language_features","features"),("validation_anchor","anchors")):
            contract[target]=copy.deepcopy(self.result[key])

    def replace(self,path,old,new):
        path=Path(path);content=path.read_text();assert old in content
        path.write_text(content.replace(old,new))


@pytest.mark.parametrize("version",["v1","v2"])
def test_default_raw_cpu_chain_is_bounded_stdlib_without_torch_or_payload_reads(tmp_path,version):
    fixture=SyntheticRawChain(tmp_path,version)
    code='''import importlib.abc,json,pathlib,sys
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname=="torch" or fullname.startswith("torch."):
            raise AssertionError("Default raw CPU envelope imported Torch")
sys.meta_path.insert(0,NoTorch())
from scripts.train.joint import imagenet,contract
request,blocked=json.loads(sys.stdin.read());blocked={pathlib.Path(p) for p in blocked}
real_sha=contract.sha;hashes=[]
def hash_spy(path):
    path=pathlib.Path(path)
    assert path not in blocked,"Gateway hashed large payload: "+str(path)
    hashes.append(str(path));return real_sha(path)
contract.sha=hash_spy
real_open=pathlib.Path.open
def no_payload_open(self,*args,**kwargs):
    assert self not in blocked,"Gateway opened large payload: "+str(self)
    return real_open(self,*args,**kwargs)
pathlib.Path.open=no_payload_open
cache_calls=[]
def pending_cache(*args,**kwargs):
    assert kwargs=={"allocated":False}
    assert args[0]=={"fixture_only_external_pending":True}
    assert args[1]=={"fixture_only_external_pending":True}
    assert args[2] is request
    cache_calls.append(True)
imagenet.verify_cache_native=pending_cache
assert imagenet.verify_dataset(request)==({"train":1281167,"val":50000},1000)
assert cache_calls==[True] and hashes and "torch" not in sys.modules
expected_components={str(pathlib.Path(request["source"])/name)
    for name in request["image_generation_contract"]["preparation_components"]}
assert expected_components <= set(hashes),"Gateway skipped the preparation code closure"
assert not any(k.startswith("egomimic.") for k in sys.modules)
print("PASS_SYNTHETIC_RAW_CHAIN_STDLIB_ONLY_CACHE_INTERFACE_PATCHED")
'''
    source=Path(imagenet.__file__).resolve().parents[3]
    result=subprocess.run([sys.executable,"-c",code],cwd=source,
        input=json.dumps([fixture.request,[str(p) for p in fixture.blocked]]),text=True,capture_output=True,
        env=dict(os.environ,PYTHONPATH=str(source)),timeout=20)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=="PASS_SYNTHETIC_RAW_CHAIN_STDLIB_ONLY_CACHE_INTERFACE_PATCHED"


@pytest.mark.parametrize("damage",["observer_running_owner","observer_running_state","observer_submit_time",
    "parent_submit_time","held_not_held","release_held_changed","capture_dependency","capture_release_dependency",
    "release_capture_identity","parent_argv","capture_argv","prepared_argv","canonical_result",
    "canonical_observer","canonical_query","canonical_component","failure_marker","owned_steps","failed_host",
    "failed_terminal","failed_query","changed_code","changed_training_component","missing_request_claim"])
def test_default_raw_chain_rejects_rehashed_native_claim_or_output_mismatches(tmp_path,monkeypatch,damage):
    fixture=SyntheticRawChain(tmp_path);root=fixture.root;claim=fixture.claim
    cache_calls=[]
    monkeypatch.setattr(imagenet,"verify_cache_native",lambda *args,**kwargs:cache_calls.append(True))
    if damage=="observer_running_owner":fixture.replace(root/"capture-running-124.txt","paphiwetsa3(3346154)","paphiwetsa3(41003)")
    if damage=="observer_running_state":fixture.replace(root/"capture-running-124.txt","JobState=RUNNING","JobState=PENDING")
    if damage=="observer_submit_time":fixture.replace(root/"capture-running-124.txt","SubmitTime=2026-10-26T22:59:00","SubmitTime=2026-10-26T22:59:01")
    if damage=="parent_submit_time":fixture.replace(root/"output/native-running.txt","SubmitTime=2026-10-26T22:59:00","SubmitTime=2026-10-26T22:59:01")
    if damage=="held_not_held":fixture.replace(claim/"HELD_NATIVE.txt","Reason=JobHeldUser","Reason=Priority")
    if damage=="release_held_changed":fixture.replace(claim/"RELEASE_HELD_NATIVE.txt","SubmitTime=2026-10-26T22:59:00","SubmitTime=2026-10-26T22:59:01")
    if damage in ("capture_dependency","capture_release_dependency"):
        name="CAPTURE_PENDING_NATIVE.txt" if damage=="capture_dependency" else "RELEASE_CAPTURE_PENDING_NATIVE.txt"
        fixture.replace(claim/name,"afterany:123","afterany:125")
    if damage=="release_capture_identity":
        path=claim/"RELEASE_ATTEMPT.json";value=json.loads(path.read_text());value["capture"]="125";write(path,value)
    if damage in ("parent_argv","capture_argv"):
        path=claim/(damage.split("_")[0]+"-sbatch.argv.json");value=json.loads(path.read_text())
        value[value.index("--hold") if damage=="parent_argv" else value.index("--dependency=afterany:123")]=("--no-hold" if damage=="parent_argv" else "--dependency=afterany:125")
        write(path,value)
    if damage=="prepared_argv":
        path=claim/"PREPARED.json";value=json.loads(path.read_text());value["parent_argv"].remove("--hold");write(path,value)
    if damage in ("canonical_result","canonical_component"):
        key="class_manifest" if damage=="canonical_result" else "source_stage"
        old=fixture.actual(fixture.result[key]);new=old.with_name("different-"+old.name);new.write_bytes(old.read_bytes())
        fixture.result[key]=fixture.bind(str(new.relative_to(root)))
    if damage=="canonical_observer":
        old=root/"capture-running-124.txt";new=root/"different-observer.txt";new.write_bytes(old.read_bytes())
        fixture.capture["observer_running"]=fixture.bind(new.name)
    if damage=="canonical_query":
        old=root/"native-123.steps.txt";new=root/"different-steps.txt";new.write_bytes(old.read_bytes())
        fixture.capture["queries"][1]["stdout"]=fixture.bind(new.name)
    if damage=="failure_marker":write(root/"output/raw-inventory.json.failure.json",{"synthetic_fixture_failure":True})
    if damage=="owned_steps":write(root/"native-123.steps.txt","StepId=123.batch State=RUNNING\n")
    if damage=="failed_host":write(root/"host-exit-123.txt","1\n")
    if damage=="failed_terminal":fixture.replace(root/"native-123.txt","ExitCode=0:0","ExitCode=0:9")
    if damage=="failed_query":fixture.capture["queries"][0]["exit_code"]=1
    if damage=="changed_code":write(root/"guard.py","SYNTHETIC ALTERED WRAPPER\n")
    if damage=="changed_training_component":
        name=next(iter(fixture.request["image_generation_contract"]["preparation_components"]))
        write(fixture.source/name,"# SYNTHETIC ALTERED TRAINING COMPONENT\n")
    if damage=="missing_request_claim":
        (root/"request-claims"/fixture.q["request_id"]/"REQUEST_SHA256").unlink()
    fixture.refresh()
    with pytest.raises(ValueError):imagenet.verify_dataset(fixture.request)
    assert cache_calls==[],"Invalid raw chain reached the pending external cache interface"
