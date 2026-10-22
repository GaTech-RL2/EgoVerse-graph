"""Synthetic evidence tests + actual535d composed-config comparison; no GPU claim."""
import copy
import json
from pathlib import Path
import uuid

import pytest

from scripts.train.joint import verify_temporal_pair as tp
from scripts.train.joint.contract import compare_configs
from scripts.train.joint.continuation import STABLE_FIELDS

BASE = Path(__file__).resolve().parent / "fixtures/joint_temporal_pair_535d"


def template(arm, phase):
    q = json.loads((BASE/f"{arm}-{phase}.template.json").read_text())
    return {k: v for k, v in q.items() if not k.startswith("_")}


def composed(arm, phase):
    return json.loads((BASE/f"{arm}-{phase}.composed.json").read_text())


@pytest.fixture
def pair(tmp_path, monkeypatch):
    requests, states = [], {}
    def put(name, obj, text=False):
        path = tmp_path/name
        path.write_text(obj if text else json.dumps(obj, sort_keys=True))
        return {"path": str(path), "sha256": tp.sha(path)}
    for idx, arm in enumerate(("discrete", "continuous")):
        q, smoke = template(arm, "full"), template(arm, "smoke")
        service = {"root": "/exact/service", "manifest": "/exact/service/FILES.sha256", "manifest_sha256": "1"*64, "revision": "unit-fixture"}
        q["operational_service"] = smoke["operational_service"] = service
        job, context = str(900+idx), ("b" if idx else "c")*64
        smoke_binding = put(f"{arm}-smoke.json", smoke)
        ranks = []
        for rank in range(4):
            ranks.append(dict(status="FIT_PASS", rank=rank, world_size=4, job_id=job,
                global_step=50, fp32_losses=50, bf16_training_outputs=50,
                fp32_gradients_verified=True, validation_batches=100, local_validation_batches=25,
                data_context_sha256=context, active_streams=["image", "proprio"],
                metrics={f"val/libero_gripper/{kind}/{metric}": 0.01 for kind in ("forecast", "completion") for metric in ("image_normalized_mse", "proprio_normalized_mse")},
                paired_rgb_batch_sha256=[tp.digest([arm, rank, draw]) for draw in range(3)],
                initial_weights=[("d" if idx else "e")*64]*4, all_rank_weights=["f"*64]*4,
                weights_sha256="f"*64, optimizer_update_lrs=[1e-4]*50,
                measured_updates=40, warm_wall_seconds=1.0,
                initial_image_weights_sha256="a"*64,
                gpu_uuid="GPU-"+str(uuid.UUID(int=rank+1))))
        completion = dict(schema="joint-mot-smoke-v1", status="PASS", job_id=job, world_size=4,
            global_step=50, validation_batches=100, strict_full_wrapper_reload=True,
            optimizer_sampler_all_rank_rng_verified=True, lr_scheduler_strictly_restored=True,
            all_rank_weights_agree=True, sample_finite=True, sample_state_steps=20,
            sample_action_steps=0, lr_warmup_steps=0, model_execution=q["model_execution"],
            active_streams=q["active_streams"], expert_parameters=q["expert_parameters"], ranks=ranks)
        preflight = dict(status="PREFLIGHT_PASS", request_sha256=smoke_binding["sha256"],
            parameters=q["parameters"], expert_parameters=q["expert_parameters"], expert_widths=q["expert_widths"],
            world_size=4, global_batch=32,
            provenance={**{key:q[key] for key in ("source_revision", "source_manifest_sha256", "data_manifest_sha256")}, "data_context_sha256":context})
        capacity = dict(status="CAPACITY_PASS", active_streams=q["active_streams"], sample_all_streams_finite=True,
                        frames_per_example=16 if arm=="discrete" else 64, per_rank_batch=8, optimizer_update_lr=1e-4)
        native = f"JobId={job} JobState=COMPLETED ExitCode=0:0 UserId=paphiwetsa3(41003) JobName={smoke['job_name']} NodeList=gpu03"
        q["smoke"] = dict(job_id=job, request=smoke_binding,
            completion=put(f"{arm}-completion.json", completion), preflight=put(f"{arm}-preflight.json", preflight),
            capacity=put(f"{arm}-capacity.json", capacity), native=put(f"{arm}-native.txt", native, True),
            host_exit=put(f"{arm}-exit.txt", "0", True), resolved=put(f"{arm}-resolved.json", composed(arm,"smoke")))
        requests.append(q);states[arm]=(completion, preflight, capacity, native)
    real_sha = tp.sha
    manifest_digests = {requests[0]["source_manifest"]:tp.SOURCE_MANIFEST, requests[0]["data_manifest"]:tp.DATA_MANIFEST}
    monkeypatch.setattr(tp, "sha", lambda p: manifest_digests.get(str(p)) or real_sha(p))
    monkeypatch.setattr(tp, "compose_request", lambda q: composed(q["arm"], "full"))
    def bind():
        return [put(f"{q['arm']}-prototype.json",q) for q in requests]
    def replace_evidence(arm,key,obj,text=False):
        q=next(q for q in requests if q['arm']==arm)
        q['smoke'][key]=put(f"{arm}-{key}-changed",obj,text)
    return requests, states, bind, put, replace_evidence


def test_actual_configs_compare_and_normalize():
    values=[]
    for arm in tp.PROFILES:
        q=template(arm,"full")
        compare_configs(composed(arm,"full"),composed(arm,"smoke"),q)
        values.append(tp.normalize_config(q,composed(arm,"full")))
    assert values[0]==values[1]


def test_pair_keeps_unpaired_proofs_and_exact_receipt_membership(pair):
    requests,_,bind,put,_=pair
    receipt=tp.verify(bind())
    assert receipt['status']=='PASS' and receipt['fairness']=='FAIR WITH CAVEATS'
    a,b=receipt['evidence']
    assert a['ranks'][0]['initial_weights']!=b['ranks'][0]['initial_weights']
    assert a['ranks'][0]['paired_rgb_batch_sha256']!=b['ranks'][0]['paired_rgb_batch_sha256']
    assert a['data_context_sha256']!=b['data_context_sha256']
    request={**requests[0], 'temporal_comparison':put('receipt.json',receipt)}
    assert tp.admit(request)==receipt


@pytest.mark.parametrize('change',[
    lambda q:q.update(arm='continuous'),
    lambda q:q.update(lr_policy=dict(peak_lr=5e-5,schedule='constant',warmup_optimizer_steps=0)),
    lambda q:q.update(world_size=2),
    lambda q:q.update(parameters=1),
    lambda q:q.update(active_streams=['image','proprio','action']),
    lambda q:q.update(source_revision='0'*40),
    lambda q:q.update(source_manifest_sha256='0'*64),
    lambda q:q.update(data_manifest_sha256='0'*64),
    lambda q:q.update(operational_service=dict(root='/different')),
    lambda q:q.update(_template_status='PENDING'),
    lambda q:q.update(continuation={}),
    lambda q:q.update(allocation_boundaries=[100000,300000]),
    lambda q:q.update(optimizer_comparison={}),
])
def test_reject_changed_identity_and_controls(pair,change):
    requests,_,bind,_,_=pair
    change(requests[0])
    with pytest.raises(ValueError):tp.verify(bind())


def test_reject_duplicate_jobs(pair):
    requests,_,bind,_,_=pair
    requests[1]['smoke']['job_id']=requests[0]['smoke']['job_id']
    with pytest.raises(ValueError,match='Shared GPU smoke'):tp.verify(bind())


def test_reject_duplicate_ids(pair):
    requests,_,bind,_,_=pair
    requests[1]['run_id']=requests[1]['intended_full_run_id']=requests[0]['run_id']
    requests[1]['overrides']=[('logger.wandb.id='+requests[0]['run_id']) if v.startswith('logger.wandb.id=') else v for v in requests[1]['overrides']]
    with pytest.raises(ValueError,match='Duplicate full'):tp.verify(bind())


@pytest.mark.parametrize('native_change',[
    lambda t:t.replace('COMPLETED','FAILED'),
    lambda t:t.replace('0:0','1:0'),
    lambda t:t.replace('JobId=900','JobId=999'),
    lambda t:t.replace('gpu03','gpu19'),
])
def test_native_proof_revalidated(pair,native_change):
    _,states,bind,_,replace=pair
    replace('discrete','native',native_change(states['discrete'][3]),True)
    with pytest.raises(ValueError):tp.verify(bind())


def test_host_failure_rejected(pair):
    _,_,bind,_,replace=pair
    replace('discrete','host_exit','1',True)
    with pytest.raises(ValueError,match='host failed'):tp.verify(bind())


@pytest.mark.parametrize('change',[
    lambda c:c['ranks'][3].update(initial_image_weights_sha256='1'*64),
    lambda c:c['ranks'][2].update(initial_image_weights_sha256='bad'),
    lambda c:c['ranks'][0].update(optimizer_update_lrs=[5e-5]*50),
    lambda c:c.update(strict_full_wrapper_reload=False),
    lambda c:c.update(global_step=49),
    lambda c:c['ranks'][0].update(metrics={'bad':float('nan')}),
])
def test_gpu_evidence_revalidated(pair,change):
    _,states,bind,_,replace=pair
    c=copy.deepcopy(states['discrete'][0]);change(c)
    replace('discrete','completion',c)
    with pytest.raises(ValueError):tp.verify(bind())


@pytest.mark.parametrize('path,value',[
    (('model','scheduler'),{'kind':'warmup'}),
    (('model','optimizer','weight_decay'),0.1),
    (('data','max_frames'),63),
    (('data','validation_discrete_frames'),12),
    (('joint','pose_dims','libero_gripper'),18),
    (('evaluator','sample_seed'),99),
    (('runtime','slurm_requeue_owner'),'runner'),
])
def test_exact_config_policy(path,value):
    q=template('continuous','full');cfg=composed('continuous','full');cur=cfg
    for key in path[:-1]:cur=cur[key]
    cur[path[-1]]=value
    with pytest.raises(ValueError):tp.normalize_config(q,cfg)


def test_unexpected_difference_rejected(pair,monkeypatch):
    _,_,bind,_,_=pair
    original=tp.normalize_config
    monkeypatch.setattr(tp,'normalize_config',lambda q,cfg:{**original(q,cfg),'unexpected':q['arm']})
    with pytest.raises(ValueError,match='Unexpected D/C'):tp.verify(bind())


def test_checkpoint_policy_rejected(pair,monkeypatch):
    requests,_,bind,_,replace=pair
    full=composed('discrete','full');smoke=composed('discrete','smoke')
    full['callbacks']['model_checkpoint']['save_top_k']=1
    smoke['callbacks']['model_checkpoint']['save_top_k']=1
    replace('discrete','resolved',smoke)
    monkeypatch.setattr(tp,'compose_request',lambda q:full if q['arm']=='discrete' else composed('continuous','full'))
    with pytest.raises(ValueError,match='retention'):tp.verify(bind())


def test_receipt_tampering_rejected_even_when_rehashed(pair):
    requests,_,bind,put,_=pair
    receipt=tp.verify(bind());receipt['initial_image_weights_sha256']='0'*64
    request={**requests[0],'temporal_comparison':put('receipt.json',receipt)}
    with pytest.raises(ValueError,match='freshly verified'):tp.admit(request)


def test_changed_member_rejected(pair):
    requests,_,bind,put,_=pair
    receipt=tp.verify(bind())
    request={**requests[0],'temporal_comparison':put('receipt.json',receipt),'extra_unbound_field':True}
    with pytest.raises(ValueError,match='exact admitted'):tp.admit(request)


def test_prototype_tampering_rejected(pair):
    _,_,bind,_,_=pair
    bindings=bind();p=Path(bindings[0]['path']);p.write_text(p.read_text()+' ')
    with pytest.raises(ValueError,match='Changed evidence'):tp.verify(bindings)


@pytest.mark.parametrize('second',['optimizer_comparison','modality_comparison','factorial_controls'])
def test_exclusive_dispatch(second):
    with pytest.raises(ValueError,match='Exactly one'):tp.admission_mode({'temporal_comparison':{},second:{}})


def test_no_implicit_dispatch():
    with pytest.raises(ValueError):tp.admission_mode({})
    assert tp.admission_mode({'optimizer_comparison':{}})=='optimizer_comparison'


def test_continuation_is_bound_and_fail_closed(pair):
    requests,_,_,_,_=pair
    assert 'temporal_comparison' in STABLE_FIELDS
    with pytest.raises(ValueError,match='single-allocation'):
        tp.admit({**requests[0],'temporal_comparison':{},'continuation':{}})


def test_path_collision_guard():
    with pytest.raises(ValueError,match='Overlapping'):tp.disjoint(['/out/x','/out/x/logs'])
    with pytest.raises(ValueError,match='Unsafe'):tp.disjoint(['/out/../data','/other'])


@pytest.mark.parametrize('override',[
    'data.forecasting_probability=0.8',
    'modality_ablation.nodes.image_loss.weight=2',
    'modality_ablation.nodes.joint_noising.noising.maximum=0.5',
])
def test_same_unrequested_change_on_both_arms_is_rejected(pair,override):
    requests,_,bind,_,_=pair
    for q in requests:q['overrides'].append(override)
    with pytest.raises(ValueError,match='Additional or changed'):tp.verify(bind())
