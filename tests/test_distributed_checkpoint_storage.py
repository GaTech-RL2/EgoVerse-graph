import copy,importlib.util,time
from pathlib import Path
import pytest
SPEC=importlib.util.spec_from_file_location('storage',Path(__file__).parents[1]/'scripts/train/check_checkpoint_storage.py');M=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(M)

def contract(tmp_path):
    rows=[]
    for name,size in [('af',100),('dp',40)]:
        root=tmp_path/name;root.mkdir()
        rows.append(dict(id=name,local_root=str(root),remote_root='/coc/flash7/paphiwetsa3/archive/'+name,measured_checkpoint_bytes=size,scheduled_count=16,additional_checkpoint_count=5,retain_local=2,checkpoint_measurement_status='STRICT_FULL_STATE_RELOAD_PASS',preserve_all_cadence=True))
    plan=dict(schema_version=1,status='DISTRIBUTED_RETENTION_READY',runs=rows,archive_uid=3346154,archive_username='paphiwetsa3',archive_quota=dict(status='UID_QUOTA_VALIDATED',uid=3346154,username='paphiwetsa3',checked_at_unix=1000,available_bytes=4000,available_inodes=100,target='/coc/flash7/paphiwetsa3',authority='native_zfs_user_and_group_quota'),mirror_cycle=dict(status='CHECKPOINT_MIRROR_CYCLE_VERIFIED',cycle_errors=0,remote_sha256_verified=True,parent_run_ids=['af','dp'],checked_at_unix=1000))
    return plan,dict(available_bytes=1000,available_inodes=100)

def check(plan,quota):return M.distributed_budget(plan,Path(plan['runs'][0]['local_root']),16,100,100,quota,1010)

def test_exact_pair_local_and_all_archive_budget(tmp_path):
    p,q=contract(tmp_path);r=check(p,q)
    assert r['pair_local_checkpoint_budget_bytes']==420
    assert r['pair_archive_checkpoint_budget_bytes']==3080
    assert r['pair_archive_checkpoint_budget_files']==44

@pytest.mark.parametrize('field,value',[('status','DRAFT_NOT_LAUNCHABLE'),('archive_uid',1)])
def test_draft_or_wrong_archive_account_rejected(tmp_path,field,value):
    p,q=contract(tmp_path);p[field]=value
    with pytest.raises(M.StorageError):check(p,q)

@pytest.mark.parametrize('section,field,value',[('archive_quota','checked_at_unix',0),('archive_quota','available_bytes',3179),('archive_quota','available_inodes',75),('archive_quota','status','UNVERIFIED'),('archive_quota','authority','df'),('mirror_cycle','cycle_errors',1),('mirror_cycle','checked_at_unix',0),('mirror_cycle','parent_run_ids',['af']),('mirror_cycle','remote_sha256_verified',False)])
def test_archive_and_mirror_fail_closed(tmp_path,section,field,value):
    p,q=contract(tmp_path);p[section][field]=value
    with pytest.raises(M.StorageError):check(p,q)

@pytest.mark.parametrize('field,value',[('available_bytes',519),('available_inodes',37)])
def test_local_pair_quota_insufficient(tmp_path,field,value):
    p,q=contract(tmp_path);q[field]=value
    with pytest.raises(M.StorageError):check(p,q)

@pytest.mark.parametrize('field,value',[('measured_checkpoint_bytes',None),('retain_local',1),('scheduled_count',15),('preserve_all_cadence',False),('additional_checkpoint_count',None),('remote_root','/other/account/af'),('checkpoint_measurement_status','estimated')])
def test_scientific_and_retention_contract_rejected(tmp_path,field,value):
    p,q=contract(tmp_path);p['runs'][0][field]=value
    with pytest.raises(M.StorageError):check(p,q)


def test_receipt_tamper_rejected_before_native_quota_query(tmp_path,monkeypatch):
    import hashlib,json
    from types import SimpleNamespace
    p,q=contract(tmp_path);receipt=tmp_path/'archive.json';receipt.write_text(json.dumps(p['archive_quota']))
    p['archive_quota_receipt']=dict(path=str(receipt),sha256='0'*64)
    plan=tmp_path/'plan.json';plan.write_text(json.dumps(p))
    monkeypatch.setenv('ICE_DISTRIBUTED_RETENTION_PLAN',str(plan));monkeypatch.setenv('ICE_DISTRIBUTED_RETENTION_PLAN_SHA256',hashlib.sha256(plan.read_bytes()).hexdigest())
    with pytest.raises(M.StorageError,match='receipt SHA mismatch'):
        M.distributed_evidence(SimpleNamespace())
