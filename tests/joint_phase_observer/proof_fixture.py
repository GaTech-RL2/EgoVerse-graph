"""Portable synthetic observer evidence builder. Never invokes native/ML code."""
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
from scripts.train.joint import phase_observer_proof as p


def put(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(value if isinstance(value,bytes) else value.encode() if isinstance(value,str) else p.encoded(value))
    return p.binding(path)


def raw(value):return ' '.join(k+'='+str(v) for k,v in value.items())+'\n'


def query(argv,stdout):return dict(schema='native-elmo-phase-observer-query-v1',argv=argv,returncode=0,stdout=stdout,stderr='',observed_at_utc=datetime.now(timezone.utc).isoformat())


def build(root,mode='inspect',arm='plain',running=True,terminal_state='COMPLETED'):
    root=Path(root);root.mkdir()
    for name in ['execution','execution-claims','templates','requests','scientific-claims','producer-claims','status','logs','observer-specs','observer-claims']:(root/name).mkdir()
    runtime=['execution_spec.py','allocated.sbatch','host_guard.py','phase_observer.py','observe-phase.sbatch']
    for name in runtime:put(root/name,"raise AssertionError('EXTERNAL OPS CODE MUST NEVER EXECUTE')\n")
    manifest=''.join(p.sha(root/name)+'  '+name+'\n' for name in runtime);ops=put(root/'FILES.sha256',manifest)
    service=Path(p.__file__).resolve().parents[3]
    sb=dict(root=str(service),manifest=str(service/'FILES.sha256'),manifest_sha256=p.sha(service/'FILES.sha256'))
    node='gpu03' if arm=='boundary' else 'gpu11';old='111';job='900001';observer='900002'
    run='mot-elmo-pickplace-continuous-large314m-'+arm+'-robot-human-camera-subtask-s42-300k-20261026-v1'
    parent=dict(schema='joint-training-v2',phase='full',kind='native-subtask-world-model-v1',run_id=run,job_name='synthetic-original',
        target_steps=300000,parameters=314187064,world_size=4,output=str(root/'original-output'),local_logs=str(root/'original-logs'),
        operational_service=sb)
    for key in ['source','source_revision','source_manifest','source_manifest_sha256','data_manifest','data_manifest_sha256','lambda_runtime','data_cache']:parent[key]='synthetic-'+key
    pb=put(root/'parent.json',parent)
    profile=dict(parent_id=old,node=node,request_path=pb['path'],request_sha256=pb['sha256'])
    schema,cpu,memory,limit,gres=p.MODES[mode]
    name=('elmo-large314m-continuous-' if mode in {'full','diagnostic'} else 'elmo-native-recovery-')+arm+'-'+mode+'-synthetic'
    native=dict(JobName=name,UserId='paphiwetsa3(41003)',Account='loaner',QOS='loaner-normal',Partition='rl2',ReqNodeList=node,NodeList=node,
        NumCPUs=cpu,MinMemoryNode=memory,TimeLimit=limit,Requeue='0',Restarts='0',Command=str(root/'allocated.sbatch'),TresPerNode=gres)
    if mode=='full':
        template=copy.deepcopy(parent);template.update(job_name=name,output=str(root/'phase-output'),local_logs=str(root/'phase-logs'),interruption=dict(parent_request=pb,admission=None))
    else:
        template=dict(schema=schema,parent_request=pb,operational_service=sb,producer_native=native,output=str(root/'phase-output'))
        for key in ['source','source_revision','source_manifest','source_manifest_sha256','data_manifest','data_manifest_sha256','lambda_runtime','data_cache']:template[key]=parent[key]
        if mode=='capture':template.update(parent_job_id=old,parent_native=dict(JobId=old))
    tb=put(root/'templates/phase.json',template)
    spec=dict(schema='native-elmo-execution-spec-v1',root=str(root),mode=mode,arm=arm,original_parent_job_id=old,parent_request=pb,
        request_template=tb,native_contract=native,operational_service=sb,ops_manifest=ops,
        execution_claim_roots=[dict(path=str(root/'execution-claims'),terminal_directory=str(root/'status'))],release_observer_path=str(root/'observer-specs/observer.json'))
    sp=put(root/'execution/phase.json',spec);ex=dict(path=sp['path'],sha256=sp['sha256'],mode=mode,ops_manifest_sha256=ops['sha256'])
    sc=root/'execution-claims'/sp['sha256'];sc.mkdir();argv=p.producer_argv(spec,Path(sp['path']),sp['sha256'],ops['sha256'])
    held=dict(native,JobId=job,JobState='PENDING',Reason='JobHeldUser',NodeList='',NumNodes='1-1',NumTasks='1',
        SubmitTime='2026-01-01T01:00:00',StartTime='Unknown',AllocTRES='(null)',ReqTRES='cpu='+cpu+',mem='+memory+',node=1,billing='+cpu,
        Comment='elmo-execution:'+sp['sha256'])
    hb=put(sc/'HELD_NATIVE.txt',raw(held))
    actual=copy.deepcopy(template)
    if mode!='full':actual['producer_native']['JobId']=job
    ab=put(root/'requests'/(sp['sha256']+'-'+job+'.json'),actual)
    claim=root/('scientific-claims' if mode in {'diagnostic','full'} else 'producer-claims')/ab['sha256'];claim.mkdir()
    put(sc/'PREPARED.json',dict(spec_sha256=sp['sha256'],ops_manifest_sha256=ops['sha256'],spec_path=sp['path'],arm=arm,parent_request=pb,argv=argv,original_parent_job_id=old,created_at=datetime.now(timezone.utc).isoformat()))
    put(claim/'PREPARED.json',dict(request_sha256=ab['sha256'],spec_sha256=sp['sha256'],spec_path=sp['path'],actual_request_path=ab['path'],argv=argv,original_parent_job_id=old))
    for directory in [sc,claim]:put(directory/'PARENT_JOB_ID',job+'\n')
    put(sc/'sbatch.stdout',job+'\n');put(sc/'sbatch.exit','0\n');put(sc/'sbatch.stderr','')
    acceptance=dict(schema='native-elmo-held-execution-binding-v1',spec_sha256=sp['sha256'],ops_manifest_sha256=ops['sha256'],job_id=job,argv=argv,request=ab,request_claim=str(claim),held_native=hb)
    put(sc/'acceptance.json',acceptance)
    ospec=dict(schema='native-elmo-phase-observer-spec-v1',root=str(root),producer_execution=ex,terminal_directory=str(root/'status'))
    ob=put(root/'observer-specs/observer.json',ospec);ledger=root/'observer-claims'/sp['sha256'];ledger.mkdir()
    ctx=dict(root=root,spec=spec,execution=ex,request=actual,job=job)
    on=p.observer_native(ctx);oa=p.observer_argv(ctx,Path(ob['path']),ob['sha256'])
    pending=dict(on,JobId=observer,JobState='PENDING',Reason='Dependency',NodeList='',NumNodes='1-1',NumTasks='1',
        SubmitTime='2026-01-01T01:00:01',StartTime='Unknown',ReqTRES='cpu=2,mem=1G,node=1,billing=2',AllocTRES='(null)',
        Dependency='afterany:'+job+'(unfulfilled)',Comment='elmo-phase-observer:'+ob['sha256'],WorkDir=str(root),
        StdOut=str(root/'logs'/(on['JobName']+'-'+observer+'.out')),StdErr=str(root/'logs'/(on['JobName']+'-'+observer+'.err')))
    pending_binding=put(ledger/'PENDING_NATIVE.txt',raw(pending));pq=query(p.native_argv(observer),raw(pending))
    put(ledger/'ACCEPTANCE_QUERY.json',pq);put(ledger/'PRODUCER_HELD_QUERY.json',query(p.native_argv(job),raw(held)))
    intent=dict(schema='native-elmo-phase-observer-intent-v1',observer_spec=ob,ops_manifest_sha256=ops['sha256'],producer_execution=ex,producer_request=ab,producer_job_id=job,argv=oa)
    put(ledger/'PREPARED.json',intent)
    for name,value in [('sbatch.stdout',observer+'\n'),('sbatch.exit','0\n'),('sbatch.stderr',''),('OBSERVER_JOB_ID',observer+'\n')]:put(ledger/name,value)
    oaccept=dict(schema='native-elmo-phase-observer-acceptance-v1',observer_spec=ob,ops_manifest_sha256=ops['sha256'],producer_execution=ex,producer_request=ab,producer_job_id=job,observer_job_id=observer,argv=oa,pending_native=pending_binding)
    oab=put(ledger/'acceptance.json',oaccept)
    ready=dict(schema='native-elmo-phase-observer-ready-v1',producer_job_id=job,producer_request_sha256=ab['sha256'],observer_job_id=observer,
        observer_spec_sha256=ob['sha256'],observer_pending_query=pq,observer_own_terminal='UNKNOWN_NOT_REQUIRED',production_completion_proven=False)
    rh=put(sc/'RELEASE_HELD_NATIVE.txt',raw(held));put(sc/'RELEASE_ATTEMPT.json',dict(job_id=job,spec_sha256=sp['sha256'],request_sha256=ab['sha256'],held_native_sha256=rh['sha256'],observer_readiness=ready))
    own=dict(on,JobId=observer,JobState='RUNNING',Reason='None',NumNodes='1',NumTasks='1',SubmitTime=pending['SubmitTime'],StartTime='2026-01-01T02:00:00',
        ReqTRES='cpu=2,mem=1G,node=1,billing=2',AllocTRES='cpu=2,mem=1G,node=1,billing=2',Comment=pending['Comment'])
    ownb=put(ledger/'RUNNING_NATIVE_QUERY.json',query(p.native_argv(observer),raw(own)))
    rn=dict(native,JobId=job,JobState='RUNNING',Reason='None',NumNodes='1',NumTasks='1',SubmitTime=held['SubmitTime'],StartTime='2026-01-01T01:00:02',
        ReqTRES=held['ReqTRES'],AllocTRES=held['ReqTRES'],Comment=held['Comment'])
    if gres:rn.update(JOB_GRES='gpu:h100:4',GRES='gpu:h100:4(IDX:0-3)')
    rb=put(claim/'RUNNING_ADMISSION.json',dict(job_id=job,request_sha256=ab['sha256'],spec_sha256=sp['sha256'],native=raw(rn))) if running else None
    term=dict(rn,JobState=terminal_state,EndTime='2026-01-01T02:00:00',ExitCode='0:0' if terminal_state=='COMPLETED' else '1:0')
    if gres:term['GRES']=''
    nb=put(root/'status'/('native-'+job+'.txt'),raw(term));qb=put(root/'status'/('native-'+job+'.query.json'),query(p.native_argv(job),raw(term)))
    eb=put(root/'status'/('native-'+job+'.exit'),'0\n');errb=put(root/'status'/('native-'+job+'.stderr'),'')
    stepb=put(root/'status'/('native-'+job+'.steps-query.json'),query(['/usr/local/bin/squeue','--steps','--noheader','--user','paphiwetsa3','--format=%i|%T'],'900999.batch|RUNNING\n'))
    result=dict(schema='native-elmo-phase-observer-result-v1',status='GENUINE_PRODUCER_TERMINAL_CAPTURED',producer_execution=ex,producer_request=ab,
        producer_job_id=job,producer_phase=mode,producer_terminal_state=terminal_state,producer_terminal_exit=term['ExitCode'],producer_native=nb,producer_query=qb,
        producer_query_exit=eb,producer_query_stderr=errb,owned_steps=stepb,producer_running_admission=rb,producer_running_admission_status='VALIDATED' if running else 'NOT_RECORDED',
        observer_spec=ob,observer_acceptance=oab,observer_running_query=ownb,observer_job_id=observer,observer_own_terminal='UNKNOWN_NOT_REQUIRED',observer_host_exit_is_native_proof=False,
        phase_completion_proven=False,zero_work_proven=False,captured_at_utc=datetime.now(timezone.utc).isoformat(),automatic_recovery_authorized=False)
    orb=put(root/'status'/('native-'+job+'.observer.json'),result)
    result_mode={v:k for k,v in p.RESULT_MODES.items()};statuses=dict(capture='CAPTURED_REQUIRES_NATIVE_PRODUCER_COMPLETION',inspect='INSPECTED_REQUIRES_NATIVE_PRODUCER_COMPLETION',diagnostic='DIAGNOSTIC_COMPLETE_REQUIRES_HOST_AND_NATIVE_SUCCESS',finalize='FINALIZED_REQUIRES_HOST_AND_NATIVE_SUCCESS')
    result_schema=result_mode.get(mode,'synthetic-full-census-only');status=statuses.get(mode,'NOT_PHASE_AUTHORITY')
    filenames=dict(capture='interruption-capture.json',inspect='checkpoint-inspection.json',diagnostic='production-recovery-diagnostic.json',finalize='interruption-finalized.json',full='full-census-not-phase-result.json')
    phase_result=put(Path(actual['output'])/filenames[mode],dict(schema=result_schema,status=status,job_id=job,request_sha256=ab['sha256']))
    host=put(root/'status'/(mode+'-'+job+'.exit'),'0\n')
    bundle=dict(request=ab,result=phase_result,native=nb,host_exit=host,observer=orb)
    return dict(root=str(root),profiles={run:profile},bundle=bundle,result_schema=result_schema,status=status,mode=mode,arm=arm,
        spec=sp,observer_spec=ob,execution_claim=str(sc),request_claim=str(claim),observer_claim=str(ledger),ops_manifest=ops)


def build_original(root,arm='plain'):
    root=Path(root);root.mkdir();job='111';observer='112';run='mot-elmo-pickplace-continuous-large314m-'+arm+'-robot-human-camera-subtask-s42-300k-20261026-v1'
    parent=dict(run_id=run,job_name='synthetic-original-parent')
    pb=put(root/'ops/requests/parent.json',parent)
    members={}
    for name in ['capture-native.sbatch','guard_full.py','submit_full_once.py','full.sbatch']:
        members[name]=put(root/name,'synthetic immutable original '+name+'\n')['sha256']
    members['ops/requests/parent.json']=pb['sha256'];mb=put(root/'FILES.sha256',''.join(v+'  '+k+'\n' for k,v in sorted(members.items())))
    claim=root/'claims'/pb['sha256'];claim.mkdir(parents=True)
    put(claim/'PREPARED.json',dict(request_sha256=pb['sha256'],manifest_sha256=mb['sha256']))
    put(claim/'PARENT_JOB_ID',job+'\n');put(claim/'CAPTURE_JOB_ID',observer+'\n')
    n=dict(JobId=job,JobName=parent['job_name'],UserId='paphiwetsa3(41003)',Account='loaner',QOS='loaner-normal',Partition='rl2',ReqNodeList='gpu11',NodeList='gpu11',NumCPUs='32',MinMemoryNode='192G',TimeLimit='7-00:00:00',Requeue='0',Restarts='0',Command=str(root/'full.sbatch'),TresPerNode='gres/gpu:h100:4')
    rn=dict(n,JobState='RUNNING',NumNodes='1',NumTasks='1',StartTime='2026-01-01T01:00:00',EndTime='2026-01-08T01:00:00',ExitCode='0:0',ReqTRES='cpu=32,mem=192G,node=1',AllocTRES='cpu=32,mem=192G,node=1',JOB_GRES='gpu:h100:4',GRES='gpu:h100:4(IDX:0-3)')
    put(claim/'RUNNING_ADMISSION.json',dict(status='ACTUAL_BOUND_FULL_NATIVE_ALLOCATION_PASS',job_id=job,request_sha256=pb['sha256'],native=raw(rn)))
    term=dict(rn,JobState='TIMEOUT',EndTime='2026-01-01T02:00:00',ExitCode='0:15',GRES='')
    put(root/('native-'+job+'.txt'),raw(term));put(root/('native-'+job+'.exit'),'0\n');put(root/('native-'+job+'.stderr'),'')
    oname='synthetic-original-observer'
    pending=dict(JobId=observer,JobName=oname,UserId=n['UserId'],Account='loaner',QOS='loaner-normal',Partition='rl2',ReqNodeList='gpu11',NumTasks='1',NumCPUs='2',TimeLimit='00:10:00',Requeue='0',Restarts='0',JobState='PENDING',Command=str(root/'capture-native.sbatch'),WorkDir=str(root),NumNodes='1-1',MinMemoryNode='1G',Dependency='afterany:'+job+'(unfulfilled)',StdOut=str(root/'logs'/(oname+'-'+observer+'.out')),StdErr=str(root/'logs'/(oname+'-'+observer+'.err')),ReqTRES='cpu=2,mem=1G,node=1',AllocTRES='(null)',TresPerTask='cpu:2')
    put(claim/'CAPTURE_PENDING_NATIVE.txt',raw(pending))
    profile=dict(root=str(root),parent_id=job,observer_id=observer,parent_name=parent['job_name'],observer_name=oname,node='gpu11',request_path=pb['path'],request_sha256=pb['sha256'],files_sha256=mb['sha256'],capture_script_sha256=members['capture-native.sbatch'],submitter_sha256=members['submit_full_once.py'],has_smoke_identity=False)
    return dict(root=str(root),terminal_directory=str(root),profiles={run:profile},request_claim=str(claim),bundle={})


if __name__=='__main__':
    print(json.dumps(build_original(sys.argv[1],sys.argv[3]) if sys.argv[2]=='original' else build(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]=='yes',sys.argv[5])))
