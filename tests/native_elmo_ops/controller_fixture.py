"""Portable in-memory remote filesystem fixture; no actual SSH or files."""
import controller as c

ROOT=c.PREFIX+"synthetic-controller-fixture"

def fixture():
    files={}
    def save(name,value,raw=False):
        location=ROOT+'/'+name
        data=value if raw else c.encode(value)
        files[location]=data
        return {'path':location,'sha256':c.digest(data)}
    original=c.encode(dict(run_id='mot-elmo-pickplace-continuous-large314m-plain-robot-human-camera-subtask-s42-300k-20261026-v1',job_name='synthetic-original',parameters=314187064,world_size=4,target_steps=300000,phase='full'))
    c.PINS=dict(c.PINS,plain=('412789','gpu11',c.digest(original)))
    parent=save('inputs/parent.json',original,True); q=c.parse(original)
    intent=save('inputs/current-intent.json',dict(schema='native-elmo-current-intent-v1',authority='human',status='ACTIVE',hold_recovery=False,
        authorized_recovery_run_ids=[q['run_id']],retired_run_ids=[]))
    terminal=save('inputs/parent-native.txt',('JobId=412789 JobName='+q['job_name']+' UserId=paphiwetsa3(41003) NodeList=gpu11 JobState=TIMEOUT EndTime=2026-10-27T00:00:00 ExitCode=0:1\n').encode(),True)
    py_script=save('inputs/python-producer.sbatch',b'#!/bin/bash\n/usr/bin/python3 -B "$root/cache.py"\n',True)
    py_native=save('inputs/python-native.txt',b'JobId=412768 NodeList=gpu11 JobState=COMPLETED ExitCode=0:0\n',True)
    py_exit=save('inputs/python-exit.txt',b'0\n',True)
    provenance=save('inputs/python-provenance.json',dict(schema='native-elmo-host-python-provenance-v1',node='gpu11',job_id='412768',
        interpreter_path='/usr/bin/python3',script=py_script,native=py_native,host_exit=py_exit))
    service_member=save('service/member.py',b'# synthetic test fixture\n',True)
    sm=save('service/FILES.sha256',(service_member['sha256']+'  member.py\n').encode(),True)
    ops={name:save(name,('# test '+name+'\n').encode(),True)['sha256'] for name in
        ('controller.py','controller.sbatch','execution_spec.py','host_guard.py','phase_observer.py','observe-phase.sbatch')}
    om=save('FILES.sha256',''.join(sha+'  '+name+'\n' for name,sha in sorted(ops.items())).encode(),True)
    scope=dict(user='paphiwetsa3',scientific_job_names=[q['job_name']],scientific_name_prefixes=['elmo-large314m-continuous-plain-'])
    template=save('templates/producer.json',{'fixture':True})
    producer=dict(root=ROOT,arm='plain',mode='capture',parent_request=parent,request_template=template,
        runtime={'admission':dict(current_intent_path=intent['path'],quiescence=scope)},
        operational_service=dict(root=ROOT+'/service',manifest=sm['path'],manifest_sha256=sm['sha256']),
        release_observer_path=ROOT+'/observer-specs/observe.json')
    pb=save('execution/producer.json',producer)
    execution=dict(pb,mode='capture',ops_manifest_sha256=om['sha256'])
    observer=save('observer-specs/observe.json',dict(schema='native-elmo-phase-observer-spec-v1',root=ROOT,
        producer_execution=execution,terminal_directory=ROOT+'/status'))
    spec=dict(schema='native-elmo-allocated-controller-v1',root=ROOT,arm='plain',parent_request=parent,ops_manifest=om,
        current_intent=intent,original_failed_native=terminal,producer_execution=execution,observer_spec=observer,
        terminal_directory=ROOT+'/status',job_name='elmo-recovery-controller-plain-fixture',
        controller_claim_root=c.PREFIX+'recovery-controller-claims/plain',queue_scope=scope,
        interpreter=dict(path='/usr/bin/python3',minimum_version=[3,10],provenance=provenance))
    sb=save('controller-specs/controller.json',spec)
    return files,spec,(sb['path'],sb['sha256'],om['sha256'])

def native(spec,state='PENDING',**changes):
    f=dict(JobId='900001',JobName=spec['job_name'],UserId='paphiwetsa3(41003)',Account='loaner',QOS='loaner-normal',
        Partition='rl2',Requeue='0',Restarts='0',JobState=state,ReqNodeList='gpu11',NumCPUs='2',NumTasks='1',
        MinMemoryNode='1G',TimeLimit='00:20:00',Command=ROOT+'/controller.sbatch',Comment='elmo-controller:'+'a'*64,
        SubmitTime='2026-10-27T00:01:00',NumNodes='1',Reason='JobHeldUser',NodeList='(null)',ReqTRES='cpu=2,mem=1G,node=1,billing=2',
        AllocTRES='(null)',JOB_GRES='(null)',StartTime='Unknown',TresPerTask='cpu:2')
    if state=='RUNNING': f.update(NodeList='gpu11',StartTime='2026-10-27T00:02:00',AllocTRES='cpu=2,mem=1G,node=1,billing=2')
    f.update(changes)
    return ' '.join(k+'='+v for k,v in f.items())+'\n'
