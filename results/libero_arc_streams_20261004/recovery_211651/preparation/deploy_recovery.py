import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path

import boto3
import yaml
from botocore.config import Config

from egomimic.utils.aws.aws_data_utils import load_env

ROOT=Path(__file__).parent
plan=json.loads((ROOT/'recovery-plan.json').read_text())
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==plan['host_commit']
assert all(r['exit_code']==0 for r in json.loads((ROOT/'service-validation.json').read_text()))
(ROOT/'submissions').mkdir(exist_ok=True)
(ROOT/'accepted').mkdir(exist_ok=True)


def query(name, path):
    for attempt in range(3):
        r=subprocess.run(['osmo','workflow','query',name,'--format-type','json'],capture_output=True,text=True,timeout=50)
        if r.returncode==0:
            q=json.loads(r.stdout);path.write_text(json.dumps(q,indent=2)+'\n');return q
    raise RuntimeError(f'Cannot confirm workflow {name}: '+r.stderr[-600:])


# Reconfirm that the old work was canceled, rather than competing with a resume.
for name in ('arc-sr-20261003-spatial-1','arc-sr-20261003-object-1','arc-sr-20261003-goal-1',
             'arc-sx-20261003-spatial-1','arc-sx-20261003-goal-1','arc-sx-20261003-10-1','arc-sf-20261004-spatial-1'):
    q=query(name,ROOT/(name+'-before-recovery.json'));assert q['status']=='FAILED_CANCELED'

for spec in plan['specs']:
    raw=Path(spec['file']).read_bytes();assert hashlib.sha256(raw).hexdigest()==spec['sha256']
    path=ROOT/'submissions'/(spec['name']+'.json')
    assert not path.exists(), 'Inspect prior submission instead of submitting twice'
    r=subprocess.run(['osmo','workflow','submit',spec['file'],'--pool','groot-l40s-03','--priority','NORMAL','--format-type','json'],capture_output=True,text=True,timeout=50)
    (ROOT/'submissions'/(spec['name']+'.stderr')).write_text(r.stderr)
    r.check_returncode();result=json.loads(r.stdout);path.write_text(json.dumps(result,indent=2)+'\n')
    assert result['name']==spec['name']+'-1'
    accepted=query(result['name'],ROOT/'accepted'/(result['name']+'.json'))
    assert accepted['status'] in {'PENDING','RUNNING','WAITING'}
    print(json.dumps({'accepted':result['name'],'role':spec['role'],'pool':accepted['pool'],'status':accepted['status']}),flush=True)

# Only the unstarted Object evaluation chain is superseded. Its live training
# lanes have no evaluation ancestors and continue with their original run IDs.
q=query('arc-sx-20261003-object-1',ROOT/'object-before-evaluation-handoff.json')
assert q['status']=='RUNNING'
tasks={t['name']:t for g in q['groups'] for t in g['tasks']}
retired={f'evaluate-{i:02d}' for i in range(3,21)}
assert all(tasks[n]['status']=='WAITING' for n in retired)
old=yaml.safe_load(Path('/tmp/libero-streams-expand/20261003T215731Z/specs/arc-sx-20261003-object.yaml').read_text())['workflow']
descendants={'evaluate-03'}
while True:
    new=descendants|{t['name'] for t in old['tasks'] if any(i['task'] in descendants for i in t.get('inputs',[]))}
    if new==descendants:break
    descendants=new
assert descendants==retired
first=next(t for t in old['tasks'] if t['name']=='evaluate-03')['environment']['RUN_ID']
prefix=f'experiments/arc-oat-20260919/{first}/'
load_env(required=True)
client=boto3.client('s3',region_name='auto',endpoint_url=os.environ['R2_ENDPOINT_URL'],aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],config=Config(connect_timeout=10,read_timeout=20,retries={'max_attempts':1},proxies=urllib.request.getproxies()))
assert not client.list_objects_v2(Bucket='rldb',Prefix=prefix,MaxKeys=1).get('KeyCount')
receipt={'operation':'superseded_unstarted_evaluation_chain','old_workflow':q['name'],
         'new_workflow':'arc-sv-20261004-object-1','first_retired_task':'evaluate-03',
         'retired_evaluation_tasks':sorted(retired),'training_tasks_untouched':[f'train-{i:02d}' for i in range(3,21)],
         'reason':'The old Object DUR reference evaluation was preempted; relayed requests use its recovered evaluation ID.',
         'expected_retirement':'The original evaluation no-overwrite guard rejects this reserved prefix if the obsolete reference gate ever becomes ready.'}
body=(json.dumps(receipt,indent=2)+'\n').encode();sha=hashlib.sha256(body).hexdigest();key=prefix+'orchestration-handoff.json'
client.put_object(Bucket='rldb',Key=key,Body=body,ContentType='application/json',Metadata={'sha256':sha},IfNoneMatch='*')
response=client.get_object(Bucket='rldb',Key=key);assert hashlib.sha256(response['Body'].read()).hexdigest()==response['Metadata']['sha256']==sha
(ROOT/'object-evaluation-handoff.json').write_text(json.dumps({**receipt,'uri':'s3://rldb/'+key,'sha256':sha},indent=2)+'\n')
print(json.dumps({'object_evaluations_relayed':18,'live_object_training_restarted':0,'handoff_verified':True}),flush=True)
