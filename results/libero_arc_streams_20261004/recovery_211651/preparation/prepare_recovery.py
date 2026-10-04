"""Compose recovery specifications from the pinned, already-tested host helpers."""
import copy
import hashlib
import json
import os
import urllib.request
from pathlib import Path

import boto3
import yaml
from botocore.config import Config

from egomimic.benchmarks.libero.arc_streams import campaign
from egomimic.benchmarks.libero.evaluate import validate_request
from egomimic.utils.aws.aws_data_utils import load_env
from scripts.benchmarks.launch_libero_osmo import evaluation_workflow
from scripts.benchmarks.launch_libero_stream_expansion import expanded_workflow
from scripts.benchmarks.launch_libero_stream_recovery import attach_resume, resume_request
from scripts.benchmarks.repair_libero_stream_tasks import retry_downloads
from scripts.benchmarks.wait_libero_stream_checkpoint import checkpoint_ready

ROOT = Path(__file__).parent
VAL = Path('/Users/rpunamiya/Desktop/GEAR/EgoVerse/scratch/libero-stream-validation-20261001')
SOURCE = '69ff3fab69c2d322b590809461c80d70878ba71f'
manifest = json.loads((VAL/'results/libero_arc_streams_20261004/status_005629/launch.json').read_text())
progress = json.loads((ROOT/'progress.json').read_text())
selection = json.loads((ROOT/'recovery-selection.json').read_text())
by_cell = {(r['suite'],r['variant'],r['mode']):r for r in manifest['runs']}
observed = {(r['suite'],r['variant'],r['mode']):r for r in progress['runs']}
plan = {'source_commit':SOURCE,'host_commit':'9c36037a5da54d5a84cd753fe57353be91e436a8',
        'assembly_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'specs':[], 'evaluation_replacements':[], 'training_replacements':[], 'object_evaluation_replacements':[]}
(ROOT/'specs').mkdir(exist_ok=True)
load_env(required=True)
client = boto3.client('s3',region_name='auto',endpoint_url=os.environ['R2_ENDPOINT_URL'],aws_access_key_id=os.environ['R2_ACCESS_KEY_ID'],aws_secret_access_key=os.environ['R2_SECRET_ACCESS_KEY'],config=Config(connect_timeout=10,read_timeout=20,retries={'max_attempts':1},proxies=urllib.request.getproxies()))


def save(spec, role):
    body = spec['workflow']
    names = {t['name'] for t in body['tasks']}
    assert len(names)==len(body['tasks'])
    assert all(i['task'] in names for t in body['tasks'] for i in t.get('inputs',[]))
    assert all(resource['platform']=='ovx-l40s' for resource in body['resources'].values())
    path=ROOT/'specs'/(body['name']+'.yaml');path.write_text(yaml.safe_dump(spec,sort_keys=False))
    plan['specs'].append({'name':body['name'],'role':role,'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    print(json.dumps({'prepared':body['name'],'role':role,'tasks':len(names)}),flush=True)


# Reconstruct completed-policy requests using checksummed exact-budget receipts.
# Reference-DUR is first so all Object candidates can use its recovered result.
completed = sorted(selection['completed_training_needing_eval'],key=lambda r:r['variant']!='reference')
new_evals = {}
for row in completed:
    cell=(row['suite'],row['variant'],row['mode'])
    suffix=row['suite'].removeprefix('libero_')
    run=f"arc-sc-20261004-{suffix}-{row['variant'].replace('_','-')}-{row['mode']}-eval"
    new_evals[cell]=run
for row in completed:
    cell=(row['suite'],row['variant'],row['mode']);obs=observed[cell]
    suffix=row['suite'].removeprefix('libero_')
    identity={'source_run':row['training_run_id'],'source_commit':SOURCE,'suite':row['suite'],
              'method':f"arc_{row['mode']}",'arc_stream_variant':row['variant'],
              'total_optimizer_steps':campaign()['optimizer_steps'][row['suite']]}
    if row['variant']!='reference':
        refcell=(row['suite'],'reference',row['mode'])
        identity['stream_reference_run']=new_evals.get(refcell,by_cell[refcell]['evaluation_run_id'])
    ready=checkpoint_ready(client,identity);assert ready
    request,proof=ready;validate_request(request)
    resume=row['evaluation_run_id'] if (obs['evaluation_status'] or {}).get('state')=='ROLLOUTS' else None
    run=new_evals[cell]
    spec=evaluation_workflow(SOURCE,run,request,workers=5,resume_evaluation_from=resume)
    spec['workflow']['resources']['default']['memory']='120Gi'
    retry_downloads(spec['workflow']['tasks'][0])
    # A completed candidate still waits on CPU for its reference if needed.
    # Object component-time DUR depends on the resumed reference-DUR result.
    if row['variant']!='reference':
        proof0=json.loads(Path('/tmp/libero-streams-20261002/parallel/replay',suffix+'.json').read_text())
        refcell=(row['suite'],'reference',row['mode']);ref=by_cell[refcell]
        references={m:by_cell[(row['suite'],'reference',m)] for m in ('stk','dur')}
        template=expanded_workflow(SOURCE,'gate-'+suffix,row['suite'],replay_proof=proof0,references=references)['workflow']
        waiter=copy.deepcopy(next(t for t in template['tasks'] if t['name']==f"reference-{row['mode']}-ready"))
        old=ref['evaluation_run_id'];new=identity['stream_reference_run'];waiter['args'][1]=waiter['args'][1].replace(old,new)
        spec['workflow']['resources']['reference_wait']=template['resources']['reference_wait']
        spec['workflow']['tasks'][0]['inputs']=[{'task':waiter['name']}]
        spec['workflow']['tasks'].insert(0,waiter)
    save(spec,'completed_policy_evaluation')
    plan['evaluation_replacements'].append({'cell':cell,'old_run':row['evaluation_run_id'],'new_run':run,'resume_from':resume,'request':request,'artifact_hashes':proof})

# Three cancelled suites each contain six interrupted and twelve unstarted cells.
partial={(r['suite'],r['variant'],r['mode']):r for r in selection['partial_training']}
for suffix in ('spatial','goal','10'):
    suite='libero_'+suffix;name='arc-sb-20261004-'+suffix
    references={m:by_cell[(suite,'reference',m)] for m in ('stk','dur')}
    assert all(observed[(suite,'reference',m)]['evaluation_status']['state']=='POLICY_EVALUATION_COMPLETE' for m in ('stk','dur'))
    proof=json.loads(Path('/tmp/libero-streams-20261002/parallel/replay',suffix+'.json').read_text())
    spec=expanded_workflow(SOURCE,name,suite,replay_proof=proof,references=references)
    resumed=0
    for task in spec['workflow']['tasks']:
        if task['name'].startswith('train-'):
            env=task['environment'];cell=(suite,env['ARC_STREAM_VARIANT'],env['ARC_STREAM_MODE'])
            old=by_cell[cell];request=None
            if cell in partial:
                request=resume_request(old,ROOT/'artifacts',SOURCE);attach_resume(task,request);resumed+=1
            else:
                assert observed[cell]['training_status'] is None
            retry_downloads(task)
            plan['training_replacements'].append({'cell':cell,'old_run':old['training_run_id'],'new_run':env['RUN_ID'],'workflow':name,'training_task':task['name'],'evaluation_task':task['name'].replace('train-','evaluate-'),'resume_request':request})
        elif task['name'].startswith('evaluate-'):
            retry_downloads(task)
    assert resumed==6
    save(spec,'cancelled_suite_recovery')

# Object training is healthy. Relay its 18 evaluations to the recovered reference
# without changing, duplicating or restarting any training task.
old_path=Path('/tmp/libero-streams-expand/20261003T215731Z/specs/arc-sx-20261003-object.yaml')
original=yaml.safe_load(old_path.read_text());spec=copy.deepcopy(original);body=spec['workflow'];body['name']='arc-sv-20261004-object'
old_tasks={t['name']:t for t in original['workflow']['tasks']};body['tasks']=[]
new_ref=new_evals[('libero_object','reference','dur')]
for mode in ('stk','dur'):
    waiter=copy.deepcopy(old_tasks[f'reference-{mode}-ready'])
    if mode=='dur':waiter['args'][1]=waiter['args'][1].replace(by_cell[('libero_object','reference','dur')]['evaluation_run_id'],new_ref)
    body['tasks'].append(waiter)
lanes=[None]*4
for index in range(3,21):
    training=old_tasks[f'train-{index:02d}'];env=training['environment'];mode=env['ARC_STREAM_MODE']
    cell=('libero_object',env['ARC_STREAM_VARIANT'],mode);row=by_cell[cell]
    reference=new_ref if mode=='dur' else by_cell[('libero_object','reference','stk')]['evaluation_run_id']
    identity={'source_run':row['training_run_id'],'source_commit':SOURCE,'suite':'libero_object',
              'method':f'arc_{mode}','arc_stream_variant':row['variant'],'total_optimizer_steps':campaign()['optimizer_steps']['libero_object'],'stream_reference_run':reference}
    waiter=copy.deepcopy(old_tasks[f'reference-{mode}-ready']);waiter['name']=f'checkpoint-{index:02d}'
    waiter['args'][1]="python -m pip install --no-cache-dir boto3==1.43.98\npython /tmp/wait-checkpoint.py --identity /tmp/policy-identity.json --output '{{output}}'"
    waiter['files']=[{'path':'/tmp/wait-checkpoint.py','contents':Path('scripts/benchmarks/wait_libero_stream_checkpoint.py').read_text()},
                     {'path':'/tmp/policy-identity.json','contents':json.dumps(identity,indent=2)+'\n'}]
    evaluation=copy.deepcopy(old_tasks[f'evaluate-{index:02d}'])
    run=f"arc-sv-20261004-object-{row['variant'].replace('_','-')}-{mode}-eval"
    evaluation['environment']['RUN_ID']=run
    evaluation['inputs']=[{'task':waiter['name']},{'task':f'reference-{mode}-ready'}]
    lane=(index-3)%4
    if lanes[lane]:evaluation['inputs'].append({'task':lanes[lane]})
    lanes[lane]=evaluation['name'];retry_downloads(evaluation)
    body['tasks'].extend((waiter,evaluation))
    plan['object_evaluation_replacements'].append({'cell':cell,'old_run':row['evaluation_run_id'],'new_run':run,'evaluation_task':evaluation['name'],'reference_run':reference,'identity':identity})
assert len(body['tasks'])==38 and not any(t['name'].startswith('train-') for t in body['tasks'])
save(spec,'healthy_object_evaluation_relay')
assert len(plan['specs'])==10 and len(plan['training_replacements'])==54 and len(plan['object_evaluation_replacements'])==18
assert sum(r['resume_request'] is not None for r in plan['training_replacements'])==18
(ROOT/'recovery-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
