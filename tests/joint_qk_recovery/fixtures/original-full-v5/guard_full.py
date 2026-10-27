"""Fresh full300k guard; replay exact sealed smoke reader before maintained C."""
from __future__ import annotations
import argparse,hashlib,json,math,os,subprocess,sys
from pathlib import Path
from bounded_census_io import CensusTransportError,MAX_COMMAND_BYTES
from host_hashlib_compat import install as install_host_hashlib
HOST_HASHLIB=install_host_hashlib()
from native_guard import REMOTE,REQUEST_NAME,CAPTURE_NAME,check_native,fields,need,same_attempt,release_observation
REQUEST=REMOTE/'ops/requests'/(REQUEST_NAME+'.json')
RUN_ID='mot-elmo-pickplace-continuous-large314m-qknorm-swiglu-rmsnorm-robot-human-camera-subtask-s42-300k-20261026-v1'
SMOKE_NAME='elmo-large314m-qknorm-swiglu-rmsnorm-smoke-v2'
SERVICE_SHA='adf45524de3c2e6dba186572e7151d48bfca0b3adc8e6cf9defd686b47744c87'
SOURCE_SHA='b5e8e898b82ea15b42e84e22a749c668c13e0c958ecba122fb10d99c3df39331'
SMOKE_CANDIDATE_SHA='2efe4758dad57c57df9198f2a0d2c9c700c979cb8eace010a1df72b9c4cc286c'
DYNAMIC_PRESERVATION={'observed_smoke_checkpoint_bytes','scheduled_checkpoint_bytes_estimate','maintained_storage_required_bytes'}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for data in iter(lambda:stream.read(8<<20),b''):h.update(data)
 return h.hexdigest()

def write_once(path,value):
 with Path(path).open('x') as f:
  json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(Path(path).parent,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)

def validate(q,b,template,*,bound=True):
 dynamic={'phase','draft','smoke','preservation_plan'}
 need({k:v for k,v in q.items() if k not in dynamic}=={k:v for k,v in template.items() if k not in dynamic},'Changed exact full science/infrastructure')
 need({k:v for k,v in q['preservation_plan'].items() if k not in DYNAMIC_PRESERVATION}=={k:v for k,v in template['preservation_plan'].items() if k not in DYNAMIC_PRESERVATION},'Changed all15 retention contract')
 need(q['run_id']==RUN_ID and q['job_name']==REQUEST_NAME and q['parameters']==314227000
      and q['target_steps']==q['allocation_stop_step']==300000 and q['allocation_boundaries']==[300000]
      and q['allocation'] is None and q['continuation'] is None and q['archive_policy'] is None
      and q['reserved_checkpoint_bytes']==80000000000,'Wrong fresh full300kTOTAL scope')
 need(q['source_revision']=='82f65498e25f41b87c954717722dcf7360e5bde8' and q['source_manifest_sha256']==SOURCE_SHA
      and q['operational_service']['manifest_sha256']==SERVICE_SHA,'Wrong exact source/service')
 for value in ['logger.wandb.resume=never','logger.wandb.id='+RUN_ID,'trainer.max_steps=300000','trainer.val_check_interval=20000','callbacks.model_checkpoint.every_n_train_steps=20000','callbacks.model_checkpoint.save_top_k=-1','callbacks.model_checkpoint.save_last=link']:
  need(value in q['overrides'] and sum(x.startswith(value.split('=',1)[0]+'=') for x in q['overrides'])==1,'Changed fresh initialization/schedule')
 need(not any(x.startswith(('ckpt_path=','model.ckpt_path=')) and not x.endswith('=null') for x in q['overrides']),'Smoke checkpoint cannot initialize fresh full')
 need(b['schema']=='elmo-architecture-full-bindings-v1' and b['remote_root']==str(REMOTE),'Wrong full binding')
 ready=b['state']=='ACTUAL_SMOKE_BOUND_ROOT_REVIEW_REQUIRED'
 if not ready:
  need(b['state']=='UNBOUND_SMOKE_PROOF_PENDING' and q['phase']=='DRAFT_UNBOUND_DO_NOT_SUBMIT' and q['draft']['executable'] is False and q['smoke'] is None
       and all(b[k] is None for k in ['smoke_parent_job_id','smoke_capture_job_id','smoke_request_sha256','smoke_payload_manifest_sha256','smoke_evidence_manifest_sha256','timing'])
       and all(q['preservation_plan'][k] is None for k in DYNAMIC_PRESERVATION),'Unbound state changed')
  need(not bound,'Actual GPU smoke completion remains UNBOUND');return False
 need(q['phase']=='full' and q['draft']['executable'] is True and str(b['smoke_parent_job_id']).isdecimal() and str(b['smoke_capture_job_id']).isdecimal(),'Missing actual successful smoke identity')
 size=q['preservation_plan']['observed_smoke_checkpoint_bytes'];need(type(size) is int and size>0,'Actual smoke checkpoint metadata required')
 need(q['preservation_plan']['scheduled_checkpoint_bytes_estimate']==15*size and q['preservation_plan']['maintained_storage_required_bytes']==15*size+1000000000<=80000000000,'80GB cannot retain all15 checkpoints')
 need(b['timing']['actual_smoke_checkpoint_bytes']==size and math.isfinite(b['timing']['warm_seconds_per_update']) and b['timing']['warm_seconds_per_update']>0,'Actual measured smoke timing required')
 return True

def manifest(path,root):
 result={}
 for line in Path(path).read_text().splitlines():
  digest,name=line.split('  ',1);file=Path(root)/name
  need(name not in result and not name.startswith('/') and '..' not in Path(name).parts and file.resolve(strict=True)==file and file.is_file() and sha(file)==digest,'Changed physical proof/reader member: '+name)
  result[name]=digest
 return result

def verify_smoke(root,q,b,*,progress_dir=None):
 root=Path(root);reader=root/'proofs/smoke';seed=root/'inputs/smoke-candidate-files.sha256'
 need(sha(seed)==SMOKE_CANDIDATE_SHA and sha(root/'SMOKE_EVIDENCE.sha256')==b['smoke_evidence_manifest_sha256'],'Smoke reader/evidence closure changed')
 evidence=manifest(root/'SMOKE_EVIDENCE.sha256',reader)
 need({str(f.relative_to(reader)) for f in reader.rglob('*') if f.is_file()}==set(evidence),'Unexpected smoke evidence file outside exact closure')
 need(sha(reader/'FILES.sha256')==b['smoke_payload_manifest_sha256'],'Actual bound smoke static manifest changed')
 deployed=manifest(reader/'FILES.sha256',reader)
 frozen={name:digest for digest,name in (line.split('  ',1) for line in seed.read_text().splitlines())}
 mutable={'bindings.json','ops/requests/'+SMOKE_NAME+'.json'}
 for name,digest in frozen.items():
  need(name in deployed and name in evidence and (name in mutable or deployed[name]==digest),'Unreviewed smoke reader/source bytes: '+name)
 need(set(deployed)==set(frozen)|{'proofs/cpu/'+x for x in json.loads((reader/'bindings.json').read_text())['proofs']},'Actual smoke static proof closure differs')
 request=reader/'ops/requests'/(SMOKE_NAME+'.json');need(sha(request)==b['smoke_request_sha256'],'Actual smoke request changed')
 claim=reader/'claims'/b['smoke_request_sha256']
 need((claim/'PARENT_JOB_ID').read_text().strip()==b['smoke_parent_job_id'] and (claim/'CAPTURE_JOB_ID').read_text().strip()==b['smoke_capture_job_id'],'Actual smoke native claim differs')
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1');env.pop('PYTHONPATH',None)
 from completed_smoke_reader import run_completed_reader
 call=run_completed_reader([sys.executable,'-B',str(reader/'guard.py'),'--root',str(reader),'--check-completed',b['smoke_parent_job_id']],env=env,cwd=reader,progress_dir=progress_dir)
 need(call.returncode==0,'Exact sealed smoke reader rejected: '+call.stderr[-2000:]);actual=json.loads(call.stdout)
 need(actual['status']=='MAINTAINED_FRESH_SMOKE_HOST_NATIVE_PASS' and actual['parameters']==314227000 and actual['timing']==b['timing'],'Actual completed smoke/timing differs')
 smoke=json.loads(request.read_text())
 for key in ('parameters','expert_parameters','expert_widths','world_size','per_rank_batch','source','source_revision','source_archive','source_archive_sha256','source_manifest','source_manifest_sha256','data_manifest','data_manifest_sha256','lambda_runtime','data_cache','quota','subtask_contract','model_execution','interfaces','lr_policy'):
  need(q.get(key)==smoke.get(key),'Full differs from actual smoke: '+key)
 output=reader/'output'/Path(smoke['output']).name;completion=json.loads((output/'smoke-completion.json').read_text());capacity=json.loads(request.with_suffix('.capacity.json').read_text())
 for rank in range(4):
  need(json.loads((output/('fit-rank'+str(rank)+'.json')).read_text())==completion['ranks'][rank],'Fit rank/aggregation mismatch')
  row=json.loads(request.with_suffix('.capacity-rank'+str(rank)+'.json').read_text())
  need(row==capacity['ranks'][rank] and row['rank']==rank and row['job_id']==b['smoke_parent_job_id'] and row['status']=='PASS'
       and row['compiled_bf16_forward_backward'] is True and math.isfinite(float(row['loss'])) and row['peak_memory_bytes']<row['device_memory_bytes'],'Actual capacity rank mismatch')
 expected={'request':request,'completion':output/'smoke-completion.json','preflight':request.with_suffix('.preflight.json'),'capacity':request.with_suffix('.capacity.json'),'resolved':request.with_suffix('.resolved.yaml'),'native':reader/('native-'+b['smoke_parent_job_id']+'.txt'),'host_exit':reader/('host-exit-'+b['smoke_parent_job_id']+'.txt')}
 need(q['smoke']['job_id']==b['smoke_parent_job_id'] and q['smoke']['native_format']=='scontrol-v1','Wrong maintained smoke interface')
 for key,path in expected.items():need(q['smoke'][key]==dict(path=str(REMOTE/'proofs/smoke'/path.relative_to(reader)),sha256=sha(path)),'Maintained smoke binding differs: '+key)
 return actual

def verify_review(root,request_sha,manifest_sha):
 for name in ('root-review.json','deployment-readback.json'):
  row=json.loads((Path(root)/name).read_text());need(row.get('status')=='PASS' and row.get('request_sha256')==request_sha and row.get('manifest_sha256')==manifest_sha and row.get('remote_root')==str(REMOTE),'Genuine root review/deployment missing')
  if name=='root-review.json':need(row.get('visible_preflight_sent') is True and row.get('measured_timing_reported') is True,'Actual resolved preflight and measured timing must be reported')

def retry_evidence(root,q):
 from prelaunch_held_retry import verify,verify_request
 proof=verify(Path(root)/'proofs/prelaunch-held-v1')
 verify_request(proof['original_request'],q)
 return proof

def live_quiescence(root,q,job,capture):
 from live_retry_census import collect
 def run(command):
  need(len(command.encode())<MAX_COMMAND_BYTES,'Allocated census command exceeds4KiB')
  try:result=subprocess.run(['bash','-c',command],capture_output=True,text=True,timeout=120)
  except subprocess.TimeoutExpired as error:
   raise CensusTransportError('Allocated census timeout',stdout=error.stdout or '',stderr=error.stderr or '',timed_out=True) from error
  if result.returncode:
   raise CensusTransportError('Allocated census failed',stdout=result.stdout,stderr=result.stderr,returncode=result.returncode)
  return result.stdout
 return collect(run,Path(root)/'proofs/prelaunch-held-v1',q,sha(Path(root)/'ops/requests'/(REQUEST_NAME+'.json')),
                phase='ALLOCATED_ENTRY',receipt_dir=Path(root)/'output/census-ALLOCATED_ENTRY',actor='ALLOCATED_REMOTE',current_job=job,current_capture=capture)


def allocated(root):
 root=Path(root);request=root/'ops/requests'/(REQUEST_NAME+'.json');q=json.loads(request.read_text());b=json.loads((root/'bindings.json').read_text());template=json.loads((root/'REQUEST_TEMPLATE.json').read_text())
 validate(q,b,template);retry_evidence(root,q);verify_review(root,sha(request),sha(root/'FILES.sha256'))
 need(os.getuid()==41003 and os.environ.get('SLURMD_NODENAME')=='gpu21' and os.environ.get('SLURM_RESTART_COUNT','0')=='0','Exact fresh owned node21 allocation required')
 job=os.environ.get('SLURM_JOB_ID','');need(job.isdecimal(),'Actual native job required');claim=root/'claims'/sha(request)
 need((claim/'PARENT_JOB_ID').read_text().strip()==job and (claim/'parent-sbatch.stdout').read_text().strip()==job and (claim/'parent-sbatch.exit').read_text().strip()=='0','Actual accepted full claim missing')
 prepared=json.loads((claim/'PREPARED.json').read_text());need(prepared['request_sha256']==sha(request) and prepared['manifest_sha256']==sha(root/'FILES.sha256'),'Prepared full payload differs')
 capture=(claim/'CAPTURE_JOB_ID').read_text().strip();need((claim/'capture-sbatch.stdout').read_text().strip()==capture and (claim/'capture-sbatch.exit').read_text().strip()=='0','Actual observer acceptance missing')
 held=[check_native((claim/filename).read_text(),job=job,state='PENDING',request_sha=sha(request)) for filename in ['HELD_NATIVE.txt','RELEASE_HELD_NATIVE.txt']]
 observer_held=[check_native((claim/filename).read_text(),job=capture,state='PENDING',capture=True,parent=job,request_sha=sha(request)) for filename in ['CAPTURE_PENDING_NATIVE.txt','RELEASE_CAPTURE_PENDING_NATIVE.txt']]
 same_attempt(*observer_held)
 release=json.loads((claim/'RELEASE_ATTEMPT.json').read_text());need(release['parent']==job and release['capture']==capture and release['request_sha256']==sha(request),'Genuine release attempt missing')
 raw=subprocess.check_output(['/usr/local/bin/scontrol','show','job',job,'-o','-dd'],text=True,timeout=20);active=check_native(raw,job=job,state='RUNNING',request_sha=sha(request))
 same_attempt(*held,active)
 observer_raw=subprocess.check_output(['/usr/local/bin/scontrol','show','job',capture,'-o','-dd'],text=True,timeout=20)
 observer_current=check_native(observer_raw,job=capture,state='PENDING',capture=True,parent=job,request_sha=sha(request))
 same_attempt(*observer_held,observer_current)
 quiescence=live_quiescence(root,q,job,capture);write_once(claim/'PRELAUNCH_RETRY_QUIESCENCE.json',quiescence)
 write_once(claim/'HOST_PYTHON.json',dict(schema='full-host-python-compatibility-v1',python_version=sys.version,executable=sys.executable,hashlib_adapter=HOST_HASHLIB,adapter_sha256=sha(root/'host_hashlib_compat.py')))
 verify_smoke(root,q,b,progress_dir=claim)
 need(not os.path.lexists(q['output']) and not os.path.lexists(q['local_logs']),'Fresh full paths consumed; preserve and inspect')
 need(sha(root/'ops/service/FILES.sha256')==SERVICE_SHA,'Maintained service changed');manifest(root/'ops/service/FILES.sha256',root/'ops/service')
 sys.path.insert(0,str(root/'ops/service'))
 from scripts.train.joint.native_interruption import destination_probe
 storage=destination_probe(q['output'],job,q['reserved_checkpoint_bytes']);write_once(claim/'STORAGE_PROBE.json',storage)
 write_once(claim/'RUNNING_ADMISSION.json',dict(status='ACTUAL_BOUND_FULL_NATIVE_ALLOCATION_PASS',job_id=job,request_sha256=sha(request),native=raw,smoke_parent=b['smoke_parent_job_id'],scope='Fresh full300kTOTAL; native completion remains external'))

def completed(root,job):
 root=Path(root).resolve();job=str(job);request=root/'ops/requests'/(REQUEST_NAME+'.json');q=json.loads(request.read_text());b=json.loads((root/'bindings.json').read_text())
 validate(q,b,json.loads((root/'REQUEST_TEMPLATE.json').read_text()));verify_smoke(root,q,b)
 digest=sha(request);claim=root/'claims'/digest
 need((claim/'PARENT_JOB_ID').read_text().strip()==job,'Full terminal belongs to another claim')
 capture=(claim/'CAPTURE_JOB_ID').read_text().strip()
 from early_native_capture import reopen_authenticated
 import native_guard as capture_api
 reopen_authenticated(root,job,capture,digest,sha(root/'FILES.sha256'),capture_api)
 terminal=check_native((root/('native-'+job+'.txt')).read_text(),job=job,state='COMPLETED',request_sha=digest)
 need((root/('native-'+job+'.exit')).read_text().strip()=='0' and (root/('native-'+job+'.stderr')).read_text()=='' and (root/('host-exit-'+job+'.txt')).read_text().strip()=='0','Successful actual host/native required')
 record=json.loads((claim/'RUNNING_ADMISSION.json').read_text());need(record['status']=='ACTUAL_BOUND_FULL_NATIVE_ALLOCATION_PASS' and record['job_id']==job and record['request_sha256']==digest,'Actual admitted full resources missing')
 active=check_native(record['native'],job=job,state='RUNNING',request_sha=digest)
 held=[check_native((claim/name).read_text(),job=job,state='PENDING',request_sha=digest) for name in ['HELD_NATIVE.txt','RELEASE_HELD_NATIVE.txt']]
 same_attempt(*held,active,terminal,terminal=terminal)
 release_observation(json.loads((root/('native-'+job+'.steps.json')).read_text()),job,digest,terminal)
 capture=(claim/'CAPTURE_JOB_ID').read_text().strip()
 observer=check_native((root/('capture-running-'+capture+'.txt')).read_text(),job=capture,state='RUNNING',capture=True,parent=job,request_sha=digest)
 observer_held=[check_native((claim/name).read_text(),job=capture,state='PENDING',capture=True,parent=job,request_sha=digest) for name in ['CAPTURE_PENDING_NATIVE.txt','RELEASE_CAPTURE_PENDING_NATIVE.txt']]
 same_attempt(*observer_held,observer)
 # This is a small proof read. The maintained allocated finalizer performed
 # actual checkpoint/model/optimizer/RNG/history verification before host exit.
 proof=json.loads((Path(q['output'])/'full-completion.json').read_text())
 need(proof['schema']=='joint-mot-full-v1' and proof['status']=='PASS' and proof['job_id']==job
      and proof['request_sha256']==digest and proof['run_id']==RUN_ID and proof['global_step']==300000
      and proof['world_size']==4 and proof['retained_milestones']==proof['validation_events']==15
      and all(proof[k] is True for k in ['strict_full_wrapper_optimizer_scheduler_reload','sampler_all_rank_rng_verified','sample_finite']),
      'Actual strict300k completion and15milestone/validation proof missing')
 return dict(status='STRICT_FULL300K_HOST_NATIVE_AND_15_MILESTONES_PASS',job_id=job,run_id=RUN_ID,
             full_completion_sha256=sha(Path(q['output'])/'full-completion.json'),observer_own_terminal='Not required or inferred; retain actual separate observation if available')


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--check-completed');a=p.parse_args()
 if a.check_completed:print(json.dumps(completed(a.root,a.check_completed),indent=2))
 else:allocated(a.root)
