"""Exact full-v3 parser-failure retry; fresh0 only after positive zero entry.

Read-only metadata. Actual current queue/claims/steps and allocated CSV absence
remain mandatory in the caller. This does not authorize checkpoint recovery.
"""
import copy,json,re
from pathlib import Path
from . import prelaunch_timeout_retry as previous
from .prelaunch_timeout_retry import need,parse,sha,canonical,bounded,reviewed_parser,instant,instant_nanoseconds,gnu_stat_nanoseconds
TASK=previous.TASK
PARENT,CAPTURE='412827','412828'
OLD_ROOT,NEW_ROOT=TASK+'/full-v3',TASK+'/full-v4'
NAME='elmo-large314m-qknorm-swiglu-rmsnorm-full-v3';NEW_NAME=NAME[:-1]+'4'
REQUEST_REL='ops/requests/'+NAME+'.json'
REQUEST_SHA='30f605a1466cd4f7e8f2cd367b034013ad767bed7623d56c57647c184551a6a0'
FILES_SHA='f3cb03872ed6bdd61b1ca8f756ffbf47ad7ddecb2bde71e1e0434701ea0fec3d'
CANONICAL_SHA='ae992614c50a648e4818d68c9608bb51ec280b5b38932c3453bcdcfb0a1fc0ee'
EVIDENCE_MANIFEST_SHA='e98939d3f88422ff2110e782d77bedaa452f4ea87d99fc3f11290d9eaa8e458f'
CLAIM_NAMES=previous.CLAIM_NAMES
PRIOR_ATTEMPTS=previous.PRIOR_ATTEMPTS+(dict(root=OLD_ROOT,name=NAME,parent=PARENT,capture=CAPTURE,request_sha=REQUEST_SHA,files_sha=FILES_SHA),)
CLAIM_ROOTS=tuple(a+'/'+s for a in [*(p['root'] for p in PRIOR_ATTEMPTS),NEW_ROOT] for s in ('claims','request-claims','ops/claims','ops/request-claims'))
MAX_BYTES=1<<20

def expected_argv(capture=False):
 args=['/usr/local/bin/sbatch','--parsable']+([] if capture else ['--hold'])+['--no-requeue','--account=loaner','--qos=loaner-normal','--partition=rl2']
 if capture:
  return args+['--job-name=elmo-native-qknorm-swiglu-rmsnorm-full-v3','--nodelist=gpu21','--nodes=1','--ntasks=1','--cpus-per-task=2','--mem=1G','--gres=none','--time=00:10:00','--export=NONE','--dependency=afterany:'+PARENT,'--comment=elmo-full-observer:'+REQUEST_SHA+':'+PARENT,'--chdir='+OLD_ROOT,'--output='+OLD_ROOT+'/logs/%x-%j.out','--error='+OLD_ROOT+'/logs/%x-%j.err',OLD_ROOT+'/capture-native.sbatch',OLD_ROOT,PARENT,REQUEST_SHA,FILES_SHA]
 return args+['--comment=elmo-full:'+REQUEST_SHA,'--job-name='+NAME,'--nodelist=gpu21','--nodes=1','--ntasks=1','--cpus-per-task=32','--mem=192G','--gres=gpu:h100:4','--time=7-00:00:00','--export=NONE','--chdir='+OLD_ROOT,'--output='+OLD_ROOT+'/logs/%x-%j.out','--error='+OLD_ROOT+'/logs/%x-%j.err',OLD_ROOT+'/full.sbatch',OLD_ROOT,REQUEST_SHA,FILES_SHA]

def verify(evidence_root):
 root=Path(evidence_root);pins_bytes=bounded(root/'EVIDENCE.json');need(sha(pins_bytes)==EVIDENCE_MANIFEST_SHA,'Changed exact parser-failure evidence manifest');pins=parse(pins_bytes)
 actual=set()
 for p in root.rglob('*'):
  need(not p.is_symlink(),'Evidence symlink forbidden')
  if p.is_file():actual.add(str(p.relative_to(root)))
 need(actual==set(pins)|{'EVIDENCE.json'},'Unexpected or missing parser-failure evidence')
 raw={}
 for name,digest in pins.items():
  need(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe evidence member')
  raw[name]=bounded(root/name);need(sha(raw[name])==digest,'Changed actual parser-failure evidence: '+name)
 need(sha(raw['original/FILES.sha256'])==FILES_SHA and sha(raw['original/'+REQUEST_REL])==REQUEST_SHA,'Wrong failed full-v3 source/identity')
 manifest={line.split('  ',1)[1]:line.split('  ',1)[0] for line in raw['original/FILES.sha256'].decode().splitlines()}
 for name in raw:
  if name.startswith('original/') and name!='original/FILES.sha256':need(manifest.get(name[9:])==sha(raw[name]),'Executed control/import file differs from sealed manifest')
 original=parse(raw['original/'+REQUEST_REL]);need(canonical(original)==CANONICAL_SHA,'Wrong full-v3 science')
 prior=previous.verify(root/'previous-v2');previous.verify_request(prior['original_request'],original)
 receipt=parse(raw['transport/receipt.json']);need(receipt['root']==OLD_ROOT and receipt['parent']==PARENT and receipt['capture']==CAPTURE and receipt['all_remote_hashes_equal_local'] is True and receipt['all_before_after_stats_equal'] is True,'Actual readback provenance differs')
 observed={name[9:] for name in raw if name.startswith('observed/')};need(set(receipt['received'])==observed,'Actual readback closure changed')
 for name,row in receipt['received'].items():
  need(row['sha256']==sha(raw['observed/'+name]) and row['bytes']==len(raw['observed/'+name]) and row['physical_owner_verified'] is True and row['before_after_stat_equal'] is True,'Readback bytes/ownership changed')
  if 'original/'+name in raw:need(raw['observed/'+name]==raw['original/'+name],'Current bytes differ from executed control closure')
 claim=lambda name:raw['observed/claims/'+REQUEST_SHA+'/'+name].decode()
 names=previous.observed(parse(raw['transport/claim-list.json'])).splitlines();need(set(names)==set(CLAIM_NAMES) and len(names)==len(CLAIM_NAMES),'Failed allocation gained admission or ambiguous claims')
 for label,job in [('parent',PARENT),('capture',CAPTURE)]:
  need(claim(label.upper()+'_JOB_ID').strip()==job and claim(label+'-sbatch.stdout').strip()==job and claim(label+'-sbatch.stderr')=='' and claim(label+'-sbatch.exit')=='0\n','Unknown accepted native outcome')
  need(parse(claim(label+'-sbatch.argv.json'))==expected_argv(label=='capture'),'Accepted argv changed')
 prepared=parse(claim('PREPARED.json'));release=parse(claim('RELEASE_ATTEMPT.json'));accepted=parse(claim('RELEASE_ACCEPTED.json'))
 need(prepared['request_sha256']==REQUEST_SHA and prepared['manifest_sha256']==FILES_SHA and prepared['parent_argv']==expected_argv(),'Prepared payload differs')
 need(release['request_sha256']==REQUEST_SHA and release['parent']==PARENT and release['capture']==CAPTURE and accepted['parent']==PARENT and accepted['capture']==CAPTURE and accepted['stdout']=='','Release binding differs')
 need(raw['observed/request-claims/'+NAME+'/REQUEST_SHA256']==(REQUEST_SHA+'\n').encode(),'Accepted request marker differs')
 native=reviewed_parser(raw['original/native_guard.py'],manifest['native_guard.py'],'_full_v3_exact_native')
 early=reviewed_parser(raw['original/early_native_capture.py'],manifest['early_native_capture.py'],'_full_v3_exact_early_native')
 def read_remote(path):
  logical=Path(path);need(logical.is_relative_to(Path(OLD_ROOT)),'Original reader escaped exact allocation')
  return raw['observed/'+str(logical.relative_to(OLD_ROOT))].decode()
 terminal=early.verify_record(parse(raw['observed/native-'+PARENT+'.early.json']),PARENT,CAPTURE,REQUEST_SHA,FILES_SHA,read_remote,native)
 need(terminal['parent_state']=='FAILED' and terminal['parent_exit_code']=='1:0' and terminal['parent_end']=='2026-10-27T03:07:51','Different genuine failure outcome')
 # Reopen the actual observer-authenticated commit and independently replay it.
 final=parse(raw['final-capture/receipt.json']);auth=raw['final-capture/native-'+PARENT+'.authenticated.json']
 need(final['status']=='READ_ONLY_BYTES_VERIFIED' and final['auth_path']==OLD_ROOT+'/native-'+PARENT+'.authenticated.json' and final['auth_sha256']==sha(auth) and final['auth_before_after_stat_equal'] is True and final['auth_owner']==41003 and final['transport_sha256']==sha(raw['final-capture/transport.stdout']),'Final authenticated readback differs')
 terminal['early_observation_sha256']=sha(raw['observed/native-'+PARENT+'.early.json'])
 need(parse(auth)==terminal,'Actual observer commit differs from independently authenticated native evidence')
 observer_record=parse(raw['final-capture/observer.json']);observer_raw=previous.observed(observer_record)
 observer=native.check_native(observer_raw,job=CAPTURE,state='COMPLETED',capture=True,parent=PARENT,request_sha=REQUEST_SHA)
 pending=[native.check_native(claim(name),job=CAPTURE,state='PENDING',capture=True,parent=PARENT,request_sha=REQUEST_SHA) for name in ['CAPTURE_PENDING_NATIVE.txt','RELEASE_CAPTURE_PENDING_NATIVE.txt']]
 running=native.check_native(raw['observed/capture-running-'+CAPTURE+'.txt'].decode(),job=CAPTURE,state='RUNNING',capture=True,parent=PARENT,request_sha=REQUEST_SHA)
 native.same_attempt(*pending,running,observer,terminal=observer)
 observed_start,observed_end=previous.observation_interval(observer_record)
 need(native.native_time(observer['StartTime'])>=native.native_time(terminal['parent_end']) and observed_start>=native.native_time(observer['EndTime']) and observed_end<=instant(final['observed_at']),'Invalid final observer interval')
 step_record=parse(raw['final-capture/steps.json']);steps=previous.observed(step_record)
 step_start,step_end=previous.observation_interval(step_record)
 need(step_start>=native.native_time(observer['EndTime']) and step_end<=instant(final['observed_at']),'Final steps precede terminal')
 verify_queue_steps('',steps)
 need(raw['observed/host-exit-'+PARENT+'.txt']==b'1\n','Exact host failure missing')
 trace=raw['observed/logs/'+NAME+'-'+PARENT+'.err'].decode()
 need(trace.count('Traceback (most recent call last):')==1 and 'line 120, in allocated\n    validate(q,b,template);retry_evidence(root,q);verify_review' in trace and 'line 107, in file_mtime\n    at = datetime.fromisoformat(parts[2])' in trace and trace.endswith("ValueError: Invalid isoformat string: '2026-10-27 01:39:55.703321584 +0000'\n"),'Not exact proved prelauncher parser failure')
 shell=raw['original/full.sbatch'].decode();gate='/usr/bin/python3 -B "$root/guard_full.py" --root "$root"';launch='bash "$root/ops/full.sbatch" "$root/ops" "$request" "$request_sha" "$ops_sha"'
 need('set -euo pipefail' in shell and shell.count(gate)==shell.count(launch)==1 and shell.index(gate)<shell.index(launch),'Failure is not a fail-fast guard before launcher')
 guard=raw['original/guard_full.py'].decode();section=guard[guard.index('def allocated(root):'):guard.index('def completed(root,job):')]
 need(section.index('validate(q,b,template);retry_evidence(root,q);verify_review')<section.index('verify_smoke(root,q,b,progress_dir=claim)')<section.index('destination_probe(')<section.index("write_once(claim/'RUNNING_ADMISSION.json'"),'Wrong exact guard ordering')
 need(previous.observed(parse(raw['transport/status-list.json'])).strip()=='','Maintained launcher status was created')
 return dict(schema='elmo-exact-prelaunch-stat-retry-v1',status='EXACT_THREE_PRELAUNCH_FAILURES_VERIFIED',failed_parent=PARENT,capture=CAPTURE,original_request=original,original_request_sha256=REQUEST_SHA,original_manifest_sha256=FILES_SHA,optimizer_updates_possible_by_reviewed_control_flow=0,maintained_training_launcher_reached=False,parent_native_state='FAILED',parent_native_exit_code='1:0',parent_native_end_time=terminal['parent_end'],genuine_parent_terminal_proof=True,terminal_authenticated_from_genuine_saved_queries=True,producer_authenticated_commit_inferred=False,observer_own_terminal='COMPLETED_0_0_ACTUAL',observer_end_time=observer['EndTime'],actual_observer_authenticated_commit_verified=True,earlier_412825_parent_terminal='UNKNOWN_NOT_INFERRED',checkpoint_recovery_authorized=False,live_launch_admitted=False)

def verify_request(original,current):
 need(canonical(original)==CANONICAL_SHA,'Wrong verified full-v3 request')
 def relocate(v):
  if isinstance(v,dict):return {k:relocate(x) for k,x in v.items()}
  if isinstance(v,list):return [relocate(x) for x in v]
  return v.replace(OLD_ROOT,NEW_ROOT) if isinstance(v,str) else v
 expected=relocate(copy.deepcopy(original));expected['job_name']=NEW_NAME
 need(original['local_logs'].endswith('-attempt-v3'),'Unexpected prior CSV path')
 expected['local_logs']=original['local_logs'][:-len('-attempt-v3')]+'-attempt-v4'
 expected['overrides']=[s.replace('logger.csv.save_dir='+original['local_logs'],'logger.csv.save_dir='+expected['local_logs']) for s in expected['overrides']]
 need(current==expected and current['output']==original['output'] and current['continuation'] is None and not current.get('interruption') and 'logger.wandb.resume=never' in current['overrides'],'Changed exact fresh0 science or retry scope')
 return dict(status='EXACT_FRESH0_FULL_V4_REQUEST_COMPATIBLE',live_launch_admitted=False)

def prior_claim_names(root,attempt):
 if attempt['root']==TASK+'/full-v2':
  return [Path(p).name for p in parse(bounded(Path(root)/'previous-v2/transport/accepted-claims-readback.json'))['claim_files']]
 return list(CLAIM_NAMES)

def expected_live_file_hashes(root):
 verify(root);expected=previous.expected_live_file_hashes(Path(root)/'previous-v2')
 for name,digest in parse(bounded(Path(root)/'EVIDENCE.json')).items():
  if name.startswith('observed/claims/') or name in {'observed/FILES.sha256','observed/'+REQUEST_REL,'observed/request-claims/'+NAME+'/REQUEST_SHA256'}:
   expected[OLD_ROOT+'/'+name[9:]]=digest
 return expected

def verify_live_file_hashes(root,observed_hashes):
 need(observed_hashes==expected_live_file_hashes(root),'Current prior files changed or missing')

def verify_claim_entries(full_roots,checked_roots,entries,*,current_request_sha=None):
 roots={a['root'] for a in PRIOR_ATTEMPTS}|{NEW_ROOT};need(len(full_roots)==len(roots) and set(full_roots)==roots,'Unknown/missing full attempt root')
 need(len(checked_roots)==len(CLAIM_ROOTS) and set(checked_roots)==set(CLAIM_ROOTS),'Incomplete claim census')
 expected={p for a in PRIOR_ATTEMPTS for p in (a['root']+'/claims/'+a['request_sha'],a['root']+'/request-claims/'+a['name'])}
 if current_request_sha is not None:
  need(re.fullmatch('[0-9a-f]{64}',current_request_sha),'Invalid current request SHA');expected|={NEW_ROOT+'/claims/'+current_request_sha,NEW_ROOT+'/request-claims/'+NEW_NAME}
 need(len(entries)==len(expected) and set(entries)==expected,'Unknown/missing/partial claim')
 return sorted(expected)

def verify_queue_steps(queue,steps,*,current_job=None,current_capture=None):
 need(isinstance(queue,str) and isinstance(steps,str) and max(len(queue.encode()),len(steps.encode()))<=MAX_BYTES,'Oversized/incomplete queue/steps')
 prior={a[k] for a in PRIOR_ATTEMPTS for k in ('parent','capture')};allowed=set()
 if current_job is not None or current_capture is not None:
  need(isinstance(current_job,str) and isinstance(current_capture,str) and current_job.isdecimal() and current_capture.isdecimal() and len(prior|{current_job,current_capture})==8,'Wrong current native IDs');allowed={current_job,current_capture}
 for line in queue.splitlines():
  row=line.split('|');need(len(row)==5 and row[2]=='paphiwetsa3','Malformed complete owned queue');job,name,_,state,_=row
  need(not any(job==old or job.startswith(old+'.') or job.startswith(old+'_') for old in prior),'Prior native identity remains queued')
  if 'qknorm-swiglu-rmsnorm' in name or job in allowed:need(job in allowed and name==(NEW_NAME if job==current_job else 'elmo-native-qknorm-swiglu-rmsnorm-full-v4') and state in {'PENDING','RUNNING'},'Another or wrongly excluded writer')
 for line in steps.splitlines():
  if line=='No job steps exist in the system.':continue
  match=re.search(r'(?:^|\s)StepId=(\S+)',line);need(match is not None,'Malformed complete steps')
  need(not any(match[1]==old or match[1].startswith(old+'.') or match[1].startswith(old+'_') for old in prior),'Prior/array-alias worker survives')
 return dict(prior_jobs_absent_from_supplied_complete_native_outputs=True,live_launch_admitted=False)
