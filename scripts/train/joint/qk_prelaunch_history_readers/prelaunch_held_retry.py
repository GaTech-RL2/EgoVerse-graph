"""Exact never-started held-v4 cancellation reader; no generic restart authority.

The v4 scheduler synthesizes StartTime=EndTime on cancellation. Positive held
state, no allocation/runtime/release and the bound operational stop distinguish
that bookkeeping value from an actual worker start. No missing log inference.
"""
import base64,copy,io,json,os,re,shlex,stat,tarfile
from datetime import datetime,timezone
from pathlib import Path
from . import prelaunch_stat_retry as previous
from .prelaunch_stat_retry import need,parse,sha,canonical,reviewed_parser,instant,instant_nanoseconds,gnu_stat_nanoseconds
TASK=previous.TASK
PARENT,CAPTURE='412829','412830'
OLD_ROOT,NEW_ROOT=TASK+'/full-v4',TASK+'/full-v5'
NAME='elmo-large314m-qknorm-swiglu-rmsnorm-full-v4';NEW_NAME=NAME[:-1]+'5'
REQUEST_REL='ops/requests/'+NAME+'.json'
REQUEST_SHA='bbe1bda1c6e5475e221b0423f4a1d1fcd552e19c6f80f70f4c912e70a14d7102'
FILES_SHA='3b5bd76e457aec7b38c9bd2aac42c6d89f287787bb7e0c7dd2dbe8734154bfec'
CANONICAL_SHA='4f435f434d451bea8b93b495fca523451622d38c16967aca1829a5553fd4ff85'
EVIDENCE_MANIFEST_SHA='15ee3a5bb152468d4616c90b9d5756a86926a799b726e877f6fdcc231cc8d7d4'
CLAIM_NAMES=previous.CLAIM_NAMES
HELD_CLAIM_NAMES=tuple(n for n in CLAIM_NAMES if n not in {'RELEASE_HELD_NATIVE.txt','RELEASE_CAPTURE_PENDING_NATIVE.txt','RELEASE_ATTEMPT.json','RELEASE_ACCEPTED.json'})
PRIOR_ATTEMPTS=previous.PRIOR_ATTEMPTS+(dict(root=OLD_ROOT,name=NAME,parent=PARENT,capture=CAPTURE,request_sha=REQUEST_SHA,files_sha=FILES_SHA),)
CLAIM_ROOTS=tuple(a+'/'+s for a in [*(p['root'] for p in PRIOR_ATTEMPTS),NEW_ROOT] for s in ('claims','request-claims','ops/claims','ops/request-claims'))
MAX_BYTES=2<<20
SCANCEL='/workspace/users/paphiwetsa3/tasks/pixel-video-gripper-optimization-20261017-v1/checkpoint-cadence-20261018-v1/tools/scancel-23.11.4'

def bounded(path):
 p=Path(path);need(p.is_absolute() and p.resolve(strict=True)==p,'Physical absolute evidence path required')
 fd=os.open(p,os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
 try:
  before=os.fstat(fd);need(stat.S_ISREG(before.st_mode) and before.st_size<=MAX_BYTES,'Unsafe or oversized exact held proof')
  with os.fdopen(fd,'rb',closefd=False) as stream:raw=stream.read(MAX_BYTES+1)
  identity=lambda x:(x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns)
  need(len(raw)<=MAX_BYTES and identity(before)==identity(os.fstat(fd))==identity(p.stat(follow_symlinks=False)),'Evidence changed during bounded read')
  return raw
 finally:os.close(fd)

def root_observed(row,*,not_before=None):
 need(row.get('exit_code')==0 and row.get('stderr')=='' and not row.get('timed_out',False) and not row.get('output_truncated',False),'Failed bounded root observation')
 begin,end=instant(row['started_utc']),instant(row['ended_utc']);need(begin<=end<=datetime.now(timezone.utc),'Invalid root observation interval')
 if not_before is not None:need(begin>=not_before,'Root observation precedes required event')
 need(len(row['stdout'].encode())<=MAX_BYTES,'Root observation oversized');return row['stdout']

def verify_readback(raw):
 receipt=parse(raw['readback/receipt.json']);archive=raw['readback/readback.tar']
 need(receipt.get('exit_code')==0 and raw['readback/stderr.txt']==b'' and receipt['archive_sha256']==sha(archive) and receipt['archive_bytes']==len(archive),'Unsuccessful actual readback')
 observed={p[9:] for p in raw if p.startswith('observed/')};need(receipt['required_files']==24 and len(receipt['files'])==24 and set(receipt['files'])==observed,'Actual readback membership differs')
 command=raw['readback/command.sh'].decode()
 need(command.startswith('set -euo pipefail; cd '+OLD_ROOT+'; for f in ') and 'test -f "$f"; test ! -L "$f"; test "$(stat -c %u -- "$f")" = 41003; test "$(stat -c %s -- "$f")" -le 2097152' in command,'Actual readback lacked exact basic physical checks')
 expected_order=command.split('; for f in ',1)[1].split('; do ',1)[0].split()
 need(len(expected_order)==24 and set(expected_order)==observed and command.split('; tar -cf - -- ',1)[1].strip().split()==expected_order,'Changed actual readback command targets')
 with tarfile.open(fileobj=io.BytesIO(archive),mode='r:') as tar:
  members=tar.getmembers();need(len(members)==24 and {m.name for m in members}==observed,'Readback archive member identity differs')
  for m in members:
   need(m.isreg() and m.uid==41003 and m.size<=MAX_BYTES and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts,'Unsafe readback archive member')
   body=tar.extractfile(m).read(MAX_BYTES+1);r=receipt['files'][m.name]
   need(body==raw['observed/'+m.name] and r['sha256']==sha(body) and r['bytes']==len(body)==m.size,'Readback archive/receipt bytes differ')
 # This transport did not produce independent remote hashes or before/after
 # stats. Its actual owner/size checks, successful tar and bytes are retained.
 return receipt

def verify_stable(raw):
 receipt=parse(raw['stable/receipt.json']);stream=raw['stable/transport.stdout'];command=raw['stable/command.sh']
 need(receipt['root']==OLD_ROOT and receipt['parent']==PARENT and receipt['capture']==CAPTURE and receipt['exit_code']==0 and receipt['all_remote_hashes_equal_local'] is True and receipt['all_before_after_stats_equal'] is True,'Final stable readback failed')
 need(receipt['transport_stdout_sha256']==sha(stream) and receipt['transport_stdout_bytes']==len(stream) and len(command)<=receipt['remote_command_bytes']<4096,'Stable transport/command bytes differ')
 need(b'test "$(readlink -e -- "$r/$f")" = "$r/$f"' in command and b'test "$(stat -c %u -- "$f")" = 41003' in command,'Stable physical ownership checks absent')
 names={name[len('stable/received/'):] for name in raw if name.startswith('stable/received/')};need(len(names)==28 and set(receipt['received'])==names,'Wrong stable exact membership')
 rows={}
 for line in stream.decode().splitlines():
  fields=line.split('|');need(len(fields)==5 and fields[0] not in rows,'Malformed or duplicate stable stream');rows[fields[0]]=fields[1:]
 need(set(rows)==names,'Missing stable transport member')
 for name,(digest,before,after,encoded) in rows.items():
  body=base64.b64decode(encoded,validate=True);entry=receipt['received'][name]
  need(body==raw['stable/received/'+name] and digest==sha(body)==entry['sha256'] and len(body)==entry['bytes'] and before==after==entry['before']==entry['after'] and entry['physical_owner_verified'] is True and entry['before_after_stat_equal'] is True,'Actual stable identity differs')
  if 'observed/'+name in raw:need(body==raw['observed/'+name],'Historical raw bytes changed before stable readback')
  if 'original/'+name in raw:need(body==raw['original/'+name],'Deployed control source differs')
 return receipt

def shell_observed(row,*,not_before):
 need(row.get('exit_code')==0 and row.get('stderr')=='' and not row.get('timed_out',False) and not row.get('output_truncated',False),'Unsuccessful final shell observation')
 start,end=instant(row['start_utc']),instant(row['end_utc']);need(not_before<=start<=end<=datetime.now(timezone.utc),'Final shell observation interval differs')
 need(len(row['stdout'].encode())<=MAX_BYTES,'Final shell observation oversized');return row['stdout']

def verify_final_metadata(raw,observer):
 end=instant(observer['EndTime']+'+00:00')
 # These historical root observations include successful actual stdout/stderr;
 # no absence authority is derived from their non-fail-fast compound commands.
 err=shell_observed(parse(raw['postfinish/stderr.json']),not_before=end)
 need(len(err.encode())<65537 and err.count('Traceback (most recent call last):')==1 and "prepared=json.loads(read(claim/'PREPARED.json'))" in err and err.endswith('ValueError: Unsafe/oversized native proof\n'),'Observer failed in a different location')
 final=parse(raw['final/closure.json']);script=final['script'];out=shell_observed(final,not_before=end)
 need(script.startswith('set -euo pipefail;') and 'LC_ALL=C sort' in script and shlex.split(final['command'])==['timeout','20s','bash','-c',script] and final['transport_argv'][-1]==final['command'],'Final compound observation must fail fast')
 need(out.splitlines()==sorted(HELD_CLAIM_NAMES),'Final exact claim names differ')
 need('r='+OLD_ROOT+';' in script and 'c='+OLD_ROOT+'/claims/'+REQUEST_SHA+';' in script,'Final exact path declarations differ')
 for name in ['RELEASE_ATTEMPT.json','RELEASE_ACCEPTED.json','RELEASE_HELD_NATIVE.txt','RELEASE_CAPTURE_PENDING_NATIVE.txt']:
  need(name in script,'Final absence command omitted release path')
 need('native-'+PARENT+'.authenticated.json' in script and all('$r/ops/'+name in script for name in ['status','claims','request-claims']),'Final absence command scope differs')

 return True

def expected_argv(capture=False):
 args=['/usr/local/bin/sbatch','--parsable']+([] if capture else ['--hold'])+['--no-requeue','--account=loaner','--qos=loaner-normal','--partition=rl2']
 if capture:
  return args+['--job-name=elmo-native-qknorm-swiglu-rmsnorm-full-v4','--nodelist=gpu21','--nodes=1','--ntasks=1','--cpus-per-task=2','--mem=1G','--gres=none','--time=00:10:00','--export=NONE','--dependency=afterany:'+PARENT,'--comment=elmo-full-observer:'+REQUEST_SHA+':'+PARENT,'--chdir='+OLD_ROOT,'--output='+OLD_ROOT+'/logs/%x-%j.out','--error='+OLD_ROOT+'/logs/%x-%j.err',OLD_ROOT+'/capture-native.sbatch',OLD_ROOT,PARENT,REQUEST_SHA,FILES_SHA]
 return args+['--comment=elmo-full:'+REQUEST_SHA,'--job-name='+NAME,'--nodelist=gpu21','--nodes=1','--ntasks=1','--cpus-per-task=32','--mem=192G','--gres=gpu:h100:4','--time=7-00:00:00','--export=NONE','--chdir='+OLD_ROOT,'--output='+OLD_ROOT+'/logs/%x-%j.out','--error='+OLD_ROOT+'/logs/%x-%j.err',OLD_ROOT+'/full.sbatch',OLD_ROOT,REQUEST_SHA,FILES_SHA]

def no_steps(raw,ids):
 need(isinstance(raw,str) and len(raw.encode())<=MAX_BYTES,'Incomplete step text')
 for line in raw.splitlines():
  if line=='No job steps exist in the system.':continue
  m=re.search(r'(?:^|\s)StepId=(\S+)',line);need(m is not None,'Malformed complete scontrol step observation')
  need(not any(m[1]==j or m[1].startswith(j+'.') or m[1].startswith(j+'_') for j in ids),'Owned or array-alias worker survives')

def no_allocation(api,row,*,terminal=False):
 f=api.check_native(row,job=PARENT,state='CANCELLED' if terminal else 'PENDING',request_sha=REQUEST_SHA)
 need(f.get('Reason')=='JobHeldUser' and f.get('RunTime')=='00:00:00' and f.get('AllocTRES')=='(null)' and f.get('NodeList') is None and not any(k in f for k in ['BatchHost','CPU_IDs','GRES','JOB_GRES']),'Parent had an actual resource allocation')
 need(f.get('WorkDir')==OLD_ROOT and f.get('StdOut')==OLD_ROOT+'/logs/'+NAME+'-'+PARENT+'.out' and f.get('StdErr')==OLD_ROOT+'/logs/'+NAME+'-'+PARENT+'.err','Wrong exact held output identity')
 if terminal:
  need(f.get('ExitCode')=='0:0' and f.get('NumNodes')=='1' and f.get('StartTime')==f.get('EndTime')=='2026-10-27T04:05:03','Wrong exact held cancellation outcome')
 else:need(f.get('StartTime')==f.get('EndTime')=='Unknown' and f.get('NumNodes')=='1-1','Parent was not genuinely held before start')
 return f

def query(row,argv,*,not_before=None):
 need(row.get('argv')==argv and row.get('exit_code')==0 and row.get('stderr')=='' and row.get('failure') is None and row.get('output_truncated') is False,'Genuine successful bounded native query required')
 begin,end=instant(row['start_utc']),instant(row['end_utc']);need(begin<=end<=datetime.now(timezone.utc),'Invalid native observation interval')
 if not_before is not None:need(begin>=not_before,'Native observation precedes required event')
 need(len(row['stdout'].encode())<=MAX_BYTES,'Native output oversized');return row['stdout']

def verify_held(raw,api):
 observation=parse(raw['held/result.json']);at=instant(observation['at_utc']);rows=observation['results']
 need(set(rows)=={'parent','capture','steps','no_release','prepared','all_current_claims'},'Incomplete actual held observation')
 for name,r in rows.items():
  body=raw['held/'+name+'.stdout'];need(r['exit_code']==0 and r['stderr']=='' and len(body)==r['stdout_bytes'] and sha(body)==r['stdout_sha256'],'Actual held output/hash differs')
  if 'stdout' in r:need(body.decode()==r['stdout'],'Held sidecar differs')
 held=no_allocation(api,raw['held/parent.stdout'].decode());observer=api.check_native(raw['held/capture.stdout'].decode(),job=CAPTURE,state='PENDING',capture=True,parent=PARENT,request_sha=REQUEST_SHA)
 need(observer.get('StartTime')=='Unknown' and observer.get('AllocTRES')=='(null)','Observer unexpectedly started before cancellation')
 no_steps(raw['held/steps.stdout'].decode(),[PARENT,CAPTURE]);need(raw['held/no_release.stdout']==b'','No-release query was not empty')
 need(api.native_time(held['SubmitTime'])<=at and api.native_time(observer['SubmitTime'])<=at,'Held observation predates submission')
 return held,observer,at

def verify_cancellation(raw,api,held,held_at):
 intent=parse(raw['cancel/intent.json']);result=parse(raw['cancel/result.json']);command=raw['cancel/command.sh'].decode()
 need(intent['job']==int(PARENT) and intent['capture_preserved']==int(CAPTURE) and intent['single_attempt'] is True and intent['unknown_outcome_requires_exact_native_inspection'] is True and 'not scientificretirement' in intent['reason'],'Wrong operational stop scope')
 need(command.count(SCANCEL+' '+PARENT)==1 and SCANCEL+' '+CAPTURE not in command and '/usr/local/bin/scontrol release' not in command,'Not exact guarded single parent-only stop')
 need(result['exit_code']==0 and result['stderr']=='' and result['timed_out'] is False and result['stdout'].count('EXACT_HELD_412829_CANCELLATION_ACCEPTED')==1,'Operational cancellation unknown')
 lines=result['stdout'].splitlines();parents=[line for line in lines if line.startswith('JobId='+PARENT+' ')];need(len(parents)==2,'Missing held and terminal records around stop')
 before=no_allocation(api,parents[0]);terminal=no_allocation(api,parents[1],terminal=True);api.same_attempt(held,before,terminal)
 no_steps('\n'.join(line for line in lines if line.startswith('StepId=')),[PARENT,CAPTURE])
 need(held_at<=instant(intent['at_utc'])<=api.native_time(terminal['EndTime'])<=instant(result['at_utc']),'Cancellation chronology differs')
 return terminal

def verify_capture(raw,api,terminal,pending):
 early=parse(raw['observed/native-'+PARENT+'.early.json']);need(early['schema']=='early-full-native-observation-v1' and early['status']=='RAW_UNAUTHENTICATED_NO_TERMINAL_AUTHORITY' and early['parent_job_id']==PARENT and early['observer_job_id']==CAPTURE and early['request_sha256']==REQUEST_SHA and early['files_manifest_sha256']==FILES_SHA and early['uid']==41003 and early['node']=='gpu21','Wrong raw afterany producer')
 want={'parent':['/usr/local/bin/scontrol','show','job',PARENT,'-o','-dd'],'steps':['/usr/local/bin/scontrol','show','step','-o'],'observer':['/usr/local/bin/scontrol','show','job',CAPTURE,'-o','-dd']};need(set(early['queries'])==set(want),'Incomplete actual capture queries')
 q={k:query(early['queries'][k],v,not_before=api.native_time(terminal['EndTime'])) for k,v in want.items()}
 captured=no_allocation(api,q['parent'],terminal=True);api.same_attempt(terminal,captured)
 active=api.check_native(q['observer'],job=CAPTURE,state='RUNNING',capture=True,parent=PARENT,request_sha=REQUEST_SHA);api.same_attempt(pending,active)
 need(api.native_time(active['StartTime'])>=api.native_time(terminal['EndTime']),'Observer is not a genuine afterany allocation')
 need(instant(early['queries']['parent']['end_utc'])<=instant(early['queries']['steps']['start_utc']) and instant(early['queries']['steps']['end_utc'])<=instant(early['queries']['observer']['start_utc']),'Native capture ordering differs')
 no_steps(q['steps'],[PARENT]);names={f'native-{PARENT}.txt',f'native-{PARENT}.stderr',f'native-{PARENT}.exit',f'native-{PARENT}.steps.json',f'capture-running-{CAPTURE}.txt'}
 need(set(early['canonical_sha256'])==names,'Wrong exact raw capture set')
 for name,digest in early['canonical_sha256'].items():need(sha(raw['observed/'+name])==digest,'Captured raw byte identity differs')
 need(raw['observed/native-'+PARENT+'.txt'].decode()==q['parent'] and raw['observed/capture-running-'+CAPTURE+'.txt'].decode()==q['observer'] and raw['observed/native-'+PARENT+'.stderr']==b'' and raw['observed/native-'+PARENT+'.exit']==b'0\n','Raw capture and query differ')
 steps=parse(raw['observed/native-'+PARENT+'.steps.json']);api.release_observation(steps,PARENT,REQUEST_SHA,captured);need(steps['stdout']==q['steps'],'Canonical steps differ from actual query')
 observer_row=parse(raw['final/observer.json']);observer_text=query(observer_row,['/usr/local/bin/scontrol','show','job',CAPTURE,'-o','-dd'],not_before=api.native_time(active['StartTime']));f=api.fields(observer_text)
 # The original authentication routine cannot attest a held cancellation: it
 # requires RELEASE_ATTEMPT and a started parent. Retain its actual outcome.
 need(f.get('JobState')=='FAILED' and f.get('ExitCode')=='1:0' and f.get('EndTime')=='2026-10-27T04:14:55','Exact actual observer failure differs')
 observer=api.check_native(observer_text,job=CAPTURE,state=f['JobState'],capture=True,parent=PARENT,request_sha=REQUEST_SHA);api.same_attempt(pending,active,observer)
 need(observer.get('NodeList')=='gpu21' and observer.get('StartTime')==active['StartTime'] and api.native_time(observer['EndTime'])>=api.native_time(active['StartTime']) and re.fullmatch(r'\d+:\d+',observer.get('ExitCode','')),'Wrong actual observer terminal interval')
 # Failed/timed-out terminal still requires exact actual allocated CPU resources.
 need(observer.get('NumNodes')=='1' and 'cpu=2' in observer.get('AllocTRES','').split(',') and 'mem=1G' in observer.get('AllocTRES','').split(','),'Observer terminal allocation differs')
 need(instant(observer_row['start_utc'])>=api.native_time(observer['EndTime']),'Observer observation predates its terminal')
 no_steps(query(parse(raw['final/steps.json']),['/usr/local/bin/scontrol','show','step','-o'],not_before=api.native_time(observer['EndTime'])),[PARENT,CAPTURE])
 return observer

def verify(evidence_root):
 root=Path(evidence_root);need(EVIDENCE_MANIFEST_SHA is not None and CANONICAL_SHA is not None,'Actual held-cancellation evidence remains UNBOUND')
 manifest_raw=bounded(root/'EVIDENCE.json');need(sha(manifest_raw)==EVIDENCE_MANIFEST_SHA,'Changed exact held evidence manifest');pins=parse(manifest_raw)
 actual=set()
 for p in root.rglob('*'):
  need(not p.is_symlink(),'Evidence symlink forbidden')
  if p.is_file():actual.add(str(p.relative_to(root)))
 need(actual==set(pins)|{'EVIDENCE.json'},'Unexpected/missing held proof file');raw={}
 for name,digest in pins.items():
  need(not Path(name).is_absolute() and '..' not in Path(name).parts,'Unsafe evidence path');body=bounded(root/name);need(sha(body)==digest,'Changed actual held evidence: '+name);raw[name]=body
 need(sha(raw['original/FILES.sha256'])==FILES_SHA and sha(raw['original/'+REQUEST_REL])==REQUEST_SHA,'Wrong held request/manifest')
 original=parse(raw['original/'+REQUEST_REL]);need(canonical(original)==CANONICAL_SHA,'Changed held request science')
 prior=previous.verify(root/'previous-v3');previous.verify_request(prior['original_request'],original)
 files={line.split('  ',1)[1]:line.split('  ',1)[0] for line in raw['original/FILES.sha256'].decode().splitlines()}
 for name,body in raw.items():
  if name.startswith('original/') and name!='original/FILES.sha256':need(files.get(name[9:])==sha(body),'Changed executed old file')
 native=reviewed_parser(raw['original/native_guard.py'],files['native_guard.py'],'_exact_full_v4_native_held')
 claim=lambda name:raw['observed/claims/'+REQUEST_SHA+'/'+name]
 # Exact named readback; no release/admission marker is permitted in this attempt.
 names={Path(p).name for p in raw if p.startswith('observed/claims/'+REQUEST_SHA+'/')};need(names==set(HELD_CLAIM_NAMES),'Held claim gained release or unknown acceptance')
 for label,jid in [('parent',PARENT),('capture',CAPTURE)]:
  need(claim(label.upper()+'_JOB_ID')==(jid+'\n').encode() and claim(label+'-sbatch.stdout')==(jid+'\n').encode() and claim(label+'-sbatch.stderr')==b'' and claim(label+'-sbatch.exit')==b'0\n','Ambiguous accepted held identity')
  need(parse(claim(label+'-sbatch.argv.json'))==expected_argv(label=='capture'),'Changed accepted argv')
 prepared=parse(claim('PREPARED.json'));need(prepared['request_sha256']==REQUEST_SHA and prepared['manifest_sha256']==FILES_SHA and prepared['parent_argv']==expected_argv(),'Wrong held preparation')
 need(claim('PREPARED.json')==raw['held/prepared.stdout'],'Prepared actual readback differs')
 need(raw['observed/request-claims/'+NAME+'/REQUEST_SHA256']==(REQUEST_SHA+'\n').encode(),'Wrong held request marker')
 held,pending,at=verify_held(raw,native);original_held=no_allocation(native,claim('HELD_NATIVE.txt').decode());native.same_attempt(original_held,held)
 original_pending=native.check_native(claim('CAPTURE_PENDING_NATIVE.txt').decode(),job=CAPTURE,state='PENDING',capture=True,parent=PARENT,request_sha=REQUEST_SHA);native.same_attempt(original_pending,pending)
 terminal=verify_cancellation(raw,native,held,at);observer=verify_capture(raw,native,terminal,pending);verify_final_metadata(raw,observer)
 # Exact successful readback, then individually framed later no-release census.
 verify_readback(raw);verify_stable(raw)
 need(raw['observed/FILES.sha256']==raw['original/FILES.sha256'] and raw['observed/'+REQUEST_REL]==raw['original/'+REQUEST_REL],'Deployed request/manifest differs')
 later=parse(raw['poststop/result.json']);rows=later['results'];after=native.native_time(terminal['EndTime'])
 claim_list=root_observed(rows['claim_files'],not_before=after).splitlines();need(len(claim_list)==len(HELD_CLAIM_NAMES) and set(claim_list)==set(HELD_CLAIM_NAMES),'Current held claim inventory differs')
 need(root_observed(rows['unused_maintained_roots'],not_before=after)=='' and root_observed(rows['no_release'],not_before=after)=='','Release or maintained admission appeared')
 error=parse(raw['controller/submission-result.json']);need(error['controller_exit']==1 and error['status']=='STOPPED_INSPECT_EXACT_CLAIM_NATIVE_BEFORE_ANY_RETRY' and error['error']=="RuntimeError('Remote inspection failed: /bin/bash: Argument list too long\\n')" and 'in prerelease' in error['traceback'] and 'prior-small-file-readback' in error['traceback'],'Different controller failure location')
 return dict(schema='elmo-exact-held-cancel-retry-v1',status='EXACT_THREE_PRELAUNCH_FAILURES_AND_NEVER_STARTED_HELD_CANCELLATION_VERIFIED',failed_parent=PARENT,capture=CAPTURE,original_request=original,original_request_sha256=REQUEST_SHA,original_manifest_sha256=FILES_SHA,optimizer_updates_possible_by_reviewed_control_flow=0,maintained_training_launcher_reached=False,parent_native_state='CANCELLED',parent_native_exit_code=terminal['ExitCode'],parent_native_end_time=terminal['EndTime'],terminal_start_is_scheduler_cancellation_bookkeeping=True,genuine_parent_terminal_proof=True,observer_own_terminal=observer['JobState']+'_'+observer['ExitCode'],observer_end_time=observer['EndTime'],original_observer_authenticated_commit=False,held_native_and_raw_afterany_queries_independently_authenticated=True,earlier_412825_parent_terminal='UNKNOWN_NOT_INFERRED',checkpoint_recovery_authorized=False,live_launch_admitted=False)

def verify_request(original,current):
 need(CANONICAL_SHA is not None and canonical(original)==CANONICAL_SHA,'Wrong verified full-v4 request')
 def move(v):
  if isinstance(v,dict):return {k:move(x) for k,x in v.items()}
  if isinstance(v,list):return [move(x) for x in v]
  return v.replace(OLD_ROOT,NEW_ROOT) if isinstance(v,str) else v
 expected=move(copy.deepcopy(original));expected['job_name']=NEW_NAME;need(original['local_logs'].endswith('-attempt-v4'),'Unexpected prior CSV')
 expected['local_logs']=original['local_logs'][:-len('-attempt-v4')]+'-attempt-v5';expected['overrides']=[s.replace('logger.csv.save_dir='+original['local_logs'],'logger.csv.save_dir='+expected['local_logs']) for s in expected['overrides']]
 need(current==expected and current['output']==original['output'] and current['continuation'] is None and not current.get('interruption') and 'logger.wandb.resume=never' in current['overrides'],'Changed exact fresh0 full-v5 science')
 return dict(status='EXACT_FRESH0_FULL_V5_REQUEST_COMPATIBLE',live_launch_admitted=False)

def prior_claim_names(root,attempt):
 return list(HELD_CLAIM_NAMES) if attempt['root']==OLD_ROOT else previous.prior_claim_names(Path(root)/'previous-v3',attempt)

def expected_live_file_hashes(root):
 verify(root);expected=previous.expected_live_file_hashes(Path(root)/'previous-v3')
 for name,digest in parse(bounded(Path(root)/'EVIDENCE.json')).items():
  if name.startswith('observed/claims/') or name in {'observed/FILES.sha256','observed/'+REQUEST_REL,'observed/request-claims/'+NAME+'/REQUEST_SHA256'}:expected[OLD_ROOT+'/'+name[9:]]=digest
 return expected

def verify_live_file_hashes(root,observed_hashes):need(observed_hashes==expected_live_file_hashes(root),'Prior current files changed or missing')

def verify_claim_entries(full_roots,checked_roots,entries,*,current_request_sha=None):
 roots={a['root'] for a in PRIOR_ATTEMPTS}|{NEW_ROOT};need(len(full_roots)==len(roots) and set(full_roots)==roots,'Unknown/missing full attempt root')
 need(len(checked_roots)==len(CLAIM_ROOTS) and set(checked_roots)==set(CLAIM_ROOTS),'Incomplete all-five claim roots')
 expected={p for a in PRIOR_ATTEMPTS for p in (a['root']+'/claims/'+a['request_sha'],a['root']+'/request-claims/'+a['name'])}
 if current_request_sha is not None:
  need(re.fullmatch('[0-9a-f]{64}',current_request_sha),'Invalid current request SHA');expected|={NEW_ROOT+'/claims/'+current_request_sha,NEW_ROOT+'/request-claims/'+NEW_NAME}
 need(len(entries)==len(expected) and set(entries)==expected,'Unknown/partial attempt claims');return sorted(expected)

def verify_queue_steps(queue,steps,*,current_job=None,current_capture=None):
 need(isinstance(queue,str) and isinstance(steps,str) and max(len(queue.encode()),len(steps.encode()))<=MAX_BYTES,'Oversized/incomplete queue/steps')
 prior={a[k] for a in PRIOR_ATTEMPTS for k in ('parent','capture')};allowed=set()
 if current_job is not None or current_capture is not None:
  need(isinstance(current_job,str) and isinstance(current_capture,str) and current_job.isdecimal() and current_capture.isdecimal() and len(prior|{current_job,current_capture})==len(prior)+2,'Wrong current native IDs');allowed={current_job,current_capture}
 for line in queue.splitlines():
  row=line.split('|');need(len(row)==5 and row[2]=='paphiwetsa3','Malformed complete queue');job,name,_,state,_=row
  need(not any(job==old or job.startswith(old+'.') or job.startswith(old+'_') for old in prior),'Prior queued identity remains')
  if 'qknorm-swiglu-rmsnorm' in name or job in allowed:need(job in allowed and name==(NEW_NAME if job==current_job else 'elmo-native-qknorm-swiglu-rmsnorm-full-v5') and state in {'PENDING','RUNNING'},'Another or wrongly excluded writer')
 no_steps(steps,prior);return dict(prior_jobs_absent_from_supplied_complete_native_outputs=True,live_launch_admitted=False)
