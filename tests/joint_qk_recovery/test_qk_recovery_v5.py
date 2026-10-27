"""Local metadata/adversarial fixtures, never native/GPU/checkpoint proof."""
import ast,copy,hashlib,importlib,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'files/scripts/train/joint/qk_recovery_profile.py').is_file() or (p/'scripts/train/joint/qk_recovery_profile.py').is_file())
SERVICE=ROOT/'files' if (ROOT/'files/scripts/train/joint/qk_recovery_profile.py').is_file() else ROOT
FIXTURES=Path(__file__).resolve().parent/'fixtures'
sys.path[:0]=[str(SERVICE),str(SERVICE/'scripts/train/joint/native_elmo_ops')]
from scripts.train.joint import qk_recovery_profile as p,qk_original_observer as o,qk_recovery_usage as u
from scripts.train.joint import original_observer,phase_observer_proof,interruption_admission,production_recovery
import execution_spec,wrapper_common,controller
ORIGINAL=FIXTURES/'original-full-v5';Q=json.loads((ORIGINAL/'ops/requests/elmo-large314m-qknorm-swiglu-rmsnorm-full-v5.json').read_text())

def encoded(x):return json.dumps(x,sort_keys=True)+'\n'
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def native(state,capture=False):
 f=o.expected_native(capture);f.update(JobState=state,NumNodes='1',NumTasks='1',WorkDir=p.ROOT,
  SubmitTime='2026-10-27T00:00:00',StartTime='2026-10-27T00:10:00' if capture else '2026-10-27T00:01:00',
  EndTime='2026-10-27T00:09:00',ExitCode='1:0' if state=='FAILED' else '0:0',
  Comment=('elmo-full-observer:'+p.REQUEST_SHA+':'+p.PARENT_ID if capture else 'elmo-full:'+p.REQUEST_SHA))
 f['StdOut']=p.ROOT+'/logs/'+f['JobName']+'-'+f['JobId']+'.out';f['StdErr']=f['StdOut'][:-3]+'err'
 f['ReqTRES']='cpu='+f['NumCPUs']+',mem='+f['MinMemoryNode']+',node=1'
 f['AllocTRES']=f['ReqTRES'];f['JOB_GRES']='(null)' if capture else 'gpu:h100:4'
 if not capture and state=='RUNNING':f['GRES']='gpu:h100:4(IDX:0-3)'
 if state=='PENDING':
  f.update(StartTime='Unknown',EndTime='Unknown',AllocTRES='(null)',NodeList='(null)',NumNodes='1-1')
  if capture:f['Dependency']='afterany:'+p.PARENT_ID+'(unfulfilled)'
  else:f['Reason']='JobHeldUser'
 return ' '.join(k+'='+v for k,v in f.items())+'\n'

def fixture():
 raw={key:(ORIGINAL/rel).read_text() for key,(rel,_) in o.CODE_PINS.items()}
 raw.update(files_manifest=(ORIGINAL/'FILES.sha256').read_text(),request_claim=p.REQUEST_SHA+'\n',
  parent_id=p.PARENT_ID+'\n',observer_id=p.OBSERVER_ID+'\n',held_native=native('PENDING'),release_held_native=native('PENDING'),
  pending_native=native('PENDING',True),release_pending_native=native('PENDING',True),observer_running=native('RUNNING',True),
  terminal_native=native('FAILED'),query_exit='0\n',query_stderr='',
  prepared=encoded(dict(request_sha256=p.REQUEST_SHA,manifest_sha256=p.FILES_SHA,parent_argv=o.accepted_argv())),
  release_attempt=encoded(dict(parent=p.PARENT_ID,capture=p.OBSERVER_ID,request_sha256=p.REQUEST_SHA)),
  running_admission=encoded(dict(status='ACTUAL_BOUND_FULL_NATIVE_ALLOCATION_PASS',job_id=p.PARENT_ID,request_sha256=p.REQUEST_SHA,native=native('RUNNING'))),
  steps=encoded(dict(schema='native-owned-step-absence-v1',parent_job_id=p.PARENT_ID,request_sha256=p.REQUEST_SHA,uid=41003,host='lambda',
   argv=['/usr/local/bin/scontrol','show','step','-o'],exit_code=0,stderr='',stdout='',observed_at_utc='2026-10-27T00:11:00+00:00')))
 for label,capture,job in [('parent_sbatch',False,p.PARENT_ID),('capture_sbatch',True,p.OBSERVER_ID)]:
  raw.update({label+'_argv':encoded(o.accepted_argv(capture)),label+'_stdout':job+'\n',label+'_stderr':'',label+'_exit':'0\n'})
 canonical={f'native-{p.PARENT_ID}.txt':raw['terminal_native'],f'native-{p.PARENT_ID}.stderr':raw['query_stderr'],f'native-{p.PARENT_ID}.exit':raw['query_exit'],f'native-{p.PARENT_ID}.steps.json':raw['steps'],f'capture-running-{p.OBSERVER_ID}.txt':raw['observer_running']}
 queries={}
 for key,argv,value,start,end in [
  ('parent',['/usr/local/bin/scontrol','show','job',p.PARENT_ID,'-o','-dd'],raw['terminal_native'],'2026-10-27T00:10:01+00:00','2026-10-27T00:10:02+00:00'),
  ('steps',['/usr/local/bin/scontrol','show','step','-o'],'','2026-10-27T00:10:03+00:00','2026-10-27T00:11:00+00:00'),
  ('observer',['/usr/local/bin/scontrol','show','job',p.OBSERVER_ID,'-o','-dd'],raw['observer_running'],'2026-10-27T00:11:01+00:00','2026-10-27T00:11:02+00:00')]:
  queries[key]=dict(argv=argv,start_utc=start,end_utc=end,stdout=value,stderr='',exit_code=0,failure=None,output_truncated=False)
 hashes={name:digest(value) for name,value in canonical.items()}
 early=dict(schema='early-full-native-observation-v1',status='RAW_UNAUTHENTICATED_NO_TERMINAL_AUTHORITY',parent_job_id=p.PARENT_ID,observer_job_id=p.OBSERVER_ID,request_sha256=p.REQUEST_SHA,files_manifest_sha256=p.FILES_SHA,uid=41003,node='gpu21',queries=queries,canonical_sha256=hashes)
 raw['early_observation']=encoded(early)
 raw['authenticated_commit']=encoded(dict(schema='authenticated-early-full-native-observation-v1',status='AUTHENTICATED_TERMINAL_EVIDENCE_ONLY',parent_job_id=p.PARENT_ID,observer_job_id=p.OBSERVER_ID,parent_state='FAILED',parent_exit_code='1:0',parent_start='2026-10-27T00:01:00',parent_end='2026-10-27T00:09:00',request_sha256=p.REQUEST_SHA,files_manifest_sha256=p.FILES_SHA,canonical_sha256=hashes,zero_work_proven=False,production_completion_proven=False,observer_own_terminal='UNKNOWN_NOT_INFERRED',early_observation_sha256=digest(raw['early_observation'])))
 return raw

def binding_fixture(raw):
 paths=o.proof_paths();proof={'schema':o.SCHEMA,**{k:dict(path=str(paths[k]),sha256=digest(v)) for k,v in raw.items()}}
 q=dict(parent_request=dict(path=p.REQUEST_PATH,sha256=p.REQUEST_SHA),parent_job_id=p.PARENT_ID,parent_native=o.expected_native(),
  parent_running_admission=proof['running_admission'],original_observer=proof)
 bypath={str(paths[k]):v for k,v in raw.items()}
 def reader(b,parsed=True):
  if b==q['parent_request']:return copy.deepcopy(Q) if parsed else (ORIGINAL/'ops/requests/elmo-large314m-qknorm-swiglu-rmsnorm-full-v5.json').read_text()
  s=bypath[b['path']];assert digest(s)==b['sha256'];return json.loads(s) if parsed else s
 return q,reader

class Profile(unittest.TestCase):
 def test_exact_actual_parent(self):self.assertEqual(p.validate_parent(Q),p.PROFILE)
 def test_science_source_runtime_and_service_changes_reject(self):
  for key,value in [('source_revision','0'*40),('parameters',314187064),('run_id','imagenet'),('target_steps',600000),('lambda_runtime',{}),('operational_service',{})]:
   q=copy.deepcopy(Q);q[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):p.validate_parent(q)
 def test_architecture_override_removed_rejects(self):
  q=copy.deepcopy(Q);q['overrides']=[v for v in q['overrides'] if 'qk_norm' not in v]
  with self.assertRaises(ValueError):p.validate_parent(q)
 def test_old_three_profiles_equal_base(self):
  base=FIXTURES/'original_observer_before_qk.py'
  old=next(ast.literal_eval(n.value) for n in ast.parse(base.read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ORIGINAL_OBSERVERS' for t in n.targets))
  self.assertEqual({k:v for k,v in original_observer.ORIGINAL_OBSERVERS.items() if k!=p.RUN_ID},old)
 def test_ops_profile_is_byte_identical(self):
  self.assertEqual((SERVICE/'scripts/train/joint/qk_recovery_profile.py').read_bytes(),(SERVICE/'scripts/train/joint/native_elmo_ops/qk_recovery_profile.py').read_bytes())
 def test_old_pins_preserved_qk_actual_published_service(self):
  for module in (execution_spec,wrapper_common):
   self.assertEqual(module.service_pins({'arm':'plain'}),('abbb00499096faabf7035fde23bf4b1eba984377','adf45524de3c2e6dba186572e7151d48bfca0b3adc8e6cf9defd686b47744c87',35))
   self.assertEqual(module.service_pins({'arm':'qk'}),('5fad76e080d04c82346e3363041edafad815dcb5','3350d6279e60acd18e19c00f4625784468f398fdf0f2fce2b1438e2bd30284ef',54))
 def test_new_attempt_keeps_science_and_total(self):
  q=copy.deepcopy(Q);q.update(output=Q['output']+'-resume',local_logs=Q['local_logs']+'-resume',job_name='new-qk-full',operational_service={'new':'unbound'})
  interruption_admission.compatible(Q,q)
  q['parameters']=314187064
  with self.assertRaises(ValueError):interruption_admission.compatible(Q,q)
 def test_only_positive_scheduled_production_steps(self):
  for step in (0,-1,50,19999,300000):
   with self.subTest(step=step),self.assertRaises(ValueError):production_recovery.phases(step,300000)
  self.assertEqual(production_recovery.phases(20000,300000),((20000,20002),(20002,20006)))
 def test_no_old_prelaunch_parent(self):
  q,r=binding_fixture(fixture());q['parent_job_id']='412823'
  with self.assertRaises(ValueError):o.validate(q,r)

class OriginalObserver(unittest.TestCase):
 def test_complete_synthetic_failure_protocol(self):
  q,r=binding_fixture(fixture());raw,proof=original_observer.validate_original(q,r)
  self.assertIn('JobState=FAILED',raw);self.assertEqual(proof['original_parent_id'],'412831');self.assertEqual(proof['observer_own_terminal'],'UNKNOWN_NOT_REQUIRED')
 def veto(self,raw):
  q,r=binding_fixture(raw)
  with self.assertRaises(ValueError):o.validate(q,r)
 def test_rehashed_submit_time_mismatch(self):
  x=fixture();x['terminal_native']=x['terminal_native'].replace('SubmitTime=2026-10-27T00:00:00','SubmitTime=2026-10-26T23:59:00');self.veto(x)
 def test_rehashed_start_time_mismatch(self):
  x=fixture();x['terminal_native']=x['terminal_native'].replace('StartTime=2026-10-27T00:01:00','StartTime=2026-10-27T00:02:00');self.veto(x)
 def test_held_release_attempt_mismatch(self):
  x=fixture();x['release_held_native']=x['release_held_native'].replace('SubmitTime=2026-10-27T00:00:00','SubmitTime=2026-10-26T23:59:00');self.veto(x)
 def test_observer_starts_before_parent_end(self):
  x=fixture();x['observer_running']=x['observer_running'].replace('StartTime=2026-10-27T00:10:00','StartTime=2026-10-27T00:08:00');self.veto(x)
 def test_observer_gpu_rejects(self):
  x=fixture();x['observer_running']=x['observer_running'].replace('JOB_GRES=(null)','JOB_GRES=gpu:h100:1');self.veto(x)
 def test_held_conflicting_gpu_resource_rejects(self):
  for extra in ['JOB_GRES=gpu:h100:2','TresPerTask=gres/gpu:1','GPUsPerTask=1']:
   x=fixture();x['held_native']=x['held_native'].replace('JOB_GRES=gpu:h100:4','')+' '+extra;self.veto(x)
 def test_native_walltime_rejects(self):
  x=fixture();x['terminal_native']=x['terminal_native'].replace('TimeLimit=7-00:00:00','TimeLimit=8-00:00:00');self.veto(x)
 def test_unknown_acceptance_rejects(self):
  x=fixture();x['parent_sbatch_exit']='255\n';self.veto(x)
 def test_forged_argv_rejects(self):
  x=fixture();a=json.loads(x['parent_sbatch_argv']);a[a.index('--no-requeue')]='--requeue';x['parent_sbatch_argv']=encoded(a);self.veto(x)
 def test_same_request_capture_for_different_parent_rejects(self):
  x=fixture();x['release_pending_native']=x['release_pending_native'].replace('afterany:412831','afterany:412823');self.veto(x)
 def test_live_worker_plain_dot_array_rejects(self):
  for step in ['412831','412831.batch','412831_7.batch']:
   x=fixture();s=json.loads(x['steps']);s['stdout']='StepId='+step+' State=RUNNING\n';x['steps']=encoded(s)
   with self.subTest(step=step):self.veto(x)
 def test_complete_steps_error_and_early_time_reject(self):
  for key,value in [('exit_code',1),('argv',['/usr/local/bin/scontrol','show','step','412831']),('observed_at_utc','2026-10-27T00:08:00+00:00'),('stdout','unparseable')]:
   x=fixture();s=json.loads(x['steps']);s[key]=value;x['steps']=encoded(s)
   with self.subTest(key=key):self.veto(x)
 def test_missing_stronger_field_rejects(self):
  q,r=binding_fixture(fixture());del q['original_observer']['observer_running']
  with self.assertRaises(ValueError):original_observer.validate_original(q,r)
 def test_claim_consumer_uses_shared_qk_builder(self):
  raw=fixture();q,reader=binding_fixture(raw);paths=o.proof_paths();by={str(v):raw[k] for k,v in paths.items()}
  def read(path):return json.loads(by[str(path)])
  def text(path):return by[str(path)]
  def bind(path):return dict(path=str(path),sha256=digest(text(path)))
  # Exercise real claim consumer, real original route and real QK validation.
  def parsed_bound(b):return b['path']
  old_read=read
  def fullread(path):return Q if str(path)==p.REQUEST_PATH else old_read(path)
  with patch.object(phase_observer_proof,'physical',side_effect=Path),patch.object(phase_observer_proof,'read_json',side_effect=fullread),patch.object(phase_observer_proof,'text',side_effect=text),patch.object(phase_observer_proof,'binding',side_effect=bind),patch.object(phase_observer_proof,'bound',side_effect=parsed_bound),patch.object(phase_observer_proof,'sha',side_effect=lambda path:digest(text(path))):
   result=phase_observer_proof.verify_claim_terminal(Path(p.ROOT)/'claims'/p.REQUEST_SHA,Path(p.ROOT),p.RUN_ID)
  self.assertEqual(result['terminal_state'],'FAILED')

class Coverage(unittest.TestCase):
 def row(self,start,robot,human):
  names=u.expected_all_parameters();initial={n:start for n in names};final={n:start+(robot+human if s=='shared' else {'eva_bimanual':robot,'human_bimanual':human}[s]) for n,s in names.items()}
  return dict(start_step=start,global_step=start+robot+human,initial_parameter_usage=initial,
   conditional_optimizer_record={'parameter_update_counts':final,'parameter_names':list(names)},embodiment_optimizer_batches={'eva_bimanual':robot,'human_bimanual':human},
   qk_private_state={'schema':'qk-private-state-preservation-v2','parameter_names':u.expected_private_names(),'before':{e:'a'*64 for e in u.EMBODIMENTS},'after':{e:'a'*64 for e in u.EMBODIMENTS}})
 def test_exact_inventory_matches_real_meta_smoke(self):
  x=json.loads((FIXTURES/'architecture-parameter-usage.json').read_text());self.assertEqual(u.expected_parameters(),x['parameters'])
 def test_shared_and_actual_private_updates(self):u.verify_row(self.row(20000,1,1))
 def test_unvisited_human_bank_allowed_unchanged(self):u.verify_row(self.row(20000,2,0))
 def test_unvisited_robot_and_action_allowed_unchanged(self):u.verify_row(self.row(20002,0,4))
 def test_missing_active_new_gradient_rejects(self):
  x=self.row(20000,1,1);name=next(iter(u.expected_parameters()));x['conditional_optimizer_record']['parameter_update_counts'][name]-=1
  with self.assertRaises(ValueError):u.verify_row(x)
 def test_inactive_counter_change_rejects(self):
  x=self.row(20000,2,0);name=next(n for n,s in u.expected_parameters().items() if s=='human_bimanual');x['conditional_optimizer_record']['parameter_update_counts'][name]+=1
  with self.assertRaises(ValueError):u.verify_row(x)
 def test_inactive_weight_moment_change_rejects(self):
  x=self.row(20000,2,0);x['qk_private_state']['after']['human_bimanual']='b'*64
  with self.assertRaises(ValueError):u.verify_row(x)
 def test_all_four_ranks_two_processes(self):
  phases=[[self.row(20000,2,0) for _ in range(4)],[self.row(20002,4,0) for _ in range(4)]];u.verify_phases(phases)
  phases[1][3]['qk_private_state']['before']['eva_bimanual']='b'*64
  with self.assertRaises(ValueError):u.verify_phases(phases)
 def test_private_names_cover_action_and_camera_adapters(self):
  self.assertEqual(u.private_scope(u.PREFIX+'action.blocks.0.q_norm.weight'),'eva_bimanual')
  self.assertEqual(u.private_scope(u.PREFIX+'camera.inputs.human_bimanual.weight'),'human_bimanual')
  self.assertIsNone(u.private_scope(u.PREFIX+'camera.blocks.0.q_norm.weight'))

class FullV5Hardening(unittest.TestCase):
 def test_no_raw_only_terminal_authority(self):
  x=fixture();x['authenticated_commit']=encoded({})
  with self.assertRaises(ValueError):o.validate(*binding_fixture(x))
 def test_commit_cannot_change_zero_work_authority(self):
  x=fixture();v=json.loads(x['authenticated_commit']);v['zero_work_proven']=True;x['authenticated_commit']=encoded(v)
  with self.assertRaises(ValueError):o.validate(*binding_fixture(x))
 def test_rehashed_early_parent_mismatch_rejects(self):
  x=fixture();v=json.loads(x['early_observation']);v['queries']['parent']['stdout']=v['queries']['parent']['stdout'].replace('ExitCode=1:0','ExitCode=0:0');x['early_observation']=encoded(v)
  with self.assertRaises(ValueError):o.validate(*binding_fixture(x))
 def test_missing_scientific_running_still_rejects(self):
  x=fixture();x['running_admission']=encoded({})
  with self.assertRaises((ValueError,KeyError)):o.validate(*binding_fixture(x))
 def test_full558_counter_and_private_language_scope(self):
  self.assertEqual(len(u.expected_all_parameters()),558)
  for suffix,scope in [('action.0.weight','eva_bimanual'),('proprio__eva_bimanual.2.bias','eva_bimanual'),('proprio__human_bimanual.0.bias','human_bimanual')]:
   self.assertEqual(u.private_scope(u.PREFIX+'language_fusions.'+suffix),scope)
  self.assertIsNone(u.private_scope(u.PREFIX+'language_fusions.camera.0.weight'))
 def test_inactive_language_counter_change_rejects(self):
  x=Coverage().row(20000,2,0);name=u.PREFIX+'language_fusions.proprio__human_bimanual.0.weight';x['conditional_optimizer_record']['parameter_update_counts'][name]+=1
  with self.assertRaises(ValueError):u.verify_row(x)
 def test_missing_private_language_name_rejects(self):
  x=Coverage().row(20000,2,0);x['qk_private_state']['parameter_names']['human_bimanual'].remove(u.PREFIX+'language_fusions.proprio__human_bimanual.0.weight')
  with self.assertRaises(ValueError):u.verify_row(x)
 def test_early_reader_missing_from_old_protocol_rejects(self):
  q,r=binding_fixture(fixture());del q['original_observer']['early_observation']
  with self.assertRaises(ValueError):o.validate(q,r)

class ReadBoundsAndScope(unittest.TestCase):
 def test_pre_read_bound_runs_before_hash(self):
  import tempfile
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder).resolve()/'oversized.json'
   with path.open('wb') as stream:stream.truncate((4<<20)+1)
   with patch.object(original_observer,'bound',side_effect=AssertionError('Must not hash oversizedmetadata')):
    with self.assertRaisesRegex(ValueError,'pre-read bound'):original_observer.read_binding({'path':str(path),'sha256':'a'*64})
 def test_unsupported_terminal_states_stay_closed(self):
  for state in ['BOOT_FAIL','DEADLINE','REVOKED']:
   with self.subTest(state=state),self.assertRaises(ValueError):o.native(native('FAILED').replace('JobState=FAILED','JobState='+state),state)
 def test_duplicate_parameter_name_rejects(self):
  x=Coverage().row(20000,2,0);x['conditional_optimizer_record']['parameter_names'].append(x['conditional_optimizer_record']['parameter_names'][0])
  with self.assertRaises(ValueError):u.verify_row(x)

if __name__=='__main__':unittest.main()
