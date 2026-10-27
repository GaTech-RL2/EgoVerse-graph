"""Historical bytes + synthetic fresh observations; never actual node proof."""
import base64
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = next(p for p in Path(__file__).resolve().parents
    if (p/'files/scripts/train/joint/qk_prelaunch_history.py').is_file()
    or (p/'scripts/train/joint/qk_prelaunch_history.py').is_file())
SERVICE = ROOT/'files' if (ROOT/'files').is_dir() else ROOT
sys.path.insert(0, str(SERVICE))
from scripts.train.joint import qk_prelaunch_history as h, original_observer as observer
from scripts.train.joint import phase_observer_proof as terminal
F = Path(__file__).resolve().parent/'fixtures'


def record(argv, stdout='', *, returncode=0, stderr='', at=None):
    return dict(schema='native-elmo-terminal-query-v1', argv=argv, returncode=returncode,
        stdout=stdout, stderr=stderr, observed_at_utc=at or datetime.now(timezone.utc).isoformat())


class CensusFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=json.loads((F/'prelaunch-census.json').read_text())
        cls.original=F/'original-full-v5'
        cls.parent_raw=(cls.original/'ops/requests/elmo-large314m-qknorm-swiglu-rmsnorm-full-v5.json').read_bytes()
        cls.files_raw=(cls.original/'FILES.sha256').read_bytes()

    def setUp(self):
        self.raw={p:base64.b64decode(v) for p,v in self.fixture['files'].items()}
        self.raw[h.REQUEST_PATH]=self.parent_raw
        self.raw[h.ROOT+'/FILES.sha256']=self.files_raw
        self.dirs={}
        self.dirs[h.ROOT+'/claims']=[Path(h.ROOT)/'claims'/h.REQUEST_SHA]
        current_marker=h.ROOT+'/request-claims/'+json.loads(self.parent_raw)['job_name']
        self.dirs[h.ROOT+'/request-claims']=[Path(current_marker)]
        self.dirs[current_marker]=[Path(current_marker)/'REQUEST_SHA256']
        self.raw[current_marker+'/REQUEST_SHA256']=(h.REQUEST_SHA+'\n').encode()
        for a in h.OLD:
            base=a['root']; claim=base+'/claims/'+a['request_sha']; marker=base+'/request-claims/'+a['name']
            self.dirs[base+'/claims']=[Path(claim)]
            self.dirs[base+'/request-claims']=[Path(marker)]
            self.dirs[claim]=[Path(claim)/name for name in self.fixture['prior_claim_names'][base]]
            self.dirs[marker]=[Path(marker)/'REQUEST_SHA256']
            for suffix in ('ops/claims','ops/request-claims','ops/status'):self.dirs[base+'/'+suffix]=[]
        self.scope=dict(user='paphiwetsa3',qk_prelaunch_history=h.expected_binding(),
            claim_roots=[dict(path=a['root']+'/claims',terminal_directory=a['root']) for a in h.OLD]
                +[dict(path=h.ROOT+'/claims',terminal_directory=h.ROOT)],
            scientific_job_names=['new-scientific'],scientific_name_prefixes=['elmo-large314m-qknorm-'])
        self.reads=[];self.absences=[];self.queue='';self.steps='';self.query_error=None
        self.extra_roots=[];self.csv_node=None

    def execute(self, *, require_node=False):
        def read(path,**kw):
            p=str(path);self.reads.append(p)
            return self.raw[p]
        def absent(path):
            self.absences.append(str(path))
            if str(path) in getattr(self,'existing',set()):raise ValueError('Prior namespace appeared')
        def query(argv):
            if self.query_error:return record(argv,**self.query_error)
            return record(argv,self.steps if '--steps' in argv else self.queue)
        with mock.patch.object(h,'read_metadata',side_effect=read), \
             mock.patch.object(h,'verify_history',return_value=copy.deepcopy(self.fixture['verified'])), \
             mock.patch.object(h,'children',side_effect=lambda p:self.dirs[str(p)]), \
             mock.patch.object(h,'full_attempt_roots',return_value=[a['root'] for a in h.OLD]+[h.ROOT]+self.extra_roots), \
             mock.patch.object(h,'absent',side_effect=absent), \
             mock.patch.object(h,'node_local_context',side_effect=lambda query,required:self.node(required)), \
             mock.patch.object(h.proof,'prior_claim_names',side_effect=lambda root,a:self.fixture['prior_claim_names'][a['root']]), \
             mock.patch.object(h.proof,'expected_live_file_hashes',return_value=self.fixture['expected_hashes']), \
             mock.patch.object(observer,'query',side_effect=query):
            return h.census(self.scope,h.RUN_ID,require_node=require_node)

    def node(self,required):
        if required and self.csv_node is None:raise ValueError('Allocated node required')
        return self.csv_node

    def test_complete_real_78_file_shape_rehashed_and_unknown_honest(self):
        value=self.execute()
        self.assertEqual(len(value['prior_file_hashes']),78)
        self.assertEqual(set(value['prior_file_hashes']),set(self.fixture['files']))
        old2=next(a for a in h.OLD if a['parent']=='412825')
        p=value['claims'][old2['root']+'/claims/'+old2['request_sha']]
        self.assertEqual(p['parent_native_state'],'UNKNOWN');self.assertIsNone(p['parent_native_exit_code'])
        self.assertFalse(p['genuine_parent_terminal_proof']);self.assertFalse(p['phase_completion_proven'])
        self.assertEqual(value['old_csv']['status'],'DEFERRED_TO_ALLOCATED_ENTRY')
        self.assertFalse(value['current_scientific_output_absence_required'])

    def test_current_full_v5_data_is_never_absence_checked(self):
        self.csv_node={'fixture':True};value=self.execute(require_node=True)
        parent=json.loads(self.parent_raw)
        self.assertNotIn(parent['output'],self.absences);self.assertNotIn(parent['local_logs'],self.absences)
        self.assertTrue(all(not p.startswith(h.ROOT+'/') for p in self.absences))
        self.assertEqual(len(value['old_csv']['paths']),4)
        self.assertEqual(value['old_csv']['status'],'ACTUAL_GPU21_ABSENT')

    def test_missing_binding_rejected(self):
        del self.scope['qk_prelaunch_history']
        with self.assertRaisesRegex(ValueError,'binding'):self.execute()

    def test_relocated_binding_rejected(self):
        self.scope['qk_prelaunch_history']['evidence']['path']='/tmp/foreign'
        with self.assertRaisesRegex(ValueError,'binding'):self.execute()

    def test_omitted_old_root_rejected(self):
        self.scope['claim_roots'].pop(0)
        with self.assertRaisesRegex(ValueError,'omitted'):self.execute()

    def test_missing_current_original_root_rejected(self):
        self.scope['claim_roots'].pop()
        with self.assertRaisesRegex(ValueError,'omitted'):self.execute()

    def test_duplicate_root_rejected(self):
        self.scope['claim_roots'].append(copy.deepcopy(self.scope['claim_roots'][0]))
        with self.assertRaisesRegex(ValueError,'omitted'):self.execute()

    def test_unknown_full_attempt_rejected(self):
        self.extra_roots=[h.TASK_ROOT+'/full-v6']
        with self.assertRaisesRegex(ValueError,'Unknown'):self.execute()

    def test_unknown_claim_rejected(self):
        self.dirs[h.OLD[1]['root']+'/claims'].append(Path(h.OLD[1]['root']+'/claims/'+'f'*64))
        with self.assertRaisesRegex(ValueError,'Unknown prior'):self.execute()

    def test_new_prior_running_admission_rejected(self):
        a=h.OLD[-1];claim=a['root']+'/claims/'+a['request_sha']
        self.dirs[claim].append(Path(claim)/'RUNNING_ADMISSION.json')
        with self.assertRaisesRegex(ValueError,'inventory'):self.execute()

    def test_new_prior_maintained_status_rejected(self):
        a=h.OLD[0];self.dirs[a['root']+'/ops/status']=[Path(a['root'])/'ops/status/full.json']
        with self.assertRaisesRegex(ValueError,'maintained'):self.execute()

    def test_new_prior_attempt_rejected(self):
        a=h.OLD[0];self.existing={a['root']+'/ops/requests/'+a['name']+'.attempt.json'}
        with self.assertRaisesRegex(ValueError,'namespace'):self.execute()

    def test_changed_historical_hash_rejected(self):
        p=next(iter(self.fixture['expected_hashes']));self.raw[p]+=b'\n'
        with self.assertRaisesRegex(ValueError,'changed|missing'):self.execute()

    def test_v1_uncaptured_acceptance_semantics_rechecked(self):
        a=h.OLD[0];p=a['root']+'/claims/'+a['request_sha']+'/capture-sbatch.argv.json'
        value=json.loads(self.raw[p]);value=[x.replace('--dependency=afterany:412823','--dependency=afterany:999999') for x in value]
        self.raw[p]=json.dumps(value).encode()
        with self.assertRaisesRegex(ValueError,'observer acceptance'):self.execute()

    def test_missing_metadata_body_fails(self):
        del self.raw[next(iter(self.fixture['expected_hashes']))]
        with self.assertRaises(KeyError):self.execute()

    def test_failed_native_query_cannot_be_empty_success(self):
        self.query_error={'returncode':1,'stderr':'controller unavailable'}
        with self.assertRaisesRegex(ValueError,'native census'):self.execute()

    def test_native_stderr_even_exit_zero_fails(self):
        self.query_error={'returncode':0,'stderr':'truncated history'}
        with self.assertRaisesRegex(ValueError,'native census'):self.execute()

    def test_prior_parent_array_alias_rejected(self):
        self.queue='412825_1|unrelated-name|RUNNING|\n'
        with self.assertRaisesRegex(ValueError,'Prior accepted'):self.execute()

    def test_prior_capture_live_step_rejected(self):
        self.steps='412830.batch|RUNNING\n'
        with self.assertRaises(ValueError):self.execute()

    def test_other_scientific_writer_rejected(self):
        self.queue='999999|elmo-large314m-qknorm-other|PENDING|\n'
        with self.assertRaisesRegex(ValueError,'scientific writer'):self.execute()

    def test_malformed_successful_native_output_rejected(self):
        self.steps='unparseable result'
        with self.assertRaisesRegex(ValueError,'step inventory'):self.execute()

    def test_allocated_entry_may_not_defer_csv(self):
        with self.assertRaisesRegex(ValueError,'Allocated node'):self.execute(require_node=True)

    def test_old_csv_conflict_rejected_on_actual_node(self):
        self.csv_node={'fixture':True};v4=self.fixture['verified']['original_request']['local_logs']
        self.existing={v4}
        with self.assertRaisesRegex(ValueError,'namespace'):self.execute(require_node=True)

    def test_serialized_census_cannot_discharge_claim(self):
        value=self.execute();claim=next(iter(h.OLD_CLAIMS))
        self.assertEqual(h.claim_disposition(value,claim,h.RUN_ID)['provenance'],'REVIEWED_PRELAUNCH_NO_TRAINING_ENTRY')
        with self.assertRaisesRegex(ValueError,'Fresh complete'):h.claim_disposition(json.loads(json.dumps(value)),claim,h.RUN_ID)
        self.assertIsNone(h.claim_disposition(value,h.ROOT+'/claims/'+h.REQUEST_SHA,h.RUN_ID))

    def test_real_service_claim_hook_returns_typed_not_fake_terminal(self):
        value=self.execute();a=h.OLD[1];claim=Path(a['root'])/'claims'/a['request_sha']
        with mock.patch.object(terminal,'physical',side_effect=Path):
            got=terminal.verify_claim_terminal(claim,a['root'],h.RUN_ID,prelaunch_census=value)
        self.assertEqual(got['parent_native_state'],'UNKNOWN');self.assertNotIn('terminal_native_sha256',got)
        with mock.patch.object(terminal,'physical',side_effect=Path):
            with self.assertRaisesRegex(ValueError,'Fresh complete'):terminal.verify_claim_terminal(claim,a['root'],h.RUN_ID)
        with mock.patch.object(terminal,'physical',side_effect=Path):
            with self.assertRaisesRegex(ValueError,'directory moved'):terminal.verify_claim_terminal(claim,'/tmp/foreign',h.RUN_ID,prelaunch_census=value)

    def test_stale_census_and_mutated_claim_context_rejected(self):
        value=self.execute();claim=next(iter(h.OLD_CLAIMS))
        value['checked_at']='2026-01-01T00:00:00+00:00'
        with self.assertRaisesRegex(ValueError,'stale'):h.claim_disposition(value,claim,h.RUN_ID)
        value['checked_at']=datetime.now(timezone.utc).isoformat()
        value['claims'][claim]['phase_completion_proven']=True
        with self.assertRaisesRegex(ValueError,'changed'):h.claim_disposition(value,claim,h.RUN_ID)

    def test_unknown_original_v5_request_marker_rejected(self):
        self.dirs[h.ROOT+'/request-claims'].append(Path(h.ROOT)/'request-claims/unknown')
        with self.assertRaisesRegex(ValueError,'partial marker'):self.execute()

    def test_original_scientific_claim_cannot_disappear(self):
        self.dirs[h.ROOT+'/claims']=[]
        with self.assertRaisesRegex(ValueError,'partial marker'):self.execute()


class PhysicalAndScope(unittest.TestCase):
    def test_other_run_cannot_opt_into_qk_history(self):
        with self.assertRaisesRegex(ValueError,'another scientific'):h.census({'qk_prelaunch_history':h.expected_binding()},'foreign')
        self.assertIsNone(h.census({},'foreign'))

    def test_metadata_symlink_and_large_file_rejected_before_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp).resolve()/'data';p.write_bytes(b'fixture')
            link=p.with_name('link');link.symlink_to(p)
            with self.assertRaisesRegex(ValueError,'Physical'):h.read_metadata(link)
            with self.assertRaisesRegex(ValueError,'oversized'):h.read_metadata(p,limit=2)
            self.assertEqual(h.read_metadata(p),b'fixture')

    def test_no_gateway_node_csv_authority(self):
        with mock.patch.dict(os.environ,{},clear=True):
            self.assertIsNone(h.node_local_context(lambda _:None,required=False))
            with self.assertRaisesRegex(ValueError,'Allocated'):h.node_local_context(lambda _:None,required=True)

    def test_actual_host_and_native_node_both_required(self):
        job='999991'
        def query(argv):
            raw=('JobId='+job+' JobState=RUNNING UserId=paphiwetsa3(41003) NodeList=gpu21 NumNodes=1') if 'job' in argv else 'NodeName=gpu21 NodeHostName=fixture-host'
            return record(argv,raw)
        with mock.patch.dict(os.environ,{'SLURM_JOB_ID':job,'SLURMD_NODENAME':'gpu21','ELMO_HISTORY_LOCATION':'ALLOCATED_HOST'},clear=True), \
             mock.patch.object(h.os,'getuid',return_value=41003),mock.patch.object(h.socket,'gethostname',return_value='fixture-host'), \
             mock.patch.object(h,'csv_parent_identity',return_value={'fixture':True}):
            self.assertEqual(h.node_local_context(query,required=True)['job_id'],job)
            with mock.patch.object(h.socket,'gethostname',return_value='gateway'):
                with self.assertRaisesRegex(ValueError,'Physical host'):h.node_local_context(query,required=True)

    def test_container_mount_must_match_real_host_directory_identity(self):
        job='999991';root=h.TASK_ROOT+'/recovery-fixture';attempt=root+'/status/inspect-'+job+'.attempt'
        fields=dict(JobId=job,UserId='paphiwetsa3(41003)',JobName='fixture',SubmitTime='2026-10-27T00:00:00',
            StartTime='2026-10-27T00:00:01',Command=root+'/allocated.sbatch',Comment='fixture',NodeList='gpu21',
            NumNodes='1',NumCPUs='4',ReqTRES='cpu=4,mem=24G',AllocTRES='cpu=4,mem=24G',TimeLimit='00:20:00',WorkDir=root)
        identity=dict(path=h.CSV_PARENT,device=37,inode=4829,uid=41003)
        context=dict(job_id=job,location='ALLOCATED_HOST',csv_parent=identity)
        host=dict(job_id=job,request_sha256='a'*64,native=' '.join(k+'='+v for k,v in dict(fields,JobState='RUNNING').items()),
            live_admission={'quiescence':{'prelaunch_history':{'node_context':context}}})
        hostraw=json.dumps(host).encode()
        container=dict(schema='native-elmo-container-auth-v1',job_id=job,request_sha256='a'*64,host_native_sha256=h.sha(hostraw))
        def read(path,**kw):return hostraw if str(path).endswith('host-native.json') else json.dumps(container).encode()
        with mock.patch.dict(os.environ,{'ELMO_HISTORY_ATTEMPT':attempt},clear=True),mock.patch.object(h,'read_metadata',side_effect=read):
            self.assertEqual(h.verify_host_csv_context(fields,job,identity,'CONTAINER_GATE')['sha256'],h.sha(hostraw))
            self.assertEqual(h.verify_host_csv_context(fields,job,identity,'CONTAINER_DISPATCH')['sha256'],h.sha(hostraw))
            with self.assertRaisesRegex(ValueError,'masks'):h.verify_host_csv_context(fields,job,dict(identity,inode=999),'CONTAINER_GATE')
            container['host_native_sha256']='f'*64
            with self.assertRaisesRegex(ValueError,'Final container'):h.verify_host_csv_context(fields,job,identity,'CONTAINER_DISPATCH')
            wrong=dict(fields,Command='/tmp/foreign/allocated.sbatch')
            with self.assertRaisesRegex(ValueError,'outside'):h.verify_host_csv_context(wrong,job,identity,'CONTAINER_GATE')



if __name__=='__main__':unittest.main()
