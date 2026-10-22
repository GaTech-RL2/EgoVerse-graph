"""Synthetic evidence rejection tests; no checkpoint/runtime/GPU certification."""
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from scripts.train.joint import crash_recovery as c
from scripts.train.joint.continuation import allocation_steps


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.counter = 0
        original = Path(__file__).parent / 'fixtures/joint_crash_recovery_535d/image-only-parent.request.json'
        self.parent = json.loads(original.read_text())
        self.parent['output'] = str(self.root / 'parent')
        self.parent['local_logs'] = str(self.root / 'parent-logs')
        self.parent['overrides'] = ['model.optimizer.lr=1e-4', 'paths.output_dir=' + self.parent['output'],
                                    'logger.csv.save_dir=' + self.parent['local_logs']]
        self.q = copy.deepcopy(self.parent)
        self.q['output'] = str(self.root / 'attempt')
        self.q['local_logs'] = str(self.root / 'attempt-logs')
        self.q['overrides'] = [v.replace(self.parent['output'],self.q['output']).replace(self.parent['local_logs'],self.q['local_logs']) for v in self.parent['overrides']]
        self.q['job_name'] = 'joint-image-lr1e4-recovery-test'
        self.q['operational_service'] = {'manifest_sha256':'a'*64}
        parent = self.binding(self.parent)
        native = self.binding('JobId=412056 JobName=joint-image-lr1e4-full-4gpu UserId=paphiwetsa3(41003) JobState=CANCELLED Requeue=0 Restarts=0', text=True)
        preflight = self.binding({'status':'FULL_ADMISSION_PASS','job_id':'412056','request_sha256':parent['sha256'],
                                 'provenance':{'run_id':self.q['run_id'],'source_revision':c.SOURCE}})
        self.failure = {'schema':'joint-crash-failure-v1','parent_request_sha256':parent['sha256'],
                        'run_id':self.q['run_id'],'classification':'worker_crash', 'recovery_authorized':True,
                        'human_retired':False,'job_id':412056,'native':native, 'logs':[self.binding('NCCL timeout/SIGABRT',text=True)],
                        'parent_preflight':preflight}
        ckpt = self.root/'parent/checkpoints/epoch_epoch=0011-step_step=000020000.ckpt'
        ckpt.parent.mkdir(parents=True);ckpt.write_bytes(b'synthetic-not-a-real-checkpoint')
        (ckpt.parent/'last.ckpt').symlink_to(ckpt.name)
        self.inspection = {'schema':'joint-crash-checkpoint-inspection-v1','status':'PASS',
                           'parent_request_sha256':parent['sha256'],'run_id':self.q['run_id'],
                           'source_revision':c.SOURCE,'world_size':4,'global_batch':32,'global_step':20000,
                           'strict_full_wrapper_optimizer_reload':True,'sampler_all_rank_rng_verified':True,
                           'highest_complete_checkpoint_inspected':True,'all_newer_candidates_reconciled':True,
                           'checkpoint':{'path':str(ckpt),'sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest()}}
        inspect = self.binding(self.inspection)
        self.proof = {'schema':'joint-crash-recovery-smoke-v1','status':'PASS','source_revision':c.SOURCE,
                      'parent_request_sha256':parent['sha256'],'inspection_sha256':inspect['sha256'],
                      'checkpoint_sha256':self.inspection['checkpoint']['sha256'],
                      'service_manifest_sha256':'a'*64,'world_size':4,'global_batch':32,
                      'all_rank_full_state_restored':True,'finite_optimizer_updates':2,
                      'scheduled_validation_anchors':100,'no_replayed_boundary_validation_or_save':True,
                      'production_science_unchanged':True,'sample_finite':True,'hidden_query_labels_nan':True,
                      'sample_steps':20,'job_id':'999','job_name':'joint-image-lr1e4-crash-diagnostic',
                      'native':self.binding('JobId=999 JobName=joint-image-lr1e4-crash-diagnostic UserId=paphiwetsa3(41003) NodeList=gpu19 Requeue=0 Restarts=0 JobState=COMPLETED ExitCode=0:0',text=True),
                      'host_exit':self.binding('0\n',text=True)}
        self.q['crash_recovery']={'schema':'joint-crash-recovery-v1','parent_request':parent,
                                 'failure':self.binding(self.failure),'inspection':inspect,
                                 'recovery_smoke':None}
        self.bind_proof()

    def tearDown(self):
        self.temp.cleanup()

    def binding(self, value, text=False):
        self.counter += 1;p=self.root/f'evidence-{self.counter}.txt'
        p.write_text(value if text else json.dumps(value))
        return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

    def update_failure(self, **values):
        self.failure.update(values);self.q['crash_recovery']['failure']=self.binding(self.failure)

    def update_inspection(self, **values):
        self.inspection.update(values);b=self.binding(self.inspection)
        self.q['crash_recovery']['inspection']=b;self.proof['inspection_sha256']=b['sha256']
        self.bind_proof()

    def bind_proof(self):
        diagnostic = copy.deepcopy(self.q)
        diagnostic.pop('crash_recovery',None)
        diagnostic.update(schema='joint-crash-recovery-diagnostic-v1',
                          parent_request=self.q['crash_recovery']['parent_request'],
                          inspection=self.q['crash_recovery']['inspection'],
                          job_name='joint-image-lr1e4-crash-diagnostic',output=str(self.root/'diagnostic'))
        request=self.binding(diagnostic);self.proof['request']=request
        self.proof['request_sha256']=request['sha256']
        self.proof['candidate']=self.binding({'status':'SYNTHETIC_TEST_ONLY'})
        records=[]
        for group in ['restore','validation','anchors']:
            for rank in range(4):
                name = (f'restore-proof/restore-step000020000-rank{rank}.json' if group=='restore' else
                        f'restore-proof/restore-validation-rank{rank}.json' if group=='validation' else
                        f'validation-proof/rank{rank}.json')
                path=self.root/'diagnostic'/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps({'status':'PASS','job_id':'999','world_size':4,
                                            'rank':rank,'start_step':20000,'global_step':20002}))
                records.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        self.proof['rank_proofs']=records
        final=self.root/'diagnostic/checkpoints/diagnostic.ckpt';final.parent.mkdir(exist_ok=True)
        final.write_bytes(b'SYNTHETIC_TEST_ONLY')
        self.proof['final_diagnostic_checkpoint']={'path':str(final),
                'sha256':hashlib.sha256(final.read_bytes()).hexdigest(),'strict_full_wrapper_optimizer_reload':True}
        self.q['crash_recovery']['recovery_smoke']=self.binding(self.proof)

    def test_bound_failure_and_actual_proof_contract(self):
        self.assertEqual(c.admission(self.q)[2],20000)
        self.assertEqual(allocation_steps(self.q),(20000,300000))

    def test_no_recovery_smoke_no_admission(self):
        del self.q['crash_recovery']['recovery_smoke']
        with self.assertRaises((KeyError,ValueError)):c.admission(self.q)

    def test_wrong_native_exit_rejected(self):
        for state in ('RUNNING','PENDING','COMPLETING','COMPLETED'):
            with self.subTest(state=state):
                self.update_failure(native=self.binding('JobId=412056 JobName=joint-image-lr1e4-full-4gpu UserId=paphiwetsa3(41003) JobState='+state+' Requeue=0 Restarts=0',text=True))
                with self.assertRaises(ValueError):c.admission(self.q)

    def test_deliberate_human_retirement_rejected(self):
        self.update_failure(human_retired=True)
        with self.assertRaises(ValueError):c.admission(self.q)

    def test_changed_science_or_lr_rejected(self):
        for field,value in [('world_size',2),('source_revision','0'*40),('lr_policy',dict(schedule='constant',peak_lr=5e-5,warmup_optimizer_steps=0)),('target_steps',320000)]:
            q=copy.deepcopy(self.q);q[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):c.admission(q)

    def test_unresolved_newer_checkpoint_rejected(self):
        self.update_inspection(all_newer_candidates_reconciled=False)
        with self.assertRaises(ValueError):c.admission(self.q)

    def test_corrupted_checkpoint_rejected(self):
        Path(self.inspection['checkpoint']['path']).write_bytes(b'changed')
        with self.assertRaises(ValueError):c.admission(self.q)

    def test_overlapping_or_reused_output_rejected(self):
        self.q['output']=self.parent['output']+'/recovery'
        with self.assertRaises(ValueError):c.admission(self.q)

    def test_override_only_log_output_relocation(self):
        self.q['overrides'].append('data.max_frames=2')
        with self.assertRaises(ValueError):c.admission(self.q)

    def test_smoke_host_native_and_service_closure_required(self):
        for field,value in [('host_exit',self.binding('1\n',text=True)),('service_manifest_sha256','b'*64),('scheduled_validation_anchors',0),('no_replayed_boundary_validation_or_save',False)]:
            proof=copy.deepcopy(self.proof);proof[field]=value;q=copy.deepcopy(self.q)
            q['crash_recovery']['recovery_smoke']=self.binding(proof)
            with self.subTest(field=field),self.assertRaises(ValueError):c.admission(q)

    def test_accepted_parent_request_preflight_required(self):
        self.update_failure(parent_preflight=self.binding({'status':'FULL_ADMISSION_PASS','job_id':'123','request_sha256':'bad','provenance':{}}))
        with self.assertRaises(ValueError):c.admission(self.q)

    def test_scheduled_path_is_unchanged(self):
        self.assertEqual(allocation_steps(self.parent),(0,300000))
        q=copy.deepcopy(self.q);q['continuation']={}
        with self.assertRaises(ValueError):c.admission(q)

    def test_original_comparison_foundation_is_retained(self):
        self.assertEqual(c.foundation_request(self.q),self.parent)
        other=copy.deepcopy(self.parent);other['run_id']='foreign'
        self.q['crash_recovery']['parent_request']=self.binding(other)
        with self.assertRaises(ValueError):c.foundation_request(self.q)

    def test_writer_quiescence_rejects_old_active_parent(self):
        run=SimpleNamespace(returncode=0,stdout='JobState=COMPLETING',stderr='')
        with patch.dict(os.environ,SLURM_JOB_ID='123'),patch.object(c.subprocess,'run',return_value=run):
            with self.assertRaises(ValueError):c.assert_native_writer_quiescence(self.q)

    def test_writer_quiescence_rejects_unknown_transport(self):
        run=SimpleNamespace(returncode=1,stdout='',stderr='connection unavailable')
        with patch.dict(os.environ,SLURM_JOB_ID='123'),patch.object(c.subprocess,'run',return_value=run):
            with self.assertRaises(ValueError):c.assert_native_writer_quiescence(self.q)

    def test_writer_quiescence_checks_entire_owned_queue(self):
        runs=[SimpleNamespace(returncode=1,stdout='',stderr='Invalid job id specified'),
              SimpleNamespace(returncode=0,stdout='123|joint-image-lr1e4-recovery-test|RUNNING\n789|joint-image-lr1e4-other|PENDING',stderr='')]
        with patch.dict(os.environ,SLURM_JOB_ID='123'),patch.object(c.subprocess,'run',side_effect=runs):
            with self.assertRaises(ValueError):c.assert_native_writer_quiescence(self.q)

    def test_exclusive_actual_destination_fsync_probe(self):
        with patch.dict(os.environ,SLURM_JOB_ID='123'),patch.object(c.shutil,'disk_usage',return_value=SimpleNamespace(free=100_000_000_000)):
            result=c.prove_destination_storage(self.q)
            self.assertTrue(result['write_fsync_readback_pass'])
            with self.assertRaises(FileExistsError):c.prove_destination_storage(self.q)


if __name__=='__main__':unittest.main()
