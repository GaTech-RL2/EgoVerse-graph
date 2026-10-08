"""Focused fingerprint invariants only; not native constructor or smoke proof."""
import copy,unittest
from scripts.train.libero_native_launch_contract import contract_sha
class Contract(unittest.TestCase):
 def setUp(self):
  self.cfg=dict(model={'pipeline':{'stages':[{'flow_aggregation':'sum_samples','inference_method':'euler','num_inference_steps':50}]},'gradient_telemetry_cadence':100},normalizer={'_target_':'native'},norm_stats={'native_saved_state_binding':{'source_commit':'fixed'}},data={'train_datasets':{'libero_panda':{'split_seed':42}},'valid_datasets':{'libero_panda':{'split_seed':42}},'train_dataloader_params':{'libero_panda':{'batch_size':32}},'valid_dataloader_params':{'libero_panda':{'batch_size':32}}},trainer={'precision':'bf16-mixed','accumulate_grad_batches':1,'max_steps':80000},seed=42,callbacks={'ema':{'validate_with_ema':True},'dit_half':{'_target_':'native'}},evaluator={'energy_sample_count':32,'energy_seed_bank_sha256':'fixed','artifact_root':'/out/a','artifact_identity':{'_target_':'factory','config_path':'/out/a/.hydra/config.yaml'},'native_diagnostic_config':{'_target_':'diagfactory','config_path':'/out/a/.hydra/config.yaml','artifact_root':'/out/a/diagnostic'}})
 def test_phase_only_equivalent(self):
  cfg=copy.deepcopy(self.cfg);cfg['trainer']['max_steps']=2;cfg['model']['gradient_telemetry_cadence']=2
  self.assertEqual(contract_sha(cfg),contract_sha(self.cfg))
 def test_output_only_equivalent(self):
  cfg=copy.deepcopy(self.cfg);cfg['evaluator']['artifact_root']='/out/b';cfg['evaluator']['artifact_identity']['config_path']='/out/b/.hydra/config.yaml';cfg['evaluator']['native_diagnostic_config']['artifact_root']='/out/b/diagnostic'
  self.assertEqual(contract_sha(cfg),contract_sha(self.cfg))
 def test_mean_rejected(self):
  cfg=copy.deepcopy(self.cfg);cfg['model']['pipeline']['stages'][0]['flow_aggregation']='mean'
  self.assertNotEqual(contract_sha(cfg),contract_sha(self.cfg))
 def test_seedbank_rejected(self):
  cfg=copy.deepcopy(self.cfg);cfg['evaluator']['energy_seed_bank_sha256']='changed'
  self.assertNotEqual(contract_sha(cfg),contract_sha(self.cfg))
 def test_k_changed_rejected(self):
  cfg=copy.deepcopy(self.cfg);cfg['evaluator']['energy_sample_count']=16
  self.assertNotEqual(contract_sha(cfg),contract_sha(self.cfg))
 def test_factory_target_rejected(self):
  cfg=copy.deepcopy(self.cfg);cfg['evaluator']['native_diagnostic_config']['_target_']='otherfactory'
  self.assertNotEqual(contract_sha(cfg),contract_sha(self.cfg))
 def test_sampling_rejected(self):
  cfg=copy.deepcopy(self.cfg);cfg['model']['pipeline']['stages'][0]['num_inference_steps']=8
  self.assertNotEqual(contract_sha(cfg),contract_sha(self.cfg))
if __name__=='__main__':unittest.main()
