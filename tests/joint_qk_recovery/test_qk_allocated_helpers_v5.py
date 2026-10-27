"""CPU tensor and config fixtures only; not an allocated restore certification."""
import copy,json,subprocess,sys,tempfile,unittest
from pathlib import Path
import torch
from omegaconf import OmegaConf
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'files/scripts/train/joint/qk_recovery_profile.py').is_file() or (p/'scripts/train/joint/qk_recovery_profile.py').is_file())
SERVICE=ROOT/'files' if (ROOT/'files/scripts/train/joint/qk_recovery_profile.py').is_file() else ROOT
FIXTURES=Path(__file__).resolve().parent/'fixtures'
sys.path.insert(0,str(SERVICE))
from scripts.train.joint.qk_recovery_usage import snapshot_private,PREFIX
from scripts.train.joint.production_recovery import phase_config

class CPUHelper(unittest.TestCase):
 def test_inactive_state_digest_covers_weight_and_adam_moments(self):
  from scripts.train.joint.qk_recovery_usage import expected_all_parameters
  named=[(name,torch.nn.Parameter(torch.tensor([1.,2.]))) for name in expected_all_parameters()]
  params=dict(named);human=params[PREFIX+'language_fusions.proprio__human_bimanual.0.weight'];robot=params[PREFIX+'language_fusions.action.0.weight']
  class Module:
   def named_parameters(self):return named
  opt=torch.optim.AdamW(list(params.values()));initial=snapshot_private(Module(),opt)
  robot.grad=torch.ones_like(robot);opt.step();after=snapshot_private(Module(),opt)
  self.assertEqual(initial['human_bimanual'],after['human_bimanual']);self.assertNotEqual(initial['eva_bimanual'],after['eva_bimanual'])
  human.grad=torch.ones_like(human);opt.step();updated=snapshot_private(Module(),opt)
  opt.state[human]['exp_avg'].add_(1);changed=snapshot_private(Module(),opt)
  self.assertNotEqual(updated['human_bimanual'],changed['human_bimanual'])
 def test_qk_callback_added_without_science_or_sampling_changes(self):
  cfg=OmegaConf.load(FIXTURES/'full-resolved.yaml')
  original=OmegaConf.to_container(cfg,resolve=True)
  phase=phase_config(cfg,output=Path('/tmp/diagnostic'),checkpoint='/tmp/production20000.ckpt',origin=20000,start=20000,stop=20002,target=300000,world=4,proof=Path('/tmp/proof'),qk_architecture=True)
  self.assertTrue(phase.callbacks.resume_proof.qk_architecture)
  self.assertEqual(OmegaConf.to_container(phase.model,resolve=True),original['model'])
  self.assertEqual(OmegaConf.to_container(phase.data,resolve=True),original['data'])
  self.assertEqual(phase.callbacks.model_checkpoint.every_n_train_steps,2)
  self.assertEqual(phase.trainer.max_steps,20006)
 def test_old_callback_profile_still_defaults_false(self):
  cfg=OmegaConf.load(FIXTURES/'full-resolved.yaml')
  phase=phase_config(cfg,output=Path('/tmp/diagnostic'),checkpoint='/tmp/production20000.ckpt',origin=20000,start=20002,stop=20006,target=300000,world=4,proof=Path('/tmp/proof'))
  self.assertFalse(phase.callbacks.resume_proof.qk_architecture)


if __name__=='__main__':unittest.main()
