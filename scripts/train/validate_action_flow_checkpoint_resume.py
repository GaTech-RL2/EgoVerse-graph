"""Audit checkpoint continuation against hash-pinned scientific configuration."""
import os,json,hashlib,subprocess
from pathlib import Path
import yaml
import sys
assert os.environ.get('SLURM_STEP_ID')
cfg=yaml.safe_load(Path(sys.argv[1]).read_text());baseline=yaml.safe_load(Path(os.environ['AF_RESUME_BASE_CONFIG']).read_text())
proof=json.load(open(os.environ['AF_RESUME_CPU_PROOF']));assert proof['status']=='PASS' and proof['strict_reload']=='passed'
assert proof['checkpoint_sha256']==os.environ['AF_INITIAL_CHECKPOINT_SHA256']
assert proof['model_source']==os.environ['AF_EXPECTED_HEAD']
start=int(os.environ['AF_RESUME_START_STEP']);assert start==proof['actual_global_step']
assert cfg['ckpt_path']==proof['selected_checkpoint']
assert cfg['run_provenance']['source_commit']==proof['model_source']
assert cfg['trainer']['precision']=='bf16' and cfg['trainer']['devices']==1
assert all(v['batch_size']==32 for v in cfg['data']['train_dataloader_params'].values())
typed=os.environ.get('AF_RESUME_TYPED_MODEL')
assert typed in (None,'noaug','av0')
if not typed: assert cfg['callbacks']['homogeneous_dithalf']['task']==os.environ['AF_HOMOGENEOUS_TASK']
assert cfg['callbacks']['model_checkpoint']['save_top_k']==-1
assert cfg['model']['pipeline']['stages'][6]['inference_method']=='euler'
assert cfg['model']['pipeline']['stages'][6]['num_inference_steps']==50
if typed!='noaug': assert cfg['speed_diagnostic']['encoding']=='scalar'
smoke=os.environ['AF_RUN_KIND']=='smoke'
assert cfg['trainer']['max_steps']==(start+4 if smoke else 80000)
assert cfg['trainer']['val_check_interval']==(4 if smoke else 15000)
assert cfg['trainer']['limit_train_batches']==1.0
assert cfg['trainer']['limit_val_batches']==(1 if smoke else 8)
assert cfg['callbacks']['model_checkpoint']['every_n_train_steps']==(1 if smoke else 5000)
assert ('resume_batch_probe' in cfg['callbacks'])==smoke
assert cfg['logger']['wandb']['resume']==('never' if smoke else 'must')
def flat(x,p=''):
 if isinstance(x,dict):
  result={}
  for k,v in x.items(): result.update(flat(v,p+'.'+str(k) if p else str(k)))
  return result
 if isinstance(x,list):
  result={}
  for k,v in enumerate(x): result.update(flat(v,p+'.'+str(k)))
  return result
 return {p:x}
a,b=flat(baseline),flat(cfg);changes=[]
allowed=['paths.','logger.','run_provenance.','callbacks.model_checkpoint.dirpath','callbacks.homogeneous_dithalf.task','ckpt_path','name','hydra.']
if smoke: allowed+=['trainer.max_steps','trainer.val_check_interval','trainer.limit_val_batches','trainer.log_every_n_steps','callbacks.model_checkpoint.every_n_train_steps','callbacks.resume_batch_probe.','model.gradient_telemetry_cadence','evaluator.energy_score_max_batches_per_rank']
# Artifact paths follow the unique output root; contracts and budgets remain checked.
allowed+=['evaluator.artifact_root','evaluator.action_flow_diagnostics.artifact_root','evaluator.unite_diagnostics.artifact_root','evaluator.energy_score_provenance.resolved_config_path','evaluator.energy_score_provenance.wandb.run_id','runtime.slurm_signal_checkpoint_dir','trainer.default_root_dir']
unexpected=[]
for key in sorted(set(a)|set(b)):
 if a.get(key)!=b.get(key):
  if not any(key==p or key.startswith(p) for p in allowed): unexpected.append((key,a.get(key),b.get(key)))
  changes.append({'field':key,'before':a.get(key),'after':b.get(key)})
assert not unexpected, unexpected
output=Path(os.environ['AF_OUTPUT_DIR'])
assert Path(cfg['trainer']['default_root_dir'])==output
assert Path(cfg['runtime']['slurm_signal_checkpoint_dir'])==output/'checkpoints'
assert Path(cfg['evaluator']['energy_score_provenance']['resolved_config_path'])==output/'.hydra/config.yaml'
assert cfg['evaluator']['energy_score_provenance']['wandb']['run_id']==os.environ['AF_WANDB_RUN_ID']
cache=json.load(open(os.environ['AF_RESUME_CACHED_PREFLIGHT']));assert cache['status']=='PASS' and cache['source']['head']==os.environ['AF_EXPECTED_HEAD']
assert cache['normalization_sha256']==os.environ['AF_EXPECTED_NORM_SHA256']
for path in [os.environ['AF_SPLIT_MANIFEST'],os.environ['AF_SECOND_SPLIT_MANIFEST'],os.environ['AF_CONTENT_MANIFEST'],os.environ['AF_SECOND_CONTENT_MANIFEST']]: assert Path(path).is_file()
# Preserve the checkpoint-bound source and all inherited model/config files.
driver=os.environ['AF_RESUME_DRIVER_REPO'];assert subprocess.check_output(['git','-C',driver,'rev-parse','HEAD'],text=True).strip()==os.environ['AF_EXPECTED_RESUME_DRIVER_HEAD'];
if not typed: subprocess.run(['git','-C',driver,'diff','--exit-code',os.environ['AF_EXPECTED_HEAD'],'HEAD','--','egomimic'],check=True,stdout=subprocess.DEVNULL)
assert subprocess.check_output(['git','-C',os.environ['AF_REPO'],'rev-parse','HEAD'],text=True).strip()==os.environ['AF_EXPECTED_HEAD']
assert not subprocess.check_output(['git','-C',os.environ['AF_REPO'],'status','--porcelain','--untracked-files=all']).strip()
assert not subprocess.check_output(['git','-C',driver,'status','--porcelain','--untracked-files=all']).strip()
for helper in (Path(driver)/'scripts/train/homogeneous_pipeline_runtime').glob('*.py'):
 assert hashlib.sha256(helper.read_bytes()).hexdigest()==hashlib.sha256((Path(os.environ.get('AF_RESUME_HELPER_DIR',os.environ['AF_HOMOGENEOUS_TASK']))/helper.name).read_bytes()).hexdigest(), helper.name
component=json.load(open(os.environ['AF_RESUME_BATCH_POLICY_PROOF']));assert component['status']=='PASS' and component['variable_batch_policy']=='native'
assert component['driver_source_commit']==os.environ['AF_EXPECTED_RESUME_DRIVER_HEAD']
callback_name='homogeneous_typed_native_training.py' if typed else 'homogeneous_dithalf_training.py'
assert component['callback_sha256']==hashlib.sha256((Path(os.environ.get('AF_RESUME_HELPER_DIR',os.environ['AF_HOMOGENEOUS_TASK']))/callback_name).read_bytes()).hexdigest()
result={'status':'PASS','model_source_commit':proof['model_source'],'driver_source_commit':subprocess.check_output(['git','-C',driver,'rev-parse','HEAD'],text=True).strip(),'resume_start_step':start,'target_global_step':cfg['trainer']['max_steps'],'checkpoint_sha256':proof['checkpoint_sha256'],'phase':os.environ['AF_RUN_KIND'],'config_sha256':hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest(),'changed_fields':changes,'cached_data_and_normalization_preflight_sha256':os.environ['AF_RESUME_CACHED_PREFLIGHT_SHA256'],'batch_shape_cpu_proof_sha256':os.environ['AF_RESUME_CPU_PROOF_SHA256']}
Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
