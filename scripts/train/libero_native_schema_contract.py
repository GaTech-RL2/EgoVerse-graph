"""Scheduled official schema audit of the maintained launcher's actual array."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--phase',choices=['preflight','smoke','full'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--argv',type=Path,required=True);p.add_argument('--expected-argv-sha256',required=True);a=p.parse_args()
 if not os.environ.get('SLURM_STEP_ID'): raise RuntimeError('scheduled srun only')
 raw=a.argv.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=a.expected_argv_sha256 or not raw.endswith(b'\0'): raise ValueError('argv identity/serialization mismatch')
 argv=[v.decode() for v in raw[:-1].split(b'\0')]
 sys.path[:0]=[str(a.repo),str(a.repo/'tools')]
 from omegaconf import OmegaConf
 from libero_maintained_dispatch_v1 import flatten,validate_resolved,EXPECTED_PARAMETER_COUNT,PROFILE
 from libero_operational_guard import validate_native,OPS
 from typed_libero_profile_v3 import COMMON,phase_bindings
 native_args=['--config-name=train_zarr_cartesian','hydra/launcher=basic','+experiment='+PROFILE]
 native=subprocess.check_output([sys.executable,'-m','egomimic.trainHydra',*native_args,'--cfg','job','--resolve'],cwd=a.repo,text=True)
 (a.output/'native-before-overrides.yaml').write_text(native)
 flat=flatten(OmegaConf.to_container(OmegaConf.create(native),resolve=True));validate_native(flat,COMMON)
 supplied={};unknown=[]
 # Canonical explicit additions audited against the exact maintained source.
 additions={'paths.root_dir','callbacks.model_checkpoint.monitor','callbacks.model_checkpoint.every_n_train_steps','callbacks.model_checkpoint.train_time_interval','callbacks.model_checkpoint.save_on_train_epoch_end','logger.wandb.offline','logger.wandb.entity','logger.wandb.project','logger.wandb.group','logger.wandb.id','logger.wandb.name','logger.wandb.resume','logger.wandb.tags','norm_stats.native_saved_state_binding'}
 provenance={'execution_cluster','run_kind','source_commit','launcher_sha256','split_manifest_path','split_manifest_sha256','content_manifest_path','content_manifest_sha256','dataset_content_aggregate_sha256','dataset_sha256','historical_split_manifest_sha256','normalization_sha256','preflight_result_sha256','smoke_result_sha256'}
 additions.update('run_provenance.'+k for k in provenance)
 for arg in argv:
  if arg.startswith('--'):
   if arg!='--config-name=train_zarr_cartesian': raise ValueError(('unexpected official flag',arg))
   continue
  if '=' not in arg: continue
  lhs,value=arg.split('=',1);key=lhs.lstrip('+');prefix=lhs[:-len(key)]
  if key in supplied: raise ValueError(('duplicate override',key))
  supplied[key]=value
  if key not in flat and key not in {'experiment','hydra/launcher'} and not key.startswith('hydra.'):
   if key not in additions or prefix!='++': unknown.append(key)
 required=(set(OPS)|set(phase_bindings(a.phase))|{'mode','ckpt_path','launch_params.gpus_per_node','launch_params.nodes','trainer.log_every_n_steps','runtime.slurm_requeue_owner','runtime.slurm_save_signal','runtime.slurm_signal_checkpoint_dir','norm_stats.native_saved_state_binding','norm_stats.precomputed_norm_path','norm_stats.save_cache_dir'})-set(COMMON)
 # Common native model values may be configured directly; canonical operational fields must be explicit.
 required.update(OPS)
 missing=sorted(required-set(supplied))
 if unknown or missing: raise ValueError({'unknown_fields':unknown,'missing_operational_fields':missing})
 if any(k.startswith('evaluator.energy_score_validation_view') for k in supplied): raise ValueError('U-Socket evaluator leak')
 binding=OmegaConf.to_container(OmegaConf.from_dotlist([next(x.lstrip('+') for x in argv if x.startswith('++norm_stats.native_saved_state_binding='))]),resolve=True)['norm_stats']['native_saved_state_binding']
 binding_keys={'path','file_sha256','normalizer_module_sha256','source_commit','dataset_receipt_path','dataset_receipt_sha256','split_receipt_path','split_receipt_sha256','data_root','normalizer_target','physical_proof_path','physical_proof_sha256'}
 if not isinstance(binding,dict) or set(binding)!=binding_keys: raise ValueError('exact native saved-state twelve-field mapping required')
 # Compose exact array without accelerator or runtime substitutions.
 resolved=subprocess.check_output([sys.executable,'-m','egomimic.trainHydra',*argv,'--cfg','job','--resolve'],cwd=a.repo,text=True)
 path=a.output/'resolved.yaml';path.write_text(resolved);cfg=OmegaConf.create(resolved)
 validate_resolved(OmegaConf.to_container(cfg,resolve=True),a.phase)
 from libero_native_scientific_contract import validate_scientific_contract
 scientific=validate_scientific_contract(OmegaConf.to_container(cfg,resolve=True),a.phase,a.repo/'assets/libero/historical_af27m_scientific_contract_v1.json','ced1ea16d71ad30fe179a371f3b6e2138765b6596bca50a3806355e05b20eb06')
 # Cheap context/hash gates; numeric saved-state proof is reused separately.
 for path_key,sha_key in [('path','file_sha256'),('dataset_receipt_path','dataset_receipt_sha256'),('split_receipt_path','split_receipt_sha256'),('physical_proof_path','physical_proof_sha256')]:
  if hashlib.sha256(Path(binding[path_key]).read_bytes()).hexdigest()!=binding[sha_key]: raise ValueError(('binding artifact SHA mismatch',path_key))
 from hydra.utils import instantiate
 from egomimic.rldb.zarr.libero_saved_state import validate_receipt_contract,make_physical_split_proof
 from egomimic.rldb.zarr import libero_action_flow
 if hashlib.sha256(Path(libero_action_flow.__file__).read_bytes()).hexdigest()!=binding['normalizer_module_sha256']: raise ValueError('normalizer module identity mismatch')
 from libero_native_metadata_probe import instantiate_metadata_dataset
 dataset=instantiate_metadata_dataset(cfg.data.train_datasets.libero_panda,omega_conf=OmegaConf,instantiate=instantiate)
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip()
 validate_receipt_contract(binding,OmegaConf.to_container(cfg.run_provenance,resolve=True),dataset,source_head=head)
 physical=json.loads(Path(binding['physical_proof_path']).read_text())
 if make_physical_split_proof(dataset,binding)!=physical: raise ValueError('actual physical split proof mismatch')
 from egomimic.rldb.zarr.libero_saved_state import bind_saved_native_state
 normalizer=instantiate(cfg.normalizer,state={},norm_mode=cfg.norm_stats.norm_mode)
 normalizer.populate_from_datasets({'libero_panda':dataset})
 normalizer.infer_shapes_from_batch(dataset[0])
 loaded=bind_saved_native_state(norm_stats=normalizer,dataset=dataset,dataset_name='libero_panda',binding=binding,run_provenance=OmegaConf.to_container(cfg.run_provenance,resolve=True))
 if type(loaded) is not type(normalizer): raise ValueError('native saved-state actual class mismatch')
 count=None
 if os.environ.get('AF_NATIVE_CONSTRUCT')=='true':
  import torch
  from hydra.utils import instantiate
  if not torch.__version__.startswith('2.7.1') or torch.cuda.is_initialized(): raise ValueError('pinned CPU runtime required')
  pipeline=instantiate(cfg.model.pipeline);count=sum(x.numel() for x in pipeline.nets.parameters())
  if count!=EXPECTED_PARAMETER_COUNT or not all(bool(torch.isfinite(x).all()) for x in pipeline.nets.parameters()): raise ValueError('full native constructor contract')
 result={'status':'OFFICIAL_SHARED_NATIVE_SCHEMA_COMPOSE_PASS_ONLY','phase':a.phase,'argv_sha256':a.expected_argv_sha256,'resolved_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'native_key_count':len(flat),'operational_key_count':len(OPS),'total_parameters':count,'native_saved_state_field_count':len(binding),'gpu_ready':False,'optimizer_validation_smoke':False,'pending_late_proof_identities':os.environ.get('AF_NATIVE_PENDING_PROOFS','').split(),'structural_audit_only':bool(os.environ.get('AF_NATIVE_PENDING_PROOFS','').strip()),'actual_native_saved_state_bound':True,'scientific_contract':scientific,'metadata_resolver_decoded_cache':False,'training_config_modified':False,'operational_verifier_pending':os.environ.get('AF_NATIVE_OPERATIONAL_VERIFIER_PENDING')=='true'}
 if count is not None:
  constructor={'status':'PASS','full_model_constructor':True,'parameter_count':count,'source_commit':head,'resolved_config_sha256':result['resolved_sha256'],'native_saved_state_binding_validated':True,'physical_proof_sha256':binding['physical_proof_sha256']}
  cp=a.output/'CPU_CONSTRUCTOR.json';cp.write_text(json.dumps(constructor,indent=2)+'\n')
  result['CPU_constructor']={'path':str(cp),'sha256':hashlib.sha256(cp.read_bytes()).hexdigest()}
  if a.phase=='preflight':
   preflight={'schema':'libero-native-launch-preflight/v1','status':'PASS','profile':PROFILE,'source_commit':head,'resolved_config_sha256':result['resolved_sha256'],'normalization_sha256':binding['file_sha256'],'split_manifest_sha256':binding['split_receipt_sha256'],'dataset_sha256':cfg.run_provenance.dataset_sha256,'CPU_constructor':result['CPU_constructor'],'physical_proof':{'path':binding['physical_proof_path'],'sha256':binding['physical_proof_sha256']},'scope':'native_schema_constructor','launch_ready':False}
   pf=a.output/'NATIVE_SCHEMA_PREFLIGHT.json';pf.write_text(json.dumps(preflight,indent=2)+'\n')
   result['native_schema_preflight']={'path':str(pf),'sha256':hashlib.sha256(pf.read_bytes()).hexdigest()}
 (a.output/'RESULT.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
