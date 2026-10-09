"""Scheduled real-data proof through the maintained generic training entry point."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
from libero_oat_pair_config import validate, SUITES


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def validate_optimizer_coverage(nets, params, family):
    """Preserve the released DP's zero-size frozen device bookkeeping parameter."""
    named = {id(p):(name,p) for name,p in nets.named_parameters()}
    active = {id(p) for p in nets.parameters() if p.requires_grad}
    selected = {id(p) for p in params}
    assert active <= selected, 'Optimizer omits trainable parameters'
    assert selected <= set(named), 'Optimizer includes parameters outside the model'
    from egomimic.models.oat.model.common.normalizer import LinearNormalizer
    frozen_normalizers = {id(p) for owner in nets.modules()
                          if type(owner) is LinearNormalizer for p in owner.parameters()}
    for identity in selected - active:
        name, parameter = named[identity]
        assert family == 'dp' and not parameter.requires_grad, (name, parameter.shape)
        if identity not in frozen_normalizers:
            assert name.endswith('._dummy_variable') and parameter.numel() == 0, (name, parameter.shape)
    return {'trainable_parameters_covered':len(active),
            'released_frozen_bookkeeping':[named[i][0] for i in sorted(selected-active)]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source-commit',required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--phase',choices=('smoke','full'),default='smoke');p.add_argument('--attempt',default='v7')
    a=p.parse_args()
    import re
    assert re.fullmatch(r'v[0-9]+(?:-retry[0-9]+)?',a.attempt)
    assert os.environ.get('SLURM_STEP_ID'), 'scheduled srun only'
    source=Path(__file__).resolve().parents[1]
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    assert actual == a.source_commit
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()
    suite=SUITES[a.index//2];family=('dp','action_flow')[a.index%2]
    out=a.root/(('proof-' if a.phase=='smoke' else 'full-')+a.attempt)/f'{family}-{suite}'
    assert not out.exists(), 'Fresh proof/full identity required; never overwrite a previous attempt'
    out.mkdir(parents=True,exist_ok=False)
    assert not (out/'RESULT.json').exists()
    if a.phase == 'full':
        gate=json.loads((a.root/'FULL_READY_V7.json').read_text())
        assert gate['status']=='READY_ALL_EIGHT_REAL_PROOFS_AND_PAIRED_ROLLOUT_PROTOCOL'
        assert gate.get('source_commits',{}).get(family,gate['source_commit'])==actual
        smoke_path=Path(gate['smoke_result_paths'][f'{family}-{suite}'])
        assert smoke_path.resolve().is_relative_to(a.root.resolve())
        smoke=json.loads(smoke_path.read_text())
        assert smoke['status']=='PASS_REAL_DATA_OPTIMIZER_VALIDATION_EMA_RELOAD'
        assert smoke['source_commit']==actual
        assert hashlib.sha256(smoke_path.read_bytes()).hexdigest()==gate['smoke_result_sha256'][f'{family}-{suite}']
    def progress(stage, **values):
        record={'stage':stage,'family':family,'suite':suite,**values}
        print(json.dumps(record),flush=True);write(out/'PROGRESS.json',record)
    progress('COMPOSE')
    with hydra.initialize_config_dir(version_base='1.2',config_dir=str(source/'egomimic/hydra_configs')):
        cfg=hydra.compose(config_name='train_zarr_cartesian',return_hydra_config=True,overrides=[
          'hydra/launcher=basic',f'+experiment=libero/{family}_{suite}_oat_dp_matched_s42',
          f'benchmark.dataset={a.root}/data/released/{suite}_N500.zarr',
          f'paths.output_dir={out}',f'paths.work_dir={source}',
          'runtime.slurm_requeue_owner=none',
        ])
    cfg.hydra.runtime.output_dir=str(out);HydraConfig.instance().set_config(cfg)
    full=OmegaConf.to_container(OmegaConf.masked_copy(cfg,[k for k in cfg if k!='hydra']),resolve=True)
    validate(full,family,suite)
    write(out/'FULL_CONFIG.json',full)
    assert Path(full['benchmark']['dataset']).is_dir()
    dependency=json.loads((a.root/'DEPENDENCY_READY_V1.json').read_text())
    assert dependency['status']=='PASS_PINNED_DEPENDENCIES_AND_RELEASED_ENCODER_CONSTRUCTION'
    assert dependency['robomimic']=='0.2.0'
    # Phase overrides preserve microbatch, accumulation, model, objective, and optimizer.
    if a.phase == 'smoke':
      with open_dict(cfg):
          cfg.trainer.max_steps=2;cfg.trainer.max_epochs=-1
          cfg.trainer.log_every_n_steps=1
          cfg.trainer.limit_val_batches=1;cfg.trainer.num_sanity_val_steps=0
          cfg.trainer.check_val_every_n_epoch=10
          cfg.val_at_end=True
          cfg.callbacks.model_checkpoint.every_n_epochs=None
          cfg.callbacks.model_checkpoint.every_n_train_steps=2
          cfg.callbacks.model_checkpoint.save_on_train_epoch_end=False
          cfg.callbacks.model_checkpoint.filename='epoch-{epoch:04d}-step-{step:09d}'
          cfg.callbacks.ema.final_checkpoint_path=str(out/'checkpoints/terminal-step000000002.ckpt')
          cfg.logger.wandb.id=f'{family}-{suite}-oat-pair-proof-s42-20261008-{a.attempt}'
          cfg.logger.wandb.name=cfg.logger.wandb.id
          cfg.logger.wandb.entity='rl2-group'
          cfg.logger.wandb.project='pushshapes-action-flow'
          cfg.logger.wandb.group='libero-four-suite-oat-dp-matched-20261008'
          cfg.logger.wandb.resume='never'
    if a.phase == 'full':
        with open_dict(cfg):
            cfg.val_at_end=True
            cfg.callbacks.ema.final_checkpoint_path=str(out/'checkpoints/terminal-epoch-{epoch:04d}-step-{step:09d}.ckpt')
            cfg.logger.wandb.id=f'{family}-{suite}-oat-matched-latent16-s42-20261008-full-{a.attempt}'
            cfg.logger.wandb.name=cfg.logger.wandb.id
            cfg.logger.wandb.entity='rl2-group';cfg.logger.wandb.project='pushshapes-action-flow'
            cfg.logger.wandb.group='libero-four-suite-oat-dp-matched-20261008';cfg.logger.wandb.resume='never'
    job=OmegaConf.masked_copy(cfg,[k for k in cfg if k!='hydra'])
    encoded=OmegaConf.to_yaml(job,resolve=True)
    config_sha=hashlib.sha256(encoded.encode()).hexdigest()
    with open_dict(cfg):
        cfg.evaluator=OmegaConf.create({'_target_':'egomimic.eval.libero_comparison_eval.LiberoComparisonEvaluator',
          'artifact_root':str(out/'validation-artifacts'),'source_commit':actual,'config_sha256':config_sha})
    OmegaConf.save(OmegaConf.masked_copy(cfg,[k for k in cfg if k!='hydra']),out/'SMOKE_CONFIG.yaml',resolve=True)
    progress('IMPORT_RUNTIME')
    import torch
    from egomimic.pl_utils.pl_model import ModelWrapper
    from egomimic.eval.energy_score import energy_score
    from egomimic.eval.libero_comparison_eval import distance
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    # Typed semantic-distance shape regression before dataset traversal or inference.
    scores=energy_score(torch.zeros(32,2,32,7),torch.zeros(2,32,7),distance_fn=distance)
    assert all(torch.isfinite(value).all() for value in scores.values())
    progress('CONSTRUCT_MODEL')
    model=ModelWrapper(config_tree=cfg,enable_grad_norm=False)
    optimizer=model.configure_optimizers()
    optimizer=optimizer.get('optimizer') if isinstance(optimizer,dict) else optimizer
    params=[p for g in optimizer.param_groups for p in g['params']]
    assert len(params)==len({id(p) for p in params})
    validate_optimizer_coverage(model.nets,params,family)
    counts={'total':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    rates=sorted({float(g['lr']) for g in optimizer.param_groups})
    if family=='dp': assert rates == [1e-5,5e-5], rates
    else: assert cfg.model.optimizer.lr == 1e-4 and cfg.model.scheduler.warmup_steps==8000
    progress('CONSTRUCTOR_PASS',parameters=counts,optimizer=type(optimizer).__name__,initial_group_lrs=rates)
    del model,optimizer,params;gc.collect();torch.cuda.empty_cache()
    from egomimic.trainHydra import train
    progress('VERIFY_REAL_EPISODE_SPLIT')
    train_view=hydra.utils.instantiate(cfg.data.train_datasets.libero_panda)
    valid_view=hydra.utils.instantiate(cfg.data.valid_datasets.libero_panda)
    train_ids=sorted(train_view.datasets);valid_ids=sorted(valid_view.datasets)
    expected_train, expected_valid = 450, 50
    assert len(train_ids)==expected_train and len(valid_ids)==expected_valid
    assert not set(train_ids)&set(valid_ids)
    assert len(set(train_ids)|set(valid_ids))==500
    train_paths={str(v.episode_path) for v in train_view.datasets.values()}
    valid_paths={str(v.episode_path) for v in valid_view.datasets.values()}
    assert not train_paths&valid_paths and len(train_paths)==expected_train and len(valid_paths)==expected_valid
    split={'train':train_ids,'valid':valid_ids,'train_episode_paths':sorted(train_paths),
           'valid_episode_paths':sorted(valid_paths),'ratio':.1,'seed':42,
           'dataset':str(train_view.resolver.folder_path)}
    write(out/'SPLIT.json',split)
    split_path=out/'SPLIT.json'
    split_path.with_suffix('.sha256').write_text(hashlib.sha256(split_path.read_bytes()).hexdigest()+'\n')
    del train_view,valid_view;gc.collect()
    seed_path=out/'SEEDS.json'
    from egomimic.eval.libero_comparison_eval import SEEDS
    write(seed_path,{'seeds':list(SEEDS)})
    if family == 'action_flow':
        diagnostic={'max_batches_per_rank':1,'artifact_root':str(out/'diagnostics'),
          'noise_seed_bank_path':str(seed_path),'noise_seed_bank_sha256':hashlib.sha256(seed_path.read_bytes()).hexdigest(),
          'raw_noise_levels':[0.,.25,.5,.75,1.],'max_samples':8,'jacobian_samples':2,
          'capture_activations':True,'activation_layer_map':{i:i for i in range(12)},'cknna_k':2,
          'validation_view':{'definition':'same_first_ordered_eight_conditions','split_manifest_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),'per_rank_batch_size':8,'world_size':1},
          'provenance':{'source_commit':actual,'latent_shape':[16,16],'sampler_steps':50,'action_horizon':32,'sampler':'euler','split_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),'metric_identity':'libero-h32-latent16-k2-all12-v1'},
          'native_error':{'enabled':True,'type':'raw_OSC7D_equal_semantic_blocks_mse','normalizer':'released_all_replay_limits'}}
        with open_dict(cfg): cfg.evaluator.diagnostic_config=OmegaConf.create(diagnostic)
    OmegaConf.save(OmegaConf.masked_copy(cfg,[k for k in cfg if k!='hydra']),out/'SMOKE_CONFIG.yaml',resolve=True)
    progress('REAL_OPTIMIZER_AND_VALIDATION')
    metrics,objects=train(cfg)
    trainer=objects['trainer'];model=objects['model'];data=objects['datamodule']
    if a.phase == 'full':
        write(out/'TRAINING_COMPLETED.json',{'status':'TRAINER_RETURNED','global_step':trainer.global_step,'source_commit':actual})
        return
    assert trainer.global_step==2
    values={k:float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k,v in metrics.items() if torch.is_tensor(v) or isinstance(v,(int,float))}
    required=['Valid/normalized_reconst_mse','Valid/reconst_mse','Valid/energy_score32','Valid/energy_accuracy32','Valid/energy_diversity32']
    if family == 'action_flow':
        required += ['Train/MSE','Optimizer/LR/Muon','Optimizer/LR/AdamW']
        assert any(key.startswith('Valid/ActionFlow/DenoisingTrajectory/') for key in values)
        assert any(key.startswith('Valid/ActionFlow/Alignment/CKA/') for key in values)
        assert any(key.startswith('Valid/ActionFlow/Alignment/CKNNA/') for key in values)
        assert any(key.startswith('Valid/ActionFlow/Alignment/FinalLatentCosine/') for key in values)
    for key in required: assert key in values and torch.isfinite(torch.tensor(values[key])),(key,values)
    train_ds=data.train_datasets['libero_panda']
    valid_ds=data.valid_datasets['libero_panda']
    train_ids=sorted(train_ds.datasets);valid_ids=sorted(valid_ds.datasets)
    expected_train, expected_valid = 450, 50
    assert len(train_ids)==expected_train and len(valid_ids)==expected_valid
    assert not set(train_ids)&set(valid_ids)
    assert len(set(train_ids)|set(valid_ids))==500
    assert train_ids == split['train'] and valid_ids == split['valid']
    checkpoint=out/'checkpoints/terminal-step000000002.ckpt'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert payload['global_step']==2 and payload['ema_num_updates']==2
    assert 'normalizer_state' in payload and 'ema_state_dict' in payload
    from egomimic.eval.checkpoint_loading import strict_load_pipeline_checkpoint
    strict_load_pipeline_checkpoint(model.model,payload,use_ema=False)
    strict_load_pipeline_checkpoint(model.model,payload,use_ema=True)
    assert all(torch.isfinite(v).all() for v in payload['state_dict'].values() if torch.is_tensor(v))
    write(out/'RESULT.json',{'status':'PASS_REAL_DATA_OPTIMIZER_VALIDATION_EMA_RELOAD','family':family,'suite':suite,'source_commit':actual,
      'parameters':counts,'metrics':values,'optimizer_steps':2,'batch_size':cfg.benchmark.batch_size,
      'accumulation':cfg.trainer.accumulate_grad_batches,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
      'checkpoint_path':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
      'normalizer_context':payload['normalizer_state']['benchmark_context'],'wandb_id':cfg.logger.wandb.id,
      'full_training_ready':False,'remaining_full_launch_gate':'launcher adoption, complete diagnostics, live storage, W&B visibility and matching immutable full bundle'})
    progress('PASS')

if __name__=='__main__': main()
