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


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source-commit',required=True);p.add_argument('--index',type=int,required=True)
    a=p.parse_args()
    assert os.environ.get('SLURM_STEP_ID'), 'scheduled srun only'
    source=Path(__file__).resolve().parents[1]
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    assert actual == a.source_commit
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()
    suite=SUITES[a.index//2];family=('dp','action_flow')[a.index%2]
    out=a.root/'proof-v2'/f'{family}-{suite}'
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'RESULT.json').exists()
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
    # Phase overrides preserve microbatch, accumulation, model, objective, and optimizer.
    with open_dict(cfg):
        cfg.trainer.max_steps=2;cfg.trainer.max_epochs=-1
        cfg.trainer.limit_val_batches=1;cfg.trainer.num_sanity_val_steps=0
        cfg.trainer.check_val_every_n_epoch=10
        cfg.val_at_end=True
        cfg.callbacks.model_checkpoint.every_n_epochs=None
        cfg.callbacks.model_checkpoint.every_n_train_steps=2
        cfg.callbacks.model_checkpoint.save_on_train_epoch_end=False
        cfg.callbacks.model_checkpoint.filename='epoch-{epoch:04d}-step-{step:09d}'
        cfg.callbacks.ema.final_checkpoint_path=str(out/'checkpoints/terminal-step000000002.ckpt')
        cfg.logger.wandb.id=f'{family}-{suite}-oat-pair-proof-s42-20261008-v2'
        cfg.logger.wandb.name=cfg.logger.wandb.id
        cfg.logger.wandb.entity='rl2-group'
        cfg.logger.wandb.project='pushshapes-action-flow'
        cfg.logger.wandb.group='libero-four-suite-oat-dp-matched-20261008'
        cfg.logger.wandb.resume='never'
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
    assert {id(p) for p in params}=={id(p) for p in model.nets.parameters() if p.requires_grad}
    counts={'total':sum(p.numel() for p in model.parameters()),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    rates=sorted({float(g['lr']) for g in optimizer.param_groups})
    if family=='dp': assert rates == [1e-5,5e-5], rates
    else: assert cfg.model.optimizer.lr == 1e-4 and cfg.model.scheduler.warmup_steps==8000
    progress('CONSTRUCTOR_PASS',parameters=counts,optimizer=type(optimizer).__name__,initial_group_lrs=rates)
    del model,optimizer,params;gc.collect();torch.cuda.empty_cache()
    from egomimic.trainHydra import train
    progress('REAL_OPTIMIZER_AND_VALIDATION')
    metrics,objects=train(cfg)
    trainer=objects['trainer'];model=objects['model'];data=objects['datamodule']
    assert trainer.global_step==2
    values={k:float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k,v in metrics.items() if torch.is_tensor(v) or isinstance(v,(int,float))}
    required=['Valid/normalized_reconst_mse','Valid/reconst_mse','Valid/energy_score32','Valid/energy_accuracy32','Valid/energy_diversity32']
    for key in required: assert key in values and torch.isfinite(torch.tensor(values[key])),(key,values)
    train_ds=data.train_datasets['libero_panda']
    valid_ds=data.valid_datasets['libero_panda']
    train_ids=sorted(train_ds.datasets);valid_ids=sorted(valid_ds.datasets)
    assert len(train_ids)==450 and len(valid_ids)==50
    assert not set(train_ids)&set(valid_ids)
    assert len(set(train_ids)|set(valid_ids))==500
    write(out/'SPLIT.json',{'train':train_ids,'valid':valid_ids,'ratio':.1,'seed':42,'dataset':str(train_ds.resolver.folder_path)})
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
