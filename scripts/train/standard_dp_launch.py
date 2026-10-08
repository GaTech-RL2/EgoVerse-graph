#!/usr/bin/env python3
"""Typed single-source extension for the canonical Planar Standard-DP launcher.

All phases share one actual Hydra argument builder. Existing co-train launch
behavior is unaffected. Portable hosts bind only the explicit environment.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, subprocess, sys

def run(argv, **kwargs):
    try:
        return subprocess.run(argv, check=True, **kwargs)
    except subprocess.CalledProcessError as error:
        if error.stdout: print(error.stdout, file=sys.stderr)
        if error.stderr: print(error.stderr, file=sys.stderr)
        raise

def digest(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def arguments(phase, output, norm):
    smoke = phase == 'smoke'
    a=['--config-name=train_zarr_cartesian','+experiment=pusht/planar_chain_manual4919_standard_dp_retimed',
       'mode=train','ckpt_path=null',f'hydra.run.dir={output}',f'++paths.root_dir={output}',
       f'paths.output_dir={output}',f'paths.work_dir={os.environ["DP_REPO"]}',
       'launch_params.gpus_per_node=1','launch_params.nodes=1','trainer.devices=1',
       'trainer.num_nodes=1','trainer.strategy=auto','trainer.precision=bf16',
       f'trainer.max_steps={2 if smoke else 80000}',
       f'trainer.val_check_interval={1 if smoke else 15000}',
       f'trainer.limit_train_batches={2 if smoke else 1.0}',
       'trainer.limit_val_batches=1','trainer.num_sanity_val_steps=0','trainer.log_every_n_steps=1',
       'norm_stats.sample_frac=0.05','norm_stats.save_cache_dir=null',f'norm_stats.precomputed_norm_path={norm}',
       '++callbacks.model_checkpoint.monitor=null','++callbacks.model_checkpoint.save_top_k=-1',
       f'++callbacks.model_checkpoint.save_last={str(smoke).lower()}',
       '++callbacks.model_checkpoint.every_n_epochs=null',
       f'++callbacks.model_checkpoint.every_n_train_steps={1 if smoke else 5000}',
       '++callbacks.model_checkpoint.train_time_interval=null','++callbacks.model_checkpoint.save_on_train_epoch_end=false',
       "++callbacks.model_checkpoint.filename='epoch-{epoch}-step-{step}'",
       '++logger.wandb.entity=rl2-group','++logger.wandb.project=pushshapes-planar-v2',
       f'++logger.wandb.id={os.environ["DP_WANDB_ID"]+ ("-smoke" if smoke else "")}',
       f'++logger.wandb.name={os.environ["DP_WANDB_ID"]+ ("-smoke" if smoke else "")}',
       '++logger.wandb.resume=never','++logger.wandb.group=standard-dp-chain-manual4919-five-rate-20261007',
       '++logger.wandb.tags=[standard-dp,chain-only,manual4919,uniform-five-rate,interpolation-only,bf16,world1]',
       'evaluator.energy_score_max_batches_per_rank=1']
    # This optional evaluator field is present in the native constructor, not YAML.
    a[-1] = '++'+a[-1]
    if phase == 'normalize':
        a = [x for x in a if not x.startswith('++logger.') ]
        a += ['norm_stats_only=true','trainer.accelerator=cpu','trainer.devices=1','~logger',
              f'norm_stats.save_cache_dir={output}','norm_stats.precomputed_norm_path=null']
    return a

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['preflight','normalize','smoke','full'])
    args=p.parse_args();repo=pathlib.Path(os.environ['DP_REPO']);task=pathlib.Path(os.environ['DP_TASK'])
    head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    assert head==os.environ['DP_EXPECTED_HEAD'];assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all'],text=True).strip()
    assert digest(os.environ['DP_SPLIT_MANIFEST'])==os.environ['DP_SPLIT_SHA256']
    os.chdir(repo);norm=str(task/'normalization/norm_stats')
    if args.mode=='preflight':
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        dest=task/'preflight-v1';dest.mkdir(exist_ok=True)
        records={}
        for phase in ['normalize','smoke','full']:
            out=task/({'normalize':'normalization','smoke':'smoke-v1','full':'full-v1'}[phase])
            command=[sys.executable,'-m','egomimic.trainHydra',*arguments(phase,str(out),norm),'--cfg','job','--resolve']
            cp=run(command,capture_output=True,text=True);path=dest/f'{phase}.yaml';path.write_text(cp.stdout)
            c=OmegaConf.load(path)
            assert str(c.data.train_datasets.pushshapes_sim_chain_gripper.resolver.folder_path)==os.environ['DP_DATASET_DIR']
            assert list(c.planar.retiming_rates)==[1.,1.25,1.5,1.75,2.]
            assert c.model.pipeline.stages[3].condition_input_dim==67
            assert c.planar.batch_size==16 and c.planar.observation_horizon==1
            assert c.data.train_datasets.pushshapes_sim_chain_gripper.expected_train_episode_count==4870
            assert c.data.valid_datasets.pushshapes_sim_chain_gripper.expected_valid_episode_count==49
            assert c.callbacks.model_checkpoint.save_top_k==-1
            assert c.paths.output_dir==str(out)
            ev=instantiate(c.evaluator);assert len(ev.seeds)==32
            assert ev.energy_score_distance is None
            records[phase]={'config_sha256':digest(path),'argv':command}
        c=OmegaConf.load(dest/'full.yaml');model=instantiate(c.model.pipeline)
        count=sum(x.numel() for x in model.nets.parameters());records['parameters']=count
        # Prove real native schema/virtual sampling before normalization.
        resolver=instantiate(c.data.train_datasets.pushshapes_sim_chain_gripper.resolver)
        split=json.loads(pathlib.Path(os.environ['DP_SPLIT_MANIFEST']).read_text())['domains']['pushshapes_sim_chain_gripper']
        from egomimic.rldb.zarr.zarr_dataset_multi import ZarrDataset
        episode=pathlib.Path(os.environ['DP_DATASET_DIR'])/split['train_ids'][0]
        if not episode.exists(): episode=episode.with_suffix('.zarr')
        leaf=ZarrDataset(episode,key_map=resolver.key_map,transform_list=resolver.transform_list,
                         embodiment_override='pushshapes_sim_chain_gripper')
        leaf.require_ordered_samples()
        samples=[leaf[i] for i in range(5)]
        import torch
        assert all(x['actions'].shape==(16,5) for x in samples)
        assert all(torch.isfinite(x['actions']).all() for x in samples)
        assert [float(x['retiming_rate'][0]) for x in samples]==[1.,1.25,1.5,1.75,2.]
        assert all(torch.equal(samples[0]['state_agent_obj'],x['state_agent_obj']) for x in samples)
        assert leaf.frame_index_at(4)==0 and leaf.frame_index_at(5)==1
        records.update(status='PASS',source_head=head,real_native_sample_contract='PASS')
        (task/'PREFLIGHT_RESULT_V1.json').write_text(json.dumps(records,indent=2)+'\n');return
    out=task/({'normalize':'normalization','smoke':'smoke-v1','full':'full-v1'}[args.mode]);assert not out.exists(),out
    if args.mode!='normalize':assert pathlib.Path(norm,'norm_stats.json').is_file()
    if args.mode=='full':
        proof=json.loads((task/'smoke-v1/SMOKE_RESULT.json').read_text());assert proof['status']=='passed'
        assert proof['repo_head']==head
    run([sys.executable,'-m','egomimic.trainHydra',*arguments(args.mode,str(out),norm)])
    if args.mode=='smoke':
        records=json.loads((task/'PREFLIGHT_RESULT_V1.json').read_text())
        run([sys.executable,'scripts/train/verify_planar_training_smoke.py',str(out),
             '--expected-head',head,'--world-size','1','--parameter-count',str(records['parameters']),
             '--expected-name','planar_chain_manual4919_standard_dp_retimed_h16',
             '--single-domain','pushshapes_sim_chain_gripper'])
if __name__=='__main__':main()
