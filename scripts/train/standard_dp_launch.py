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
    smoke = phase in {'smoke', 'resume-smoke'}
    resume = phase in {'resume', 'resume-smoke'}
    start = int(os.environ['DP_RESUME_STEP']) if resume else 0
    checkpoint = os.environ['DP_RESUME_CHECKPOINT'] if resume else 'null'
    a=['--config-name=train_zarr_cartesian','+experiment=pusht/planar_chain_manual4919_standard_dp_retimed',
       'mode=train',f'ckpt_path={checkpoint}','++model.train_log_on_step=true',f'hydra.run.dir={output}',f'++paths.root_dir={output}',
       f'paths.output_dir={output}',f'paths.work_dir={os.environ["DP_REPO"]}',
       'launch_params.gpus_per_node=1','launch_params.nodes=1','trainer.devices=1',
       'trainer.num_nodes=1','trainer.strategy=auto','trainer.precision=bf16',
       f'trainer.max_steps={start + 2 if smoke else 80000}',
       f'trainer.val_check_interval={1 if smoke else 15000}',
       f'trainer.limit_train_batches={2 if smoke and not resume else 1.0}',
       'trainer.limit_val_batches=1','trainer.num_sanity_val_steps=0','trainer.log_every_n_steps=1',
       'norm_stats.sample_frac=0.05','norm_stats.save_cache_dir=null',f'norm_stats.precomputed_norm_path={norm}',
       '++callbacks.model_checkpoint.monitor=null','++callbacks.model_checkpoint.save_top_k=-1',
       f'++callbacks.model_checkpoint.save_last={str(smoke).lower()}',
       '++callbacks.model_checkpoint.every_n_epochs=null',
       f'++callbacks.model_checkpoint.every_n_train_steps={1 if smoke else 5000}',
       '++callbacks.model_checkpoint.train_time_interval=null','++callbacks.model_checkpoint.save_on_train_epoch_end=false',
       "++callbacks.model_checkpoint.filename='epoch-{epoch}-step-{step}'",
       '++logger.wandb.entity=rl2-group','++logger.wandb.project=pushshapes-planar-v2',
       f'++logger.wandb.id={(os.environ.get("DP_SMOKE_WANDB_ID", os.environ["DP_WANDB_ID"]+"-smoke") if smoke else os.environ["DP_WANDB_ID"])}',
       f'++logger.wandb.name={(os.environ.get("DP_SMOKE_WANDB_ID", os.environ["DP_WANDB_ID"]+"-smoke") if smoke else os.environ["DP_WANDB_ID"])}',
       f'++logger.wandb.resume={"must" if resume and not smoke else "never"}','++logger.wandb.group=standard-dp-chain-manual4919-five-rate-20261007',
       '++logger.wandb.tags=[standard-dp,chain-only,manual4919,uniform-five-rate,interpolation-only,bf16,world1]',
       'evaluator.energy_score_max_batches_per_rank=1']
    # This optional evaluator field is present in the native constructor, not YAML.
    a[-1] = '++'+a[-1]
    if smoke:
        a += ['evaluator.limit_val_batches=1']
    if phase != 'normalize':
        run_id = os.environ.get('DP_SMOKE_WANDB_ID', os.environ['DP_WANDB_ID']+'-smoke') if smoke else os.environ['DP_WANDB_ID']
        provenance = {
            'source_commit': os.environ['DP_EXPECTED_HEAD'],
            'normalization_sha256': os.environ['DP_NORM_SHA256'],
            'split_manifest_sha256': os.environ['DP_SPLIT_SHA256'],
            'resolved_config_path': str(pathlib.Path(output)/'.hydra/config.yaml'),
            'wandb.entity': 'rl2-group', 'wandb.project': 'pushshapes-planar-v2', 'wandb.run_id': run_id,
            'dataset_content.manifest_path': os.environ['DP_CONTENT_MANIFEST'],
            'dataset_content.manifest_sha256': os.environ['DP_CONTENT_SHA256'],
            'dataset_content.aggregate_sha256': os.environ['DP_CONTENT_AGGREGATE_SHA256'],
        }
        a += [f'++evaluator.energy_score_provenance.{key}={value}' for key,value in provenance.items()]
        a += [f'++evaluator.energy_score_validation_view.split_manifest_sha256={os.environ["DP_SPLIT_SHA256"]}']

    if phase == 'normalize':
        a = [x for x in a if not x.startswith('++logger.') ]
        a += ['norm_stats_only=true','trainer.accelerator=cpu','trainer.devices=1','~logger',
              f'norm_stats.save_cache_dir={output}','norm_stats.precomputed_norm_path=null']
    return a

def verify_resume_checkpoint():
    """Bind continuation to payload state; filenames never establish progress."""
    import torch
    path = pathlib.Path(os.environ['DP_RESUME_CHECKPOINT']).resolve()
    assert path.is_file()
    expected = os.environ['DP_RESUME_CHECKPOINT_SHA256']
    assert len(expected) == 64 and all(c in '0123456789abcdef' for c in expected)
    assert digest(path) == expected
    checkpoint = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    step = int(checkpoint['global_step'])
    assert 0 < step < 80000
    assert checkpoint['optimizer_states'] and len(checkpoint['lr_schedulers']) == 1
    assert checkpoint['state_dict'] and checkpoint.get('data_context')
    result = {'path': str(path), 'sha256': expected, 'global_step': step,
              'optimizer_states': len(checkpoint['optimizer_states']),
              'lr_schedulers': len(checkpoint['lr_schedulers'])}
    os.environ['DP_RESUME_STEP'] = str(step)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['preflight','normalize','smoke','full','resume-smoke','resume'])
    args=p.parse_args();repo=pathlib.Path(os.environ['DP_REPO']);task=pathlib.Path(os.environ['DP_TASK'])
    head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    assert head==os.environ['DP_EXPECTED_HEAD'];assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=all'],text=True).strip()
    assert digest(os.environ['DP_SPLIT_MANIFEST'])==os.environ['DP_SPLIT_SHA256']
    os.chdir(repo);norm=str(task/'normalization/norm_stats')
    resume_identity = verify_resume_checkpoint() if os.environ.get('DP_RESUME_CHECKPOINT') else None
    if args.mode in {'resume', 'resume-smoke'}: assert resume_identity is not None
    if args.mode=='preflight':
        from omegaconf import OmegaConf
        from hydra.utils import instantiate
        dest=task/'preflight-v1';dest.mkdir(exist_ok=True)
        records={'resume_identity': resume_identity}
        phases = ['normalize','smoke','full'] + (['resume-smoke','resume'] if resume_identity else [])
        for phase in phases:
            out=task/({'normalize':'normalization','smoke':'smoke-v1','full':'full-v1','resume-smoke':'resume-smoke-v1','resume':'resume-v1'}[phase])
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
            assert c.model.train_log_on_step is True
            if phase in {'resume', 'resume-smoke'}:
                assert str(c.ckpt_path) == resume_identity['path']
                assert int(c.trainer.max_steps) == (resume_identity['global_step'] + 2 if phase == 'resume-smoke' else 80000)
                assert str(c.logger.wandb.resume) == ('must' if phase == 'resume' else 'never')
                assert float(c.trainer.limit_train_batches) == 1.0
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
    out=task/({'normalize':'normalization','smoke':'smoke-v1','full':'full-v1','resume-smoke':'resume-smoke-v1','resume':'resume-v1'}[args.mode]);assert not out.exists(),out
    if args.mode!='normalize':assert pathlib.Path(norm,'norm_stats.json').is_file()
    if args.mode=='resume':
        proof=json.loads((task/'resume-smoke-v1/SMOKE_RESULT.json').read_text())
        assert proof['status']=='passed' and proof['repo_head']==head
        assert proof['resume_start_step']==resume_identity['global_step']
        assert proof['resume_checkpoint_sha256']==resume_identity['sha256']
    if args.mode=='full':
        proof=json.loads((task/'smoke-v1/SMOKE_RESULT.json').read_text());assert proof['status']=='passed'
        assert proof['repo_head']==head
    run([sys.executable,'-m','egomimic.trainHydra',*arguments(args.mode,str(out),norm)])
    if args.mode in {'smoke', 'resume-smoke'}:
        records=json.loads((task/'PREFLIGHT_RESULT_V1.json').read_text())
        run([sys.executable,'scripts/train/verify_planar_training_smoke.py',str(out),
             '--expected-head',head,'--world-size','1','--parameter-count',str(records['parameters']),
             '--expected-name','planar_chain_manual4919_standard_dp_retimed_h16',
             '--single-domain','pushshapes_sim_chain_gripper',
             '--start-step',str(resume_identity['global_step'] if args.mode == 'resume-smoke' else 0)])
if __name__=='__main__':main()
