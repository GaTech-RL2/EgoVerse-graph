import runpy
from pathlib import Path
import pytest

@pytest.fixture
def api():
    return runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/train/standard_dp_launch.py'))

def bind(monkeypatch, restart='0'):
    for k,v in dict(DP_SUPERVISED='1', ICE_REQUEUE_OWNER='runner', ICE_CHILD_REQUEUE_DISABLED='1', DP_SUPERVISED_JOB_ID='123', SLURM_JOB_ID='123', SLURM_RESTART_COUNT=restart).items(): monkeypatch.setenv(k,v)
    for k in ['ICE_RESUME_CHECKPOINT','ICE_RESUME_CHECKPOINT_SHA256','ICE_RESUME_GLOBAL_STEP']: monkeypatch.delenv(k,raising=False)

def checkpoint(monkeypatch, out, path=None):
    p=path or out/'checkpoints'/'epoch-0-step-5000.ckpt';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
    for k,v in dict(ICE_RESUME_CHECKPOINT=str(p), ICE_RESUME_CHECKPOINT_SHA256='a'*64, ICE_RESUME_GLOBAL_STEP='5000').items(): monkeypatch.setenv(k,v)
    return p

def test_fresh_and_collision(api, monkeypatch, tmp_path):
    bind(monkeypatch);out=tmp_path/'full';api['guard_output']('full',out)
    out.mkdir()
    with pytest.raises(AssertionError, match='collision'): api['guard_output']('full',out)

def test_bound_restart(api,monkeypatch,tmp_path):
    bind(monkeypatch,'1');out=tmp_path/'full';p=checkpoint(monkeypatch,out)
    api['guard_output']('full',out); assert api['supervised_checkpoint'](out)==str(p)

@pytest.mark.parametrize('failure',['missing','foreign-path','wrong-owner','wrong-job','bad-sha','terminal-step'])
def test_restart_fails_closed(api,monkeypatch,tmp_path,failure):
    bind(monkeypatch,'1');out=tmp_path/'full';out.mkdir()
    if failure!='missing': checkpoint(monkeypatch,out,tmp_path/'foreign.ckpt' if failure=='foreign-path' else None)
    if failure=='wrong-owner': monkeypatch.setenv('ICE_REQUEUE_OWNER','child')
    if failure=='wrong-job': monkeypatch.setenv('DP_SUPERVISED_JOB_ID','999')
    if failure=='bad-sha': monkeypatch.setenv('ICE_RESUME_CHECKPOINT_SHA256','a'*63)
    if failure=='terminal-step': monkeypatch.setenv('ICE_RESUME_GLOBAL_STEP','80000')
    with pytest.raises((AssertionError,ValueError)): api['guard_output']('full',out)

def test_legacy_remains_fresh_only(api,monkeypatch,tmp_path):
    monkeypatch.delenv('DP_SUPERVISED',raising=False);out=tmp_path/'full';out.mkdir()
    with pytest.raises(AssertionError): api['guard_output']('full',out)

def test_supervised_full_arguments_keep_science(api,monkeypatch,tmp_path):
    from hydra import compose,initialize_config_dir
    root=Path(__file__).resolve().parents[1]
    for k,v in dict(DP_COTRAIN_STANDARD='usocket_chain_manual4919_af_obs_multiplier_261m_v1',DP_REPO=str(root),DP_WANDB_ID='fixture',DP_EXPECTED_HEAD='a'*40,DP_NORM_SHA256='b'*64,DP_SPLIT_SHA256='c'*64,DP_CONTENT_MANIFEST='/content',DP_CONTENT_SHA256='d'*64,DP_CONTENT_AGGREGATE_SHA256='e'*64).items():monkeypatch.setenv(k,v)
    bind(monkeypatch);out=tmp_path/'full'
    first=api['arguments']('full',str(out),'/norm')
    bind(monkeypatch,'1');path=checkpoint(monkeypatch,out)
    resumed=api['arguments']('full',str(out),'/norm')
    for argv in [first,resumed]:
        with initialize_config_dir(config_dir=str(root/'egomimic/hydra_configs'),version_base=None):
            cfg=compose(config_name='train_zarr_cartesian',overrides=argv[1:])
            assert cfg.runtime.slurm_requeue_owner=='runner'
            assert cfg.runtime.slurm_signal_checkpoint_dir==str(out/'checkpoints')
            assert cfg.trainer.max_steps==80000 and cfg.trainer.limit_train_batches==1.0
            assert cfg.trainer.precision=='bf16' and cfg.planar.batch_size==32
            assert cfg.callbacks.get('ema') is None
            assert cfg.model.pipeline.stages[2].conditioning_input=='retiming_multiplier'
            assert cfg.ckpt_path==(str(path) if argv==resumed else None)
            assert cfg.logger.wandb.resume==('must' if argv==resumed else 'never')
