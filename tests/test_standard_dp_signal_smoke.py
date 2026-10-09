import runpy
from pathlib import Path
import pytest

def api():return runpy.run_path(str(Path(__file__).parents[1]/'scripts/train/standard_dp_launch.py'))

def bind(monkeypatch):
    root=Path(__file__).parents[1]
    for k,v in dict(DP_COTRAIN_STANDARD='usocket_chain_manual4919_af_obs_multiplier_261m_v1',DP_REPO=str(root),DP_WANDB_ID='fixture',DP_EXPECTED_HEAD='a'*40,DP_NORM_SHA256='b'*64,DP_SPLIT_SHA256='c'*64,DP_CONTENT_MANIFEST='/content',DP_CONTENT_SHA256='d'*64,DP_CONTENT_AGGREGATE_SHA256='e'*64,ICE_REQUEUE_OWNER='runner',ICE_CHILD_REQUEUE_DISABLED='1',SLURM_JOB_ID='123',SLURM_RESTART_COUNT='0').items():monkeypatch.setenv(k,v)
    monkeypatch.delenv('DP_SUPERVISED',raising=False)

def test_signal_smoke_only_changes_runtime_callback(monkeypatch,tmp_path):
    from hydra import compose,initialize_config_dir
    bind(monkeypatch);m=api();monkeypatch.delenv('DP_SIGNAL_AUDIT_SMOKE',raising=False)
    before=m['arguments']('smoke',str(tmp_path/'smoke'),'/norm')
    monkeypatch.setenv('DP_SIGNAL_AUDIT_SMOKE','1');after=m['arguments']('smoke',str(tmp_path/'smoke'),'/norm')
    assert after[:len(before)]==before
    assert after[len(before):]==['++runtime.slurm_requeue_owner=runner','++runtime.slurm_save_signal=SIGUSR2','++runtime.slurm_signal_checkpoint_dir=${paths.output_dir}/checkpoints']
    root=Path(__file__).parents[1]
    with initialize_config_dir(config_dir=str(root/'egomimic/hydra_configs'),version_base=None):
        cfg=compose(config_name='train_zarr_cartesian',overrides=after[1:])
    assert cfg.runtime.slurm_requeue_owner=='runner'
    assert cfg.trainer.max_steps==2 and cfg.trainer.precision=='bf16' and cfg.planar.batch_size==32
    assert cfg.callbacks.get('ema') is None
    assert cfg.model.pipeline.stages[2].conditioning_input=='retiming_multiplier'
    assert not m['signal_audit_smoke']('normalize') and not m['signal_audit_smoke']('full')

@pytest.mark.parametrize('key,value',[('ICE_REQUEUE_OWNER','child'),('ICE_CHILD_REQUEUE_DISABLED','0'),('SLURM_JOB_ID',''),('SLURM_RESTART_COUNT','1')])
def test_wrong_owner_or_restart_refused(monkeypatch,key,value):
    bind(monkeypatch);monkeypatch.setenv('DP_SIGNAL_AUDIT_SMOKE','1');monkeypatch.setenv(key,value)
    with pytest.raises(AssertionError):api()['signal_audit_smoke']('smoke')
