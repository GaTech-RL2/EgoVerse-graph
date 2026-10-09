"""Exact shared phase composition protects the maintained Standard-DP entry."""
import os, runpy
from pathlib import Path
from hydra import compose, initialize_config_dir

def test_exact_standard_dp_phases(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    for key,value in dict(DP_REPO=str(root),DP_DATASET_DIR='/data/chain',DP_SPLIT_MANIFEST='/data/split.json',DP_SPLIT_SHA256='a'*64,DP_WANDB_ID='fixture',DP_SMOKE_WANDB_ID='fixture-smoke-retry',DP_EXPECTED_HEAD='a'*40,DP_NORM_SHA256='b'*64,DP_CONTENT_MANIFEST='/data/content.json',DP_CONTENT_SHA256='c'*64,DP_CONTENT_AGGREGATE_SHA256='d'*64,DP_RESUME_CHECKPOINT='/checkpoint/epoch-epoch=0-step-step=30000.ckpt',DP_RESUME_STEP='30000').items():
        monkeypatch.setenv(key,value)
    builder=runpy.run_path(str(root/'scripts/train/standard_dp_launch.py'))['arguments']
    for phase in ['normalize','smoke','full','resume-smoke','resume']:
        argv=builder(phase,'/output/'+phase,'/normalization')
        with initialize_config_dir(config_dir=str(root/'egomimic/hydra_configs'),version_base=None):
            cfg=compose(config_name='train_zarr_cartesian',overrides=argv[1:])
            assert cfg.data.train_datasets.pushshapes_sim_chain_gripper.expected_train_episode_count==4870
            assert cfg.data.valid_datasets.pushshapes_sim_chain_gripper.expected_valid_episode_count==49
            assert cfg.model.pipeline.stages[3].condition_input_dim==67
            assert cfg.planar.batch_size==16
            assert cfg.model.train_log_on_step is True
            assert cfg.callbacks.model_checkpoint.save_top_k==-1
            if phase=='smoke':
                assert cfg.logger.wandb.id=='fixture-smoke-retry'
                assert cfg.evaluator.limit_val_batches==1
                assert cfg.evaluator.energy_score_provenance.wandb.run_id=='fixture-smoke-retry'
            if phase=='full':
                assert cfg.logger.wandb.id=='fixture'
                assert cfg.evaluator.limit_val_batches==1.0
                assert cfg.evaluator.energy_score_provenance.normalization_sha256=='b'*64
            if phase=='normalize': assert '~logger' in argv and not any(x.startswith('++logger.') for x in argv)

            if phase in {'resume', 'resume-smoke'}:
                assert cfg.ckpt_path=='/checkpoint/epoch-epoch=0-step-step=30000.ckpt'
                assert cfg.trainer.max_steps==(30002 if phase=='resume-smoke' else 80000)
                assert cfg.trainer.limit_train_batches==1.0
                assert cfg.logger.wandb.resume==('never' if phase=='resume-smoke' else 'must')
                assert cfg.logger.wandb.id==('fixture-smoke-retry' if phase=='resume-smoke' else 'fixture')
                assert cfg.evaluator.limit_val_batches==(1 if phase=='resume-smoke' else 1.0)


def test_multiplier_profile_all_phases(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    env = dict(DP_COTRAIN_STANDARD='usocket_chain_manual4919_af_obs_multiplier_261m_v1',
               DP_REPO=str(root), DP_WANDB_ID='multiplier-fixture', DP_EXPECTED_HEAD='a'*40,
               DP_NORM_SHA256='b'*64, DP_SPLIT_SHA256='c'*64, DP_CONTENT_MANIFEST='/content',
               DP_CONTENT_SHA256='d'*64, DP_CONTENT_AGGREGATE_SHA256='e'*64,
               DP_RESUME_STEP='20000', DP_RESUME_CHECKPOINT='/checkpoint')
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    builder = runpy.run_path(str(root/'scripts/train/standard_dp_launch.py'))['arguments']
    for phase in ['normalize', 'smoke', 'full', 'resume-smoke', 'resume']:
        argv = builder(phase, '/output/'+phase, '/normalization')
        with initialize_config_dir(config_dir=str(root/'egomimic/hydra_configs'), version_base=None):
            c = compose(config_name='train_zarr_cartesian', overrides=argv[1:])
            assert c.name == 'planar_uc_manual4919_dp_261m_af_obs_multiplier'
            assert c.planar.batch_size == 32
            assert c.model.pipeline.stages[5].condition_input_dim == 128
            assert c.model.pipeline.stages[2].conditioning_input == 'retiming_multiplier'
            assert dict(c.model.pipeline.stages[6].active_action_dims_by_embodiment) == {
                'pushshapes_sim_u_socket': 4, 'pushshapes_sim_chain_gripper': 5}
            assert c.callbacks.get("ema") is None
            assert c.norm_stats.sample_frac == 1.0
            assert c.trainer.precision == 'bf16'
            assert c.callbacks.model_checkpoint.save_top_k == -1
