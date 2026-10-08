"""Exact shared phase composition protects the maintained Standard-DP entry."""
import os, runpy
from pathlib import Path
from hydra import compose, initialize_config_dir

def test_exact_standard_dp_phases(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    for key,value in dict(DP_REPO=str(root),DP_DATASET_DIR='/data/chain',DP_SPLIT_MANIFEST='/data/split.json',DP_SPLIT_SHA256='a'*64,DP_WANDB_ID='fixture',DP_SMOKE_WANDB_ID='fixture-smoke-retry',DP_EXPECTED_HEAD='a'*40,DP_NORM_SHA256='b'*64,DP_CONTENT_MANIFEST='/data/content.json',DP_CONTENT_SHA256='c'*64,DP_CONTENT_AGGREGATE_SHA256='d'*64).items():
        monkeypatch.setenv(key,value)
    builder=runpy.run_path(str(root/'scripts/train/standard_dp_launch.py'))['arguments']
    for phase in ['normalize','smoke','full']:
        argv=builder(phase,'/output/'+phase,'/normalization')
        with initialize_config_dir(config_dir=str(root/'egomimic/hydra_configs'),version_base=None):
            cfg=compose(config_name='train_zarr_cartesian',overrides=argv[1:])
            assert cfg.data.train_datasets.pushshapes_sim_chain_gripper.expected_train_episode_count==4870
            assert cfg.data.valid_datasets.pushshapes_sim_chain_gripper.expected_valid_episode_count==49
            assert cfg.model.pipeline.stages[3].condition_input_dim==67
            assert cfg.planar.batch_size==16
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
