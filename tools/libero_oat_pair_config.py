"""Compose and audit explicit OAT-DP/Action-Flow comparisons without Torch/data IO."""
import argparse
import json
from pathlib import Path
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

SUITES = ('libero10', 'libero_spatial', 'libero_object', 'libero_goal')

def validate(cfg, family, suite):
    assert cfg['benchmark']['suite'] == suite
    assert cfg['benchmark']['horizon'] == 32
    stages = cfg['model']['pipeline']['stages']
    if family == 'action_flow':
        assert cfg['model']['action_horizon'] == 32
        assert cfg['model']['num_latent_tokens'] == 16
        assert cfg['model']['latent_dim'] == 16
        assert stages[2]['num_tokens'] == 16
        for stage in stages:
            if 'num_latent_tokens' in stage:
                assert stage['num_latent_tokens'] == 16
    else:
        assert stages[2]['action_horizon'] == stages[3]['policy']['action_horizon'] == 32
    assert cfg['benchmark']['n_obs_steps'] == 2
    assert cfg['benchmark']['n_action_steps'] == 16
    valid_ratio = .1
    assert cfg['data']['train_datasets']['libero_panda']['valid_ratio'] == valid_ratio
    assert cfg['data']['valid_datasets']['libero_panda']['valid_ratio'] == valid_ratio
    assert cfg['callbacks']['model_checkpoint']['save_top_k'] == -1
    batch = cfg['benchmark']['batch_size'] * cfg['trainer']['accumulate_grad_batches']
    assert cfg['callbacks']['batch_budget']['global_batch_size'] == batch
    if family == 'action_flow':
        assert batch == 32 and cfg['trainer']['max_steps'] == 80000
        assert cfg['trainer']['gradient_clip_val'] == 3.0
        assert cfg['callbacks']['ema']['_target_'].endswith('ActionFlowFixedEMACallback')
        assert cfg['callbacks']['ema']['decay'] == .9978
        assert cfg['callbacks']['ema']['use_warmup'] is False
        assert cfg['model']['optimizer_named_parameters'] is True
        assert cfg['model']['optimizer']['_target_'].endswith('ReleasedUniteCompositeOptimizer')
        assert cfg['model']['optimizer']['lr'] == .0001
        assert cfg['model']['scheduler']['warmup_steps'] == 8000
        assert cfg['model']['training_behavior']['_target_'].endswith('oat_matched_training.OATObservationActionFlowTrainingBehavior')
        assert cfg['model']['pipeline']['stages'][-1]['action_velocity_weight'] == 1.
    else:
        assert batch == 1024 and cfg['trainer']['max_epochs'] == 5001
        assert cfg['trainer']['gradient_clip_val'] == 1.0
        assert cfg['callbacks']['ema']['decay'] == .9999
        assert cfg['callbacks']['ema']['use_warmup'] is True
        assert cfg['model']['training_behavior']['learning_rate'] == 5e-5
        assert cfg['model']['training_behavior']['obs_enc_lr'] == 1e-5
    return batch

def compose_all(source, output):
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with hydra.initialize_config_dir(version_base='1.2', config_dir=str(source / 'egomimic/hydra_configs')):
        for suite in SUITES:
            pair = {}
            for family in ('dp', 'action_flow'):
                cfg = hydra.compose(config_name='train_zarr_cartesian', return_hydra_config=True, overrides=[
                    'hydra/launcher=basic', f'+experiment=libero/{family}_{suite}_oat_dp_matched_s42',
                    'benchmark.dataset=/NEVER_LAUNCH_UNBOUND_DATA',
                    'paths.output_dir=/NEVER_LAUNCH_UNBOUND_OUTPUT',
                    f'paths.work_dir={source}',
                ])
                cfg.hydra.runtime.output_dir = '/NEVER_LAUNCH_UNBOUND_OUTPUT'
                HydraConfig.instance().set_config(cfg)
                # Hydra sweep-only job.num is unset in a single-job envelope.
                job = OmegaConf.masked_copy(cfg, [k for k in cfg if k != 'hydra'])
                data = OmegaConf.to_container(job, resolve=True)
                batch = validate(data, family, suite)
                (output / f'{family}-{suite}.json').write_text(json.dumps(data, indent=2)+'\n')
                pair[family] = data
                results.append({'suite': suite, 'family': family, 'effective_batch': batch, 'status':'PASS_CONFIG_ONLY'})
            for key in ('normalizer','norm_stats','seed'):
                assert pair['dp'][key] == pair['action_flow'][key], (suite,key)
            for split in ('train_datasets', 'valid_datasets'):
                dp_dataset = dict(pair['dp']['data'][split]['libero_panda'])
                af_dataset = dict(pair['action_flow']['data'][split]['libero_panda'])
                assert dp_dataset.pop('valid_ratio') == .1
                assert af_dataset.pop('valid_ratio') == .1
                assert dp_dataset == af_dataset
            assert pair['dp']['model']['pipeline']['stages'][0] == pair['action_flow']['model']['pipeline']['stages'][0]
            for key in ('suite','dataset','horizon','n_obs_steps','n_action_steps'):
                assert pair['dp']['benchmark'][key] == pair['action_flow']['benchmark'][key]
    return {'status':'PASS_CONFIG_ONLY', 'fairness':'FAIR WITH CAVEATS: matched episode split, intentionally different family optimizers and training budgets', 'full_training_ready':False, 'rows': results}

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    source=Path(__file__).resolve().parents[1]
    result=compose_all(source,a.output)
    (a.output/'CONFIG_PROOF.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
