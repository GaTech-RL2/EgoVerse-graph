"""Runtime native identities from the actual Hydra snapshot; no self hash field.

Factories hydrate only scalar/path descriptors. They do not instantiate a
model, resolver, normalizer or dataset and perform no stochastic computation.
"""
import hashlib
import json
import subprocess
from pathlib import Path

_SNAPSHOTS={}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _snapshot(config_path):
    from omegaconf import OmegaConf
    path=Path(config_path).resolve()
    if path.name!='config.yaml' or path.parent.name!='.hydra':
        raise ValueError('actual .hydra/config.yaml required')
    raw=path.read_bytes();sha=hashlib.sha256(raw).hexdigest()
    if str(path) in _SNAPSHOTS:
        previous_sha,cfg=_SNAPSHOTS[str(path)]
        if previous_sha!=sha:raise ValueError('Hydra snapshot changed after native identity binding')
        return path,sha,cfg
    cfg=OmegaConf.to_container(OmegaConf.load(path),resolve=True)
    if path.read_bytes()!=raw:raise ValueError('Hydra snapshot changed while reading')
    if not isinstance(cfg,dict):raise ValueError('resolved config mapping required')
    _SNAPSHOTS[str(path)]=(sha,cfg)
    return path,sha,cfg


def _source_identity():
    root=Path(__file__).resolve().parents[3]
    head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True,timeout=10).strip()
    if subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=all'],text=True,timeout=10):
        raise ValueError('native artifact source must be immutable clean checkout')
    module=root/'egomimic/rldb/zarr/libero_action_flow.py'
    return head,_sha(module)


def _checked_json(path,sha):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('native bound artifact SHA mismatch')
    return json.loads(raw)


def _identity_from_resolved(cfg, snapshot_sha, source_head, normalizer_module_sha, receipt, split):
    """Pure contract guard; runtime factories supply independently hashed inputs."""
    model=cfg['model'];data=cfg['data'];bench=cfg['benchmark'];evalcfg=cfg['evaluator']
    identity_descriptor=evalcfg.get('artifact_identity',{})
    diagnostics_descriptor=evalcfg.get('native_diagnostic_config',{})
    if (set(identity_descriptor)!={'_target_','config_path'}
        or identity_descriptor['_target_']!='egomimic.benchmarks.libero.native_artifact_identity.build_native_artifact_identity_from_config'
        or set(diagnostics_descriptor)!={'_target_','config_path','artifact_root'}
        or diagnostics_descriptor['_target_']!='egomimic.benchmarks.libero.native_artifact_identity.build_native_diagnostic_config_from_config'):
        raise ValueError('native scalar factory descriptors required; no literal/self hash or nested model hydration')
    expected_config=Path(cfg['paths']['output_dir']).resolve()/'.hydra/config.yaml'
    if any(Path(d['config_path']).resolve()!=expected_config for d in (identity_descriptor,diagnostics_descriptor)):
        raise ValueError('factory descriptors must bind actual output Hydra snapshot')
    binding=cfg['norm_stats']['native_saved_state_binding'];provenance=cfg['run_provenance']
    target='egomimic.rldb.zarr.libero_action_flow.LiberoActionFlowNormalizer'
    if cfg['normalizer'].get('_target_')!=target or binding.get('normalizer_target')!=target:
        raise ValueError('native train-only normalizer target binding required')
    if cfg['norm_stats'].get('precomputed_norm_path') is not None:
        raise ValueError('native saved state cannot use generic precomputed normalization')
    if Path(bench['dataset']).resolve()!=Path(binding['data_root']).resolve() or Path(receipt['replay_path']).resolve()!=Path(binding['data_root']).resolve():
        raise ValueError('native cached receipt/benchmark physical dataset path mismatch')
    if cfg.get('seed')!=42 or bench.get('suite')!='libero10' or bench.get('batch_size')!=32 or bench.get('horizon')!=16:
        raise ValueError('native seed/suite/batch/horizon mismatch')
    if cfg['trainer'].get('precision') not in ('bf16','bf16-mixed'):
        raise ValueError('actual native BF16 precision required')
    if cfg['trainer'].get('accumulate_grad_batches')!=1 or cfg['trainer'].get('num_nodes')!=1:
        raise ValueError('native one-process effective32 contract required')
    if cfg['trainer'].get('devices') not in (1,[0], '1'):
        raise ValueError('native single-device source contract required')
    if evalcfg.get('_recursive_',True) is False:
        raise ValueError('evaluator recursive factory hydration cannot be disabled')
    for key,value in {'action_dim':7,'action_horizon':16,'num_latent_tokens':8,'latent_dim':16,
                      'hidden_dim':240,'num_inference_steps':50,'flow_samples_per_content':14}.items():
        if model.get(key)!=value:raise ValueError(('native model contract mismatch',key))
    stages=model['pipeline']['stages']
    if len(stages)!=9 or any(s['_target_'].endswith('.SharedSpeedCondition') for s in stages):
        raise ValueError('native noaug nine-stage profile required')
    owners={name:[s for s in stages if s['_target_'].endswith('.'+name)] for name in
            ('ContentEncoderStage','ConditionalVelocityStage','ContentDecoderStage','ActionFlowObjectiveStage')}
    if any(len(v)!=1 for v in owners.values()):raise ValueError('native typed stage ownership mismatch')
    encoder=owners['ContentEncoderStage'][0]['encoder']['backbone']
    field=owners['ConditionalVelocityStage'][0]
    decoder=owners['ContentDecoderStage'][0]['decoder']
    for owner in (encoder,field['field']['backbone'],decoder):
        if owner.get('depth')!=12 or owner.get('num_heads')!=8:
            raise ValueError('native H240 twelve-layer/eighthead source owners required')
    if field.get('inference_method')!='euler' or field.get('num_inference_steps')!=50:
        raise ValueError('actual native Euler50 pathway required')
    objective=owners['ActionFlowObjectiveStage'][0]
    if (model.get('flow_loss_aggregation')!='sum_samples'
        or objective.get('flow_aggregation')!='sum_samples'
        or objective.get('flow_samples_per_content')!=14):
        raise ValueError('historical native sum_samples/14 flow aggregation must be preserved')
    weights={k:objective[k] for k in ('reconstruction_weight','flow_weight','action_velocity_weight')}
    if any(value!=1. for value in weights.values()):raise ValueError('native objective weights mismatch')
    for name,mode in [('train_datasets','train'),('valid_datasets','valid')]:
        datasets=data[name]
        if set(datasets)!={'libero_panda'}:raise ValueError('native single logical source required')
        dataset=datasets['libero_panda']
        if dataset.get('mode')!=mode or dataset.get('split_seed')!=42 or dataset.get('valid_ratio')!=.01:
            raise ValueError('native exact episode split required')
        if Path(dataset['resolver']['folder_path']).resolve()!=Path(binding['data_root']).resolve():
            raise ValueError('native dataset resolver/binding path mismatch')
    for key in ('train_dataloader_params','valid_dataloader_params'):
        if data[key]['libero_panda']['batch_size']!=32:raise ValueError('native source batch32 required')
    if source_head!=binding['source_commit'] or source_head!=provenance['source_commit']:
        raise ValueError('actual source HEAD/provenance/binding mismatch')
    if normalizer_module_sha!=binding['normalizer_module_sha256']:
        raise ValueError('actual native normalizer module SHA mismatch')
    if binding['file_sha256']!=provenance['normalization_sha256'] or binding['split_receipt_sha256']!=provenance['split_manifest_sha256']:
        raise ValueError('actual normalization/split provenance binding mismatch')
    if receipt['dataset_logical_sha256']!=provenance['dataset_sha256'] or split['dataset_logical_sha256']!=receipt['dataset_logical_sha256']:
        raise ValueError('native dataset identity mismatch')
    if (receipt['split_seed']!=42 or receipt['valid_ratio']!=.01 or receipt['episodes']!=500
        or len(receipt['train_episode_indices'])!=495 or len(receipt['valid_episode_indices'])!=5
        or split['train_episode_indices']!=receipt['train_episode_indices']
        or split['valid_episode_indices']!=receipt['valid_episode_indices']):
        raise ValueError('native bound receipt split mismatch')
    identity={'suite':'libero10','source':'libero_panda','action_dim':7,'action_horizon':16,
        'seed':cfg['seed'],'sample_count':evalcfg['energy_sample_count'],
        'energy_seed_bank_sha256':evalcfg['energy_seed_bank_sha256'],
        'inference_method':field['inference_method'],'inference_steps':field['num_inference_steps'],
        'normalization_scope':'training_episodes_only','effective_batch_size':32,
        'homogeneous':'not_applicable_single_source','source_commit':source_head,
        'resolved_config_sha256':snapshot_sha,'split_sha256':binding['split_receipt_sha256'],
        'normalizer_state_sha256':binding['file_sha256'],
        'dataset_logical_sha256':receipt['dataset_logical_sha256'],
        'precision':cfg['trainer']['precision'],'flow_loss_aggregation':model['flow_loss_aggregation'],
        'flow_samples_per_content':objective['flow_samples_per_content'],'native_objective_weights':{
            'reconstruction':weights['reconstruction_weight'], 'flow':weights['flow_weight'],
            'action_velocity':weights['action_velocity_weight']},
        'source_normalizer_module_sha256':normalizer_module_sha,
        'dataset_receipt_sha256':binding['dataset_receipt_sha256'],
        'physical_split_proof_sha256':binding['physical_proof_sha256'],
        'historical_split_sha256':provenance['historical_split_manifest_sha256']}
    return identity


def _runtime_identity(path,sha,cfg):
    source_head,module_sha=_source_identity();binding=cfg['norm_stats']['native_saved_state_binding']
    if path!=Path(cfg['paths']['output_dir']).resolve()/'.hydra/config.yaml':
        raise ValueError('factory actual snapshot/output mismatch')
    if _sha(binding['path'])!=binding['file_sha256']:raise ValueError('native state payload SHA mismatch')
    if _sha(cfg['evaluator']['energy_seed_bank_path'])!=cfg['evaluator']['energy_seed_bank_sha256']:
        raise ValueError('native seed bank bytes mismatch')
    receipt=_checked_json(binding['dataset_receipt_path'],binding['dataset_receipt_sha256'])
    split=_checked_json(binding['split_receipt_path'],binding['split_receipt_sha256'])
    proof=_checked_json(binding['physical_proof_path'],binding['physical_proof_sha256'])
    if proof.get('status')!='PASS' or proof.get('split_manifest_sha256')!=binding['split_receipt_sha256']:
        raise ValueError('native physical proof binding mismatch')
    identity=_identity_from_resolved(cfg,sha,source_head,module_sha,receipt,split)
    from egomimic.benchmarks.libero.action_flow_artifacts import validate_identity
    validate_identity(identity)
    identity['resolved_config_path']=str(path)
    return identity


def build_native_artifact_identity_from_config(config_path):
    path,sha,cfg=_snapshot(config_path)
    return _runtime_identity(path,sha,cfg)


def build_native_diagnostic_config_from_config(config_path, artifact_root):
    path,sha,cfg=_snapshot(config_path)
    identity=_runtime_identity(path,sha,cfg)
    from egomimic.benchmarks.libero.native_diagnostic_config import build_native_diagnostic_config
    config=build_native_diagnostic_config(identity,artifact_root,
        cfg['evaluator']['energy_seed_bank_path'],cfg['model'])
    return _bind_shared_provenance(config,identity)


def _bind_shared_provenance(config,identity):
    if set(identity['native_objective_weights'])!={'reconstruction','flow','action_velocity'}:
        raise ValueError('native objective identity key inventory mismatch')
    config['provenance'].update(normalization_sha256=identity['normalizer_state_sha256'],
        split_manifest_sha256=identity['split_sha256'])
    if (config['provenance']['normalization_sha256']!=identity['normalizer_state_sha256']
        or config['provenance']['split_manifest_sha256']!=identity['split_sha256']):
        raise ValueError('shared provenance alias binding mismatch')
    return config
