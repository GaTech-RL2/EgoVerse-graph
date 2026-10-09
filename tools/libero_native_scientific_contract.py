"""Source-grounded resolved noaug contract. Reject silent mean/optimizer drift."""
import copy,hashlib,json
from pathlib import Path

def flatten(value,prefix=''):
    out={}
    if isinstance(value,dict):
        for key,child in value.items():out.update(flatten(child,f'{prefix}.{key}' if prefix else key))
    elif isinstance(value,list):
        for i,child in enumerate(value):out.update(flatten(child,f'{prefix}.{i}'))
    else:out[prefix]=value
    return out

def validate_scientific_contract(config,phase,contract_path,expected_contract_sha256):
    if phase not in {'preflight','smoke','full'}:raise ValueError('unknown native phase')
    raw=Path(contract_path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_contract_sha256:raise ValueError('historical scientific-contract hash mismatch')
    reference=json.loads(raw);expected=copy.deepcopy(reference['historical_model'])
    from egomimic.benchmarks.libero.native_launch_profiles import profile_for_config
    profile=profile_for_config(config)
    expected['pipeline']['stages'][6]['inference_method']='euler'
    expected['pipeline']['stages'][8]['action_velocity_weight']=profile.action_velocity_weight
    actual=flatten(config['model']);wanted=flatten(expected)
    # Instrumentation is separately proven by native diagnostics tests; it may
    # change logging/provider schema, never the actual pipeline/optimizer path.
    instrumentation={'gradient_telemetry_cadence','enable_grad_norm'}
    allowed=lambda k:k in instrumentation or k.startswith('diagnostic_provider.')
    changes={k:{'expected':wanted.get(k),'actual':actual.get(k)} for k in sorted(set(wanted)|set(actual)) if not allowed(k) and (k not in wanted or k not in actual or wanted[k]!=actual[k])}
    if changes:raise ValueError({'native scientific model contract changed':changes})
    # Explicit dual assertion protects both metadata and constructor argument.
    if config['model']['flow_loss_aggregation']!='sum_samples' or config['model']['pipeline']['stages'][8]['flow_aggregation']!='sum_samples':raise ValueError('historical sum_samples objective mandatory at both boundaries')
    trainer=config['trainer'];historical=reference['historical_trainer']
    for name in ['devices','precision','accumulate_grad_batches','gradient_clip_val','gradient_clip_algorithm','deterministic']:
        if trainer.get(name)!=historical[name]:raise ValueError(('native trainer scientific contract',name,historical[name],trainer.get(name)))
    from egomimic.benchmarks.libero.native_launch_profiles import profile_for_config
    profile=profile_for_config(config)
    allowed_seeds={42,43} if profile.suite in {'libero_goal','libero_object'} else {reference['historical_seed']}
    if type(config.get('seed')) is not int or config['seed'] not in allowed_seeds:raise ValueError('native training seed mismatch')
    for mode,field in [('train','historical_train_batch'),('valid','historical_valid_batch')]:
        params=config['data'][mode+'_dataloader_params']
        if set(params)!={'libero_panda'} or params['libero_panda']['batch_size']!=reference[field]:raise ValueError('native logical source/batch mismatch')
    expected_phase={'max_steps':2,'val_check_interval':2,'limit_train_batches':2,'limit_val_batches':1} if phase=='smoke' else {'max_steps':80000,'val_check_interval':15000,'limit_train_batches':1.0,'limit_val_batches':8}
    for name,value in expected_phase.items():
        if trainer.get(name)!=value:raise ValueError(('native phase contract',name,value,trainer.get(name)))
    evaluator=config['evaluator'];he=reference['historical_evaluator']
    for name in ['_target_','energy_sample_count','energy_seed_bank_sha256','diagnostic_raw_noise_levels']:
        if evaluator.get(name)!=he[name]:raise ValueError(('native evaluator contract',name))
    if any(k.startswith('energy_score_validation_view') for k in evaluator):raise ValueError('U-Socket-only evaluator contract leaked')
    return {'model_contract':'historical_exact_except_requested_euler50_and_instrumentation_and_typed_velocity_ablation','action_velocity_weight':profile.action_velocity_weight,'objective':'sum_samples_K14','optimizer':'historical_exact','batch':32,'source_count':1,'phase':phase,'gpu_ready':False}
