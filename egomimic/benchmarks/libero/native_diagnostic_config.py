"""Explicit native diagnostic-only profile; independent metric identity.

k2/native8 and all12 identity block pairs are approved diagnostic settings.
They are not comparable to U-Socket k10 or historical native RMS-only artifacts.
"""

def build_native_diagnostic_config(identity, artifact_root, seed_bank_path, model_config):
    if model_config['num_latent_tokens']!=8 or model_config['latent_dim']!=16:
        raise ValueError('resolved native latent source must be [8,16]')
    stages=model_config['pipeline']['stages']
    encoder=[s for s in stages if s['_target_'].endswith('.ContentEncoderStage')]
    field=[s for s in stages if s['_target_'].endswith('.ConditionalVelocityStage')]
    if len(encoder)!=1 or len(field)!=1 or encoder[0]['encoder']['backbone']['depth']!=12 or field[0]['field']['backbone']['depth']!=12:
        raise ValueError('resolved native encoder/field depths must both be12')
    if field[0]['inference_method']!='euler' or model_config['num_inference_steps']!=50:
        raise ValueError('native diagnostics requires resolved Euler50')
    return {'max_batches_per_rank':1,'artifact_root':artifact_root,
        'noise_seed_bank_path':seed_bank_path,
        'noise_seed_bank_sha256':identity['energy_seed_bank_sha256'],
        'raw_noise_levels':[0.,.25,.5,.75,1.], 'max_samples':8,
        'jacobian_samples':2,'capture_activations':True,
        'activation_layer_map':{i:i for i in range(12)},'cknna_k':2,
        'validation_view':{'definition':'same_first_native_evaluator_validation_batch',
            'split_manifest_sha256':identity['split_sha256'],'per_rank_batch_size':32,'world_size':1},
        'provenance':dict(identity,latent_shape=[8,16],sampler='euler',sampler_steps=50,
            diagnostic_metric_identity='libero-native8-k2-all12-equalblocks-mse/v1',
            comparability='new_diagnostic_identity_not_usocket_k10_or_historical_native_rms'),
        'native_error':{'enabled':True,'type':'libero_delta_osc7d_equal_semantic_blocks_mse_v1',
            'space':'native_translation_rotation_commands_gripper',
            'reduction':'mean_of_translation_xyz_MSE_rotation_command_xyz_MSE_gripper_MSE',
            'normalizer':'bound_train_only_native_normalizer',
            'rotation':'native_OSC_rotation_commands_no_SO3_orientation_wrapping'}}


def assert_native_diagnostic_owners(module):
    stages=tuple(module.model.pipeline.stages)
    encoder=[s for s in stages if type(s).__name__=='ContentEncoderStage']
    field=[s for s in stages if type(s).__name__=='ConditionalVelocityStage']
    if len(stages)!=9 or len(encoder)!=1 or len(field)!=1:
        raise ValueError('native noaug topology mismatch')
    if len(encoder[0].encoder.backbone.blocks)!=12 or len(field[0].field.backbone.blocks)!=12:
        raise ValueError('actual encoder/field diagnostic owners must both have12 blocks')
