"""Typed extension of the maintained scalar-speed contract, scheduled only."""
import os
from pathlib import Path
import validate_speed_config as native

contract=native.validate_speed_contract
pipeline=native.validate_speed_pipeline

def validate(cfg, encoding):
    contract(cfg,encoding)
    assert encoding=='scalar'
    assert os.environ['AF_HOMOGENEOUS_NUMERICAL_DELTA_ACCEPTED']=='true'
    field=cfg.model.pipeline.stages[6]
    assert field.inference_method=='euler' and field.num_inference_steps==50
    callback=cfg.callbacks.homogeneous_dithalf
    assert callback._target_=='homogeneous_dithalf_training.HomogeneousDiTHalf'
    assert str(callback.task)==os.environ['AF_HOMOGENEOUS_TASK']
    assert Path(callback.task).is_absolute()
    assert cfg.run_provenance.validation_inference_sampler=='euler'
    assert cfg.run_provenance.validation_inference_steps==50

def validate_pipeline(cfg,algo):
    count=pipeline(cfg,algo)
    fields=[s for s in algo.pipeline.stages if type(s).__name__=='ConditionalVelocityStage']
    assert len(fields)==1
    assert fields[0].inference_method=='euler' and fields[0].num_inference_steps==50
    from homogeneous_dithalf_training import HomogeneousDiTHalf
    assert HomogeneousDiTHalf(cfg.callbacks.homogeneous_dithalf.task).state_dict()['user_accepted_numerical_delta']
    return count

if __name__=='__main__':
    assert os.environ.get('SLURM_STEP_ID'), 'cluster Python must be scheduled'
    native.validate_speed_contract=validate
    native.validate_speed_pipeline=validate_pipeline
    native.main()
