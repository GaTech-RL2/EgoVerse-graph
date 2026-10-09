"""Fresh typed route rejects physical speed and preserves the training contract."""
from pathlib import Path
import hashlib
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
import runpy

def test_multiplier_training_contract():
    root = Path(__file__).resolve().parents[1]
    with initialize_config_dir(config_dir=str(root/'egomimic/hydra_configs'), version_base='1.3'):
        cfg = compose(config_name='train_zarr_cartesian', overrides=[
            'hydra/launcher=basic', '+experiment=pusht/action_flow_cotrain_uc_multiplier_interpolation',
            'trainer.precision=bf16', 'trainer.devices=1', 'trainer.num_nodes=1',
            'model.pipeline.stages.6.inference_method=euler', 'model.pipeline.stages.6.num_inference_steps=50'])
    validator = runpy.run_path(str(root/'scripts/train/validate_multiplier_config.py'))
    validator['validate_speed_contract'](cfg, 'scalar')
    assert cfg.callbacks.ema.decay == 0.9978
    assert cfg.callbacks.ema.use_warmup is False
    assert cfg.callbacks.homogeneous_dithalf._target_.endswith('HomogeneousDiTHalf')
    assert cfg.norm_stats.sample_frac == 1.0
    assert cfg.norm_stats.norm_mode == 'minmax'

def test_shared_prefix_assets_are_pinned():
    root = Path(__file__).resolve().parents[1]/'egomimic/pl_utils/homogeneous_training/prefix_source'
    for name, expected in {
        'core.py': 'f15bb44f9f20fa1eed34cb89059a44587f19beeeb58bdc743245de994eff7b4b',
        'sampler.py': '152b674184e0b93580ca0e6db7f58e9471a48a50838df502af032b52b8334ac0',
        'batch_utils.py': 'f00aaf14e85b39c72f2a6db0527e3a636739a2053268869350af38e4532c8737',
    }.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected


def test_partial_batches_recapture_native_rng_without_padding(monkeypatch):
    import torch
    from egomimic.pipeline.algo import PipelineAlgo
    from egomimic.pl_utils.homogeneous_training import callback
    class Plan:
        def __init__(self, task):
            self.ready=False; self.grouped_encoder_calls=0; self.previous=None
        def install_capture(self, algo):
            self.algo=algo; self.previous=algo._execute
            def capture(batch, *, mode):
                self.signature=tuple((k,v['actions'].shape[0]) for k,v in batch.items())
                self.ready=True
                return self.previous(batch,mode=mode)
            algo._execute=capture
        def restore(self):
            if self.previous is not None:
                self.algo._execute=self.previous; self.previous=None
        def install_candidate(self, algo):
            self.algo=algo; self.previous=algo._execute
            def grouped(batch, *, mode):
                assert tuple((k,v['actions'].shape[0]) for k,v in batch.items())==self.signature
                self.grouped_encoder_calls+=1
                return self.previous(batch,mode=mode)
            algo._execute=grouped
    monkeypatch.setattr(callback,'HomogeneousNativeReplay',Plan)
    algo=object.__new__(PipelineAlgo)
    pipeline=torch.nn.Module()
    pipeline.stages=[type(name,(),{})() for name in ('KeyedFeatureProjection','FusedObsEncoder','SharedSpeedCondition')]
    algo.nets=torch.nn.ModuleDict({'pipeline':pipeline})
    native=lambda batch,mode: {k:v['actions'].clone() for k,v in batch.items()}
    algo._execute=native
    module=type('Module',(),{'model':algo,'nets':algo.nets})()
    cb=callback.HomogeneousDiTHalf()
    cb.on_fit_start(None,module)
    for rows in [(32,32),(32,32),(32,5),(32,5),(32,32),(32,32)]:
        batch={name:{'actions':torch.randn(n,16,4)} for name,n in zip(('u','chain'),rows)}
        got=algo._execute(batch,mode='train')
        for name in batch: torch.testing.assert_close(got[name],batch[name]['actions'],rtol=0,atol=0)
    assert cb.capture_updates==3 and cb.grouped_updates==3 and cb.updates==6
    cb.restore()
    assert algo._execute is native


def test_launcher_child_helper_exists_and_parses():
    import subprocess
    root = Path(__file__).resolve().parents[1]
    helper = root/'scripts/train/action_flow_run_child.sh'
    assert helper.is_file()
    subprocess.run(['bash','-n',str(helper)],check=True)
    assert '--cpu-bind=none' in helper.read_text()
