import json
from pathlib import Path
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation
from egomimic.rldb.zarr.libero_arc_arrival import LiberoArcArrivalCodec
from egomimic.rldb.zarr.libero_arc import _from_rotation6d
from egomimic.rldb.zarr.cartesian_arc import TokenizeBimanualArcLengthCartesian, absolute_times_to_seconds

ROOT=Path.cwd()
VARIANTS=['reference','gripper','xyz_scalar','component_velocity','component_time','x_yz','y_xz','z_xy','angular_driver','angular_scalar','all_scalar']

def codec(variant,mode,**kw):
 spec=yaml.safe_load((ROOT/'egomimic/hydra_configs/arc_stream'/f'{variant}.yaml').read_text())['spec']
 return LiberoArcArrivalCodec(stream_spec=spec,mode=mode,**{**dict(num_waypoints=36,max_translation=1.6,max_rotation_degrees=384),**kw})

@pytest.mark.parametrize('variant',VARIANTS)
@pytest.mark.parametrize('mode',['stk','dur'])
def test_constant_commands_and_still(variant,mode):
 c=codec(variant,mode); a=np.zeros((32,7)); a[:,:3]=[.2,.1,.03]; a[:,3]=.01; a[:,6]=1
 decoded=c.decode(c.encode(a))
 np.testing.assert_allclose(decoded,a,atol=2e-6)
 a[:,:6]=0
 np.testing.assert_allclose(c.decode(c.encode(a)),a,atol=2e-6)

@pytest.mark.parametrize('variant',['reference','gripper'])
@pytest.mark.parametrize('mode',['stk','dur'])
@pytest.mark.parametrize('motion',['random','pause','still','event'])
def test_matches_main(variant,mode,motion):
 c=codec(variant,mode); rng=np.random.default_rng(42)
 a=rng.uniform(-.3,.3,(32,7)); a[:,6]=np.where(np.arange(32)<15,-1.,1.)
 if motion=='pause':a[5:15,:6]=0
 if motion=='still':a[:,:6]=0
 if motion=='event':a[:,:6]=0;a[:,0]=.2
 raw=np.zeros((32,14)); raw[:,:3]=np.cumsum(a[:,:3]*c.translation_scale,axis=0)
 prev=Rotation.identity(); rots=[]
 for row in a:
  prev=Rotation.from_rotvec(row[3:6]*c.rotation_scale)*prev;rots.append(prev.as_quat())
 raw[:,3:6]=Rotation.from_quat(rots).as_euler('ZYX');raw[:,6]=a[:,6]
 main=TokenizeBimanualArcLengthCartesian(action_key='actions',output_action_key='actions',resampled_vector_length=c.num_waypoints,min_distance_unit=c.max_translation,rotation_distance_unit=np.deg2rad(c.max_rotation_degrees),dt=c.dt,clock_mode='duration' if mode=='dur' else 'velocity',streams='full_separation' if variant=='gripper' else 'translation_rotation')
 token=main.transform({'actions':raw,'arc.current_action':np.zeros(14)})['actions']
 ours=c.encode(a)
 np.testing.assert_allclose(ours[:,:3],token[:,:3],atol=1e-7)
 np.testing.assert_allclose(_from_rotation6d(ours[:,3:9]).as_matrix(),Rotation.from_euler('ZYX',token[:,3:6]).as_matrix(),atol=1e-7)
 np.testing.assert_allclose(ours[:,9],token[:,6],atol=1e-7)
 np.testing.assert_allclose(ours[:,10:],token[:,14::2],atol=1e-5)
 if mode=='dur':
  for i,name in enumerate(main._stream_names):
   for arm,off in enumerate((0,7)):
    col=14+2*i+arm
    token[:,col]=absolute_times_to_seconds(token[:,col],main._distances(token,off,name,np.zeros(14)),max_speed=None)
 poses=main.detokenize(token,32,initial_action=np.zeros(14),clocks_in_seconds=mode=='dur')
 poses=np.vstack((np.zeros(14),poses));r=Rotation.from_euler('ZYX',poses[:,3:6])
 expected=np.c_[np.diff(poses[:,:3],axis=0)/c.translation_scale,(r[1:]*r[:-1].inv()).as_rotvec()/c.rotation_scale,poses[1:,6]]
 np.testing.assert_allclose(c.decode(ours),expected,atol=4e-5)

@pytest.mark.parametrize('variant',VARIANTS)
def test_absolute_correction_all_groups(variant):
 c=codec(variant,'dur');a=np.zeros((32,7));a[:,0]=.5;a[:,6]=1;t=c.encode(a)
 for sl in c.timing_slices.values():t[:,sl]=np.linspace(.2,1.6,c.num_waypoints)[:,None];t[3:8,sl]=-.2
 for clock in c.clocks(t).values():assert np.all(np.diff(clock,axis=0)>=0)
 assert np.isfinite(c.decode(t)).all()

@pytest.mark.parametrize('variant',VARIANTS)
@pytest.mark.parametrize('mode',['stk','dur'])
def test_bad_predictions_stay_finite(variant,mode):
 c=codec(variant,mode);t=np.zeros((c.num_waypoints,c.action_dim));t[:,0]=np.arange(c.num_waypoints)
 assert np.isfinite(c.decode(t)).all()


def test_profiles_and_campaign_use_arrival_factory():
 from egomimic.rldb.zarr.libero_arc_timed import make_libero_arc_codec,codec_source_files
 from egomimic.benchmarks.libero.arc_streams import candidates
 assert len(candidates())==22
 for mode in ('stk','dur'):
  assert isinstance(make_libero_arc_codec(mode=mode),LiberoArcArrivalCodec)
  assert 'egomimic/rldb/zarr/cartesian_arc.py' in codec_source_files(mode)
 from hydra import initialize_config_dir,compose
 with initialize_config_dir(config_dir=str(ROOT/'egomimic/hydra_configs'),version_base=None):
  c=compose(config_name='train_zarr_cartesian',overrides=['+experiment=oat/libero_arc_streams'])
  for s in c.model.pipeline.stages:
   if s['_target_'].endswith('LiberoArcStage'):assert s.clock_contract=='libero_osc_main_arrival_v2_unbounded'

@pytest.mark.parametrize('manifest', ['libero_arc_streams.yaml']+[f'libero_arrival_{p}_{b}.yaml' for p in ['stk_1','stk_2','dur_1','dur_2','shared'] for b in ['unet','oat_dp'] if (p,b)!=('shared','unet')])
def test_full_training_recipe_and_dag(manifest,monkeypatch):
 from egomimic.benchmarks.libero.arc_streams import campaign,stream_training_arguments,validate_stream_config
 from scripts.benchmarks.launch_libero_streams import stream_workflow
 from hydra import initialize_config_dir,compose
 from omegaconf import OmegaConf
 path='egomimic/hydra_configs/benchmark/'+manifest
 monkeypatch.setenv('LIBERO_STREAM_CAMPAIGN',path)
 spec=campaign()
 for mode in spec['modes']:
  args=stream_training_arguments('libero_spatial',Path('/tmp/data'),Path('/tmp/evidence'),'full',variant='reference',arc_mode=mode,gpus=4)
  with initialize_config_dir(config_dir=str(ROOT/'egomimic/hydra_configs'),version_base=None):
   cfg=compose(config_name='train_zarr_cartesian',overrides=args[3:])
   config={'model':OmegaConf.to_container(cfg.model,resolve=True)}
   proof=validate_stream_config(config,suite='libero_spatial',variant='reference',arc_mode=mode)
   assert proof['clock_contract']=='libero_osc_main_arrival_v2_unbounded'
   assert cfg.trainer.max_epochs==5001 and cfg.benchmark.batch_size*4==1024
 w=stream_workflow('a'*40,'arrival-test','libero_spatial',gpus=4)['workflow']
 assert all(t['environment']['LIBERO_STREAM_CAMPAIGN']==path for t in w['tasks'])
 assert w['tasks'][0]['name']=='replay'
 assert all('replay' in [x['task'] for x in t.get('inputs',[])] for t in w['tasks'] if t['name'].startswith('train'))

@pytest.mark.parametrize('variant',VARIANTS)
def test_dense_absolute_clocks_preserve_noncommuting_commands(variant):
 c=codec(variant,'dur',num_waypoints=32)
 rng=np.random.default_rng(9);a=rng.uniform(-.2,.2,(32,7));a[:,6]=np.where(np.arange(32)<10,-1.,1.)
 t=np.zeros((32,c.action_dim));t[:,:3]=np.cumsum(a[:,:3]*c.translation_scale,axis=0)
 if c.rotation_representation=='so3':
  from egomimic.rldb.zarr.libero_arc import _rotation6d
  rs=[];r=Rotation.identity()
  for row in a:r=Rotation.from_rotvec(row[3:6]*c.rotation_scale)*r;rs.append(r.as_quat())
  t[:,3:9]=_rotation6d(Rotation.from_quat(rs))
 else:t[:,3:6]=np.cumsum(a[:,3:6]*c.rotation_scale,axis=0)
 t[:,c.grip_column]=a[:,6]
 for sl in c.timing_slices.values():t[:,sl]=(np.arange(32)+1)[:,None]*c.dt
 np.testing.assert_allclose(c.decode(t),a,atol=2e-6)

@pytest.mark.parametrize('variant',['shared','separate','shape_masked','mot_xyz_rot_gripper','mot_shape_velocity','mot_shape_velocity_masked'])
def test_decoder_ablation_recipes(variant,monkeypatch):
 test_full_training_recipe_and_dag('libero_arrival_decoder_'+variant+'.yaml',monkeypatch)
 from egomimic.models.arc_diffusion import SHAPE_COLUMNS,TIMING_COLUMNS
 assert SHAPE_COLUMNS==tuple(range(10)) and TIMING_COLUMNS==(10,11)
 if variant.startswith('mot_'):
  from egomimic.benchmarks.libero.arc_decoders import mot_architecture
  cols=mot_architecture(variant)['modalities']
  assert sorted(x for v in cols.values() for x in v)==list(range(12))
  if 'velocity' in cols:assert cols['velocity']==[10,11]
