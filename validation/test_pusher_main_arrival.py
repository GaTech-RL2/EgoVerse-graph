import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.rldb.zarr.planar_arc_clock import PlanarArcClockCodec, current_action_from_state, get_planar_clock_transform_list
from egomimic.rldb.zarr.arc_clock_bridge import ARRIVAL_CONTRACT
from egomimic.pipeline.planar_clock import PlanarArcClockNativeDecoder

ROOT = Path.cwd()

def codec(dim=3, mode='duration', **kw):
    return PlanarArcClockCodec(native_action_dim=dim, clock_mode=mode, num_waypoints=8, horizon=8, **kw)

def path(dim=3):
    a=np.zeros((8,dim)); a[:,0]=np.arange(10,18); a[:,1]=np.arange(20,28)
    if dim>=3: a[:,2]=np.arange(8)*.01+.1
    if dim==4: a[:,3]=[0,0,0,1,1,1,1,1]
    return a

def test_upstream_parity():
    # Verified byte/AST-identical to main c7e110461650d09a88b2c2dff8f02db54b1deace.
    assert hashlib.sha256((ROOT/'egomimic/rldb/zarr/cartesian_arc.py').read_bytes()).hexdigest()=='6a93876939af29538a744053642766b488c690acf0e7e4e3d76c61a908d08362'
    expected={'_ypr_to_rotation': 'f2b85a88ec9d58ec00b16f7e0161e507460955f29a9d9327f75f204487e89cb7', '_rotation_to_ypr': 'df7944d9c98a656ee94a792ed9d838c21a211da10a6f6a8a6e32baa12269a54e', 'cumulative_arc_length': '9da5d8d0a6ecf5761a0c423c921d79fc12d4719719af5a599d400190af595fdb', 'cumulative_rotation_length': '2284a00fd09ebc70ce1eb86cb22b4d6757868b7e88dbc8eaa1fd6624f85a6265', '_bracket_segments': '1adaa96cddcb17e5417a2a8481d8e184ec43a9ad8c5a7dce03b0c6a3892b9fcc', '_first_crossing_brackets': 'c1bfe0175d5bc940f977837b556f05e5c5ed86d66ccce51c6fdc64c86b996819', '_slerp_segments_ypr': '8ae7602015e6fbe6b704765c4e0978fb2674755d571539f04da0eed2749704a3'}
    actual={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in ast.parse((ROOT/'egomimic/rldb/zarr/arc_length_tokenizer.py').read_text()).body if isinstance(n,ast.FunctionDef)}
    for name,digest in expected.items(): assert actual[name]==digest, name

@pytest.mark.parametrize('dim',[2,3,4])
@pytest.mark.parametrize('mode',['duration','velocity'])
def test_roundtrip_and_incoming_anchor(dim,mode):
    c=codec(dim,mode); a=path(dim); anchor=np.zeros(dim)
    t=c.encode(a,initial_action=anchor)
    decoded=c.decode(t,initial_action=anchor)
    np.testing.assert_allclose(decoded[:,:min(dim,3)],a[:,:min(dim,3)],atol=1e-8)
    if dim==4 and mode=="duration": np.testing.assert_allclose(decoded[:,3],a[:,3])
    if mode=='duration': assert t[0,5]==pytest.approx(1/30)
    else: assert t[0,5]==pytest.approx(np.linalg.norm(a[0,:2])*30)
    assert c.waypoint_clocks(t,initial_action=anchor)['translation']['seconds'][0]==pytest.approx(1/30)

@pytest.mark.parametrize('mode',['duration','velocity'])
def test_future_budget_excludes_anchor(mode):
    c=codec(2,mode,min_distance_unit=2)
    a=np.c_[np.arange(100,108),np.zeros(8)]
    t=c.encode(a,initial_action=np.zeros(2))
    assert t[0,0]==100 and t[-1,0]==102
    assert c.decode(t,initial_action=np.zeros(2))[0,0]==pytest.approx(100)

@pytest.mark.parametrize('mode',['duration','velocity'])
@pytest.mark.parametrize('streams',['full_separation','translation_rotation','translation','rotation'])
def test_stationary_finite_and_source_duration(mode,streams):
    c=codec(3,mode,streams=streams); a=np.tile([50.,60.,.2],(8,1)); anchor=a[0]
    t=c.encode(a,initial_action=anchor)
    np.testing.assert_allclose(c.decode(t,initial_action=anchor),a,atol=1e-8)
    for v in c.waypoint_clocks(t,initial_action=anchor).values():
        assert v['seconds'][-1]==pytest.approx(8/30 if mode=='duration' else 0)

@pytest.mark.parametrize('mode',['duration','velocity'])
def test_wrapped_rotation(mode):
    c=codec(3,mode); a=np.zeros((8,3)); a[:,2]=np.linspace(3.13,3.23,8)
    t=c.encode(a,initial_action=[0,0,3.12])
    np.testing.assert_allclose(c.decode(t,initial_action=[0,0,3.12]),a,atol=1e-8)

def test_unbounded_correction():
    c=codec(2); a=path(2)*100; t=c.encode(a,initial_action=[0,0])
    assert c.decode(t,initial_action=[0,0])[0,0]==pytest.approx(1000)
    t[:,5]=[.1,-.1,.05,.04,.2,.15,.3,.4]
    times=c.waypoint_clocks(t,initial_action=[0,0])['translation']['seconds']
    assert (np.diff(times)>=0).all() and times[0]>=0
    assert times[-1]<=.4+1e-8

def test_anchor_required_and_snapshot():
    c=codec(2)
    with pytest.raises(TypeError): c.encode(path(2))
    with pytest.raises(ValueError): c.encode(path(2),initial_action=[np.nan,0])
    with pytest.raises(ValueError): current_action_from_state(np.zeros(6),4)
    with pytest.raises(ValueError): PlanarArcClockNativeDecoder(codec_contract='first_delay',native_action_dim=2)
    d=PlanarArcClockNativeDecoder(codec_contract=ARRIVAL_CONTRACT,native_action_dim=2,num_waypoints=8,action_horizon=8)
    state=torch.tensor([[[1.,2.,0.],[3.,4.,0.]],[[5.,6.,0.],[7.,8.,0.]]])
    ctx=d.context_from_observation({'state_agent_obj':state}); state.zero_()
    np.testing.assert_equal(ctx['initial_action'],[[3,4],[7,8]])
    a=path(2); tokens=torch.tensor(np.stack([d.codec.encode(a,initial_action=x) for x in ctx['initial_action']]),dtype=torch.float32)
    out=d.decode(tokens,context=ctx)
    assert out.dtype==tokens.dtype and out.shape==(2,8,2)
    np.testing.assert_allclose(out[:,0],np.tile(a[0],(2,1)),atol=1e-5)
    with pytest.raises(ValueError): d.decode(tokens)

@pytest.mark.parametrize('alignment,offset',[('pre_step',1),('post_step',2)])
@pytest.mark.parametrize('mode',['duration','velocity'])
def test_loader_alignment_and_end_padding(tmp_path,alignment,offset,mode):
    from egomimic.rldb.zarr.numpy_episode import NumpyEpisodeResolver
    row=dict(episode_id='e',embodiment='pushshapes_sim_triangle',total_frames=12,split='train',source='test',observation_alignment=alignment,action_target_offset_obs2=offset)
    (tmp_path/'e').mkdir(); a=np.c_[np.arange(12)+100,np.arange(12)+200,np.arange(12)*.01]
    s=np.c_[np.arange(12),np.arange(12)+1,np.arange(12)*.001,np.zeros((12,3))]
    np.save(tmp_path/'e/actions.npy',a); np.save(tmp_path/'e/state.npy',s)
    raw=json.dumps({'episodes':[row]}).encode(); (tmp_path/'manifest.json').write_bytes(raw)
    transform=get_planar_clock_transform_list(raw_action_horizon=8,native_action_dim=3,num_waypoints=8,clock_mode=mode)
    resolver=NumpyEpisodeResolver(tmp_path,hashlib.sha256(raw).hexdigest(),row['embodiment'],'train',{'actions':{'zarr_key':'actions','horizon':9},'state_agent_obj':{'zarr_key':'state','horizon':2}},transform)
    episode=resolver.resolve()['e']
    for index in (0,4,11):
        b=episode[index]; anchor=s[min(index+1,11),:3]
        np.testing.assert_allclose(b['state_agent_obj'][-1,:3],anchor,atol=1e-7)
        out=codec(3,mode).decode(b['actions'],initial_action=anchor)
        np.testing.assert_allclose(out[0],a[min(index+offset,11)],atol=2e-5)

@pytest.mark.parametrize('arm',['dp16','xy_velocity_m56','xy_absdur_m56'])
@pytest.mark.parametrize('regime',['cotrain','clean'])
def test_recipes_and_matched_budget(arm,regime):
    def load(prefix):
        exp=f'pusht/{prefix}_{arm}'
        if regime=='clean': exp=f'[{exp},pusht/pusher_clean_only_overlay]'
        return compose(config_name='train_zarr_cartesian',overrides=['+experiment='+exp])
    with initialize_config_dir(config_dir=str(ROOT/'egomimic/hydra_configs'),version_base=None):
        c=load('pusher_arrival'); old=load('pusher_clock')
        for key in ('model','trainer','norm_stats'):
            assert OmegaConf.to_container(c[key],resolve=False)==OmegaConf.to_container(old[key],resolve=False)
        assert c.run_provenance.optimizer_updates==240000
        assert c.run_provenance.effective_batch_size==128
        assert c.run_provenance.action_contract.execution_steps==8
        if arm!='dp16':
            contract=OmegaConf.to_container(c.run_provenance.action_contract,resolve=True)
            assert contract['codec_contract']==ARRIVAL_CONTRACT
            for name, dc in c.evaluator.native_decoders.items():
                d=instantiate(dc); assert d.codec.shape==(56,8)
                assert d.representation_context()['timing_layout']==contract['timing_layout']
                for split in ('train_datasets','valid_datasets'):
                    transforms=instantiate(c.data[split][name].resolver.transform_list)
                    actions=np.zeros((81,d.native_action_dim)); actions[:,0]=np.arange(81)+100
                    states=np.array([[0,0,0,999,999,999],[1,0,0,999,999,999]],dtype=float)
                    batch={'actions':actions,'state_agent_obj':states}
                    for transform in transforms: batch=transform.transform(batch)
                    assert batch['actions'].shape==(56,8)
                    decoded=d.decode(batch['actions'],context={'initial_action':states[-1,:d.native_action_dim]})
                    np.testing.assert_allclose(decoded[0],actions[1],atol=1e-5)

        else:
            assert c.run_provenance.action_contract==old.run_provenance.action_contract

@pytest.mark.parametrize('mode',['duration','velocity'])
@pytest.mark.parametrize('kind',['initial_pause','internal_pause','terminal_pause','gripper'])
def test_pause_and_gripper_match_main(mode,kind):
    c=codec(4,mode); a=path(4); anchor=np.zeros(4)
    if kind=='initial_pause': a[:3]=anchor
    if kind=='internal_pause': a[3:6]=a[2]
    if kind=='terminal_pause': a[4:]=a[3]
    raw=np.zeros((8,14)); raw[:,:2]=a[:,:2]; raw[:,3]=a[:,2]; raw[:,6]=a[:,3]
    ref=c.cartesian.transform({'actions':raw,'arc.current_action':np.zeros(14)})['actions']
    token=c.encode(a,initial_action=anchor)
    expected=np.c_[ref[:,:2],np.cos(ref[:,3]),np.sin(ref[:,3]),ref[:,6],ref[:,c.timing_columns]]
    np.testing.assert_allclose(token,expected,atol=1e-12)
    # Duration bridge deliberately removes main's metric speed bound.
    from egomimic.rldb.zarr.cartesian_arc import absolute_times_to_seconds, STREAM_NAMES
    if mode=='duration':
        for i,name in enumerate(STREAM_NAMES[c.streams]):
            for arm,off in enumerate((0,7)):
                col=14+2*i+arm
                ref[:,col]=absolute_times_to_seconds(ref[:,col],c.cartesian._distances(ref,off,name,np.zeros(14)),max_speed=None)
    reference=c.cartesian.detokenize(ref,8,initial_action=np.zeros(14),clocks_in_seconds=mode=='duration')
    decoded=c.decode(token,initial_action=anchor)
    np.testing.assert_allclose(decoded,np.c_[reference[:,:2],reference[:,3],reference[:,6]],atol=1e-10)

def test_rollout_snapshots_raw_anchor_before_normalization_and_fixed8():
    from egomimic.eval.planar_rollout import PlanarGraphPolicy, PlanarActionQueue
    from egomimic.rldb.embodiment.embodiment import get_embodiment_id
    domain='pushshapes_sim_triangle'; eid=get_embodiment_id(domain)
    decoder=PlanarArcClockNativeDecoder(codec_contract=ARRIVAL_CONTRACT,native_action_dim=3,num_waypoints=8,action_horizon=8)
    anchors=[]; original=decoder.decode
    def decode(actions,context=None):
        anchors.append(context['initial_action'].copy()); return original(actions,context)
    decoder.decode=decode
    class Norm:
        embodiments=[eid]; key_types={eid:{'state_agent_obj':'proprio_keys','actions':'action_keys'}}; norm_mode='none'
        def key_shape(self,key,embodiment): return (8,8)
        def normalize(self,obs,e):
            obs['state_agent_obj'].fill_(-999) # deliberately in place
            return obs
        def unnormalize(self,obs,e): return obs
    token=torch.tensor(decoder.codec.encode(path(3),initial_action=[0,0,0]),dtype=torch.float32)[None]
    graph=SimpleNamespace(device='cpu',process_batch_for_training=lambda b:b,forward_eval=lambda b:{domain:{'pred_action':token}})
    policy=PlanarGraphPolicy(graph=graph,normalizer=Norm(),decoder=decoder,embodiment_name=domain,observation_horizon=2,token_shape=(8,8),native_shape=(8,3))
    def obs(i): return {'agent_pos':np.array([i,2*i]),'agent_angle':np.array([.01*i]),'object_pose':np.array([1000,2000,3]),'image':np.zeros((4,4,3),dtype=np.uint8)}
    queue=PlanarActionQueue(policy,execution_horizon=8); queue.reset(obs(0))
    for i in range(9): queue.next_action(); queue.observe(obs(i+1))
    assert len(anchors)==2
    np.testing.assert_allclose(anchors[0],[[0,0,0]])
    np.testing.assert_allclose(anchors[1],[[8,16,.08]],atol=1e-8)

def test_old_normalizer_rejected_before_model_allocation(tmp_path):
    from egomimic.eval.planar_rollout import load_planar_graph_policy
    with initialize_config_dir(config_dir=str(ROOT/'egomimic/hydra_configs'),version_base=None):
        cfg=compose(config_name='train_zarr_cartesian',overrides=['+experiment=pusht/pusher_arrival_xy_absdur_m56'])
    config=tmp_path/'config.yaml'; OmegaConf.save(cfg,config)
    checkpoint=tmp_path/'model.ckpt'; torch.save({'hyper_parameters':{'config_tree':OmegaConf.to_container(cfg,resolve=False)}},checkpoint)
    norm=tmp_path/'norm.json'; norm.write_text('{"normalizer_state":{}}')
    args=dict(checkpoint_path=checkpoint,config_path=config,normalizer_path=norm,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),normalizer_sha256=hashlib.sha256(norm.read_bytes()).hexdigest(),embodiment_name='pushshapes_sim_triangle',device='cpu',use_ema=True)
    with pytest.raises(ValueError,match='Normalizer has incompatible ARC'):
        load_planar_graph_policy(**args)

def test_launch_matrix_preserves_l40s_and_frozen_data():
    from scripts.benchmarks.launch_pusher_holdout import workflow
    import yaml
    campaign='egomimic/hydra_configs/benchmark/pusher_main_arrival.yaml'
    spec=yaml.safe_load((ROOT/campaign).read_text())
    training=evals=0
    for regime in ('cotrain','clean'):
        for seed in (42,43,44):
            w=workflow('a'*40,f'arrival-{regime}-{seed}',regime,seed,campaign)['workflow']
            assert w['resources']['training']['platform']=='ovx-l40s'
            assert w['resources']['training']['gpu']==4
            training+=sum(t['name'].startswith('train-') for t in w['tasks'])
            evals+=sum(t['name'].startswith('eval-') for t in w['tasks'])
    assert training==18 and evals==18*spec['evaluation_shards']
    assert spec['data_release_prefix']=='experiments/pusher-holdout-20261012/data-v1/release/'
    assert spec['geometry']==dict(D_pixels=80.,R_degrees=26.,M=56,raw_horizon=80,fps=30)
