import os,json,hashlib,gc
from pathlib import Path
from collections import OrderedDict
from types import SimpleNamespace
import torch
from egomimic.pipeline.algo import PipelineAlgo
from egomimic.pipeline.core import Stage
from egomimic.pipeline.stages_sampler import KeyedFeatureProjection,FusedObsEncoder,DPStyleObsEncoder
from egomimic.models.stems.visual_core import VisualCore
from egomimic.pl_utils.pl_model import ModelWrapper
from homogeneous_dithalf_training import HomogeneousDiTHalf,batch_shape_signature
from homogeneous_rng_replay_v2 import rng_state,set_rng,same_rng
fix=Path(os.environ['REPAIR_TASK'])

def test_native_crop_rng_and_shape_cache():
    class SharedSpeedCondition(Stage):
        reads=['condition'];writes=['result']
        def forward(self,batch):
            return {'result':batch['condition']+torch.randn_like(batch['condition'])}
    v=VisualCore.__new__(VisualCore);torch.nn.Module.__init__(v)
    v.embed_dim=4;v.resize_to=None;v.crop_aug=True;v.crop_height=v.crop_width=8;v.crop_scope='frame';v.crop_train_mode='random';v.crop_eval_mode='center';v.crop_sample_mode='v02';v.backbone=torch.nn.Identity();v.pool=torch.nn.AdaptiveAvgPool2d(1);v.head=torch.nn.Linear(3,4)
    stages=[KeyedFeatureProjection('selector','state','projected',{'a':{'input_dim':4},'b':{'input_dim':4}},output_dim=4),FusedObsEncoder(DPStyleObsEncoder({'state':{'input_dim':4}},{'image':v}),{'state':'projected','image':'image'},n_obs_steps=1),SharedSpeedCondition()]
    algo=PipelineAlgo(stages,device='cpu');mod=SimpleNamespace(model=algo,nets=algo.nets)
    cb=HomogeneousDiTHalf(fix);cb.on_fit_start(None,mod)
    native=cb.native_execute
    observations=[]
    for na,nb in [(32,32),(32,32),(13,32),(13,32),(32,32),(32,7),(32,7),(32,32)]:
        batch=OrderedDict((key,{'state':torch.linspace(0,1,n*4).reshape(n,4),'image':torch.linspace(0,1,n*3*12*12).reshape(n,3,12,12),'selector':key}) for key,n in [('a',na),('b',nb)])
        start=rng_state();baseline=native(batch,mode='train');end=rng_state();set_rng(start)
        candidate=algo.forward_training(batch);assert same_rng(end,rng_state())
        for key in batch: torch.testing.assert_close(candidate[key]['result'],baseline[key]['result'],rtol=1e-6,atol=1e-6)
        loss=sum(o['result'].square().mean() for o in candidate.values());loss.backward();assert all(torch.isfinite(p.grad).all() for p in algo.nets.parameters() if p.grad is not None);algo.nets.zero_grad()
        observations.append({'rows':[na,nb],'captures':cb.capture_updates,'grouped_updates':cb.grouped_updates,'rng_exact':True,'finite_gradients':True})
    assert cb.capture_updates==3 and cb.grouped_updates==5 and cb.updates==8
    assert batch_shape_signature({'a':{'x':torch.zeros(2,4),'id':['x','y']}})==batch_shape_signature({'a':{'x':torch.ones(2,4),'id':['z','w']}})
    cb.restore();assert '_execute' not in vars(algo)
    del algo,mod,cb,baseline,candidate,loss,stages,v;gc.collect()
