import copy
import pytest
import torch
from egomimic.models.mot_components import RMSNorm,SwiGLU,swiglu_hidden_dim
from egomimic.models.world_model import MoTBlock
from mot_architecture_fixture import model,world,activate,install_reference_attention


@pytest.fixture(autouse=True)
def one_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_disabled_options_strict_state_rng_and_outputs():
    torch.manual_seed(72);a=model();rng=torch.get_rng_state().clone()
    torch.manual_seed(72);b=model(explicit=True)
    assert torch.equal(rng,torch.get_rng_state())
    assert a.state_dict().keys()==b.state_dict().keys()
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    b.load_state_dict(a.state_dict(),strict=True)
    torch.manual_seed(90);activate(a);b.load_state_dict(a.state_dict(),strict=True)
    for emb in ['robot','human','imagenet']:
        w,v,l,language=world(emb,image_only=emb=='imagenet')
        x,y=a(w,emb,v,l,language),b(w,emb,v,l,language)
        assert all(torch.equal(x[k],y[k]) for k in x)


@pytest.mark.parametrize('width,hidden',[(1024,2752),(512,1344),(384,1024)])
def test_aligned_swiglu_budget(width,hidden):
    with torch.device('meta'):ffn=SwiGLU(width)
    assert ffn.hidden_dim==hidden==swiglu_hidden_dim(width,4)
    assert sum(p.numel() for p in ffn.parameters())==3*width*hidden+2*hidden+width


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
@pytest.mark.parametrize('affine',[False,True])
def test_rmsnorm_fp32_reference_dtype_and_grad(dtype,affine):
    n=RMSNorm(8,elementwise_affine=affine)
    x=(torch.randn(2,3,4,8)*100).to(dtype).requires_grad_()
    y=n(x)
    expected=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).to(dtype)
    assert y.dtype==dtype and torch.equal(y,expected)
    y.float().square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert (n.weight is None)==(not affine)


def test_qk_norm_is_private_per_head_and_before_rope(monkeypatch):
    m=model(enabled=True)
    banks=[m.image_model,*m.proprio_banks.values(),m.action,m.camera]
    assert len(banks)==5
    params=[]
    import egomimic.models.world_model as image_module
    import egomimic.models.joint_image_proprio as vector_module
    for index,bank in enumerate(banks):
        assert isinstance(bank.norm[0] if index==0 else bank.norm,RMSNorm)
        for block in bank.blocks:
            qn=block.q_norm[0] if index==0 else block.q_norm
            kn=block.k_norm[0] if index==0 else block.k_norm
            assert qn.weight.shape==kn.weight.shape==(8,)
            assert qn.weight is not kn.weight
            params.extend([id(qn.weight),id(kn.weight)])
            assert not (block.norm1[0] if index==0 else block.norm1).elementwise_affine
            seen=[]
            def rope(value,table):
                seen.append(value)
                return value+7
            monkeypatch.setattr(image_module if index==0 else vector_module,'apply_temporal_rope',rope)
            width=16 if index==0 else bank.inputs[next(iter(bank.inputs))].out_features
            x=torch.randn(1,3,width);groups=torch.arange(3)[None]
            mods=tuple(torch.zeros(3,width) for _ in range(6))
            qkv=(block._project_inputs([x],[torch.zeros(1,3)],(None,),mods,groups)
                 if index==0 else block.project(x,None,mods,groups))
            assert len(seen)==2
            for value in seen:torch.testing.assert_close(value.float().square().mean(-1),torch.ones_like(value[...,0]),rtol=2e-4,atol=2e-4)
            assert torch.equal(qkv[0],seen[0]+7) and torch.equal(qkv[1],seen[1]+7)
    assert len(params)==len(set(params))


@pytest.mark.parametrize('emb',['robot','human','imagenet'])
def test_new_full_family_forward_backward_active_banks(emb):
    torch.manual_seed(72);m=model(enabled=True);activate(m)
    w,v,l,language=world(emb,image_only=emb=='imagenet');out=m(w,emb,v,l,language)
    assert set(out)==set(m.active_streams(emb))
    loss=sum(x.square().mean() for x in out.values());loss.backward()
    assert all(torch.isfinite(x).all() and torch.equal(x[~w[k]['valid']],torch.zeros_like(x[~w[k]['valid']])) for k,x in out.items())
    active=[p for p in m.parameters() if p.grad is not None]
    assert active and all(torch.isfinite(p.grad).all() for p in active)
    assert m.image_model.blocks[0].q_norm[0].weight.grad.abs().sum()>0
    if emb=='imagenet':
        assert all(p.grad is None for p in m.proprio_banks.parameters())
        assert all(p.grad is None for p in m.action.parameters())
        assert all(p.grad is None for p in m.camera.parameters())
        assert all(p.grad is None for p in m.camera_identity.parameters())
        assert m.language_projection.weight.grad is not None
        assert set(out)=={'image'}
    else:
        assert m.proprio_banks[emb].blocks[0].q_norm.weight.grad.abs().sum()>0
        inactive='human' if emb=='robot' else 'robot'
        assert all(p.grad is None for p in m.proprio_banks[inactive].parameters())


@pytest.mark.parametrize('emb',['robot','human','imagenet'])
def test_new_packed_dense_forward_backward_parity(monkeypatch,emb):
    calls=install_reference_attention(monkeypatch)
    torch.manual_seed(75);a=model(enabled=True,flash=True);activate(a)
    b=model(enabled=True,packed=True,flash=True);b.load_state_dict(a.state_dict())
    w,v,l,language=world(emb,image_only=emb=='imagenet')
    x,y=a(w,emb,v,l,language),b(w,emb,v,l,language)
    for k in x:torch.testing.assert_close(x[k],y[k],rtol=.035,atol=.003)
    sum(z.square().mean() for z in x.values()).backward()
    sum(z.square().mean() for z in y.values()).backward()
    for (_,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=.08,atol=.002)
    assert 'packed' in calls


@pytest.mark.parametrize('kw',[dict(qk_norm=1),dict(use_rmsnorm='yes'),dict(use_swiglu=None),dict(swiglu_multiple_of=0)])
def test_invalid_architecture_controls_fail_closed(kw):
    with pytest.raises(ValueError):MoTBlock(16,2,**kw)


@pytest.mark.parametrize('feature',['qk_norm','use_rmsnorm','use_swiglu'])
def test_each_feature_is_independently_optional(feature):
    block=MoTBlock(16,2,modalities=('image',),**{feature:True})
    x=torch.randn(2,4,16,requires_grad=True)
    out=block([x],[torch.zeros(2,4)],[torch.ones(2,4,dtype=torch.bool)])[0]
    assert out.shape==x.shape and torch.isfinite(out).all()
    out.square().mean().backward()
    assert torch.isfinite(x.grad).all()


def test_new_checkpoint_roundtrip_requires_matching_architecture(tmp_path):
    torch.manual_seed(73);a=model(enabled=True);activate(a)
    path=tmp_path/'new-architecture.pt';torch.save(a.state_dict(),path)
    state=torch.load(path,weights_only=True)
    b=model(enabled=True);b.load_state_dict(state,strict=True)
    with pytest.raises(RuntimeError):model().load_state_dict(state,strict=True)
    w,v,l,language=world('robot')
    x,y=a(w,'robot',v,l,language),b(w,'robot',v,l,language)
    assert all(torch.equal(x[k],y[k]) for k in x)


@pytest.mark.parametrize('emb',['robot','imagenet'])
def test_mixed_precision_keeps_parameters_fp32_and_gradients_finite(emb):
    torch.manual_seed(74);m=model(enabled=True);activate(m)
    w,v,l,language=world(emb,image_only=emb=='imagenet')
    with torch.autocast('cpu',dtype=torch.bfloat16):
        out=m(w,emb,v,l,language)
        loss=sum(x.float().square().mean() for x in out.values())
    loss.backward()
    assert all(p.dtype==torch.float32 for p in m.parameters())
    assert all(p.grad.dtype==torch.float32 and torch.isfinite(p.grad).all()
               for p in m.parameters() if p.grad is not None)
    assert all(torch.isfinite(x).all() for x in out.values())
