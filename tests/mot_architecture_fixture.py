"""Small real native-model fixtures; optional CUDA attention uses CPU references."""
import torch
from torch.nn import functional as F
from egomimic.models.image_proprio import ImageProprioModel
from egomimic.models.joint_image_proprio import JointImageProprioModel
from egomimic.models.attention_execution import FlashAttentionKernels


def model(*, enabled=False, packed=False, flash=False, explicit=False):
    options = dict(qk_norm=enabled, use_rmsnorm=enabled, use_swiglu=enabled,
                   swiglu_multiple_of=8) if enabled or explicit else {}
    image = ImageProprioModel(
        objective='visual_only', phase_a_condition_on_proprio=False,
        image_dim=12, width=16, depth=2, heads=2, mlp_ratio=4,
        max_cameras=3, max_tokens=64, require_observed_context=False,
        use_phase_embedding=False, noise_conditioning='adaln_zero',
        position_encoding='spatiotemporal_rope', rope_axis_dims=(4,2,2),
        spatial_grid_size=(2,2), token_execution='packed_valid' if packed else 'padded',
        attention_backend='flash4' if flash else 'sdpa',
        attention_dtype='bfloat16' if flash else 'native', **options)
    return JointImageProprioModel(image, {'robot':3, 'human':5}, proprio_width=12,
        action_dims={'robot':2}, action_width=8, camera_dims={'robot':9,'human':9},
        camera_width=8, active_streams=('image','proprio','action','camera'),
        separate_proprio_experts=True, language_feature_dim=1024)


def world(embodiment='robot', image_only=False):
    valid=torch.tensor([[1,1,0],[1,1,1]],dtype=torch.bool)
    times=torch.tensor([[0.,.3,.8],[0.,.3,.8]])
    result={}
    for name,repeat,dim in [('image',4,12),('proprio',1,3 if embodiment=='robot' else 5),
                            ('action',1,2),('camera',1,9)]:
        if (image_only and name!='image') or (embodiment=='human' and name=='action'):continue
        v=valid.repeat_interleave(repeat,1)
        result[name]=dict(value=torch.randn(2,3*repeat,dim),valid=v,
                          observed=torch.zeros_like(v),target_valid=v.clone(),
                          time=times.repeat_interleave(repeat,1))
        result[name]['value'][~v]=float('nan')
    image=result['image']
    image.update(xy=torch.tensor([[-.5,-.5],[.5,-.5],[-.5,.5],[.5,.5]]).repeat(2,3,1),
                 camera=torch.zeros(2,12,dtype=torch.long))
    # Non-prefix padding exercises actual packing and RoPE metadata selection.
    order=torch.tensor([0,8,4,1,9,5,2,10,6,3,11,7])
    result['image']={k:v.index_select(1,order) for k,v in image.items()}
    if 'camera' in result:result['camera']['camera']=torch.zeros(2,3,dtype=torch.long)
    values={k:torch.randn_like(v['value']) for k,v in result.items()}
    levels={k:torch.full_like(v['time'],.7).masked_fill(~v['valid'],0) for k,v in result.items()}
    return result,values,levels,torch.randn(2,1024)


def activate(model):
    """Open zero-init gates/heads so tests cover the full backward path."""
    with torch.no_grad():
        model.image_model.image_out.weight.normal_(std=.08)
        for block in model.image_model.blocks:
            block.adaln_image[-1].weight.normal_(std=.05)
            block.adaln_image[-1].bias.normal_(std=.03)
        for expert in [*model.proprio_banks.values(),model.action,model.camera]:
            for layer in expert.outputs.values():layer.weight.normal_(std=.08)
            for block in expert.blocks:
                block.adaln[-1].weight.normal_(std=.05)
                block.adaln[-1].bias.normal_(std=.03)


def install_reference_attention(monkeypatch):
    calls=[]
    def dense(q,k,v,*,causal):
        assert causal is False
        return F.scaled_dot_product_attention(q.float().transpose(1,2),k.float().transpose(1,2),
            v.float().transpose(1,2),dropout_p=0.).transpose(1,2).to(q.dtype)
    def load(backend,device):
        assert torch.device(device).type=='cpu'
        def padded(q,k,v,**kw):
            calls.append('dense');out=dense(q,k,v,**kw)
            return (out,None) if backend=='flash4' else out
        def packed(q,k,v,*,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,causal):
            calls.append('packed');qs=cu_seqlens_q.tolist();ks=cu_seqlens_k.tolist()
            out=torch.cat([dense(q[a:b][None],k[c:d][None],v[c:d][None],causal=causal)[0]
                           for a,b,c,d in zip(qs,qs[1:],ks,ks[1:])])
            return (out,None) if backend=='flash4' else out
        return FlashAttentionKernels(backend,padded,packed)
    for name in ['image_proprio','world_model','joint_image_proprio']:
        monkeypatch.setattr('egomimic.models.'+name+'.load_flash_attention_kernels',load)
    return calls
