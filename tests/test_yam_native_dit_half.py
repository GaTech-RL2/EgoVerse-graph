"""Native depth6/depth12 DiT-half: raw derivatives, JVP, RNG and state."""
import copy
import pytest
import torch
from egomimic.models.unite_dit import UniteDiTBackbone


def make(depth,policy):
 return UniteDiTBackbone(input_dim=4,output_dim=4,horizon=2,max_condition_tokens=1,condition_dim=4,hidden_dim=16,depth=depth,num_heads=4,in_context_len=2,in_context_start=4,max_content_tokens=2,time_fourier_dim=8,dropout=.1,gradient_checkpointing=True,checkpoint_policy=policy).float().train()

def raw_equal(a,b):
 if a is None or b is None:
  assert a is None and b is None;return
 assert a.shape==b.shape and a.dtype==b.dtype
 assert torch.isfinite(a).all() and torch.isfinite(b).all()
 assert torch.equal(a.detach().contiguous().view(torch.uint8),b.detach().contiguous().view(torch.uint8))

@pytest.mark.parametrize("depth",[6,12])
def test_native_half_checkpoint_raw_higher_order_rng_and_state(depth):
 torch.set_num_threads(1)
 torch.manual_seed(42)
 model=make(depth,"all")
 with torch.no_grad():
  for p in model.parameters(): p.normal_(0,.1)
 state={k:v.detach().clone() for k,v in model.state_dict().items()}
 def evaluate(policy):
  model.checkpoint_policy=policy;model.checkpoint_policy_counts={"direct":0,"checkpoint":0}
  torch.manual_seed(73)
  x=torch.linspace(-1,1,8).reshape(1,2,4).requires_grad_()
  t=torch.tensor([.4],requires_grad=True)
  c=torch.linspace(-.2,.3,4).reshape(1,4).requires_grad_()
  params=tuple(model.parameters())
  def forward(value):return model(value,t,c,content_tokens=value)
  with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
   y=forward(x)
   first=torch.autograd.grad(y.square().sum(),(x,t,c)+params,create_graph=True,retain_graph=True,allow_unused=True)
   second=torch.autograd.grad(sum(v.square().sum() for v in first if v is not None),(x,t,c),retain_graph=True)
   repeat=torch.autograd.grad(y.sum(),(x,t,c),retain_graph=True)
   torch.manual_seed(79)
   jy,jvp=torch.autograd.functional.jvp(forward,x,torch.ones_like(x))
  return [y,*first,*second,*repeat,jy,jvp,torch.get_rng_state()],dict(model.checkpoint_policy_counts)
 baseline,counts_all=evaluate("all");candidate,counts_half=evaluate("dit_half")
 for a,b in zip(baseline,candidate):raw_equal(a,b)
 assert counts_all["direct"]==0 and counts_all["checkpoint"]>0
 assert counts_half["direct"]>0 and counts_half["checkpoint"]>0
 assert counts_half["direct"]==counts_half["checkpoint"]
 for k,v in model.state_dict().items():raw_equal(v,state[k])
 restored=make(depth,"dit_half");restored.load_state_dict(state,strict=True)
 for k,v in restored.state_dict().items():raw_equal(v,state[k])
 assert not torch.cuda.is_initialized()

@pytest.mark.parametrize("policy,depth,checkpointing",[("invalid",6,True),("dit_half",5,True),("dit_half",6,False)])
def test_native_policy_fails_closed(policy,depth,checkpointing):
 with pytest.raises(ValueError):
  UniteDiTBackbone(input_dim=4,output_dim=4,horizon=2,max_condition_tokens=1,condition_dim=4,hidden_dim=16,depth=depth,num_heads=4,gradient_checkpointing=checkpointing,checkpoint_policy=policy)
