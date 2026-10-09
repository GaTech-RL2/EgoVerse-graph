"""Multiplier semantics must not depend on physical speed or future actions."""
import torch
from egomimic.pipeline import stages_speed as m

def test_multiplier_input_contract():
    c=m.SharedMultiplierCondition()
    # Exercise learned dependence rather than only the zero-initialized branch.
    with torch.no_grad():
     c.mlp[0].weight.fill_(1);c.mlp[0].bias.zero_();c.mlp[-1].weight.fill_(.1)
    obs=torch.randn(5,256,requires_grad=True);rates=torch.tensor([.2,.4,.6,.8,1.])
    a=c({"condition":obs,"retiming_rate":rates,"requested_speed":torch.ones(5,1)})["speed_condition"]
    b=c({"condition":obs,"retiming_rate":rates[:,None],"requested_speed":torch.ones(5,1)*999})["speed_condition"]
    assert a.shape==(5,256) and torch.equal(a,b)
    assert not torch.equal(a[0]-obs[0],a[-1]-obs[-1])
    a.sum().backward();assert obs.grad is not None and c.mlp[0].weight.grad.abs().sum()>0
    assert c({"condition":obs.detach()[:,None],"retiming_rate":rates})["speed_condition"].shape==(5,1,256)
    for bad in (torch.zeros(5),-rates,torch.ones(5)*float("nan"),torch.ones(5)*float("inf"),torch.ones(5,2)):
     try:c({"condition":obs,"retiming_rate":bad})
     except ValueError:pass
     else:raise AssertionError("invalid multiplier accepted")
    try:c({"condition":obs,"requested_speed":torch.ones(5,1)})
    except KeyError:pass
    else:raise AssertionError("physical-speed fallback accepted")
    # Preserve historical conditioner behavior and parameter surface.
    old=m.SharedSpeedCondition(1.,condition_dim=256)
    assert old({"condition":obs,"requested_speed":torch.ones(5,1)})["speed_condition"].shape==(5,256)
    print("MULTIPLIER_CONDITION_PASS shapes, telemetry independence, direct rate dependence, gradients, missing/invalid rejection, historical compatibility")
