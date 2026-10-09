from pathlib import Path
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from egomimic.pipeline.stages_matched_dp import SharedNativeDPStage


def test_shared_native_dp_both_domains_train_infer_and_gradients():
    torch.set_num_threads(1)
    stage=SharedNativeDPStage(down_dims=(8,16,32),step_embed_dim=8,n_groups=4,inference_steps=2,condition_dropout_probability=0)
    robot=stage.branches["yam_bimanual"].stages[1].policy.model
    human=stage.branches["human_bimanual"].stages[1].policy.model
    assert robot.mid_modules is human.mid_modules
    assert robot.down_modules[1] is human.down_modules[1]
    assert robot.down_modules[0][0] is not human.down_modules[0][0]
    assert robot.final_conv[1] is not human.final_conv[1]
    for eid,width in ((7,14),(3,138)):
        stage.zero_grad(set_to_none=True)
        condition=torch.randn(1,256,requires_grad=True)
        out=stage.execute({"embodiment":torch.tensor([eid]),"condition":condition,
            "target":torch.randn(1,100,width)},mode="train")
        loss=out["loss/diffusion_noise"];assert torch.isfinite(loss)
        loss.backward();assert condition.grad.abs().sum()>0
        assert sum(p.grad.abs().sum() for p in robot.mid_modules.parameters() if p.grad is not None)>0
        with torch.no_grad():
            pred=stage.execute({"embodiment":torch.tensor([eid]),"condition":condition.detach()},mode="inference")["pred_action"]
        assert pred.shape==(1,100,width) and torch.isfinite(pred).all()


def comparable(value):
    return OmegaConf.to_container(value,resolve=False) if OmegaConf.is_config(value) else value


def test_matched_configs():
    root=Path(__file__).parents[1]/"egomimic/hydra_configs"
    with initialize_config_dir(config_dir=str(root),version_base="1.3"):
        a,d=[compose(config_name="train_zarr_cartesian",overrides=["hydra/launcher=basic",
            "+experiment=e1/yam_human_matched_"+name]) for name in ("af64_multiplier","dp_noaug")]
    for key in ("trainer","callbacks","hpt","e1","seed","model.scheduler"):
        assert comparable(OmegaConf.select(a,key))==comparable(OmegaConf.select(d,key)),key
    for x,y in zip(a.model.pipeline.stages[:3],d.model.pipeline.stages[:3]):assert comparable(x)==comparable(y)
    assert d.model.num_latent_tokens is None and d.model.latent_dim is None
    assert d.run_provenance.objective.epsilon_samples_per_content==1
    assert a.evaluator.action_flow_diagnostics.provenance.latent_shape==[64,16]
    assert a.model.pipeline.flow_inference_method=="euler" and d.model.pipeline.dp_inference_steps==50
    assert (a.model.num_latent_tokens,a.model.latent_dim,a.model.action_horizon)==(64,16,100)
    assert a.data.train_datasets.yam_bimanual.expected_train_episode_count==375
    assert a.data.train_datasets.human_bimanual.expected_train_episode_count==37
    assert a.stationary_speed.human_rates==[.2,.4,.6,.8,1.]
    assert d.stationary_speed.human_rates==[1.]
    assert d.model.pipeline.conditioning_input=="none"
    assert d.run_provenance.speed_augmentation.enabled is False
    for domain in ("yam_bimanual","human_bimanual"):
        aug=a.data.train_datasets[domain];native=d.data.train_datasets[domain]
        assert comparable(aug.resolver.transform_list.native_transforms)==comparable(native.resolver.transform_list.native_transforms)
        assert native.resolver.transform_list.window_transform._target_.endswith("CompleteNativeWindow")
        for key in ("mode","valid_ratio","split_seed","expected_train_episode_count","expected_valid_episode_count"):
            assert aug[key]==native[key]
        assert comparable(aug.resolver)["folder_path"]==comparable(native.resolver)["folder_path"]
    assert a.stationary_speed.yam_rates==d.stationary_speed.yam_rates==[1.]


def test_native_af_and_dp_optimizers_and_schedule_contracts():
    from hydra.utils import instantiate
    root=Path(__file__).parents[1]/"egomimic/hydra_configs"
    schedules=[]
    with initialize_config_dir(config_dir=str(root),version_base="1.3"):
        for name in ("af64_multiplier","dp_noaug"):
            cfg=compose(config_name="train_zarr_cartesian",overrides=["hydra/launcher=basic",
                "+experiment=e1/yam_human_matched_"+name])
            if name=="af64_multiplier":
                assert cfg.model.optimizer._target_=="egomimic.utils.unite_optim.ReleasedUniteCompositeOptimizer"
                assert cfg.model.optimizer_named_parameters is True
                assert cfg.model.optimizer.muon_momentum==0.95
                continue
            opt=instantiate(cfg.model.optimizer,params=[torch.nn.Parameter(torch.zeros(1))])()
            assert isinstance(opt,torch.optim.AdamW)
            schedule=instantiate(cfg.model.scheduler,optimizer=opt)
            if callable(schedule):schedule=schedule()
            schedules.append([schedule.lr_lambdas[0](step) for step in (0,7999,8000,12000,20000,80000)])
    assert len(schedules)==1


def test_strict_dp_checkpoint_preserves_shared_core():
    args=dict(down_dims=(8,16,32),step_embed_dim=8,n_groups=4,inference_steps=2)
    old=SharedNativeDPStage(**args);new=SharedNativeDPStage(**args)
    new.load_state_dict(old.state_dict(),strict=True)
    a=new.branches["yam_bimanual"].stages[1].policy.model
    b=new.branches["human_bimanual"].stages[1].policy.model
    assert a.mid_modules is b.mid_modules


def test_unaugmented_window_guard_preserves_values_and_complete_anchors():
    import numpy as np
    from egomimic.rldb.zarr.physical_retiming import CompleteNativeWindow
    from egomimic.rldb.zarr.episode_split import complete_window_count
    for horizon in (30,100):
        guard=CompleteNativeWindow(horizon,["action"])
        guard.bind_episode({}, {"action":{"horizon":horizon}})
        action=np.arange(horizon)[:,None]
        batch={"action":action,"_retiming_view":0}
        result=guard.transform(batch)
        assert result["action"] is action
        assert "retiming_rate" not in result and "requested_speed" not in result
        assert guard.sample_views==1
        assert complete_window_count(200,guard.required_frames)==201-horizon
