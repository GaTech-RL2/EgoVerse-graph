"""Real CPU pipeline and scheduled Lightning validation on synthetic images.

Small widths and CPU SDPA deliberately do not certify full-model CUDA/FA4 smoke.
"""
import copy
from pathlib import Path

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir
from lightning import Trainer
from omegaconf import OmegaConf

from egomimic.eval.image_proprio_eval import ImageProprioEval
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.imagenet_imagefolder import ImageNetImageFolderDataModule
from imagenet_fixture import make_fixture


def resolved_config():
    source = Path(__import__('egomimic').__file__).parent
    with initialize_config_dir(version_base=None, config_dir=str(source / 'hydra_configs')):
        return compose(config_name='train_zarr_cartesian', overrides=[
            '+experiment=pixel_proprio/native_joint/imagenet_image_generation',
            'paths.output_dir=/tmp/fixture-only-imagenet-output'])


def small_config():
    cfg = resolved_config()
    OmegaConf.set_struct(cfg, False)
    cfg.model.pipeline.device = 'cpu'
    m = cfg.model.pipeline.stages[1].model
    m.image_model.update(dict(width=16,depth=1,heads=2,rope_axis_dims=[4,2,2],
        attention_backend='sdpa',attention_dtype='native',strict_fp32=False,
        compile_mode='none',token_execution='padded',swiglu_multiple_of=8,
        allow_patch_compression=True))
    m.proprio_width=12;m.action_width=8;m.camera_width=8
    cfg.model.scheduler=None
    return cfg


def wrapper(cfg, evaluator=None):
    return ModelWrapper(config_tree=cfg,evaluator=evaluator,enable_grad_norm=False,
                        train_log_on_step=True)


def fixture_dm(tmp_path):
    f=make_fixture(tmp_path)
    kwargs=dict(f['datamodule_kwargs'],distributed_world_size=1,batch_size=2)
    dm=ImageNetImageFolderDataModule(**kwargs)
    context=dm.prepare_context(mode='train',normalization=None,normalizer=None)
    return f,dm,context


def test_composed_full_architecture_and_explicit_data_unbound():
    cfg=resolved_config()
    assert OmegaConf.is_missing(cfg.data,'manifest_path')
    assert OmegaConf.is_missing(cfg.data,'validation_anchor_path')
    assert cfg.data.expected_classes==1000
    assert cfg.data.expected_train_images==1281167 and cfg.data.expected_val_images==50000
    assert cfg.trainer.strategy.find_unused_parameters is True
    assert cfg.model.pipeline.stages[1].model.active_streams==['image','proprio','action','camera']
    assert cfg.evaluator.objective=='image_generation'
    assert cfg.evaluator.pixel_diagnostics.max_samples_per_source_kind==32
    with torch.device('meta'):
        model=hydra.utils.instantiate(cfg.model.pipeline.stages[1].model)
    assert sum(p.numel() for p in model.parameters())==314227000
    assert model.active_streams('imagenet')==('image',)
    counts=model.expert_parameter_counts()
    assert counts['image']+counts['language_shared']==159006592
    assert model.pose_dims=={'eva_bimanual':20,'human_bimanual':126}
    assert model.action_dims=={'eva_bimanual':20}


def test_real_optimizer_only_image_and_shared_language_receive_gradients(tmp_path):
    torch.manual_seed(42);_,dm,context=fixture_dm(tmp_path)
    cfg=small_config();model=wrapper(cfg);context.bind(model.model);model.data_context=context
    batch=next(iter(dm.train_dataloader()))
    result=model.model.forward_training(batch)
    loss=model.model.compute_losses(result,batch)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    native=model.model.pipeline.stage_by_id('joint_denoiser').model
    assert native.image_model.image_out.weight.grad is not None
    assert native.language_projection.weight.grad is not None
    assert all(p.grad is None for name,p in native.named_parameters()
               if name.startswith(('proprio_banks.','action.','camera.')))
    opt=torch.optim.AdamW(model.parameters(),lr=5e-6)
    opt.step()
    assert all(p.dtype==torch.float32 for p in native.parameters())
    assert all(v.dtype==torch.float32 for st in opt.state.values() for k,v in st.items()
               if k in ('exp_avg','exp_avg_sq'))


def test_actual_20step_sampler_no_target_leak_and_rng_preserved(tmp_path):
    torch.manual_seed(42);_,dm,context=fixture_dm(tmp_path)
    evaluator=ImageProprioEval(objective='image_generation')
    model=wrapper(small_config(),evaluator);context.bind(model.model,evaluator);model.data_context=context
    batch=next(iter(dm.val_dataloader()))
    native=model.model.pipeline.stage_by_id('joint_denoiser').model
    calls=[]
    def hook(module,args,kwargs):
        assert set(args[0])=={'image'}
        calls.append(args[3]['image'].detach().clone())
    h=native.register_forward_pre_hook(hook,with_kwargs=True)
    model.trainer=Trainer(accelerator='cpu',devices=1,logger=False,enable_checkpointing=False,
                          enable_progress_bar=False,enable_model_summary=False)
    evaluator.model=model;evaluator.on_validation_start()
    before=torch.random.get_rng_state().clone()
    out=evaluator.on_validation_step(batch,0)
    assert torch.equal(before,torch.random.get_rng_state())
    assert len(calls)==20
    assert all(torch.allclose(level,torch.full_like(level,(20-j)/20)) for j,level in enumerate(calls))
    assert torch.isfinite(out['imagenet']['image_proprio_prediction']['image']).all()
    changed=copy.deepcopy(batch);changed['imagenet']['world']['image']['value'].neg_()
    again=evaluator.on_validation_step(changed,0)
    torch.testing.assert_close(out['imagenet']['image_proprio_prediction']['image'],
                               again['imagenet']['image_proprio_prediction']['image'])
    h.remove()


def test_shared_lightning_real_scheduled_generation_validation(tmp_path):
    torch.manual_seed(42);_,dm,context=fixture_dm(tmp_path/'data')
    evaluator=ImageProprioEval(objective='image_generation',pixel_diagnostics=dict(
        artifact_dir=str(tmp_path/'previews'),max_samples_per_source_kind=1,
        contact_sheet_frames=1,write_video=False,log_wandb=False))
    model=wrapper(small_config(),evaluator);context.bind(model.model,evaluator);model.data_context=context
    trainer=Trainer(accelerator='cpu',devices=1,max_steps=2,max_epochs=-1,
        val_check_interval=1,check_val_every_n_epoch=None,num_sanity_val_steps=0,
        logger=False,enable_checkpointing=False,enable_progress_bar=False,
        enable_model_summary=False,default_root_dir=str(tmp_path/'lightning'))
    trainer.fit(model,datamodule=dm)
    assert trainer.global_step==2 and dm.consumed_batches==2
    assert evaluator.has_completed_validation(2)
    assert evaluator.global_validation_batches==2
    key='val/imagenet/image_generation/image_normalized_mse'
    assert torch.isfinite(trainer.callback_metrics[key])
    assert list((tmp_path/'previews').rglob('*.png'))
    checkpoint={};model.on_save_checkpoint(checkpoint)
    assert 'mot_rng_state' in checkpoint
