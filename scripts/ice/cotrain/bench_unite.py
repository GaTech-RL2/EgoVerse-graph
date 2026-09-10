"""Where does a UNITE cotrain step spend its time? One H200, synthetic batches.

Times (ms, median of 20 after 5 warm-up) for the pieces of one optimizer step of the
topology-A row at batch 32 per embodiment, and for the candidate speed-ups. Compute-only:
no data loading, no DDP.
"""
import statistics
import sys
import time

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

D = "/home/hice1/agao81/scratch/EgoVerse-graph-ct/egomimic/hydra_configs"
U, C = "pushshapes_sim_u_socket", "pushshapes_sim_chain_gripper"
dev = torch.device("cuda")


def cfg_for(model):
    with initialize_config_dir(config_dir=D, version_base=None):
        return compose(
            config_name="train_zarr_cartesian",
            overrides=[
                "+experiment=pusht/unite_cotrain_usocket_chain_val01_h16",
                f"model={model}",
                # the FAST launcher block
                "model.pipeline.stages.4.flow_mini_batch=14",
                "model.pipeline.stages.4.generative_encoder.gradient_checkpointing=false",
                "model.pipeline.stages.4.generative_encoder.backbone_config.gradient_checkpointing=false",
                f"model.pipeline.stages.4.decoders.{U}.gradient_checkpointing=false",
                f"model.pipeline.stages.4.decoders.{C}.gradient_checkpointing=false",
            ],
        )


def timeit(fn, n=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def build(model_name):
    cfg = cfg_for(model_name)
    algo = instantiate(cfg.model.pipeline, _recursive_=True)
    pipeline = algo.nets  # the nn.Module holding every stage
    pipeline.to(dev).train()
    stages = list(algo.pipeline.stages)
    policy = [s for s in stages if hasattr(s, "generative_encoder")][0]
    obs = [s for s in stages if hasattr(s, "n_obs_steps")][0]
    return cfg, pipeline, policy, obs


def step_pieces(policy, obs, B, flow_steps, label):
    ge = policy.generative_encoder
    policy.flow_steps_per_reconstruction = flow_steps
    policy.flow_mini_batch = flow_steps
    out = {}
    for emb, adim in ((U, 4), (C, 6)):
        target = torch.randn(B, 16, adim, device=dev)
        noise = torch.randn(B, 8, 16, device=dev)
        cond = torch.randn(B, 128, device=dev)

        def full():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                batch = {"target": target, "sampler/noise": noise, "condition": cond}
                batch = policy._forward_training(batch, emb)
                loss = batch["unite/flow_loss"] + (batch["unite/reconstructed_action"] - target).square().mean()
            loss.backward()
            policy.zero_grad(set_to_none=True)

        def tok():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = ge.tokenize(target, emb, noise)
            z.float().square().mean().backward()
            policy.zero_grad(set_to_none=True)

        def dec():
            z = torch.randn(B, 8, 16, device=dev, requires_grad=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                a = policy._decode(z, emb)
            a.float().square().mean().backward()
            policy.zero_grad(set_to_none=True)

        R = flow_steps

        def den():
            z = torch.randn(B * R, 8, 16, device=dev)
            t = torch.rand(B * R, device=dev)
            c = cond.repeat(R, 1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = ge.denoise(z, t, c, emb)
            p.float().square().mean().backward()
            policy.zero_grad(set_to_none=True)

        out[emb] = {"full_step": timeit(full), "tokenizer": timeit(tok), "decoder": timeit(dec), f"denoiser_x{R}": timeit(den)}
    # observation encoder: image + proprio for both embodiments (single frame, 96x96, crop inside)
    img = torch.rand(B, 3, 96, 96, device=dev)
    proprio = torch.randn(B, 64, device=dev)

    def obs_fwd_bwd():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = obs.encoder.forward_packed(obs_packed={"front_img_1": img, "proprio_condition": proprio}, T_total=B)
        feats.float().square().mean().backward()
        obs.zero_grad(set_to_none=True)

    try:
        out["obs_encoder_per_emb"] = timeit(obs_fwd_bwd)
    except Exception as exc:  # signature differs; report and move on
        out["obs_encoder_per_emb"] = f"skipped ({type(exc).__name__}: {str(exc)[:80]})"
    print(f"\n== {label}: B={B} per embodiment, flow_steps={flow_steps}")
    for k, v in out.items():
        print(f"  {k}: {v}")
    return out


def muon_cost(cfg, pipeline):
    opt_fn = instantiate(cfg.model.optimizer)
    opt = opt_fn(list(pipeline.named_parameters()))
    for p in pipeline.parameters():
        p.grad = torch.randn_like(p) * 1e-3

    def step():
        opt.step()

    ms = timeit(step, n=10, warm=3)
    nparams = sum(p.numel() for p in pipeline.parameters())
    print(f"\n== optimizer step (Muon+AdamW) on {nparams/1e6:.1f} M params: {ms:.1f} ms")
    for p in pipeline.parameters():
        p.grad = None
    return ms


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    print(torch.cuda.get_device_name())
    cfg, pipeline, policy, obs = build("bf/ct_unite_register_separate_nt8_h384_s42")
    print("params:", sum(p.numel() for p in pipeline.parameters()) / 1e6, "M")
    base = step_pieces(policy, obs, 32, 14, "topology A, as launched")
    muon = muon_cost(cfg, pipeline)
    step_pieces(policy, obs, 32, 7, "flow_steps 7")
    step_pieces(policy, obs, 64, 14, "batch 64 per embodiment")
    step_pieces(policy, obs, 128, 14, "batch 128 per embodiment")
    # torch.compile on the two DiT backbones (steady state, compile time excluded)
    try:
        ge = policy.generative_encoder
        ge.tokenization_module = torch.compile(ge.tokenization_module, dynamic=False)
        ge.denoising_module = torch.compile(ge.denoising_module, dynamic=False)
        for emb in (U, C):
            dec = policy.action_decoder.decoder_for(emb)
            policy.action_decoder.decoders[emb] = torch.compile(dec, dynamic=False)
        step_pieces(policy, obs, 32, 14, "torch.compile on tokenizer, denoiser, decoders")
    except Exception as exc:
        print("torch.compile variant failed:", repr(exc)[:300])
    cfgB, pipelineB, policyB, obsB = build("bf/ct_unite_register_split_tok_nt8_h384_s42")
    step_pieces(policyB, obsB, 32, 14, "topology B, as launched")
    print("\nsummary: total per-step compute (both embodiments) at B=32, flow 14 =",
          f"{base[U]['full_step'] + base[C]['full_step']:.0f} ms + muon {muon:.0f} ms")


if __name__ == "__main__":
    main()
