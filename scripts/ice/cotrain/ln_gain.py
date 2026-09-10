import torch, glob, sys, re, os
d = sys.argv[1]
cks = sorted(glob.glob(os.path.join(d, "checkpoints", "*.ckpt")), key=lambda p: int(re.search(r"step=(\d+)", p).group(1)) if re.search(r"step=(\d+)", p) else -1)
print("checkpoints:", [os.path.basename(c) for c in cks])
want = None
for c in cks:
    sd = torch.load(c, map_location="cpu", mmap=True, weights_only=False)
    sd = sd.get("state_dict", sd)
    if want is None:
        want = [k for k in sd if ("output_norm" in k or "denoising_output_norm" in k) and not k.startswith("ema")]
        print("keys:", want)
    step = re.search(r"step=(\d+)", c).group(1)
    row = []
    for k in want:
        t = sd[k].float()
        row.append(f"{k.split('.')[-2][:12]}.{k.split('.')[-1][0]}: rms={t.pow(2).mean().sqrt():.3f} max={t.abs().max():.2f}")
    print(f"step {step}: " + " | ".join(row))
