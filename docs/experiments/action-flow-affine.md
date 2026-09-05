# Action-flow affine and nonlinear torus comparison

This experiment implements variants A--F from the Action Flow Affine Test Plan.
All adapter variants use an eight-dimensional clean-to-noise latent bridge and
generate by integrating the learned field backward from `t=1` to `t=0` before
decoding. The direct baseline retains its existing three-dimensional flow.

The training dataset must be generated with `--source-dim 8`. Its
`source_gaussian_latent` is independent of the paired torus target, and its first
three dimensions are exactly `source_gaussian_3d`, so direct FM and the fixed
lift see the same action-space corruption samples during evaluation.

Use a separate large frozen evaluation dataset and generate the three-seed
config matrix with:

```bash
python scripts/data/generate_gaussian_torus.py \
  --output /path/to/gaussian_torus_latent8_train.npz \
  --count 4096 --source-dim 8 --seed 42
python scripts/data/generate_gaussian_torus.py \
  --output /path/to/gaussian_torus_latent8_eval.npz \
  --count 32768 --source-dim 8 --seed 4242
python scripts/experiments/build_action_flow_affine_configs.py \
  --output-dir /path/to/experiment/config \
  --experiment-root /path/to/experiment \
  --training-dataset /path/to/gaussian_torus_latent8_train.npz \
  --evaluation-dataset /path/to/gaussian_torus_latent8_eval.npz
```

The primary comparison is symmetric nearest-neighbor squared distance computed
on the identical 1,536-point target cloud for every run. Report all three seeds
and their mean and standard deviation; there is no separate pass threshold.
Energy Distance, surface error, angular coverage, reconstruction, path error,
latent scale, Jacobian singular values, and decoded-radius quantiles are
diagnostics. The config manifest records both dataset SHA-256 values, and config
generation fails if either frozen dataset is missing. Checkpoints default to
every 50,000 optimizer steps.

Dry-run note: an attempted custom evaluation split with the default 0.9 train
fraction plus 0.2 validation fraction was correctly rejected. The generator's
split-sum validator is the durable guard; keep the documented default 0.9/0.05
split (or set both fractions so their sum is below one).

Clean-clone note: the first Skynet preflight found that path-invoked scripts
could resolve an older environment-installed `egomimic` package. Every
synthetic data/train/export entry point now prepends its own checkout root, and
a subprocess regression test runs the generator from outside the repository.

Smoke retry note: the first nonlinear-path smoke exposed validation tensors
created under inference mode and then reused by an autograd Jacobian. Evaluation
data now loads outside inference mode, and the real subprocess test covers both
fixed-lift and nonlinear-path diagnostics. Use `--run-suffix` for immutable,
collision-free retry output and W&B identities.
