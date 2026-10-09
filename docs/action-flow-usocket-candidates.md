# Conditional U-Socket ports of the torus candidates

These are three research candidates, not three equivalent reconstruction-weight
ablations. All use U-Socket BC, fresh seed42, H16 normalized `[x,y,cosθ,sinθ]`,
latent8 per token, one observation frame, a shared 12×512 AdaLN field, private
context-free two-layer width20 codecs, AdamW3e-5, 8K warmup, cosine floor3e-6,
240K updates, batch32 and BF16 on one H100 or H200. No warm-start or
reconstruction-only phase. The generic pipeline runner is unchanged.

## Configs and gradients

### FM-only clean stop-gradient (method 2)

Config: `pusht/action_flow_bc_usocket_latent_fm_sg_recon1_s42`.

- Reconstruction: `A → E → g → MSE(A)`; updates E and g, weight1.
- Latent FM: `A → E → stop-gradient → latent bridge → v(z,t,O) → latent MSE`;
  both the bridge endpoint and velocity target detach. Updates v and the
  observation encoder, not E or g.
- Action Flow: `A → E → attached latent bridge → v → residual → J_g(residual)`;
  squared residual mean updates E, v, g and the observation encoder, weight1.

The two v forwards share time, Gaussian noise and conditioning-dropout mask.
Fourteen samples per content means 28 field sample-equivalents, not 14, for
this variant. A zero FM/reconstruction parameter intersection is intentional;
its gradient cosine is undefined, not evidence of broken training.
Inference: `Gaussian latent → reverse Euler16(v,O) → g → action`.

### Learned Gaussian-bridge likelihood (method 3)

Config: `pusht/action_flow_bc_usocket_bridge_likelihood_s42`.

Private reference mean `mu(A,t)=(1-t)E_raw(A,t)` ends exactly at zero. The
reference stage, Gaussian noising stage, conditional reverse-mean transformer,
decoder and objective are separate nodes.

Interior: `A → mu_k,mu_(k-1) → Gaussian z_k and attached reverse target →
M(z,k,O)=z+field(z,k/32,O) → equal-variance Gaussian KL`.
Boundary: `A → mu_1 + .1*noise → M(z,1,O) → g → action Gaussian NLL`.
Both mean and target gradients remain attached. Both terms update the private
mean encoder, field and observation encoder; the boundary also updates g.
There is no latent-FM, clean reconstruction, JVP or decoded-noise scale loss.

K32, sigma1=.1, sigma32=1, geometric schedule, rho=.95, tau=.02. Draw 14
interior levels/noises per content plus one independent boundary noise. The
batched field sees 15 sample-equivalents in one forward. Interior loss uses the
31× uniform-level correction and variance `(1-rho²)*sigma_(k-1)²`.

All squared coordinate residuals are **summed over the full H16 chunk**, then
averaged over examples/level samples. Boundary weighting is therefore 80000×
per-coordinate action MSE for H16×D4, not the toy three-coordinate coefficient.
Logged likelihood terms omit parameter-independent Gaussian constants; they
are not exact reported marginal log-likelihood estimates.

Inference: `z32~N(0,I) → 31 stochastic Gaussian reverse transitions →
final learned mean M(z1,1,O) → g → add tau*action_noise`. Exactly32 field calls,
31 latent innovations, and one action-noise draw. No deterministic Euler
substitution. Finite output noise models a smoothed action distribution.

`Train/MSE` is decoded noisy-boundary-mean error; `Valid/MSE` is actual generated
action error. Neither is clean-codec reconstruction. Ordinary CFM trajectory/
clean-latent/Jacobian diagnostics are disabled; normalized/native generated
errors and fixed-bank stochastic EnergyScore@32 remain required.

### Exact graph section (method 4, restricted diagnostic)

Config: `pusht/action_flow_bc_usocket_graph_section_s42`.

`E(A)=(A,f(A))`; split each latent token into x4,h4 and decode
`g(x,h)=x + [R(x,h)-R(x,f(x))]`. The subtraction is grouped before adding x.
The same f parameter objects are used at both boundaries. A model-specific
Hydra factory injects them once: repeating a Hydra config is not weight sharing.
E owns only f; decoder-only R is not included in E's parameter set.

Training: `A → E → latent bridge → v → latent FM`, plus
`latent residual → J_g → Action Flow loss`, both fully attached. No reconstruction
optimizer term; `g(E(A))=A` is checked and logged as a diagnostic. Inference uses
Gaussian latent → reverse Euler16 → g. Fixed-level/Jacobian diagnostics remain;
activation matching is disabled with an empty map and cknna_k=0.

**R6 FAIL / restricted diagnostic:** explicit action coordinates are embedded in
the latent. Exact reconstruction is not a general flexible-interface solution
and is not evidence of good generation.

## Comparison and operational boundaries

Methods2/4 retain the historical U-Socket scale-regularizer weight0; the torus
winners used scale1. That difference is explicit, not a faithful toy ablation.
Conditional quality, scale stability and transfer remain OPEN. R2–R5/R8 are
implemented contracts, not demonstrated cross-embodiment transfer; R7 requires
observed GPU cost. R1/R9/R10 are not passed merely by finite smoke metrics.

Use the maintained `scripts/train/launch_action_flow_usocket.sbatch` typed
entrypoint with exact source/runtime/config/data/norm bindings. Reuse hashed
2999-episode content, 2970/29 split, and train-only normalization receipts;
never recompute unchanged statistics to prepare another row. Validation10K,
immutable checkpoint40K, all scheduled/signal saves retained. Full training
requires that candidate's real optimizer+validation smoke and strict reload.

An explicitly authorized ICE/Phoenix queue race needs distinct attempt paths and
tracked scheduler IDs. Keep the first allocation, confirm loser cancellation,
and never let two writers share W&B/output. Preserve the original queue deadline.
Restart testing must prove checkpoint-based optimizer continuation and the same
W&B ID; a passing ordinary smoke alone does not prove requeue continuity.
