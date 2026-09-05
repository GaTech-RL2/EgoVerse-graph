# UNITE-AV

UNITE-AV keeps the released observation-conditioned UNITE architecture and adds
the decoder-aware action-velocity objective tested by synthetic Variant G.  It
is selected explicitly with:

```text
model=bf/us_unite_av_register_shared_nt8_s42
```

The node visualizer graph is generated from this exact config and records both
identities at the graph root. The policy and objective node sidebars expose the
four decoder-JVP samples and action-velocity loss weight:

```text
python tools/config_graph.py docs/experiments/unite-av-config-graph.json \
  egomimic/hydra_configs/model/bf/us_unite_av_register_shared_nt8_s42.yaml \
  --mode both \
  --override data=pusht/unite_usocket_val01_h16_per_emb_proprio \
  --lint
```

[Open the renderer input](unite-av-config-graph.json).

Do not use `model=G`.  `unite_register_v1` is the architecture identity;
`unite_action_velocity_v1` is the objective identity.  Baseline UNITE remains
`unite_baseline_v1` and executes no decoder JVP.

For UNITE's clean-endpoint parameterization,

```text
Z_t = t Z_0 + (1-t) epsilon
R_v = (predicted_Z_0 - Z_0) / max(1-t, train_eps)
L_AV = mean(||J_decoder(Z_t) R_v||^2).
```

The configured profile evaluates four uniformly distributed bridge samples
already present in each 14-sample flow update.  This is an unbiased stochastic
estimate of the same expectation; it avoids constructing a full Jacobian and
limits higher-order differentiation cost.  The JVP preserves gradients through
the action decoder, the shared denoiser, and the attached clean-latent bridge.
Because fused scaled-dot-product attention and activation checkpointing do not
provide the required stable double backward, only this decoder-JVP pass uses
the math attention kernel with checkpointing bypassed.  Baseline UNITE and
ordinary inference retain their configured fast paths.

The synthetic decoded-noise moment penalty is deliberately not enabled.  It is
a diagnostic-specific regularizer, not part of the accepted PushShapes design.

Full-state checkpoints carry `unite_contract`.  UNITE-AV refuses a baseline or
unidentified full-state checkpoint.  A deliberate baseline-to-AV transfer must
be a separately recorded weights-only warm start.
