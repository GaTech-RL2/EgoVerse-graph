# DP / Action Flow observation parity

The user requested checking the 261M DP comparison against PushT Action Flow,
especially the observation encoder, and making the observation path comparable.
The discovered corrected DP co-training source is db7e2840 (PR220); its recorded
full run409210 has51506981parameters, not261M. A separately identified261M run
has not been supplied. This new fresh candidate extends the discovered corrected
DP stack; it does not modify or resume its checkpoint-bound source.

## Findings and repaired contract

The old DP and AF use the same96px RGB input,84px crop, no pretrained weights,
GroupNorm ResNet18,32-keypoint spatial-softmax and64 image features. Their
VisualCore source differs only in the preload-on-restoration guard, which is
inactive when pretrained=false. The old DP feeds native3D agent pose directly
with image features (67conditions). AF converts agent state into4D
[x,y,cos(theta),sin(theta)], then uses one private4→64→64 projection per
embodiment and concatenates64 image features (128conditions).

Select `planar_uc_manual4919_dp_261m_af_obs_multiplier`:

- Copies the exact resolved AF two-stage observation prefix and RGB settings.
- Converts dataset state into the same4D proprio while retaining native state
  as evaluator metadata, and updates U/Chain training and validation together.
- Feeds the same raw augmentation multiplier into a shared1→32→128 MLP.
  The exact maintained multiplier stage is ported from PR223; the generic graph
  runner is unchanged. This stage updates condition before DP denoising.
- Uses DDIM100/H16/common5/U4-only+Chain5 masks, preserving corrected DP targets.
- Uses32+32 source batches, BF16 and80ksteps. DP homogeneous optimization remains
  unsupported; equal batches do not imply an identical fused execution path.
- Uses U-Net widths648/1296/2592 for261382885 total parameters, counted from
  constructed modules on metadata-only devices without allocating weight arrays.

New observation keys and larger U-Net invalidate the old model smoke, optimizer,
normalization and checkpoint contract. Fresh train-only per-source normalization,
typed official launcher adoption, real optimizer/validation smoke, strict reload,
storage and full launch gates are required. No training/evaluation is submitted
by this source change. Model-owned deployment remains unsupported until audited;
this config is not a canonical-rollout command or training READY verdict.

## Parameter-budget limits

Exact AF75k resolved reference instantiated on metadata-only devices:
observation prefix11206048; private encoders106857632 and106857888;
shared field106856992; private decoders69882580 and69883974.
Scalar/multiplier conditioner4288. Total training471549402.
Excluding both training-only encoders gives257833882 resident inference
parameters for both embodiments. The active U/Chain inference paths use
187949908/187951302 respectively because each uses only its selected decoder.

The261382885 DP is close to the *combined resident* AF inference budget but has
more active parameters per embodiment and fewer total training parameters.
It is not a pure equal-training-capacity or equal-compute comparison. Keep
training total, resident inference and active per-domain counts separate.
DP predicts common5; AF uses U4/Chainpoints6 private codecs. Optimizers/objectives,
training computations and samplers differ. Compare native closed-loop quality,
latency/memory and disclosed cost using identical task/seeds/budgets/BF16;
do not compare normalized losses across action representations as equivalent.

## Regression prevention

`test_dp_action_flow_observation_parity.py` checks resolved dimensions/crops,
per-source observations and validation configuration, real CNN/proprio/multiplier
gradients in training and inference graph modes for both embodiments, retained
native state/common5 action targets, masks, and the exact real-module count.
Legacy data helpers default to their previous behavior. This avoids changing
historical runs while making the new recipe explicit.
