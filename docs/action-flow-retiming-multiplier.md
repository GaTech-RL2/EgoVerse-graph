# Retiming multiplier conditioning

The user corrected the conditioning contract on 2026-10-08: use the selected
augmentation multiplier, not the measured commanded-XY speed of the retimed
chunk. `PlanarCommandRetiming` already emits `retiming_rate`; no new speed
estimator or change to interpolation is needed. A multiplier of 1 is original
timing; 2 samples the original command trajectory twice as quickly.

Select `action_flow_cotrain_uc_multiplier_interpolation` for the new diagnostic
recipe. It keeps H16 actions and the existing 8-token x 16-feature latent.
`SharedSpeedCondition(conditioning_input="retiming_multiplier",
speed_reference=None)` reads `retiming_rate`, requires finite positive [B,1]
values, and feeds the raw dimensionless multiplier to the conditioning MLP.
The MLP output is added to observation features for the shared latent bridge
and conditional velocity field. U-Socket remains 1x; ChainGripper keeps the
existing 1/1.25/1.5/1.75/2x augmentation views.

For rollout this recipe requires explicit `deployment.requested_multiplier`;
1.0 requests original timing. It rejects `deployment.requested_speed` and
never infers conditioning from predicted actions. Result metadata records
the multiplier and dimensionless units. BF16 model inference is unchanged.

The user subsequently authorized retirement of the old precomputed-speed
code/recipe, this scalar run's checkpoints and saved rollout results. The old
recipe and native-speed implementation are removed from this development
branch; existing checkpoint-bound source commits remain historical provenance.
Retiming no longer calculates measured XY speed. This
new input meaning requires fresh initialization and new experiment identity;
an additional persistent multiplier-contract buffer makes strict loading reject
physical-speed weights even though scalar MLP parameter dimensions match.
do not relabel, resume or evaluate an existing speed-conditioned checkpoint as
multiplier-conditioned. The user requested a code change, not a fresh launch.
No running job or historical result is changed. This recipe is a source
candidate, not a passing training preflight or approved complete architecture.

## Design contract audit

| Requirement | Verdict | Evidence / remaining work |
|---|---|---|
| R1 | OPEN | Inherited latent FM; actual guided sampler distribution and achieved joint fitting remain unproven. |
| R2 | PASS | One shared latent field and Gaussian latent corruption; multiplier changes only its context. |
| R3 | OPEN | Regression verifies multiplier reaches the shared field in train/inference and receives gradients; full joint encoder/decoder/generator updates not retested. |
| R4 | OPEN | Fresh joint initialization intended; inherited FM stop-gradient plus action-velocity coupling must pass actual full-model gradient gates. |
| R5 | PASS | Within-domain pairs and observation/multiplier context at shared generator; context-free private action decoders. |
| R6 | PASS | Nonlinear private codecs retain 4D U-Socket and 6D ChainGripper model actions. |
| R7 | OPEN | Inherited reconstruction/FM/action-velocity losses, 14 flow samples and derivative work; new measured cost unavailable. |
| R8 | PASS | No new intermediate action-path constraint; inherited action-velocity loss remains an explicitly selected diagnostic bias. |
| R9 | OPEN | Fresh optimizer/validation smoke, collapse/scale/decoder-sensitivity and strict checkpoint gates required. |
| R10 | OPEN | No new training or matched baseline/torus result. Historical native-speed scores do not evaluate this contract. |

The scalar conditioner has one small MLP evaluation per chunk. No additional
flow evaluations are introduced. Longer-run convergence is not established.

## Prevention

`tests/test_retiming_multiplier_condition.py` checks different physical speeds
produce identical raw multiplier inputs, missing multipliers cannot fall back
to speed, invalid multipliers fail closed, gradients and inference reach the
shared field, and the resolved recipe preserves compression. The rollout
boundary test covers both embodiments with an explicit multiplier. Operational
validation still uses live canonical launchers and separate official gates.
