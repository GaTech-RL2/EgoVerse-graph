# Action-flow: stop-gradient, action velocity, and noise augmentation

Runs by Aidan on ICE, 2026-09-05/06, on the frozen latent-8 torus sets
(`gaussian_torus_latent8_train_4096_seed42.npz` / `..._eval_40960_seed4242_val2048.npz`,
hash-verified), 60k steps, three seeds, symmetric NN MSE on the identical
2,048-particle validation cloud (mean +- sample std). Model = the G/E layout
(D=8, nonlinear residual adapters 32x2, field 128x4, lambda_scale=1).

## Stop-gradient x action-velocity (`build_action_flow_stop_gradient_configs.py`)

| variant | rec | clean gradient | NN MSE |
|---|---:|---|---:|
| a-direct (3-D FM) | - | - | 0.0120 +- 0.0012 |
| g (E + L_act) | 100 | full | 0.0128 +- 0.0012 |
| g | 100 | all_stopgrad | 0.0655 +- 0.0128 |
| e (reconstruction only) | 100 | full | 0.0850 +- 0.0025 |
| e | 100 | all_stopgrad | 0.3707 +- 0.1666 |
| g | 1 | all_stopgrad | 0.1690 +- 0.0280 |
| g | 1 | full | 0.2187 +- 0.0055 |
| e | 1 | full | 0.2466 +- 0.0385 |
| e | 1 | all_stopgrad | 0.3297 +- 0.1756 |

With an independent Gaussian source the attached (`full`) gradient is not a
shortcut: it spreads the code so an 8-D Gaussian can be transported onto it.
Under `all_stopgrad` the code stays the thin affine lift (latent RMS 0.75) and
the field cannot hit it. L_act helps in both arms (6.6x / 5.7x).
`all_stopgrad` + L_act shows a gauge runaway (latent RMS 18 at rec 100, 111 at
rec 1). Reconstruction weight 100 is needed in every arm.

## Follow-ups (`build_action_flow_followup_configs.py`, rec 100)

| variant | NN MSE |
|---|---:|
| e, all_stopgrad, **noise-augmented reconstruction** | **0.0114 +- 0.0005** |
| g, all_stopgrad, noise-aug | 0.0122 +- 0.0007 |
| g, full, noise-aug | 0.0158 +- 0.0021 |
| e, full, noise-aug | 0.0164 +- 0.0013 |
| g, flow full / AV all_stopgrad | 0.0217 +- 0.0102 |
| g, flow all_stopgrad / AV full | 0.0789 +- 0.0073 |

Noise augmentation (`reconstruction_noise_aug`: decode `z = t z0 + (1-t) eps`,
`t ~ U[0.7, 1]`, for half the batch -- released UNITE's recipe) turns the
stop-gradient from the worst arm into the best and makes L_act redundant in
both attachment modes; L_act and noise augmentation are substitutes. The
co-adaptation that the attached suite relies on rides on the *flow* gradient
into the encoder, not the AV gradient (flow-sg / AV-attached = 0.079).

## Euler step sweep (`eval_synthetic_inference_steps.py`, same checkpoints)

| steps | a-direct | g full rec100 |
|---:|---:|---:|
| 32 | 0.0120 | 0.0128 |
| 16 | 0.0122 | 0.0147 |
| 8 | 0.0149 | 0.0249 |
| 4 | 0.0361 | 0.0679 |
| 2 | 0.1496 | 0.1459 |
| 1 | 0.3600 | 0.2282 |

No few-step advantage for the latent path on this target.

## Codec rows (`SyntheticSharedLatentFlow`, paired `source_latent`)

The `unite`/`vfm` codec configs use the paired source, where an attached
encoder can zero the flow loss by copying the source. `clean_gradient_mode`
"all_stopgrad": NN MSE 0.203 -> 0.018 (latent 2), 0.070 -> 0.016 (latent 4),
reconstruction -> ~0, latent 2 ~ latent 4; direct 3-D FM 0.012. The paired /
unpaired coupling decides which way the stop-gradient cuts.
