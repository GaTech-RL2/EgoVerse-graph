# OAT tokens of ARC action chunks

**Deferred by the September 24 priority update.** Finish the original ARC/OAT
evaluations, then ARC with OAT's released diffusion policy, before running this
hybrid. The submitted hybrid workflows were canceled before their GPU tasks
started. The implementation is retained; real GPU smoke validation remains
pending. See [experiment priorities](LIBERO_EXPERIMENT_PRIORITIES.md).

This experiment trains the native OAT learned tokenizer on normalized ARC
supports. Its autoregressive observation policy then predicts those learned
tokens. The inference path is:

`observations → OAT autoregressive tokens → learned ARC supports → ARC decoding → dense actions`.

The tokenizer's reconstruction target is the ARC representation, rather than
the original seven action channels. The tokenizer remains the pinned OAT
register encoder, FSQ quantizer and causal decoder. It uses eight learned
tokens with a 1,000-entry codebook, width 256, two encoder layers and four
decoder layers. The policy uses the existing observation encoder and OAT
autoregressive Transformer (four layers, width 256, four heads).

The first experiment uses Spatial and two existing replay selections:

| Variant | R (degrees) | D (meters) | ARC supports M | Channels | OAT tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| STK2 + OAT | 192 | 0.8 | 32 | 12 | 8 |
| DUR2 + OAT | 384 | 0.8 | 32 | 12 | 8 |

Both preserve ARC's separate translation and rotation clocks, rotation-6D
orientation representation, gripper channel, fixed physical normalization,
and the existing conversion between LIBERO controller commands and paths.
STK represents interval speed; DUR represents interval duration. Neither is
passed directly to LIBERO as a controller command.

Training first fits the tokenizer for 5,001 epochs. Held-out reconstruction
then measures the complete encode/quantize/decode path using 1, 2, 4 and 8 OAT
tokens, against the same ARC codec without learned quantization. The tokenizer
is frozen, including dropout and quantizer corruption, while the observation
policy trains for another 5,001 epochs. Both use the released OAT optimizer,
EMA and effective batch size 1,024. The full policy evaluation uses 2,500
episodes, two observations, a 32-action horizon and 16 executed actions per
replan. These are separate experiment runs; previous ARC/OAT weights are not
used to initialize the models.

The YAML recipes are `oat/libero_arc_oattok` and
`oat/libero_arc_oatpolicy`. They compose the existing `LiberoArcStage`,
`OATTokenizerStage`/`OATPolicyStage`, and ARC decoder in `PipelineAlgo`.
The shared trainer, decoded replay loader, normalizer and evaluator are used.
Validation reports dense controller-action errors after decoding, while the
tokenizer training loss is MSE on normalized ARC supports.

Checkpoints record the input representation's mode, R/D/M, physical scales,
horizon and normalization factors. Loading/resuming rejects incompatible
representations even when tensor dimensions agree. Policy checkpoints contain
the frozen tokenizer configuration and weights, so inference does not require
the original tokenizer file. Legacy raw-action OAT checkpoints remain supported.

`scripts.benchmarks.launch_libero_osmo.arc_oat_workflow` renders a pinned GPU
workflow with `RUN_KIND=arc_oat`. It accepts one fixed profile and its existing
mode-specific replay receipt. The entry point
`egomimic.benchmarks.libero.arc_oat` checks the replay selection before full
training, persists content-addressed checkpoints and supports resuming the
tokenizer and policy. Full runs evaluate automatically with five repetition
workers. Rollout records use method `arc_oat`, separately from `arc` and `oat`.

The comparison against an existing OAT evaluation requires the normal full
protocol and identical initial-state hashes. A failed comparison check is
recorded; it is not bypassed to claim a paired result. The earlier U-Net scores
and the separate ARC diffusion-Transformer campaign are not hybrid results.

Tests cover the actual ARC-space loss and gradients, stationary/gripper-only
chunks, STK/DUR prefix decoding, real shared training with float32/bfloat16,
EMA resume, inference without target actions, independent checkpoint reload,
representation mismatches, reconstruction metrics, complete training budgets,
and rejection of unpaired evaluation states. Before full training starts, GPU
smoke runs must additionally train both networks and execute simulator rollouts.
