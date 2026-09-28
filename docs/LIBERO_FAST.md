# Native FAST comparison

The September 26 request adds the released OAT-codebase FAST baseline on
Spatial, Object, Goal and LIBERO-10. Existing ARC/OAT/plain-DP jobs continue;
LIBERO-90 and ARC+OAT remain deferred. All new jobs request L40S at NORMAL
priority. No other worktrees are used.
The [initial launch table](results/fast_libero_20260926.md) records the original
four workflows. The [September 28 progress](results/libero_progress_20260928.md)
records verified checkpoints and accepted recoveries after cluster quota
interruptions. Goal has completed training; its evaluation resumes from saved
trials. Spatial, Object and LIBERO-10 resume policy training. FAST full-suite
success rates remain pending.

Sources are OAT commit `1da92695ef12c23b7000a0b1a76cab0aef4750e6`, specifically
[`FASTPolicy`](https://github.com/Chaoqi-LIU/oat/blob/1da92695ef12c23b7000a0b1a76cab0aef4750e6/oat/policy/fastpolicy.py),
[`train_fasttok`](https://github.com/Chaoqi-LIU/oat/blob/1da92695ef12c23b7000a0b1a76cab0aef4750e6/oat/workspace/train_fasttok.py),
and the corresponding FAST YAMLs. The original
[Physical Intelligence processor](https://huggingface.co/physical-intelligence/fast/blob/ec4d7aa71691cac0b8bed6942be45684db2110f4/processing_action_tokenizer.py)
is vendored with its license and source hash. Runtime fitting and evaluation
do not fetch or execute mutable Hugging Face code.

| Setting | Released configuration |
| --- | --- |
| Representation | DCT over 32 dense action steps, followed by BPE |
| Quantization scale / vocabulary | 10 / 1024 |
| Policy | Autoregressive Transformer, 4 layers, width 256, 4 heads |
| Maximum supervised sequence | 127 action tokens plus EOS; BOS prepended |
| Generation | At most 128 tokens, temperature 1, top-k 10, stop at EOS |
| Observations | Same two RGB cameras, state/task ID, two frames as OAT |
| Replanning | Predict 32 actions, execute 16, then replan |
| Optimization | AdamW; policy 5e-5, observation encoder 1e-5; betas 0.9/0.95 |
| Training | Seed 42, 5001 epochs, global batch 1024, bf16, gradient clip 1, EMA |
| Evaluation | 10 tasks × 50 trials × 5 repetitions = 2500 episodes per suite |

FAST fits a fresh BPE vocabulary for each suite. This is OAT's dataset-specific
FAST experiment, not the pretrained FAST+ vocabulary. Its released workspace
fits BPE on raw **training** chunks; its policy wrapper normalizes actions
before tokenization. We preserve that convention, including OAT's action
limits computed over all replay frames. The 90/10 episode split, padding,
limits and data fingerprints use the shared native loader. No validation
chunks enter BPE fitting. The decoded replay cache is enabled for training.

`fast-reconstruction.json` reports held-out action MSE/MAE, translation and
rotation errors, gripper sign accuracy, token-length percentiles, the fraction
longer than the supervised 127-token limit, and invalid round trips. The
upstream decoder's fallback to zero normalized actions for malformed generated
strings is retained. Fitting records the precise normalizer and dataset split;
policies reject mismatched BPE/data on resume.

`egomimic.benchmarks.libero.fast` fits, reconstructs, validates GPU training,
trains through `trainHydra`/`PipelineAlgo`, and publishes a final checkpoint.
`experiment=oat/libero_fastpolicy` selects the graph recipe. Existing rollout
and evaluation entry points recognize `method=fast`, restore its embedded BPE,
and use the same reset/observation refresh, seeds and 550-step cap as ARC/OAT.

```bash
source emimic/bin/activate
PYTHONPATH=. python scripts/benchmarks/launch_libero_osmo.py \
  --commit <immutable-40-character-commit> --run-id <unique-run-id> \
  --fast --suite libero_spatial --mode full --gpus 4 --gpu-type L40S \
  --output /tmp/fast-spatial.yaml
osmo workflow submit /tmp/fast-spatial.yaml --pool groot-l40s-03 --priority NORMAL
```

Each full workflow gates training on native DDP/bf16/checkpoint and real
simulator smoke tests. After training, four GPUs are released and a dependent
one-GPU evaluation runs five workers. Final scores require all 5001 epochs,
matching optimizer/EMA counts, the pinned checkpoint hash, and all 2500 trials.
`--resume-from-run` restores optimizer/EMA and the embedded BPE; it does not fit
a new vocabulary over the resumed policy's token IDs.

CPU checks in `tests/test_fast_native.py` compare tokenization, reconstruction,
cross-entropy, gradients and generation with the pinned sources. They also
exercise empty sequences, EOS/PAD, overflow truncation, no movement, small
movement, gripper changes, native float32/bfloat16 training, optimizer/EMA
resume, and inference after deleting the external tokenizer artifact. Eight
FAST checks, thirteen OAT source-parity checks and 145 shared training/evaluation
regressions passed. These are separate from GPU/simulator checks and full SR.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OAT_REFERENCE_ROOT=/path/to/pinned/oat \
FAST_PROCESSOR_REFERENCE=/path/to/pinned/processing_action_tokenizer.py \
pytest -q tests/test_fast_native.py
```
