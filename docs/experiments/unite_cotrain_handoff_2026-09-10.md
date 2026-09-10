# UNITE cotrain on PushShapes sim_v2 — reproduction and handoff (2026-09-10)

Branch: `aidan/unite-cotrain-repro` (this file's branch), the consolidation of
the ICE working stack `aidan/unite-cotrain-1 → -2 → unite-compile → -3 → -4 → -5`
plus the rollout-driver branch `aidan/rollout-eval-3`, re-based as a small set of
commits on `main` (75525eb, PR #20). The raw stack keeps the exact per-row
commits that every run's provenance records; this branch has the same content
at different SHAs. Nothing here has been merged; the experiments are still
running as of 13:30 ET.

Question under test: does UNITE (released recipe, un-tied tokenizer/denoiser,
hidden 384, topology A = shared tokenizer body + per-embodiment decoders),
co-trained on U-Socket (`[x, y, cos θ, sin θ]`) and ChainGripper (six-point
`[x1 y1 x2 y2 x3 y3]`), beat Paper-DP behaviour cloning on each embodiment and
Paper-DP co-training, under Elmo's closed-loop protocol?

Short answer so far: on chain, yes (UNITE cotrain 0.637 vs DP cotrain 0.624 vs
DP BC 0.583); on U-Socket, no (0.719 at 120k vs DP cotrain 0.748 at 240k), and
the reason is a training-dynamics defect diagnosed on 9-10 (section 7), with
five fix variants in flight.

---

## 1. What is in this branch

| area | files | notes |
|---|---|---|
| UNITE cotrain pipeline | `egomimic/pipeline/stages_unite_released.py`, `stages_unite_separate.py`, `stages_sampler.py`, `egomimic/eval/planar_action_eval.py`, `egomimic/rldb/embodiment/pushshapes.py`, `egomimic/rldb/zarr/*` | topology A/B, six-point chain actions, `semantic_blocks_by_embodiment` for the energy score, decoded-action / action-velocity opt-ins (both off), `latent_norm_affine` flag (section 7), opt-in `compile_backbones` |
| configs | `egomimic/hydra_configs/{model/bf,experiment/pusht,data/pusht}` | see the row table |
| ICE launcher | `scripts/ice/launch_unite_cotrain.sbatch` | one launcher for every row (UNITE and DP); allowlisted experiment×model pairs; `ICE_UNITE_FAST`, `ICE_UNITE_LR/LR_FINAL/LR_TERMINAL`, `ICE_UNITE_COMPILE`, `ICE_EXTRA_OVERRIDES`; re-checks `HEAD == ICE_EXPECTED_HEAD` and a clean tree on every requeue |
| rollout driver | `scripts/eval/rollout_pushshapes.py`, `rollout_pushshapes.sbatch`, `rollout_multi.sbatch` | protocol mode: `--budget-json`, `--full-horizon`, `--chunk-start`, `--seeds a+b+c`, `--label`, `--ensemble-samples`, video capture; forces eager inference for compiled checkpoints; writes a `protocol` provenance block into every result JSON |
| campaign helpers | `scripts/ice/cotrain/` | `launch_ct2.sh` (row table), `rollout_ct3.sh` (protocol submitter), `monitor_ct3.py` + `monitor_ct3_rows_s3.json` + `monitor_ct3.sbatch` (self-chaining checkpoint scorer), `cotrain_predicate_rev3.py` (the success predicate), `export_peaks.py` + `plot_protocol.py` (figure), `oec_summary.py`, `ttfc.py`, `ln_gain.py`, `wb_*.py`, `bench_unite.py`, `bad_gpu_nodes.txt` |
| protocol data | `scripts/ice/cotrain/budgets/*.json`, `seeds/eval_side_access_seeds_newgeom_150.json`, `protocol/EVAL_PROTOCOL.md`, `protocol/episode_budget.py`, `protocol/derive_budget.py` | rev-3 budgets and the OEC-56 obstacle bank, mirrored from `/coc/flash7/paphiwetsa3/scripts/eval/` on Skynet |
| tests | `tests/test_unite_*.py`, `tests/test_chain_gripper_points.py` | 51 pass on this tree; `tests/test_unite_register.py::test_only_registered_rows_and_no_legacy_or_diagnostic_surface` fails on `main` too |

The helper scripts hard-code the ICE account's paths (`/home/hice1/agao81/scratch/...`,
cedar run root `/storage/cedar/cedar0/cedarp-dxu345-0/agao81/runs`). They are kept
verbatim because every recorded run went through them; search for `hice1/agao81`
when porting.

## 2. Environment (ICE, PACE)

- Partition `coe-gpu`, QOS `coe-ice`, account `ece`, 2 × H200 per training row
  (`--gres=gpu:h200:2 --ntasks-per-node=2 --cpus-per-task=8 --mem=128G`). QOS caps
  a job at 960 GPU-minutes, so 2-GPU jobs request 8 h and the launcher requeues
  from its own checkpoints (`signal-job=...ckpt`).
- Python 3.11.16, torch 2.7.1+cu126, venv `~/scratch/EgoVerse-graph-unite/.venv`
  (installed from the repo lock; also holds the simulator deps; run the sim
  headless with `SDL_VIDEODRIVER=dummy`).
- Simulator: `~/scratch/sim_chain` = the `Tsimulation` tree of EgoVerse-graph
  `2133a92` (2026-09-03) with `ChainGripperAgent` + point control (see its
  `ORIGIN.txt`). Protocol identity recorded per result: env
  `Tsimulation/sim_v2/pushshapes/env.py` sha256 `cd8f78e8…`, obstacles
  `12b64ad2…`; both match Elmo's pinned values. A second tree
  (`EgoVerse-graph-rollout`'s `Tsimulation`) gives bit-identical U-Socket
  level-0 physics; all protocol rollouts used `sim_chain`.
- Data (ICE mirrors of Skynet `/coc/flash7/paphiwetsa3/datasets/Tsim_v2/`):
  - `~/scratch/data/Tsim_v2/u_socket_3000_v2_clean` — split manifest sha256
    `3683e346…`
  - `~/scratch/data/Tsim_v2/chain_gripper_3000_v2` — `3ced944e…`
  - `~/scratch/data/Tsim_v2/chain_gripper_3000_v2_plus_gen_1919` — symlink
    merge of the 3000 set with `chain_gripper_gen` (1,919 obstacle episodes,
    64 per level for levels 1–30, 63 at level 7); manifest
    `egomimic/hydra_configs/data/pusht/planar_v2_chain_gripper_3000_plus_gen1919_split_seed42_v1.json`
    sha256 `e320fefd…` (train 4,870 / valid 49).
  - All three verified PRE-step action alignment → `--chunk-start 0`.
- Normalisation statistics (fixed per sweep, quantile/min-max per embodiment):
  sweep 2 `runs/ct-smoke/ct-smoke-cotrain-5739042/norm_stats` (cedar, sha
  `2b0de224…`); sweep 3 `runs/unite-cotrain-3/norm_stats_chaingen_minmax/norm_stats.json`
  (sha `64e296a4…`, 1,707,818 chain frames).
- W&B: entity `rl2-group`, project `pushshapes-flow-transfer` (UNITE) and
  `pushshapes-planar-v2` (DP), run ids `aidan-ct2-<tag>`, group = sweep name.
  Pull histories per key (`r.history(keys=[k, "trainer/global_step"])`); a
  multi-key call returns zero rows because keys log at different steps.

## 3. Recipe facts that matter

- Global batch 64 = 32 per GPU × 2 (DDP, sharded sampler); Elmo's 2-GPU
  convention keeps 32 global, so step clocks differ from his by 2×.
- UNITE rows: released two-stage schedule, warmup 8k, 1e-4 → 5e-5 between 12k
  and 20k, then flat 5e-5 (second decay at 1.2 M is never reached);
  `ICE_UNITE_FAST=true` = `flow_mini_batch 14` (one chunk) and no gradient
  checkpointing, nothing else; EMA on; CFG 4.0 embedded in the checkpoint;
  `n_obs_steps 1`; 8 register tokens × latent 16; 14 flow draws per sample per
  step; Muon (matrices) + AdamW, weight decay 0.
- DP rows: Paper-DP UNet (`down_dims [512, 1024, 2048]`, 262 M), DDPM 100
  steps, `n_obs_steps 2`, cosine LR to 0 over 240k, EMA. Cotrain: one six-wide
  head, U-Socket common-five `[x, y, cos θ, sin θ, grip]` zero-padded to six
  (`PadActionWidth`), chain six points, no embodiment token, per-embodiment
  norm stats, `CombinedLoader max_size_cycle` (one 32-batch per embodiment per
  step, losses averaged), `PaddedPlanarCommon5NativeDecoder` decodes U-Socket.
- Inference: UNITE at CFG 1.0 (declared 9-10; the embedded 4.0 is reported
  alongside as `cfgemb`; 1.5 and 2.0 were worse; K-sample chunk averaging is
  worse), dopri5, replan every 8 of a 16-step chunk, EMA weights.
- `torch.compile` (`ICE_UNITE_COMPILE=true`) is bit-equivalent in fp32 and
  +39 % throughput, but needs `functorch donated_buffer=False` because the
  telemetry calls `autograd.grad(retain_graph=True)` every 100 steps; not
  adopted as default until the compiled replica `ctAc` passes the paired
  closed-loop check.
- `env.reset(seed=k)` does not reproduce dataset episode k; rollout seeds are
  not training initial states.

## 4. Evaluation protocol (Elmo, `EVAL_PROTOCOL_2026-09-08`, horizon rev 3)

- Budget = `ceil(1.1 × p99(total_frames | level))` of the model's own training
  set: U-Socket 318, chain 688 (same for the 3000 and 3000+gen chain sets).
- Full horizon (`--full-horizon`, ignore env termination, coverage threshold
  1.01), peak coverage, SR@0.80 and SR@0.95 from per-episode peaks.
- Level 0, seeds 0–39 canonical; 40–79 added as a labelled extension, never
  mixed into a canonical number.
- Obstacle generalisation: OEC-56 bank, 30 levels × 5 seeds
  (`eval_side_access_seeds_newgeom_150.json`, sha `499560f2…`); U-Socket
  obstacle budgets derived at the level-0 ratio 0.4624 from the chain
  3000+gen budgets (`u_socket_3000_v2_clean_p99_derived_from_chain3000_gen1919.json`).
- Replan 8, chunk start 0, EMA. Every result JSON carries a `protocol` block
  (budget file, seeds, chunk window, sampler, checkpoint step, sim SHAs,
  driver git head, label). Never pool across horizon revisions; rev-1 numbers
  from the earlier loop are kept separately and are not cited.

## 5. Rows

All rows: `scripts/ice/cotrain/launch_ct2.sh <row> <tag> [steps]` with
`REPO=<worktree>` pinned at the SHA in the table (the launcher refuses a
dirty tree or a moved HEAD on requeue). Checkpoints every 30k to 240k.

Sweep 2 (U-Socket 3000 + chain 3000):

| row | model | experiment | env / overrides | pinned HEAD (raw stack) | run dir (`…/runs/unite-cotrain-2/`) | job |
|---|---|---|---|---|---|---|
| ctA | `bf/ct_unite_register_separate_nt8_h384_s42` | `pusht/unite_cotrain_usocket_chain_val01_h16` | FAST, LR 1e-4/5e-5 | 729e320 | `ctA-h384-240k-09091542` | 5748787 |
| ctB | `bf/ct_unite_register_split_tok_nt8_h384_s42` | same | same | 729e320 | `ctB-h384-240k-09091542` | 5748788 |
| ctAc | ctA + `ICE_UNITE_COMPILE=true` | same | same | 6b91df9 | `ctA-compiled-h384-240k-09092002` | 5751322 |
| ctA768 | `bf/ct_unite_register_separate_nt8_h768_s42` (441 M) | same | same | 065e116 | `ctA768-h768-240k-09100127` | 5751457 |
| ctAema | ctA + EMA 0.9999 | same | + override | d630638 | `ctA-ema9999-240k-09100403` | 5752343 |
| ctAann | ctA + LR 5e-5 → 1e-5 over 200k–240k | same | `ICE_UNITE_LR_TERMINAL=1e-5`, `decay_start_2 200000`, `decay_end_2 240000` | d630638 | `ctA-anneal200k-240k-09100403` | 5752344 |
| ctApin | ctA + `latent_norm_affine=false` | same | `+model.pipeline.stages.4.generative_encoder.latent_norm_affine=false` | 9dece7c | `ctA-pinlatent-240k-09101218` | 5753581 |
| ctApinAnn | ctApin + cosine 5e-5 → 1e-6 over 120k–240k | same | + `LR_TERMINAL=1e-6`, `decay_start_2 120000`, `decay_end_2 240000` | 9dece7c | `ctA-pinlatent-cos120k-240k-09101218` | 5753582 |
| ctAwd | ctA + weight decay 0.1 (AdamW and Muon) | same | `model.optimizer.adamw_weight_decay=0.1 model.optimizer.muon_weight_decay=0.1` | 9dece7c | `ctA-wd0.1-240k-09101218` | 5753583 |
| uniteus | `bf/us_unite_register_separate_nt8_h384_s42` | `pusht/unite_usocket_register_sweep_val01_h16` | FAST, LR 1e-4/5e-5 | 8b40f6b | `unite-usocket-h384-240k-09091808` | 5749211 |
| uniteusema / uniteusann | uniteus + the EMA / anneal variants | | | d630638 | `unite-usocket-{ema9999,anneal200k}-240k-…` | 5752377 / 5752378 |
| unitech | `bf/ch_unite_register_separate_nt8_h384_s42` | `pusht/unite_chain_points_val01_h16` | FAST, LR 1e-4/5e-5 | 8b40f6b | `unite-chain-points6-h384-240k-09091808` | 5749212 |
| dpct | `bf/bf_planar_v2_dp_paper_points6` | `pusht/planar_v2_cotrain_dp_paper_points6` | `ICE_UNITE_FAST=false` | 729e320 | `dp_paper-cotrain-points6-240k-09091540` | 5748772 (done) |
| dpus | `bf/bf_planar_v2_dp_paper` | `pusht/planar_v2_usocket_dp_paper` | same | 729e320 | `dp_paper-usocket-240k-09091540` | 5748773 (done) |
| dpch | `bf/bf_planar_v2_dp_paper_points6` | `pusht/planar_v2_chain_points_dp_paper` | same | 729e320 | `dp_paper-chain-points6-240k-09091540` | 5748774 (done) |

Sweep 3 (U-Socket 3000 + chain 3000+gen 1,919; run dirs under `…/runs/unite-cotrain-3/`):

| row | model | experiment | pinned HEAD | run dir | job |
|---|---|---|---|---|---|
| s3ctA | `ct_unite_register_separate_nt8_h384_s42` | `pusht/unite_cotrain_usocket_chaingen_val01_h16` | 065e116 | `s3-ctA-chaingen-240k-09100222` | 5751830 |
| s3ctA768 | `…_h768_s42` | same | 065e116 | `s3-ctA768-chaingen-240k-09100222` | 5751832 |
| s3ctApin / s3ctApinAnn | ctApin / ctApinAnn on sweep-3 data | same | 9dece7c | `s3-ctA-pinlatent{,-cos120k}-chaingen-240k-09101222` | 5753590 / 5753591 |
| s3unitech | `ch_unite_register_separate_nt8_h384_s42` | `pusht/unite_chaingen_points_val01_h16` | 065e116 | `s3-unite-chaingen-points6-240k-09100222` | 5751831 |
| s3dpct | `bf_planar_v2_dp_paper_points6` | `pusht/planar_v2_cotrain_dp_paper_points6_chaingen` | 065e116 | `s3-dp_paper-cotrain-chaingen-240k-09100222` | 5751646 (done) |
| s3dpch | `bf_planar_v2_dp_paper_points6` | `pusht/planar_v2_chaingen_points_dp_paper` | 065e116 | `s3-dp_paper-chaingen-points6-240k-09100222` | 5751647 (done) |

Throughput (H200 pair, shared node): DP BC 19 it/s, DP cotrain 12, UNITE BC
5.2, UNITE cotrain A 3.05 (compiled 4.23, h768 2.6, B 2.18). The UNITE step is
kernel-launch bound (GPU util 46–83 % at 11 GB), see `bench_unite.py` and the
sweep-2 note. The launcher's 1-GPU path hangs at datamodule instantiation and
is unmeasured.

## 6. Commands

```bash
# training row (from a clean worktree pinned at the row's HEAD)
cd ~/scratch/autoresearch/orchestrator-260909-1520   # or scripts/ice/cotrain/
REPO=~/scratch/EgoVerse-graph-s5 NORM_STATS=<cedar norm_stats dir> \
  [LR_TERMINAL=1e-6] [EXTRA="<hydra overrides>"] ./launch_ct2.sh ctA <tag> [240000]

# protocol rollouts for one checkpoint (level 0 seeds 0-39 or 40-79, or OEC level ranges)
./rollout_ct3.sh <run_dir> <ckpt file> usocket|chain <level|a-b> <tag> \
  [--cfg 1.0] [--budget-set s2|s3] [--seed-block 0|40] [--ens K]
# → ~/scratch/rollouts/CT2-rev3/<tag>.json (or <tag>-L<level>.json)

# scorer: submits missing rollouts for every landed checkpoint of every registered row,
# rewrites summary.tsv; self-chaining ice-cpu job
sbatch monitor_ct3.sbatch          # rows: monitor_ct3.py ROWS + monitor_ct3_rows_s3.json

# success predicate (exit 0 iff one UNITE-cotrain checkpoint beats best DP BC and
# best DP cotrain on both embodiments, canonical seeds, CFG 1.0)
python3 cotrain_predicate_rev3.py ~/scratch/rollouts/CT2-rev3

# figure
python3 export_peaks.py ~/scratch/rollouts/CT2-rev3 peaks_export.json
python3 plot_protocol.py peaks_export.json protocol_curves.png "<stamp>"

# rev-3 budgets from a dataset (Elmo's scripts)
python3 protocol/episode_budget.py --dataset <dir> --statistic p99 --multiplier 1.1 --out budgets/<name>_p99.json
python3 protocol/derive_budget.py ...   # U-Socket obstacle budgets from the chain 3000+gen file
```

`monitor_ct3.py` scores each checkpoint on level 0 for seed blocks 0 and 40 with
both CFG settings for UNITE rows, and the OEC-56 obstacle levels in six chunks
at 120k/180k/240k; results and `summary.tsv` live in `~/scratch/rollouts/CT2-rev3/`
(columns: row, step, emb, level, variant, n, mean_peak, sr80, sr95,
never_moved, budget, seeds0-39_mean).

## 7. Results as of 2026-09-10 13:30 ET (protocol, level 0)

Bars = best checkpoint per embodiment on the canonical seeds 0–39 (selection-
biased upward, i.e. conservative for UNITE). UNITE at CFG 1.0.

| | U-Socket | chain |
|---|---:|---:|
| DP BC | 0.678 (dpus 180k) | 0.583 (dpch 240k) |
| DP cotrain | **0.748** (dpct 240k) | 0.624 (dpct 240k) |
| UNITE BC | 0.567 (uniteusann 120k) | 0.535 (unitech 90k) |
| UNITE cotrain A h384 | 0.719 (ctA 120k) | **0.637** (ctA 90k) |
| UNITE cotrain A compiled replica | 0.690 (120k) | 0.679 (90k) |
| UNITE cotrain B (split tokenizer) | 0.651 (150k) | 0.523 (150k) |
| UNITE cotrain A h768 | 0.592 (60k) | 0.571 (30k) |
| sweep 3: DP cotrain + chain obstacles | 0.739 (s3dpct 180k) | 0.605 (240k) |
| sweep 3: UNITE cotrain A + chain obstacles | 0.540 (90k, still training) | 0.603 (90k) |

Predicate: units remaining 0.0616 (ctA 120k: U-Socket 0.719 vs 0.748, chain
0.591 on that checkpoint vs 0.624). Pooled seeds 0–79: U-Socket DP cotrain
0.735 > UNITE cotrain 0.715 > DP BC 0.682 > UNITE BC 0.613; chain UNITE cotrain
0.623 ≥ DP cotrain 0.610 > DP BC 0.567 > UNITE BC 0.530.

Scene generalisation (U-Socket on the 30 OEC-56 obstacle levels, 150 episodes):
every U-Socket policy freezes in 124–148 of 150 scenes; mean peak 0.071 (dpct),
0.063 (dpus), 0.021 (ctA), 0.012 (uniteus), 0.010 (s3dpct). Chain obstacle
data does not transfer to U-Socket for DP either. Chain on its own obstacle
levels: s3dpct 0.448 (240k), s3dpch 0.336, s3unitech 0.303 (120k).

Diagnostics that closed hypotheses: UNITE is not hesitant (first-contact time
matches DP; `ttfc.py`) but finishes later (peak at median step 410 vs 254);
CFG 1.5/2.0 and K-sample chunk averaging are worse than single-sample CFG 1.0;
topology B trails A everywhere; width (h768) does not help.

**The plateau (9-10 12:30).** UNITE cotrain climbs fast (U-Socket 0.60 at 60k
when DP is at 0.36) and flattens from 90–120k, then slips; DP rises to 240k.
W&B shows why: the tokenizer's output LayerNorm gain grows monotonically (rms
1.22 at 30k → 1.67 at 180k, read directly from the checkpoints with
`ln_gain.py`), so the detached flow target inflates all run; train flow loss
2.0 → 3.4, validation flow loss 1.8 → 7.6, and validation action MSE, energy
score and sample diversity are all best at 60k and degrade after. Reconstruction
loss is ≈ 1e-5 from 30k on, the LR is flat 5e-5, weight decay is 0, and the
reconstruction-noising term rewards a larger latent norm. h768 shows the same
signature earlier and larger, so it is not capacity. UNITE's 60k checkpoint has
a better validation MSE (0.0062) than any DP checkpoint (0.0093).

## 8. In flight and next steps

Running (all 2 × H200, 240k steps, scored automatically by the monitor):
ctA (164k+), ctB, ctAc, ctA768, ctAema, ctAann, uniteusema, uniteusann;
s3ctA, s3ctA768, s3unitech; the five plateau variants ctApin, ctApinAnn,
ctAwd, s3ctApin, s3ctApinAnn (launched 12:18–12:22, first checkpoints ≈ 15:30).

Decision rule for the variants: if ctApin's validation flow loss stays within
≈ 1.5× of train through 150k and its closed-loop keeps rising past 120k, the
drift was the cause and `latent_norm_affine=false` becomes the recipe; if
ctAwd alone closes the gap, it is plain overfitting and the fix is
regularisation plus an anneal; if neither, the next levers are a two-phase
schedule (freeze the tokenizer at 60k), `decoded_action_samples_per_reconstruction > 0`
(already implemented, untested past 60k), and the 2-frame observation window
(`bedbbba1` added `n_obs_steps 2` for U-Socket; DP uses 2 frames, UNITE 1).

Autoresearch loop ledger: `~/scratch/autoresearch/orchestrator-260910-0200/`
(`orchestrator-state.json`, `pipeline.tsv`, `units-history.txt`). Vault notes:
"UNITE Cotrain Sweep 2 9-9", "UNITE Cotrain Sweep 3 9-10", "PushShapes Eval
Protocol sim_v2 9-8", and the handoff note "UNITE Cotrain Handoff 9-10".

## 9. Traps (each cost time)

- Never commit on a worktree that has running or pending training jobs: the
  launcher re-checks `HEAD == ICE_EXPECTED_HEAD` and a clean tree on every
  requeue, and a moved HEAD kills the requeue. Worktrees on ICE: `ct` 729e320,
  `ct2` 8b40f6b, `ct3` 6b91df9, `eval` (rollout driver), `s3` 065e116, `s4`
  d630638, `s5` 9dece7c, `repro` (this branch).
- `sbatch --export` splits on commas: pass seed lists with `+`; the driver
  splits on `[,+ ]`.
- The compiled training path needs `donated_buffer=False`; rollouts of compiled
  checkpoints force eager inference.
- `ICE_UNITE_FAST` once silently overrode the UNITE LR (3e-5 → 3e-6) and halved
  closed-loop success; the launcher no longer replicates that override, LR
  comes from `ICE_UNITE_LR/LR_FINAL`.
- Login nodes differ per `ssh` (`login-ice-gnr-1`, `-2`, …): `/tmp` is not shared,
  `pgrep` will not see another node's processes.
- Node `atl1-1-03-014-16-0` reported "No CUDA GPUs are available" once;
  `atl1-1-03-010-10-0` has a GPU that throws "device busy"; both are in
  `bad_gpu_nodes.txt`.
- Single 40-seed readings carry ± 0.05 (DP BC 240k scored 0.708 and 0.599 on the
  same seeds in two runs); decisions use both seed blocks and repeats.
- ICE cannot push over HTTPS (no credentials). Since 9-10 13:35 the ICE
  account has an SSH key registered on GitHub (`~/.ssh/id_ed25519_github`,
  push URL `git@github.com:GaTech-RL2/EgoVerse-graph.git`), and pushes work
  from any of the worktrees. `gt` is not installed on ICE; Graphite PRs are
  opened from the WSL clone `~/EgoVerse-graph`, where the stack is tracked.
