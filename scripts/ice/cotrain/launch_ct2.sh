#!/bin/bash
# Cotrain sweep 2 (2026-09-09): five rows through one launcher, 2xH200 DDP each, checkpoints
# streamed straight to cedar. Usage:
#   launch_ct2.sh <row> <tag> [MAX_STEPS]      row in {ctA, ctB, dpct, dpus, dpch}
#   env: WORLD (2), CKPT_EVERY (30000), VAL_EVERY (30000), NORM_STATS (path), SMOKE=1 (tiny dirs),
#        LR/LR_FINAL (UNITE, 1e-4/5e-5), EXTRA (hydra overrides)
#   ctA  = shared-tokenizer-body UNITE (topology A, best row of loop 1)
#   ctB  = per-embodiment tokenizer + shared denoiser UNITE (topology B)
#   dpct = Paper-DP cotrain (U-Socket common5 pad6 + chain points6)
#   dpus = Paper-DP BC U-Socket        dpch = Paper-DP BC chain points6
#   uniteus / unitech = single-embodiment UNITE (cotrain row restricted to one domain)
#   ctAc = topology A with ICE_UNITE_COMPILE=true (torch.compile replica; needs REPO on aidan/unite-cotrain-2)
#   refctA / refdpct = Action Flow ladder references (2026-09-15): the sweep-3 s3ctA / s3dpct recipes
#     retrained on the ladder data (same experiment, model, data, LR env); checkpoints under
#     runs/af-ladder-20260913/; norm stats pinned to each original row's own file (s3ctA: the sweep-3
#     minmax file; s3dpct: the quantile file its run computed); 2 GPUs of any H100|H200 (the
#     launcher's constraint) instead of h200 only.
#     refctA sets model.pipeline.stages.4.cfg_scale=1.0 (labelled deviation, 2026-09-15 CFG-1.0-only rule):
#     cfg_scale is read only by the inference-mode sampler (Valid/MSE, EnergyScore, rollouts); training
#     losses, condition dropout 0.1 and the optimizer are unchanged. s3ctA validated at the embedded 4.0.
#   refctP / refdpctP = the same two recipes retrained 2026-09-16 on the corrected chain data
#     (chain_gripper_3000_v2_plus_gen_prestep1199: 3,000 human + 1,199 pre-step obstacle episodes; the 720
#     post-step chain_gripper_gen episodes excluded); runs under runs/cotrain-prestep-20260916/. No norm-stats
#     default: each row's smoke computes its own file (UNITE minmax, DP quantile) and the full run takes it
#     via NORM_STATS=<smoke run>/norm_stats. refctP keeps refctA's CFG 1.0 validation deviation.
#   cl1dec..cl6ld8 = 2026-09-25 change loop: the refctP recipe + one change each (decoded-action loss,
#     Min-SNR train_eps cap, contrastive flow matching, dispersive loss, flow-latent BatchNorm, latent_dim 8);
#     runs under runs/unite-change-loop-20260925/; NORM_STATS defaults to refctP's pinned file.
#   PREFLIGHT=1 runs the launcher's preflight-only mode inline instead of submitting (use on ice-cpu).
#   identity env (defaults unchanged): RUN_NAME (run dir, default <tag>-<stamp>), WANDB_RUN_ID
#     (default aidan-ct2-<tag>-<stamp>), WANDB_GROUP (default the sweep), GRES (default gpu:h200:WORLD),
#     TEST_ONLY=1 (sbatch --test-only)
set -euo pipefail
ROW=${1:?}; TAG=${2:?}; STEPS=${3:-240000}
REPO=${REPO:-/home/hice1/agao81/scratch/EgoVerse-graph-ct}
PY=/home/hice1/agao81/scratch/EgoVerse-graph-unite/.venv/bin/python
HEAD=$(git -C "$REPO" rev-parse HEAD)
test -z "$(git -C "$REPO" status --porcelain=v1 --untracked-files=all)" || { echo "repo dirty"; exit 64; }
US=/home/hice1/agao81/scratch/data/Tsim_v2/u_socket_3000_v2_clean
CH=/home/hice1/agao81/scratch/data/Tsim_v2/chain_gripper_3000_v2
SHA_US=3683e3461596eef8df2432fa865779b3c77b2a2057dabd0fea125595729cf313
SHA_CH=3ced944ea3af8e875ea88fc5c2df3a5d2865a9f95223d109fb4bd28c8be7cf69
EXCL=$(paste -sd, /home/hice1/agao81/scratch/autoresearch/bad_gpu_nodes.txt)
STAMP=$(date +%m%d%H%M)
CEDAR=/storage/cedar/cedar0/cedarp-dxu345-0/agao81/runs
SWEEP=unite-cotrain-2
LOG=/home/hice1/agao81/scratch/logs/ct2-${TAG}-%j.log
WORLD=${WORLD:-2}
# QOS coe-ice caps gres/gpu minutes per job at 960: 2 GPUs -> 8 h wall, 1 GPU -> 16 h.
if [ -z "${TIME:-}" ]; then if [ "$WORLD" -ge 2 ]; then TIME=$((960 / WORLD / 60)):00:00; else TIME=16:00:00; fi; fi
DATA_US="ICE_DATASET_DIR_USOCKET=$US,ICE_EXPECTED_SPLIT_MANIFEST_SHA256_USOCKET=$SHA_US"
DATA_CH="ICE_DATASET_DIR_CHAIN=$CH,ICE_EXPECTED_SPLIT_MANIFEST_SHA256_CHAIN=$SHA_CH"
# sweep 3: chain source = chain_gripper_3000_v2 + chain_gripper_gen (4,919 episodes), its own manifest
CHG=/storage/ice1/4/5/agao81/data/Tsim_v2/chain_gripper_3000_v2_plus_gen_1919
SHA_CHG=e320fefd2e1e5d491d74a9910e7eb641fcb9b295a4c7e6e58a3c8234f15332af
MAN_CHG=$REPO/egomimic/hydra_configs/data/pusht/planar_v2_chain_gripper_3000_plus_gen1919_split_seed42_v1.json
DATA_CHG="ICE_DATASET_DIR_CHAIN=$CHG,ICE_SPLIT_MANIFEST_CHAIN=$MAN_CHG,ICE_EXPECTED_SPLIT_MANIFEST_SHA256_CHAIN=$SHA_CHG"
# 2026-09-16 corrected chain source: chain_gripper_3000_v2 + chain_gripper_gen slots 0-1199 (4,199 pre-step episodes)
CHP=/storage/ice1/4/5/agao81/data/Tsim_v2/chain_gripper_3000_v2_plus_gen_prestep1199
SHA_CHP=e380e3c7a00084651dd8554f570394fd007ea7f3fe0442819f85b28de9f908e6
MAN_CHP=$REPO/egomimic/hydra_configs/data/pusht/planar_v2_chain_gripper_3000_plus_gen1199_prestep_split_seed42_v1.json
DATA_CHP="ICE_DATASET_DIR_CHAIN=$CHP,ICE_SPLIT_MANIFEST_CHAIN=$MAN_CHP,ICE_EXPECTED_SPLIT_MANIFEST_SHA256_CHAIN=$SHA_CHP"
UNITE_ENV="ICE_UNITE_FAST=true,ICE_UNITE_LR=${LR:-1e-4},ICE_UNITE_LR_FINAL=${LR_FINAL:-5e-5}"
[ -n "${LR_TERMINAL:-}" ] && UNITE_ENV="$UNITE_ENV,ICE_UNITE_LR_TERMINAL=$LR_TERMINAL"
case "$ROW" in
  ctA)  EXP=pusht/unite_cotrain_usocket_chain_val01_h16; MODEL=bf/ct_unite_register_separate_nt8_h384_s42;  DATA="$DATA_US,$DATA_CH"; FAM=$UNITE_ENV; PROJ=pushshapes-flow-transfer ;;
  ctAc) EXP=pusht/unite_cotrain_usocket_chain_val01_h16; MODEL=bf/ct_unite_register_separate_nt8_h384_s42;  DATA="$DATA_US,$DATA_CH"; FAM="$UNITE_ENV,ICE_UNITE_COMPILE=true"; PROJ=pushshapes-flow-transfer ;;
  ctA768)   EXP=pusht/unite_cotrain_usocket_chain_val01_h16;    MODEL=bf/ct_unite_register_separate_nt8_h768_s42;  DATA="$DATA_US,$DATA_CH";  FAM=$UNITE_ENV; PROJ=pushshapes-flow-transfer ;;
  s3ctA768) EXP=pusht/unite_cotrain_usocket_chaingen_val01_h16; MODEL=bf/ct_unite_register_separate_nt8_h768_s42;  DATA="$DATA_US,$DATA_CHG"; FAM=$UNITE_ENV; PROJ=pushshapes-flow-transfer; SWEEP=unite-cotrain-3 ;;
  ctB)  EXP=pusht/unite_cotrain_usocket_chain_val01_h16; MODEL=bf/ct_unite_register_split_tok_nt8_h384_s42; DATA="$DATA_US,$DATA_CH"; FAM=$UNITE_ENV; PROJ=pushshapes-flow-transfer ;;
  dpct) EXP=pusht/planar_v2_cotrain_dp_paper_points6;    MODEL=bf/bf_planar_v2_dp_paper_points6;             DATA="$DATA_US,$DATA_CH"; FAM=ICE_UNITE_FAST=false; PROJ=pushshapes-planar-v2 ;;
  dpus) EXP=pusht/planar_v2_usocket_dp_paper;            MODEL=bf/bf_planar_v2_dp_paper;                     DATA="$DATA_US";          FAM=ICE_UNITE_FAST=false; PROJ=pushshapes-planar-v2 ;;
  uniteus) EXP=pusht/unite_usocket_register_sweep_val01_h16; MODEL=bf/us_unite_register_separate_nt8_h384_s42; DATA="$DATA_US"; FAM=$UNITE_ENV; PROJ=pushshapes-flow-transfer ;;
  unitech) EXP=pusht/unite_chain_points_val01_h16;          MODEL=bf/ch_unite_register_separate_nt8_h384_s42; DATA="$DATA_CH"; FAM=$UNITE_ENV; PROJ=pushshapes-flow-transfer ;;
  s3ctA)     EXP=pusht/unite_cotrain_usocket_chaingen_val01_h16;    MODEL=bf/ct_unite_register_separate_nt8_h384_s42; DATA="$DATA_US,$DATA_CHG"; FAM=$UNITE_ENV; PROJ=pushshapes-flow-transfer; SWEEP=unite-cotrain-3 ;;
  s3ctB)     EXP=pusht/unite_cotrain_usocket_chaingen_val01_h16;    MODEL=bf/ct_unite_register_split_tok_nt8_h384_s42; DATA="$DATA_US,$DATA_CHG"; FAM=$UNITE_ENV; PROJ=pushshapes-flow-transfer; SWEEP=unite-cotrain-3 ;;
  s3unitech) EXP=pusht/unite_chaingen_points_val01_h16;             MODEL=bf/ch_unite_register_separate_nt8_h384_s42; DATA="$DATA_CHG";          FAM=$UNITE_ENV; PROJ=pushshapes-flow-transfer; SWEEP=unite-cotrain-3 ;;
  s3dpct)    EXP=pusht/planar_v2_cotrain_dp_paper_points6_chaingen; MODEL=bf/bf_planar_v2_dp_paper_points6;            DATA="$DATA_US,$DATA_CHG"; FAM=ICE_UNITE_FAST=false; PROJ=pushshapes-planar-v2; SWEEP=unite-cotrain-3 ;;
  s3dpch)    EXP=pusht/planar_v2_chaingen_points_dp_paper;          MODEL=bf/bf_planar_v2_dp_paper_points6;            DATA="$DATA_CHG";          FAM=ICE_UNITE_FAST=false; PROJ=pushshapes-planar-v2; SWEEP=unite-cotrain-3 ;;
  refctA)    EXP=pusht/unite_cotrain_usocket_chaingen_val01_h16;    MODEL=bf/ct_unite_register_separate_nt8_h384_s42; DATA="$DATA_US,$DATA_CHG"; FAM=$UNITE_ENV; PROJ=pushshapes-flow-transfer; SWEEP=af-ladder-20260913; GRES=${GRES:-gpu:$WORLD}; NORM_STATS=${NORM_STATS:-$CEDAR/unite-cotrain-3/norm_stats_chaingen_minmax}; EXTRA="model.pipeline.stages.4.cfg_scale=1.0${EXTRA:+ $EXTRA}" ;;
  refdpct)   EXP=pusht/planar_v2_cotrain_dp_paper_points6_chaingen; MODEL=bf/bf_planar_v2_dp_paper_points6;            DATA="$DATA_US,$DATA_CHG"; FAM=ICE_UNITE_FAST=false; PROJ=pushshapes-planar-v2; SWEEP=af-ladder-20260913; GRES=${GRES:-gpu:$WORLD}; NORM_STATS=${NORM_STATS:-$CEDAR/unite-cotrain-3/s3-dp_paper-cotrain-chaingen-240k-09100200/norm_stats} ;;
  refctP)    EXP=pusht/unite_cotrain_usocket_chainpre_val01_h16;    MODEL=bf/ct_unite_register_separate_nt8_h384_s42; DATA="$DATA_US,$DATA_CHP"; FAM=$UNITE_ENV; PROJ=pushshapes-flow-transfer; SWEEP=cotrain-prestep-20260916; GRES=${GRES:-gpu:$WORLD}; EXTRA="model.pipeline.stages.4.cfg_scale=1.0${EXTRA:+ $EXTRA}" ;;
  refdpctP)  EXP=pusht/planar_v2_cotrain_dp_paper_points6_chainpre; MODEL=bf/bf_planar_v2_dp_paper_points6;            DATA="$DATA_US,$DATA_CHP"; FAM=ICE_UNITE_FAST=false; PROJ=pushshapes-planar-v2; SWEEP=cotrain-prestep-20260916; GRES=${GRES:-gpu:$WORLD} ;;
  cl1dec|cl2msnr|cl3cfm|cl4disp|cl5bn|cl6ld8)
    # 2026-09-25 change loop (orchestrator-260925-0105): refctP + exactly one change each, refctP's
    # pinned norm stats, CFG-1.0 validation. S4=policy stage, S5=objective stage.
    S4=model.pipeline.stages.4; S5=model.pipeline.stages.5
    case "$ROW" in
      cl1dec)  CL="+$S4.decoded_action_samples_per_reconstruction=4 +$S5.decoded_action_weight=0.1" ;;
      cl2msnr) CL="$S4.train_eps=0.447" ;;
      cl3cfm)  CL="+$S4.contrastive_flow_weight=0.05" ;;
      cl4disp) CL="+$S4.dispersive_weight=0.5 +$S4.dispersive_tau=0.5 +$S4.dispersive_block=3" ;;
      cl5bn)   CL="+$S4.flow_latent_batchnorm=true" ;;
      cl6ld8)  CL="model.latent_dim=8" ;;
    esac
    EXP=pusht/unite_cotrain_usocket_chainpre_val01_h16; MODEL=bf/ct_unite_register_separate_nt8_h384_s42; DATA="$DATA_US,$DATA_CHP"; FAM=$UNITE_ENV; PROJ=pushshapes-flow-transfer; SWEEP=unite-change-loop-20260925; GRES=${GRES:-gpu:$WORLD}
    NORM_STATS=${NORM_STATS:-$CEDAR/cotrain-prestep-20260916/norm_stats_refctP}; EXTRA="model.pipeline.stages.4.cfg_scale=1.0 $CL${EXTRA:+ $EXTRA}" ;;
  dpch) EXP=pusht/planar_v2_chain_points_dp_paper;       MODEL=bf/bf_planar_v2_dp_paper_points6;             DATA="$DATA_CH";          FAM=ICE_UNITE_FAST=false; PROJ=pushshapes-planar-v2 ;;
  *) echo "row must be ctA|ctAc|ctA768|ctB|dpct|dpus|dpch|uniteus|unitech|s3ctA|s3ctB|s3ctA768|s3unitech|s3dpct|s3dpch|refctA|refdpct|refctP|refdpctP|cl1dec|cl2msnr|cl3cfm|cl4disp|cl5bn|cl6ld8"; exit 64 ;;
esac
[ "${SMOKE:-0}" = 1 ] && SWEEP=${SWEEP}-smoke
OUT=$CEDAR/$SWEEP/${RUN_NAME:-${TAG}-${STAMP}}
mkdir -p "$CEDAR/$SWEEP" /home/hice1/agao81/scratch/logs
NS=""; [ -n "${NORM_STATS:-}" ] && NS=",ICE_NORM_STATS_PATH=$NORM_STATS"
EO=""; [ -n "${EXTRA:-}" ] && EO=",ICE_EXTRA_OVERRIDES=$EXTRA"
COMMON="ICE_LAUNCH_MODE=run,ICE_REPO=$REPO,ICE_EXPECTED_HEAD=$HEAD,ICE_PYTHON=$PY,ICE_WANDB_ENTITY=rl2-group,ICE_WANDB_PROJECT=$PROJ,ICE_WANDB_GROUP=${WANDB_GROUP:-$SWEEP},ICE_EXPECTED_ACCOUNT=ece,ICE_EXPECTED_PARTITION=coe-gpu,ICE_EXPECTED_QOS=coe-ice,ICE_OUTPUT_DIR=$OUT,ICE_WANDB_RUN_ID=${WANDB_RUN_ID:-aidan-ct2-${TAG}-${STAMP}},ICE_LIMIT_TRAIN_BATCHES=1.0,ICE_LIMIT_VAL_BATCHES=8,ICE_TRAIN_BATCH_SIZE=32,ICE_VALID_BATCH_SIZE=32"
EXPORTS="$COMMON,$DATA,ICE_EXPERIMENT=$EXP,ICE_MODEL=$MODEL,ICE_WORLD_SIZE=$WORLD,ICE_MAX_STEPS=$STEPS,ICE_VAL_CHECK_INTERVAL=${VAL_EVERY:-30000},ICE_CHECKPOINT_EVERY_N_STEPS=${CKPT_EVERY:-30000},$FAM$NS$EO"
if [ "${PREFLIGHT:-0}" = 1 ]; then
  # The launcher's official ICE_LAUNCH_MODE=preflight, run inline in this shell with the row's exact
  # exports (no sbatch, no GPU): dataset validation, resolved-config asserts, runner dry-run. Run it
  # inside an ice-cpu allocation -- login nodes kill large processes.
  ( IFS=,; set -f; export ${EXPORTS/ICE_LAUNCH_MODE=run/ICE_LAUNCH_MODE=preflight}; exec bash "$REPO/scripts/ice/launch_unite_cotrain.sbatch" )
  exit
fi
JOB=$(sbatch --parsable ${TEST_ONLY:+--test-only} --account=ece --partition=coe-gpu --qos=coe-ice --exclude="$EXCL" --time=$TIME \
  --gres=${GRES:-gpu:h200:$WORLD} --ntasks-per-node=$WORLD --cpus-per-task=8 --mem=128G \
  --job-name="ct2-$TAG" --output="$LOG" \
  --export="ALL,$EXPORTS" \
  "$REPO/scripts/ice/launch_unite_cotrain.sbatch")
echo "$JOB $OUT"
