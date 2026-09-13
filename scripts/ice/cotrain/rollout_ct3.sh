#!/bin/bash
# Protocol rollout submitter (PushShapes eval protocol sim_v2, horizon revision 3).
#   rollout_ct3.sh <run_dir> <ckpt_file> <usocket|chain> <level> <tag> [--cfg X] [--budget-set s2|s3] [--seed-block 0|40] [extra driver args]
# level 0: 40 seeds (block 0 = canonical seeds 0-39; block 40 = extension seeds 40-79, non-canonical).
# level 1-30: the five OEC-56 bank seeds for that level. Budget = the model's own training set's
# rev-3 p99 file (chain: s2 = chain_gripper_3000_v2, s3 = chain 3000 + gen 1919; U-Socket levels
# 1-30 are DERIVED from the chain 3000+gen set at the level-0 speed ratio). Full horizon, EMA,
# replan 8, chunk start 0, eager inference, sim_chain tree (env/obstacles SHA pinned by the bank).
# UNITE rows run at the checkpoint's embedded CFG unless --cfg is given (then labeled NON-PROTOCOL).
set -euo pipefail
RUN=${1:?}; CKPT=${2:?}; EMB=${3:?}; LEVEL=${4:?}; TAG=${5:?}; shift 5
CFG_OVR=""; BSET=s2; SEEDBLOCK=0; EXTRA=""; ENS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --cfg) CFG_OVR=$2; shift 2 ;;
    --ens) ENS=$2; shift 2 ;;
    --budget-set) BSET=$2; shift 2 ;;
    --seed-block) SEEDBLOCK=$2; shift 2 ;;
    *) EXTRA="$EXTRA $1"; shift ;;
  esac
done
SOURCE=/home/hice1/agao81/scratch/EgoVerse-graph-eval
PY=/home/hice1/agao81/scratch/EgoVerse-graph-unite/.venv/bin/python
SIM=/home/hice1/agao81/scratch/sim_chain
PM=/home/hice1/agao81/scratch/protocol_mirror
BANK=$PM/seeds/eval_side_access_seeds_newgeom_150.json
CFG=$RUN/provenance/restart-0/resolved_config.yaml
test -s "$RUN/checkpoints/$CKPT" || { echo "missing checkpoint $RUN/checkpoints/$CKPT"; exit 64; }
test -s "$CFG" || { echo "missing $CFG"; exit 64; }
case "$EMB" in
  usocket) EARGS="--pusher u_socket --embodiment-name pushshapes_sim_u_socket --embodiment-id 19"
           BUDGET=$PM/budgets_ice/u_socket_3000_v2_clean_p99_derived_from_chain3000_gen1919.json ;;
  chain)   EARGS="--pusher chain_gripper --embodiment-name pushshapes_sim_chain_gripper --embodiment-id 20 --chain-control-mode points"
           case "$BSET" in
             s2) BUDGET=$PM/budgets_ice/chain_gripper_3000_v2_p99.json ;;
             s3) BUDGET=$PM/budgets_ice/chain_gripper_3000_v2_plus_gen_1919_p99.json ;;
             *) echo "budget-set must be s2|s3"; exit 64 ;;
           esac ;;
  *) echo "emb must be usocket|chain"; exit 64 ;;
esac
test -s "$BUDGET" || { echo "missing budget $BUDGET"; exit 64; }
build_args() {  # $1 = level -> sets ARGS_ONE and LABEL_ONE for that level
  local level=$1 seedargs label
  if [ "$level" = 0 ]; then
    seedargs="--n-episodes 40 --seed-base $SEEDBLOCK"
    if [ "$SEEDBLOCK" = 0 ]; then label=CANONICAL_LEVEL0_SEEDS_0_39; else label=EXTENSION_SEEDS_${SEEDBLOCK}_NON_CANONICAL; fi
  else
    local seeds
    seeds=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('+'.join(str(int(e['seed'])) for e in d['levels'][sys.argv[2]]))" "$BANK" "$level")
    test -n "$seeds" || { echo "no bank seeds for level $level"; exit 64; }
    seedargs="--seeds $seeds"
    label=OEC56_LEVEL_${level}
  fi
  local cfgs=""
  if grep -q ReleasedRecipeUniteLatentPolicy "$CFG"; then
    if [ -n "$CFG_OVR" ]; then cfgs="--cfg-scale $CFG_OVR"; label=${label}_CFG${CFG_OVR}_NON_PROTOCOL; else label=${label}_CFG_EMBEDDED; fi
  elif [ -n "$CFG_OVR" ]; then echo "--cfg applies to latent-flow rows only"; exit 64; fi
  if [ -n "$ENS" ]; then cfgs="$cfgs --ensemble-samples $ENS"; label=${label}_ENS${ENS}_NON_PROTOCOL; fi
  local out=$OUTD/${TAG}
  [ "$MULTI" = 1 ] && out=$OUTD/${TAG}-L${level}
  ARGS_ONE="--ckpt $RUN/checkpoints/$CKPT --config-path $CFG --output-dir $RUN $EARGS --use-ema $cfgs --replan-every 8 --chunk-start 0 $seedargs --obstacle-level $level --budget-json $BUDGET --full-horizon --image-size 96 --success-threshold 0.80 --label $label --out ${out}.json $EXTRA"
}
OUTD=${OUTDIR:-/home/hice1/agao81/scratch/rollouts/CT2-rev3}; mkdir -p "$OUTD"
EXCL=$(paste -sd, /home/hice1/agao81/scratch/autoresearch/bad_gpu_nodes.txt)
case "$LEVEL" in
  *-*)  # a level range, e.g. 1-5: one GPU job, one invocation per level, result <tag>-L<level>.json
    MULTI=1; LIST=""
    for level in $(seq ${LEVEL%-*} ${LEVEL#*-}); do build_args "$level"; LIST="$LIST${LIST:+ ;; }$ARGS_ONE"; done
    sbatch --parsable --account=ece --partition=coe-gpu --qos=coe-ice --exclude="$EXCL" --time=3:00:00 --job-name="ro3-$TAG" \
      --export="ALL,ROLLOUT_ARGS_LIST=$LIST,SOURCE=$SOURCE,PYTHON=$PY,SIM_ROOT=$SIM" "$SOURCE/scripts/eval/rollout_multi.sbatch" ;;
  *)
    MULTI=0; build_args "$LEVEL"
    sbatch --parsable --account=ece --partition=coe-gpu --qos=coe-ice --exclude="$EXCL" --time=3:00:00 --job-name="ro3-$TAG" \
      --export="ALL,ROLLOUT_ARGS=$ARGS_ONE,SOURCE=$SOURCE,PYTHON=$PY,SIM_ROOT=$SIM" "$SOURCE/scripts/eval/rollout_pushshapes.sbatch" ;;
esac
