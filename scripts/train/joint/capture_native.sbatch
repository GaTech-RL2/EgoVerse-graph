#!/bin/bash
set -euo pipefail
root=${1:?Operations root}
parent=${2:?Exact own GPU job}
expected_name=${3:?Exact job name}
phase=${4:?smoke or full}
[[ "$parent" =~ ^[0-9]+$ && "${SLURM_RESTART_COUNT:-0}" == 0 ]]
[[ "$phase" == smoke || "$phase" == full ]]
[[ "$expected_name" =~ ^joint-[a-z0-9-]+$ ]]
destination="$root/status/$phase-$parent.native-completion.txt"
[[ ! -e "$destination" ]]
state=$(/usr/local/bin/scontrol show job "$parent" -o)
[[ "$state" == *"JobId=$parent "* && "$state" == *"UserId=paphiwetsa3("* ]]
[[ "$state" == *"JobName=$expected_name "* ]]
(set -o noclobber; printf '%s\n' "$state" > "$destination")
