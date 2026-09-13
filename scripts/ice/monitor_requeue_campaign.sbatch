#!/usr/bin/env bash
# Bounded, self-requeueing watcher for a Submitit campaign.
#
# The watcher is deliberately campaign-scoped: it discovers only numeric
# Submitit job directories below ICE_REQUEUE_CAMPAIGN_ROOT, so it cannot
# requeue unrelated jobs owned by the user.  A terminal job is requeued at
# most ICE_REQUEUE_MAX_ATTEMPTS times.  Every decision is recorded in the
# campaign's .requeue-monitor directory.

#SBATCH --job-name=ice-requeue-watch
#SBATCH --output=/storage/cedar/cedar0/cedarp-dxu345-0/egoverse-graph-runs/_ice-log-archive/requeue-watch-%j.out
#SBATCH --error=/storage/cedar/cedar0/cedarp-dxu345-0/egoverse-graph-runs/_ice-log-archive/requeue-watch-%j.err
#SBATCH --partition=ice-cpu
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:30:00
#SBATCH --requeue

set -euo pipefail

CAMPAIGN_ROOT=${ICE_REQUEUE_CAMPAIGN_ROOT:?set ICE_REQUEUE_CAMPAIGN_ROOT}
INTERVAL_SECONDS=${ICE_REQUEUE_INTERVAL_SECONDS:-300}
MAX_ATTEMPTS=${ICE_REQUEUE_MAX_ATTEMPTS:-3}
WATCH_SECONDS=${ICE_REQUEUE_WATCH_SECONDS:-1500}
TARGET_JOB_IDS=${ICE_REQUEUE_JOB_IDS:-}
STATE_ROOT="$CAMPAIGN_ROOT/.requeue-monitor"
LOG_FILE="$STATE_ROOT/requeue.log"
LOCK_FILE="$STATE_ROOT/lock"

mkdir -p "$STATE_ROOT"
exec 9>"$LOCK_FILE"
flock -n 9 || exit 0

log() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >>"$LOG_FILE"
}

numeric_job_dir() {
  if [[ -n "$TARGET_JOB_IDS" ]]; then
    # Semicolons are used instead of commas because sbatch --export treats
    # commas as environment-variable separators.
    tr ';' '\n' <<<"$TARGET_JOB_IDS" | awk '/^[0-9]+$/' | sort -n -u
    return
  fi
  find "$CAMPAIGN_ROOT" -type d -path '*/.submitit/[0-9]*' -printf '%f\n' 2>/dev/null \
    | awk '/^[0-9]+$/' | sort -n -u
}

job_submission_script() {
  local job_id=$1
  find "$CAMPAIGN_ROOT" -type f \
    -path "*/.submitit/$job_id/${job_id}_submission.sh" -print -quit 2>/dev/null
}

application_failure_marker() {
  local submission=$1 run_dir file
  [[ -n "$submission" ]] || return 1
  run_dir=$(dirname "$(dirname "$(dirname "$submission")")")
  while IFS= read -r -d '' file; do
    if grep -Eiq 'Traceback|InstantiationException|FileNotFoundError|PermissionError|MisconfigurationException|OutOfMemory|out[[:space:]]+of[[:space:]]+memory|CUDA.*memory|Killed' "$file"; then
      return 0
    fi
  done < <(
    find "$run_dir" -type f \( -name 'trainHydra.log' -o -name '*_log.err' -o -name '*_log.out' \) -print0 2>/dev/null
  )
  return 1
}

retry_count_file() {
  printf '%s/%s.attempts' "$STATE_ROOT" "$1"
}

read_attempts() {
  local file
  file=$(retry_count_file "$1")
  if [[ -s "$file" ]] && [[ $(<"$file") =~ ^[0-9]+$ ]]; then
    printf '%s' "$(<"$file")"
  else
    printf '0'
  fi
}

write_attempts() {
  local job_id=$1 attempts=$2 file tmp
  file=$(retry_count_file "$job_id")
  tmp="$file.tmp.$$"
  printf '%s\n' "$attempts" >"$tmp"
  mv -f -- "$tmp" "$file"
}

requeueable_state() {
  case "$1" in
    TIMEOUT|PREEMPTED|NODE_FAIL|BOOT_FAIL|DEADLINE|FAILED|FAILED_APP|OUT_OF_MEMORY)
      return 0 ;;
    *)
      return 1 ;;
  esac
}

scan_once() {
  local job_id active state elapsed exit_code attempts submission
  while IFS= read -r job_id; do
    [[ -n "$job_id" ]] || continue

    active=$(squeue -h -j "$job_id" -o '%T|%M|%R' 2>/dev/null || true)
    if [[ -n "$active" ]]; then
      log "observe job=$job_id active=$active"
      continue
    fi

    IFS='|' read -r state elapsed exit_code < <(
      sacct -X -n -P -j "$job_id" \
        -o State,Elapsed,ExitCode 2>/dev/null | head -n 1
    ) || true
    state=${state%%+*}
    [[ -n "$state" ]] || { log "observe job=$job_id state=UNKNOWN"; continue; }

    submission=$(job_submission_script "$job_id")
    if [[ "$state" == COMPLETED || "$state" == CANCELLED ]]; then
      if [[ "$state" == COMPLETED ]] && application_failure_marker "$submission"; then
        log "observe job=$job_id state=$state application-failure-marker=1 action=requeue-candidate"
        state=FAILED_APP
      else
        log "observe job=$job_id state=$state elapsed=${elapsed:-unknown} exit=${exit_code:-unknown} action=none"
        continue
      fi
    fi
    if ! requeueable_state "$state"; then
      log "observe job=$job_id state=$state elapsed=${elapsed:-unknown} action=none"
      continue
    fi

    if [[ -z "$submission" ]] || ! grep -q '^#SBATCH --requeue' "$submission"; then
      log "observe job=$job_id state=$state action=skip-no-requeue-capability"
      continue
    fi

    attempts=$(read_attempts "$job_id")
    if (( attempts >= MAX_ATTEMPTS )); then
      log "observe job=$job_id state=$state attempts=$attempts action=skip-max-attempts"
      continue
    fi

    if scontrol requeue "$job_id" >/dev/null 2>&1; then
      attempts=$((attempts + 1))
      write_attempts "$job_id" "$attempts"
      log "requeue job=$job_id state=$state attempt=$attempts/$MAX_ATTEMPTS elapsed=${elapsed:-unknown} exit=${exit_code:-unknown}"
    else
      log "requeue-failed job=$job_id state=$state attempts=$attempts"
    fi
  done < <(numeric_job_dir)
}

end=$((SECONDS + WATCH_SECONDS))
while (( SECONDS < end )); do
  scan_once
  remaining=$((end - SECONDS))
  (( remaining > 0 )) || break
  sleep "$(( remaining < INTERVAL_SECONDS ? remaining : INTERVAL_SECONDS ))"
done

# Keep one watcher alive without a user crontab.  A self-requeue is safe here:
# the lock is released as the shell exits and the next allocation gets a new
# watcher process.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol requeue "$SLURM_JOB_ID" >/dev/null 2>&1 || \
    log "watcher-requeue-failed job=$SLURM_JOB_ID"
fi
