#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source this script so exported cluster paths reach the caller" >&2
  exit 2
fi

profile=${1:?usage: source scripts/clusters/common/load_profile.sh PROFILE}
case "$profile" in
  skynet|ice|phoenix) ;;
  *)
    echo "unsupported EgoVerse cluster profile: $profile" >&2
    return 2
    ;;
esac

profile_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck source=/dev/null
source "$profile_root/$profile/profile.sh"

for variable in \
  EGOVERSE_CLUSTER_PROFILE \
  PUSHSHAPES_DATA_ROOT \
  EGOVERSE_RUN_ROOT \
  EGOVERSE_SOURCE_ROOT
do
  if [[ -z "${!variable:-}" ]]; then
    echo "$profile profile did not bind required variable $variable" >&2
    return 2
  fi
done

export EGOVERSE_CLUSTER_PROFILE
export PUSHSHAPES_DATA_ROOT
export EGOVERSE_RUN_ROOT
export EGOVERSE_SOURCE_ROOT
