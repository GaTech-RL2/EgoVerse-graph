#!/bin/bash
set -euo pipefail
root=/run/elmo-slurm-client
unset LD_PRELOAD LD_LIBRARY_PATH LD_AUDIT
export SLURM_CONF="$root/etc/slurm.conf"
exec "$root/lib/ld-linux-x86-64.so.2" --library-path "$root/lib" "$root/bin/scontrol" "$@"
