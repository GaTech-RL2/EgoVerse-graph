#!/bin/bash
# The outer wrapper supplies a pinned scientific Python prefix for every rank.
set -euo pipefail
test -r /run/pace-proof/control.json
exec /usr/bin/python3.12 /run/pace-boundary/gpu_admission.py --bootstrap
