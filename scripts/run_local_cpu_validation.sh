#!/usr/bin/env bash
# Local CPU regression runner, not a cluster launcher or exact-runtime proof.
set -euo pipefail

# Large synthetic lifecycle fixtures write real full-state checkpoints. Fail
# before optimizer work rather than turn exhausted local storage into a model
# failure. Callers may raise (but not lower) this floor for larger suites.
required_kib=${LOCAL_CPU_TEST_MIN_FREE_KIB:-4194304}
[[ "$required_kib" =~ ^[0-9]+$ ]] && (( required_kib >= 4194304 )) || {
    echo "Local CPU validation requires at least 4 GiB free-space floor" >&2
    exit 64
}
available_kib=$(df -Pk "${TMPDIR:-/tmp}" | awk 'NR == 2 {print $4}')
[[ "$available_kib" =~ ^[0-9]+$ ]] && (( available_kib >= required_kib )) || {
    echo "Local CPU validation blocked: insufficient temporary-storage headroom" >&2
    exit 64
}

# The Rosetta Torch/OpenMP validation environment stalled inside libiomp5 with
# multiple threads. Set these before Python imports and emit stalled-test stacks.
# No training config, GPU runtime, source identity or approval is changed here.
diagnostic_timeout=60
for argument in "$@"; do
    case "$argument" in
        tests/test_recursive_config_audit.py|tests/test_recursive_config_audit.py::*)
            # This intentionally long, all-YAML test takes >60s. Native-arm
            # CPython's concurrent frame dump stalled in dump_frame and held
            # pytest's cancellation lock. Preserve diagnostics at its own
            # bounded 10-minute threshold rather than dump during healthy work.
            diagnostic_timeout=600
            ;;
    esac
done
exec env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMBA_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue \
    "${LOCAL_CPU_TEST_PYTHON:-python3}" -m pytest \
    -p scripts.local_cpu_storage_guard \
    -o "faulthandler_timeout=$diagnostic_timeout" "$@"
