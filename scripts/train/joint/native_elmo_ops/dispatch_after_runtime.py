"""Fixed phase adapter, reached after exact runtime_bootstrap/service_bootstrap."""
import os
from pathlib import Path
import runpy
import sys
from wrapper_common import MODES, bound, need, read_json, sha, write_once, now
from runtime_guard import attempt_path, capsule, native_auth, prerequisite_path


def main(request_path):
    from execution_spec import load_execution
    args = (os.environ['ELMO_EXECUTION_SPEC'], os.environ['ELMO_EXECUTION_SPEC_SHA256'],
            os.environ['ELMO_EXECUTION_MODE'], os.environ['ELMO_OPS_MANIFEST_SHA256'])
    spec, path, q, job, claim = load_execution(*args)
    mode = spec['mode']
    need(mode in {'inspect', 'diagnostic', 'finalize', 'full'} and str(path) == request_path,
         'Fixed non-capture phase adapter required')
    from scripts.train.joint.service_bootstrap import activate
    root = activate(q)
    need(root == Path(spec['operational_service']['root']), 'Active operational service differs')
    import torch
    parent = bound(spec['parent_request'])
    need(Path(torch.__file__).is_relative_to(Path(parent['lambda_runtime']['fa4_root']) / 'torch-runtime'),
         'Pinned runtime bootstrap did not select exact Torch')
    attempt = attempt_path(spec, job)
    actual = read_json(attempt / 'pre-torch-prerequisites.json')
    need(actual['job_id'] == job and actual['request_sha256'] == sha(path), 'Pre-Torch actual evidence differs')
    auth = native_auth(spec, path, q, job)
    gpu = None
    if mode in {'diagnostic', 'full'}:
        from gpu_health_set import admit_gpu_set
        gpu = admit_gpu_set(attempt / 'gpu-health.json', 4)
        need(gpu['status'] == 'PASS', 'Actual four-H100 health admission failed')
    else:
        need(not os.environ.get('SLURM_JOB_GPUS') and not os.environ.get('SLURM_STEP_GPUS'), 'CPU phase owns GPUs')
    receipt = dict(schema='native-elmo-allocated-prerequisites-v1',
        status='ACTUAL_PREREQUISITES_AND_NATIVE_ALLOCATION_PASS',
        request_sha256=sha(path), job_id=job, native=auth['native'], checked_at=now(),
        parent_request=spec['parent_request'], execution_spec_sha256=args[1],
        ops_manifest_sha256=args[3], operational_service=spec['operational_service'],
        actual_prerequisites=actual, actual_gpu_health=gpu,
        native_client_manifest_sha256=capsule(spec, job),
        host_auth={'path': str(attempt / 'host-native.json'), 'sha256': sha(attempt / 'host-native.json')},
        container_auth={'path': str(attempt / 'container-native.json'), 'sha256': sha(attempt / 'container-native.json')},
        actual_runtime_bootstrap=True, actual_service_bootstrap=True)
    write_once(prerequisite_path(spec, q), receipt)
    module = 'scripts.train.joint.' + MODES[mode]
    sys.argv = [module, str(path)]
    runpy.run_module(module, run_name='__main__', alter_sys=True)


if __name__ == '__main__':
    main(sys.argv[1])
