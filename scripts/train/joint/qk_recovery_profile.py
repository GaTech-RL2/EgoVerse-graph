"""Exact, immutable original QK Elmo allocation profile; stdlib metadata only.

This is not an ImageNet profile, a new submission authority, or a step-zero
retry. A later failed replacement requires a separately reviewed original pin.
"""
import hashlib
import json

ARM = 'qk'
RUN_ID = 'mot-elmo-pickplace-continuous-large314m-qknorm-swiglu-rmsnorm-robot-human-camera-subtask-s42-300k-20261026-v1'
TASK_ROOT = '/workspace/users/paphiwetsa3/tasks/elmo-qknorm-swiglu-rmsnorm-imagenet-20261026-v1'
ROOT = TASK_ROOT + '/full-v5'
PARENT_ID, OBSERVER_ID, NODE = '412831', '412832', 'gpu21'
PARAMETERS = 314227000
REQUEST_SHA = '8760409ca77b6888fe700b4741d93da654c99c6ed9f3fa328194693ef5aa6592'
REQUEST_PATH = ROOT + '/ops/requests/elmo-large314m-qknorm-swiglu-rmsnorm-full-v5.json'
PARENT_CANONICAL_SHA = '5fe43f09dc0daef914ea809c923b5f14ab37a92d27f3a8d8488a1408899c8fd1'
FILES_SHA = '69a06292fb80e84b84dbaaa10025968f62e371f5b621d2348d0fbd1573c8a5f2'
PROFILE = dict(parent_id=PARENT_ID, observer_id=OBSERVER_ID, node=NODE, root=ROOT,
    parent_name='elmo-large314m-qknorm-swiglu-rmsnorm-full-v5',
    observer_name='elmo-native-qknorm-swiglu-rmsnorm-full-v5',
    request_path=REQUEST_PATH, request_sha256=REQUEST_SHA, files_sha256=FILES_SHA,
    parameters=PARAMETERS, has_smoke_identity=False, protocol='qk-full-v5-early-authenticated-v1',
    capture_script_sha256='8bda1c0b5b05e03ebf785bc478c13f8a86c924a150dca984cfd98022d1abccd8',
    submitter_sha256='564fa80ef96887399fc58675e88d4bef0322fc420fd5255949dfd99f922e36af')


def need(value, message):
    if not value: raise ValueError(message)


def is_qk(parent): return parent.get('run_id') == RUN_ID


def validate_parent(parent):
    """All original science, paths, private mounts, source and smoke stay exact."""
    payload = json.dumps(parent, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    need(hashlib.sha256(payload).hexdigest() == PARENT_CANONICAL_SHA,
         'QK recovery requires the exact original 412831 production request')
    return PROFILE


def supported_parameters(parent):
    if is_qk(parent):
        validate_parent(parent)
        return PARAMETERS
    need(parent.get('parameters') == 314187064, 'Unsupported recovery scientific architecture')
    return 314187064


def profile_run_id(arm):
    return RUN_ID if arm == ARM else ('mot-elmo-pickplace-continuous-large314m-' + arm +
                                      '-robot-human-camera-subtask-s42-300k-20261026-v1')
