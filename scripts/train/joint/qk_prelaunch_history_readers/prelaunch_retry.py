"""Read-only admission evidence for exact failed prelauncher 412823, once.

This is not interruption/checkpoint recovery and grants no live launch authority.
The guarded training launcher was unreachable in this exact observed failure.
The caller must separately validate current intent, all claims, native queue and
steps, exact own held/RUNNING allocation, destination freshness and storage.
No remote queries, ML imports, source mutation or old submitter replay occur here.
"""
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from .reviewed_parsers import reviewed_parser

PARENT = '412823'
CAPTURE = '412824'
OLD_ROOT = '/workspace/users/paphiwetsa3/tasks/elmo-qknorm-swiglu-rmsnorm-imagenet-20261026-v1/full-v1'
NEW_ROOT = OLD_ROOT[:-1] + '2'
NAME = 'elmo-large314m-qknorm-swiglu-rmsnorm-full-v1'
REQUEST_REL = 'ops/requests/' + NAME + '.json'
REQUEST_SHA = '39d1059efc6292fcaf0a8c2b6b438bb84496ba947ee4b198b70cf3ad28108d37'
FILES_SHA = 'c1d356ba049931c479eaf886873a3ca6168611baf67e07f69898ecd3c1439ffb'
SOURCE = '82f65498e25f41b87c954717722dcf7360e5bde8'
RUN_ID = 'mot-elmo-pickplace-continuous-large314m-qknorm-swiglu-rmsnorm-robot-human-camera-subtask-s42-300k-20261026-v1'
ORIGINAL_CANONICAL_SHA = '6dc45ea2d9a541b40d26df7fd7a9d5b44e7b69443bdb510dc97fa58d6f84801a'
MAX_FILE_BYTES = 1 << 20
# These are immutable actual original/observed bytes, not caller-set bindings.
EVIDENCE_HASHES = {'observed/capture-running-412824.txt': '7091d0f91eeaffd69d4036547154868f35eccdfee395c0ce00ab218a27305cf2',
 'observed/claims/39d1059efc6292fcaf0a8c2b6b438bb84496ba947ee4b198b70cf3ad28108d37/CAPTURE_JOB_ID': '0160a48366d96df8183b32c705c51a8589b09e0fe165f4173b31ba9160e8be94',
 'observed/claims/39d1059efc6292fcaf0a8c2b6b438bb84496ba947ee4b198b70cf3ad28108d37/CAPTURE_PENDING_NATIVE.txt': '23f930f675ada9fa450d8e474f9b6a4d8ab6b120f64ef5645273840ed7840037',
 'observed/claims/39d1059efc6292fcaf0a8c2b6b438bb84496ba947ee4b198b70cf3ad28108d37/HELD_NATIVE.txt': '8b3d5257a8252fb27cd27b6edbd7a6738b1d0f9666fffe7b6c83f6aaec2c6952',
 'observed/claims/39d1059efc6292fcaf0a8c2b6b438bb84496ba947ee4b198b70cf3ad28108d37/PARENT_JOB_ID': '3a27209f7594b239c87d2d3e397670c6022f7e8a7e48926a579e92c88bd2e265',
 'observed/claims/39d1059efc6292fcaf0a8c2b6b438bb84496ba947ee4b198b70cf3ad28108d37/PREPARED.json': '363761939e03f1f4e51130b66797336394f65b04890750b71a33926d915a3305',
 'observed/claims/39d1059efc6292fcaf0a8c2b6b438bb84496ba947ee4b198b70cf3ad28108d37/RELEASE_ATTEMPT.json': 'd4142a7f9bbcaef83726cb217c42ad66aefbeab3a1ad46ba03a137fd970b4044',
 'observed/claims/39d1059efc6292fcaf0a8c2b6b438bb84496ba947ee4b198b70cf3ad28108d37/RELEASE_CAPTURE_PENDING_NATIVE.txt': '23f930f675ada9fa450d8e474f9b6a4d8ab6b120f64ef5645273840ed7840037',
 'observed/claims/39d1059efc6292fcaf0a8c2b6b438bb84496ba947ee4b198b70cf3ad28108d37/RELEASE_HELD_NATIVE.txt': '8b3d5257a8252fb27cd27b6edbd7a6738b1d0f9666fffe7b6c83f6aaec2c6952',
 'observed/host-exit-412823.txt': '4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865',
 'observed/logs/elmo-large314m-qknorm-swiglu-rmsnorm-full-v1-412823.err': 'de6b99f3287e76bcb5eff739e7f9ac754a39ae7ac967ef8977f46dcb9fa7fea5',
 'observed/logs/elmo-large314m-qknorm-swiglu-rmsnorm-full-v1-412823.out': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
 'observed/logs/elmo-native-qknorm-swiglu-rmsnorm-full-v1-412824.err': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
 'observed/logs/elmo-native-qknorm-swiglu-rmsnorm-full-v1-412824.out': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
 'observed/native-412823.exit': '9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa',
 'observed/native-412823.stderr': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855',
 'observed/native-412823.steps.json': 'bf9d2b14fb8126de91f14fad383e9f2953ed96c8ae51c6672f45c92959730b8c',
 'observed/native-412823.txt': '1dfea4347ff0f9db7355d8b04ba04b2aaf963188784473f64f487fcf31076963',
 'original/FILES.sha256': 'c1d356ba049931c479eaf886873a3ca6168611baf67e07f69898ecd3c1439ffb',
 'original/capture-native.sbatch': '5b319e794b6dcaecd53a3f7154e23f270207cc5e9cdc027235bba8e1e347e99a',
 'original/full.sbatch': '2b054973c44a98ff244c426c5001eaeac46a6f4c4ca91db5bdeac5497e0b0410',
 'original/guard_full.py': '0d8f85071dc8ce9d7364339b3c4ce777cbc8cfa6df4474b84769c9e41f2ea8ff',
 'original/native_guard.py': '08d5bd354a0eb81dad4b9664037e711ac4f7ecc2b57355db2e4154571c49697c',
 'original/ops/full.sbatch': '102acbf6ebb55ca63b1f7174a943bb0a63f93aa157e533b65aad1f2ea705d08d',
 'original/ops/requests/elmo-large314m-qknorm-swiglu-rmsnorm-full-v1.json': '39d1059efc6292fcaf0a8c2b6b438bb84496ba947ee4b198b70cf3ad28108d37',
 'original/ops/service/FILES.sha256': 'adf45524de3c2e6dba186572e7151d48bfca0b3adc8e6cf9defd686b47744c87',
 'original/ops/service/scripts/train/joint/contract.py': 'cd29edc148c0acbcc2ab31889eb4162dde57b834255d0966986af23ad21e2fed',
 'original/ops/service/scripts/train/joint/native_interruption.py': '53341ef0962883c45ef7df2ffa44f73a9cbd5ac32d7b847a163e938534f75364',
 'original/proofs/smoke/guard.py': '833f87f3e35b05fe2fb78eea2ba5038b8fc3c8535aba1d50777d1bf26d8eb87c',
 'original/submit_full_once.py': 'f2f55d3f0e9f41b3d85480ca9d8e0ec313fbe16e5343dffc461154bf88012afd',
 'transport/destination-observation.json': 'a54abdce28ac8fae28496d756cef412385394449c79b239560a91c90a5cfe88b',
 'transport/destination-observation.stderr': 'd86ce97e384891c38579d4b66b4fbf5c2dbd8935aced91891668cec9b835fced',
 'transport/destination-observation.stdout': '3d727b6ee94cbcd06e76b40117e764947b9e1a5649eb9cb8e3442dd7c212141c',
 'transport/diagnosis-final.json': 'b63992ec2988e62957d08a32a5cbcdbcf0b2579abba322f37c34b41cc0e9d445',
 'transport/failure-small-artifacts.json': '2382c54c7dcbec4e334ab7739703246b3b62802150065fed98272d2d135c6acc',
 'transport/status-and-gateway-python.json': 'f91b087807625ae98c890437456eefd41037343adbfc4484757edf20ceec5fb6',
 'transport/status-and-gateway-python.stderr': '66300fdf3fd8cb15a7a80def35c8b62c2c9686589c3e92f72818abbbd79e119c',
 'transport/status-and-gateway-python.stdout': 'a74ea7507ff30662000ed6dd89acdb54496d3582f92044041b3dac878a64a897',
 'transport/terminal-read-1.json': '34a5fe206445583ce1108e0bcb57cd3a6b9b004b5c6d6d9f5eb8ae4078a716b7',
 'transport/terminal-read-1.stderr': '5d4fd32cb4e030ab8e522e9dc0be35be7806278f01d698a2dadf5c8ffa56d94a',
 'transport/terminal-read-1.stdout': 'ac74be5921d83fd78ec57df8cb113eb152f837a8d6da80958b42141025cdec71'}


def need(value, message):
    if not value:
        raise ValueError(message)


def unique(pairs):
    value = {}
    for key, item in pairs:
        need(key not in value, 'Duplicate metadata key')
        value[key] = item
    return value


def read_json(raw):
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def bounded(path):
    need(path.resolve(strict=True) == path, 'Evidence path must be physical, not a symlink')
    descriptor = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        before = os.fstat(descriptor)
        need(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_FILE_BYTES, 'Unsafe or oversized evidence')
        with os.fdopen(descriptor, 'rb', closefd=False) as stream:
            raw = stream.read(MAX_FILE_BYTES + 1)
        after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        need(len(raw) <= MAX_FILE_BYTES and identity(before) == identity(after) == identity(current),
             'Evidence changed while reading')
        return raw
    finally:
        os.close(descriptor)


def native_time(value):
    need(isinstance(value, str) and re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d', value),
         'Missing native timestamp')
    return datetime.fromisoformat(value).replace(tzinfo=timezone.utc)


def verify_prelaunch_retry(evidence_root):
    """Authenticate positive prelauncher failure proof; return no live authority."""
    root = Path(evidence_root)
    need(root.is_absolute() and '..' not in root.parts and root.resolve(strict=True) == root
         and root.is_dir(), 'Physical absolute evidence root required')
    observed = set()
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs:
            need(not (Path(directory) / name).is_symlink(), 'Symlink evidence directory')
        for name in names:
            relative = str((Path(directory) / name).relative_to(root))
            observed.add(relative)
            need(relative in EVIDENCE_HASHES, 'Unreviewed file in original failure evidence')
    need(observed == set(EVIDENCE_HASHES), 'Original failure evidence is incomplete')
    raw = {}
    for name, digest in EVIDENCE_HASHES.items():
        value = bounded(root / name)
        need(hashlib.sha256(value).hexdigest() == digest, 'Changed original failure evidence: ' + name)
        raw[name] = value.decode('utf-8')
    original = read_json(raw['original/' + REQUEST_REL])
    need(hashlib.sha256(raw['original/' + REQUEST_REL].encode()).hexdigest() == REQUEST_SHA
         and hashlib.sha256(raw['original/FILES.sha256'].encode()).hexdigest() == FILES_SHA,
         'Wrong original request or sealed wrapper')
    need(original['run_id'] == RUN_ID and original['source_revision'] == SOURCE
         and original['parameters'] == 314227000 and original['phase'] == 'full',
         'Wrong scientific identity')
    manifest = {}
    for line in raw['original/FILES.sha256'].splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  (.+)', line)
        need(match is not None and match[2] not in manifest and '..' not in Path(match[2]).parts
             and not Path(match[2]).is_absolute(), 'Malformed original closure')
        manifest[match[2]] = match[1]
    for name, digest in EVIDENCE_HASHES.items():
        if name.startswith('original/') and name != 'original/FILES.sha256':
            need(manifest.get(name[len('original/'):]) == digest,
                 'Original reviewed control file is outside sealed closure')
    # Original parser source remains hash-checked evidence data. Execute only
    # the reviewed service-owned parser; preserve its extra failure checks below.
    guard = reviewed_parser(raw['original/native_guard.py'].encode(),
                            EVIDENCE_HASHES['original/native_guard.py'],
                            '_exact_prelaunch_native_guard')
    prefix = 'observed/claims/' + REQUEST_SHA + '/'
    need(raw[prefix + 'PARENT_JOB_ID'].strip() == PARENT
         and raw[prefix + 'CAPTURE_JOB_ID'].strip() == CAPTURE, 'Original accepted identities differ')
    prepared = read_json(raw[prefix + 'PREPARED.json'])
    release = read_json(raw[prefix + 'RELEASE_ATTEMPT.json'])
    need(prepared['request_sha256'] == REQUEST_SHA and prepared['manifest_sha256'] == FILES_SHA
         and release['request_sha256'] == REQUEST_SHA and release['parent'] == PARENT
         and release['capture'] == CAPTURE, 'Original accepted payload/release differs')
    held = [guard.check_native(raw[prefix + name], job=PARENT, state='PENDING', request_sha=REQUEST_SHA)
            for name in ('HELD_NATIVE.txt', 'RELEASE_HELD_NATIVE.txt')]
    observer_held = [guard.check_native(raw[prefix + name], job=CAPTURE, state='PENDING', capture=True,
                                      parent=PARENT, request_sha=REQUEST_SHA)
                     for name in ('CAPTURE_PENDING_NATIVE.txt', 'RELEASE_CAPTURE_PENDING_NATIVE.txt')]
    terminal = guard.check_native(raw['observed/native-' + PARENT + '.txt'], job=PARENT,
                                  state='FAILED', request_sha=REQUEST_SHA)
    need(terminal['NodeList'] == 'gpu21' and terminal['NumNodes'] == '1'
         and terminal['ExitCode'] == '1:0' and terminal['Reason'] == 'NonZeroExitCode'
         and terminal['WorkDir'] == OLD_ROOT
         and native_time(terminal['SubmitTime']) <= native_time(terminal['StartTime'])
         < native_time(terminal['EndTime']), 'Actual failed prelaunch interval differs')
    need({'cpu=32', 'mem=192G', 'node=1'} <= set(terminal['AllocTRES'].split(',')),
         'Failed parent actual resources differ')
    need(all(row['SubmitTime'] == terminal['SubmitTime'] for row in held), 'Failed parent is another submission')
    need(raw['observed/host-exit-' + PARENT + '.txt'].strip() == '1'
         and raw['observed/native-' + PARENT + '.exit'].strip() == '0'
         and not raw['observed/native-' + PARENT + '.stderr'], 'Actual failed host/native query missing')
    steps = read_json(raw['observed/native-' + PARENT + '.steps.json'])
    guard.release_observation(steps, PARENT, REQUEST_SHA, terminal)
    observer = guard.check_native(raw['observed/capture-running-' + CAPTURE + '.txt'], job=CAPTURE,
                                  state='RUNNING', capture=True, parent=PARENT, request_sha=REQUEST_SHA)
    guard.same_attempt(*observer_held, observer)
    transport = read_json(raw['transport/terminal-read-1.json'])
    need(transport['exit_code'] == 0 and transport['failure'] is None and transport['parent'] == PARENT
         and transport['capture'] == CAPTURE and transport['request_sha256'] == REQUEST_SHA
         and transport['stdout_sha256'] == hashlib.sha256(raw['transport/terminal-read-1.stdout'].encode()).hexdigest()
         and transport['stderr_sha256'] == hashlib.sha256(raw['transport/terminal-read-1.stderr'].encode()).hexdigest(),
         'Actual terminal readback provenance differs')
    rows = [line for line in raw['transport/terminal-read-1.stdout'].splitlines()
            if line.startswith('JobId=' + CAPTURE + ' ')]
    need(len(rows) == 2 or len(rows) == 1, 'Missing captured observer native record')
    completed = [line for line in rows if 'JobState=COMPLETED ' in line]
    need(len(completed) == 1, 'Observer own successful terminal record missing')
    done = guard.check_native(completed[0], job=CAPTURE, state='COMPLETED', capture=True,
                              parent=PARENT, request_sha=REQUEST_SHA)
    guard.same_attempt(*observer_held, observer, done, terminal=done)
    need(native_time(observer['StartTime']) >= native_time(terminal['EndTime']), 'Afterany ordering differs')
    # Positive failure proof: exact complete stderr contains this single traceback
    # through destination_probe -> sha. No CSV/log absence is used to infer zero.
    trace = raw['observed/logs/' + NAME + '-' + PARENT + '.err']
    need(trace.count('Traceback (most recent call last):') == 1
         and trace.endswith("AttributeError: module 'hashlib' has no attribute 'file_digest'\n")
         and 'storage=destination_probe(' in trace and 'in sha\n' in trace, 'Not the exact prelauncher failure')
    shell = raw['original/full.sbatch']
    gate = '/usr/bin/python3 -B "$root/guard_full.py" --root "$root"'
    launch = 'bash "$root/ops/full.sbatch" "$root/ops" "$request" "$request_sha" "$ops_sha"'
    need('set -euo pipefail' in shell and shell.count(gate) == shell.count(launch) == 1
         and shell.index(gate) < shell.index(launch), 'Guard no longer precedes unconditional training launcher')
    return dict(schema='elmo-exact-prelauncher-retry-evidence-v1',
                status='EXACT_412823_PRELAUNCH_FAILURE_VERIFIED',
                run_id=RUN_ID, failed_parent=PARENT, capture=CAPTURE,
                original_request_sha256=REQUEST_SHA, original_files_manifest_sha256=FILES_SHA,
                source_revision=SOURCE, parameters=314227000,
                optimizer_updates_observed=None, optimizer_updates_possible_by_reviewed_control_flow=0,
                maintained_training_launcher_reached=False,
                zero_basis='Exact complete traceback at hash-bound guard destination_probe; shell set -e prevents subsequent launcher; actual host1/native FAILED1:0 and successful post-terminal complete-step query.',
                missing_csv_used_as_zero_evidence=False,
                checkpoint_recovery_authorized=False, live_launch_admitted=False,
                permitted_route='Fresh immutable full-v2 at step0 with same ID, unchanged science and resume=never; caller must revalidate current intent/claims/queues/steps/path freshness/storage/native allocation.',
                original_request=original, evidence_file_count=len(EVIDENCE_HASHES),
                parent_terminal_end_time=terminal['EndTime'], observer_terminal_end_time=done['EndTime'])


def verify_retry_request(original, current):
    """Only root's exact full-v2 relocation and new CSV namespace are allowed."""
    need(hashlib.sha256(json.dumps(original, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest() == ORIGINAL_CANONICAL_SHA, 'Unverified original request')
    need(original.get('run_id') == RUN_ID and original.get('source_revision') == SOURCE,
         'Wrong original request identity')
    def relocate(value):
        if isinstance(value, dict):
            return {k: relocate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [relocate(v) for v in value]
        return value.replace(OLD_ROOT, NEW_ROOT) if isinstance(value, str) else value
    expected = relocate(copy.deepcopy(original))
    expected['job_name'] = NAME[:-1] + '2'
    expected['local_logs'] = original['local_logs'] + '-attempt-v2'
    expected['overrides'] = [x.replace('logger.csv.save_dir=' + original['local_logs'],
                                     'logger.csv.save_dir=' + expected['local_logs'])
                             for x in expected['overrides']]
    need(current == expected, 'Prelaunch retry changed more than exact new operational/log namespaces')
    need(current['output'] == original['output'] and current['target_steps'] == 300000
         and current['parameters'] == 314227000 and current['continuation'] is None
         and not current.get('interruption') and 'logger.wandb.resume=never' in current['overrides'],
         'This is only a zero-update prelauncher retry, never checkpoint recovery')
    return dict(status='EXACT_FRESH0_FULL_V2_REQUEST_COMPATIBLE', live_launch_admitted=False)
