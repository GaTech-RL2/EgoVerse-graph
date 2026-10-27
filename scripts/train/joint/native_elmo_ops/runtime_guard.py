"""Allocated host/container authentication and fixed maintained dispatch.

Capture is evidence-only and stays stdlib-only. Every scientific/inspection
phase enters the pinned runtime and exact operational service before dispatch.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
from wrapper_common import MODES, bound, command, fields, need, physical, read_json, service, sha, write_once, now
from host_guard import admit_before_submission

LOGICAL = Path('/run/elmo-slurm-client')


def attempt_path(spec, job):
    return Path(spec['root']) / 'status' / (spec['mode'] + '-' + job + '.attempt')


def prerequisite_path(spec, q):
    p = Path(spec['runtime']['allocated_prerequisites_path'])
    need(p.parent == Path(spec['root']) / 'status' and p.suffix == '.json',
         'Fresh prerequisite path must be prechosen under bound ROOT/status')
    if spec['mode'] in {'inspect', 'diagnostic', 'finalize'}:
        need(q['allocated_prerequisites_path'] == str(p), 'Producer prerequisite path differs')
    return p


def native_auth(spec, path, q, job):
    native = service(spec, 'native_interruption')
    expected = dict(spec['native_contract'], JobId=job)
    native.process_identity(expected, cpu=spec['mode'] not in {'diagnostic', 'full'})
    raw = command(['/usr/local/bin/scontrol', 'show', 'job', job, '-o', '-dd'])
    f = native.native_identity(raw, expected, 'RUNNING')
    queue = command(['/usr/local/bin/squeue', '--noheader', '--jobs', job, '--format=%i|%j|%T|%u'])
    rows = [x.strip().split('|') for x in queue.splitlines() if x.strip()]
    owner = re.fullmatch(r'([^()]+)\((\d+)\)', expected['UserId'])
    need(rows == [[job, expected['JobName'], 'RUNNING', owner[1]]], 'Native squeue authentication/identity differs')
    return dict(native=raw, own_queue=queue, job_id=job, request_sha256=sha(path),
                node=f['NodeList'], checked_at=now())


def capsule(spec, job):
    staged = read_json(attempt_path(spec, job) / 'native-client-stage.json')
    need(staged['schema'] == 'native-elmo-client-stage-v1' and staged['job_id'] == job
         and staged['parent_request'] == spec['parent_request']
         and staged['logical_container_root'] == str(LOGICAL)
         and sha(physical(LOGICAL / 'FILES.sha256')) == staged['manifest_sha256'],
         'Native capsule is not the exact allocated-host closure')
    found = set()
    for line in (LOGICAL / 'FILES.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1); relative = Path(name)
        need(not relative.is_absolute() and '..' not in relative.parts and name not in found,
             'Native capsule manifest escapes or duplicates members')
        found.add(name)
        need(sha(physical(LOGICAL / relative)) == digest == staged['members'][name]['sha256'],
             'Copied native capsule member changed')
    need(found == set(staged['members']), 'Native capsule closure incomplete')
    for name in ('scontrol', 'squeue'):
        need(sha('/usr/local/bin/' + name) == sha(Path(spec['root']) / ('native_' + name + '.sh')),
             'Container native executable is not the isolated reviewed wrapper')
    socket = Path(spec['runtime']['native_client']['munge_socket'])
    need(socket.is_socket() and socket.parent == Path('/run/munge'), 'Munge socket mount absent')
    return staged['manifest_sha256']


def env_exports(spec, q, job, path):
    parent = bound(spec['parent_request']); root = Path(spec['root']); runtime = parent['lambda_runtime']
    values = dict(ELMO_REQUEST=str(path), ELMO_PARENT_REQUEST=spec['parent_request']['path'],
        ELMO_ATTEMPT=str(attempt_path(spec, job)), ELMO_OUTPUT=q['output'],
        ELMO_SOURCE=parent['source'], ELMO_MODE=spec['mode'],
        ELMO_CPUS=spec['native_contract']['NumCPUs'], ELMO_MEMORY=spec['native_contract']['MinMemoryNode'],
        ELMO_CONTAINER_IMAGE=runtime['container_image'], ELMO_SCRATCH=runtime['scratch_root'],
        ELMO_DATA_ROOT=str(Path(parent['data_manifest']).parent), ELMO_CACHE_ROOT=parent['data_cache']['root'],
        ELMO_ARCHIVE=parent['source_archive'], ELMO_ARCHIVE_SHA=parent['source_archive_sha256'],
        ELMO_SOURCE_MANIFEST=parent['source_manifest'], ELMO_SOURCE_MANIFEST_SHA=parent['source_manifest_sha256'],
        ELMO_SERVICE_BOOTSTRAP=str(Path(spec['operational_service']['root']) / 'scripts/train/joint/service_bootstrap.py'))
    if spec['mode'] in {'diagnostic', 'full'}:
        local_logs = q.get('local_logs') if spec['mode'] == 'full' else str(Path(runtime['scratch_root']) / 'recovery-logs' / job)
        need(local_logs and Path(local_logs).is_absolute() and not Path(local_logs).exists(), 'Fresh local log namespace required')
        values['ELMO_LOCAL_LOGS'] = local_logs
    return '\n'.join('export ' + key + '=' + shlex.quote(str(value)) for key, value in values.items())


def verify_scientific_prerequisites(spec, q, parent, job):
    """Called only after both authenticated host and container native queries."""
    from verify_joint_data_cache import verify_cache
    from lambda_runtime import exports
    need(spec['mode'] != 'capture', 'Capture cannot certify model/cache prerequisites')
    exports(parent, os.environ, inside=True)
    cache = verify_cache(parent, os.environ['SLURMD_NODENAME'])
    staged = read_json(attempt_path(spec, job) / 'source-stage.json')
    need(staged['status'] == 'PASS' and str(staged['job_id']) == job
         and staged['node'] == spec['native_contract']['NodeList']
         and staged['archive_sha256'] == parent['source_archive_sha256']
         and staged['manifest_sha256'] == parent['source_manifest_sha256']
         and staged['all_manifest_files_verified'] is True, 'Actual source staging receipt differs')
    need(sha(parent['source_manifest']) == parent['source_manifest_sha256'], 'Source manifest changed')
    seen = set()
    for line in Path(parent['source_manifest']).read_text().splitlines():
        digest, name = line.split('  ', 1); rel = Path(name)
        need(not rel.is_absolute() and '..' not in rel.parts and name not in seen, 'Unsafe/duplicate frozen source member')
        seen.add(name)
        need(sha(physical(Path(parent['source']) / rel)) == digest, 'Frozen source changed in container')
    need(seen, 'Empty source manifest')
    return {'cache': cache, 'source_stage': staged, 'container_source_files_verified': len(seen),
            'runtime_mounts_sha256': parent['lambda_runtime']['mounts']['sha256']}


def storage_probe(spec, q, parent, job, request_sha):
    output = Path(q['output']); physical(output.parent)
    need(not output.exists() and not output.is_symlink(), 'Preserve existing output')
    reserved = parent['reserved_checkpoint_bytes']; free = shutil.disk_usage(output.parent).free
    need(free > reserved >= 80_000_000_000, 'Actual persistent storage reserve unavailable')
    probe = output.parent / ('.elmo-recovery-fsync-' + job + '-' + request_sha[:16] + '.json')
    value = dict(job_id=job, request_sha256=request_sha, output=str(output))
    write_once(probe, value)
    need(read_json(probe) == value, 'Exclusive persistent write/fsync/readback failed')
    return dict(probe=str(probe), probe_sha256=sha(probe), output=str(output),
                available_bytes=free, reserved_bytes=reserved, actual_write_fsync_readback=True)


def dispatch_argv(spec, request_path):
    mode = spec['mode']; need(mode in MODES, 'Unknown phase')
    root = Path(spec['root']); service_boot = Path(spec['operational_service']['root']) / 'scripts/train/joint/service_bootstrap.py'
    if mode == 'capture':
        # Evidence-only exception approved by root: avoid Torch in 2CPU/1GiB capture.
        return [sys.executable, str(service_boot), str(request_path), '--', '-m',
                'scripts.train.joint.native_interruption', str(request_path)]
    return [os.environ['JOINT_BOOTSTRAP_PYTHON'], str(root / 'runtime_bootstrap.py'),
            str(service_boot), str(request_path), '--', str(root / 'dispatch_after_runtime.py'),
            str(request_path)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('host', 'inside', 'dispatch'))
    parser.add_argument('spec'); parser.add_argument('spec_sha'); parser.add_argument('mode', choices=MODES)
    parser.add_argument('manifest_sha'); args = parser.parse_args(argv)
    from execution_spec import load_execution, record_running_admission
    # Authenticate capsule bytes before calling any container native wrapper.
    if args.stage != 'host':
        spec_early = bound({'path': args.spec, 'sha256': args.spec_sha})
        capsule(spec_early, os.environ['SLURM_JOB_ID'])
    spec, path, q, job, claim = load_execution(args.spec, args.spec_sha, args.mode, args.manifest_sha)
    parent = bound(spec['parent_request']); attempt = attempt_path(spec, job)
    auth = native_auth(spec, path, q, job)
    # The reviewed entrypoint, rather than request metadata, sets this boundary.
    os.environ['ELMO_HISTORY_LOCATION'] = {'host':'ALLOCATED_HOST', 'inside':'CONTAINER_GATE',
                                          'dispatch':'CONTAINER_DISPATCH'}[args.stage]
    os.environ['ELMO_HISTORY_ATTEMPT'] = str(attempt)
    gate = admit_before_submission(spec, q, job, sha(path), running=True)
    prerequisite_path(spec, q)
    if args.stage == 'host':
        attempt.mkdir()
        running = record_running_admission(args.spec, args.spec_sha, args.mode, args.manifest_sha)
        write_once(attempt / 'host-native.json', dict(auth, running_admission=running, live_admission=gate))
    elif args.stage == 'inside':
        host = read_json(attempt / 'host-native.json')
        need(host['request_sha256'] == sha(path) and host['job_id'] == job, 'Actual host authentication missing')
        record = dict(auth, schema='native-elmo-container-auth-v1', status='CONTAINER_NATIVE_AUTHENTICATED',
            parent_request=spec['parent_request'], original_parent_job_id=str(spec['original_parent_job_id']),
            native_client_manifest_sha256=capsule(spec, job), host_native_sha256=sha(attempt / 'host-native.json'))
        write_once(attempt / 'container-native.json', record)
        if args.mode != 'capture':
            actual = verify_scientific_prerequisites(spec, q, parent, job)
            actual['storage'] = storage_probe(spec, q, parent, job, sha(path))
            write_once(attempt / 'pre-torch-prerequisites.json', dict(actual, request_sha256=sha(path), job_id=job))
    else:
        need((attempt / 'container-native.json').is_file(), 'Container native authentication not performed')
        if args.mode != 'capture':
            need((attempt / 'pre-torch-prerequisites.json').is_file(), 'Pre-Torch source/cache/storage gate missing')
        os.environ.update(ELMO_EXECUTION_SPEC=args.spec, ELMO_EXECUTION_SPEC_SHA256=args.spec_sha,
                          ELMO_OPS_MANIFEST_SHA256=args.manifest_sha, ELMO_EXECUTION_MODE=args.mode)
        os.execv(dispatch_argv(spec, path)[0], dispatch_argv(spec, path))
    print(env_exports(spec, q, job, path))


if __name__ == '__main__':
    main()
