"""Exact pre-launch-only timeout retry. No native-parent terminal is invented.

Pure metadata readers: no remote calls, ML, writes or submission. The caller must
invoke fresh census at pre-submit, pre-release and allocated entry, and retain
its ordinary resource, native ownership, source/runtime/data/smoke/storage gates.
This route cannot admit interruption recovery or any later failed attempt.
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

PARENT, CAPTURE = '412825', '412826'
TASK = '/workspace/users/paphiwetsa3/tasks/elmo-qknorm-swiglu-rmsnorm-imagenet-20261026-v1'
OLD_ROOT, NEW_ROOT = TASK + '/full-v2', TASK + '/full-v3'
NAME = 'elmo-large314m-qknorm-swiglu-rmsnorm-full-v2'
NEW_NAME = NAME[:-1] + '3'
REQUEST_REL = 'ops/requests/' + NAME + '.json'
REQUEST_SHA = '7445a2c8d4e47c13495567ef4d84ced4c8dc4e1991c6e86f9e70cce01eeec327'
FILES_SHA = 'e9d1cf93624ab94709baf3ba7d36bbadf7adf4fc72aba4a333bda011cd248153'
CANONICAL_SHA = 'fd817a28f638a00208bbab8ee68f25637fe736364b1ca3d087defd4fc17aad59'
PRIOR_READER_SHA = 'bc278ad53d6701d69b2df0b2df37a1286d75b1abd9be5d6c726b1c35539254f0'
NATIVE_READER_SHA = 'f6df73a12f7d4401376fd26929ee2678d46c50ed52c48be95c7ca50f0a36f71f'
EVIDENCE_MANIFEST_SHA = '9d613c6e90788c7d55b3f6049f24fa16705928dac988ad6401277307bbdc0e09'
MAX_BYTES = 1 << 20
CLAIM_NAMES = ('PREPARED.json', 'PARENT_JOB_ID', 'CAPTURE_JOB_ID',
               'parent-sbatch.argv.json', 'parent-sbatch.stdout', 'parent-sbatch.stderr', 'parent-sbatch.exit',
               'capture-sbatch.argv.json', 'capture-sbatch.stdout', 'capture-sbatch.stderr', 'capture-sbatch.exit',
               'HELD_NATIVE.txt', 'RELEASE_HELD_NATIVE.txt', 'CAPTURE_PENDING_NATIVE.txt',
               'RELEASE_CAPTURE_PENDING_NATIVE.txt', 'RELEASE_ATTEMPT.json', 'RELEASE_ACCEPTED.json')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def unique(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def bounded(path):
    path = Path(path)
    need(path.is_absolute() and path.resolve(strict=True) == path, 'Physical absolute evidence path required')
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_BYTES, 'Unsafe or oversized evidence')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(MAX_BYTES + 1)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        need(len(raw) <= MAX_BYTES and identity(before) == identity(os.fstat(fd)) == identity(path.stat(follow_symlinks=False)), 'Evidence changed while reading')
        return raw
    finally:
        os.close(fd)


def instant(value):
    need(isinstance(value, str), 'Missing observation time')
    at = datetime.fromisoformat(value.replace('Z', '+00:00'))
    need(at.tzinfo is not None, 'Observation time must identify UTC offset')
    return at


def observation_interval(row):
    start = instant(row.get('start_utc', row.get('started_at_utc')))
    end = instant(row.get('end_utc', row.get('ended_at_utc')))
    need(start <= end <= datetime.now(timezone.utc), 'Observation interval is invalid or future')
    return start, end


def instant_nanoseconds(value):
    """Exact aware datetime-to-epoch conversion; no floating point."""
    need(isinstance(value,datetime) and value.tzinfo is not None,'Aware instant required')
    delta=value.astimezone(timezone.utc)-datetime(1970,1,1,tzinfo=timezone.utc)
    return (delta.days*86400+delta.seconds)*1000000000+delta.microseconds*1000


def gnu_stat_nanoseconds(value):
    """Accept only the exact GNU stat UTC nanosecond format in this evidence."""
    match=re.fullmatch(r'([0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2})\.([0-9]{9}) \+0000',value)
    need(match is not None,'Expected exact GNU stat UTC nanosecond timestamp')
    base=datetime.strptime(match[1],'%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    return instant_nanoseconds(base)+int(match[2])


def file_mtime(row):
    output = row['stdout_stderr_merged']
    lines = [line for line in output.splitlines() if line.startswith('__STAT__|')]
    need(len(lines) == 1, 'Missing exact observed stat')
    parts = lines[0].split('|')
    need(len(parts) == 5 and parts[3] == '41003', 'Observed file owner differs')
    at = gnu_stat_nanoseconds(parts[2])
    need(at <= instant_nanoseconds(observation_interval(row)[0]), 'Observed stat time differs')
    return at


def observed(row, path=None):
    observation_interval(row)
    need(row.get('exit_code') == 0 and row.get('output_truncated', False) is False and not row.get('timed_out', False), 'Failed or truncated actual observation')
    if 'stdout_stderr_merged' in row:
        output = row['stdout_stderr_merged']
        if path is not None:
            need(output.startswith('__PATH__|' + path + '\n') and output.count('__CONTENT__\n') == 1, 'Observed file provenance differs')
            return output.split('__CONTENT__\n', 1)[1]
        return output
    need(row.get('stderr', '') == '', 'Observation stderr is not empty')
    return row['stdout']


def expected_argv(capture=False):
    root = OLD_ROOT
    args = ['/usr/local/bin/sbatch', '--parsable']
    if not capture:
        args += ['--hold']
    args += ['--no-requeue', '--account=loaner', '--qos=loaner-normal', '--partition=rl2']
    if capture:
        args += ['--job-name=elmo-native-qknorm-swiglu-rmsnorm-full-v2', '--nodelist=gpu21',
                 '--nodes=1', '--ntasks=1', '--cpus-per-task=2', '--mem=1G', '--gres=none', '--time=00:10:00',
                 '--export=NONE', '--dependency=afterany:' + PARENT, '--comment=elmo-full-observer:' + REQUEST_SHA + ':' + PARENT, '--chdir=' + root,
                 '--output=' + root + '/logs/%x-%j.out', '--error=' + root + '/logs/%x-%j.err',
                 root + '/capture-native.sbatch', root, PARENT, REQUEST_SHA, FILES_SHA]
    else:
        args += ['--comment=elmo-full:' + REQUEST_SHA, '--job-name=' + NAME, '--nodelist=gpu21',
                 '--nodes=1', '--ntasks=1', '--cpus-per-task=32', '--mem=192G', '--gres=gpu:h100:4',
                 '--time=7-00:00:00', '--export=NONE', '--chdir=' + root,
                 '--output=' + root + '/logs/%x-%j.out', '--error=' + root + '/logs/%x-%j.err',
                 root + '/full.sbatch', root, REQUEST_SHA, FILES_SHA]
    return args


def _protocol(raw, native):
    """Semantic checks after exact evidence closure; separately adversarially tested."""
    claim = lambda name: raw['observed/claims/' + REQUEST_SHA + '/' + name].decode()
    for label, job in [('parent', PARENT), ('capture', CAPTURE)]:
        need(claim(label.upper() + '_JOB_ID').strip() == job and claim(label + '-sbatch.stdout').strip() == job
             and claim(label + '-sbatch.stderr') == '' and claim(label + '-sbatch.exit').strip() == '0', 'Accepted native identity/outcome differs')
        need(parse(claim(label + '-sbatch.argv.json')) == expected_argv(label == 'capture'), 'Accepted immutable argv differs')
    prepared, release = parse(claim('PREPARED.json')), parse(claim('RELEASE_ATTEMPT.json'))
    need(prepared['request_sha256'] == REQUEST_SHA and prepared['manifest_sha256'] == FILES_SHA
         and prepared['parent_argv'] == expected_argv(), 'Prepared immutable payload differs')
    need(release['request_sha256'] == REQUEST_SHA and release['parent'] == PARENT and release['capture'] == CAPTURE, 'Release binding differs')
    accepted = parse(claim('RELEASE_ACCEPTED.json'))
    need(accepted['parent'] == PARENT and accepted['capture'] == CAPTURE, 'Accepted release identities differ')
    held = [native.check_native(claim(x), job=PARENT, state='PENDING', request_sha=REQUEST_SHA)
            for x in ('HELD_NATIVE.txt', 'RELEASE_HELD_NATIVE.txt')]
    pending = [native.check_native(claim(x), job=CAPTURE, state='PENDING', capture=True,
                                   parent=PARENT, request_sha=REQUEST_SHA)
               for x in ('CAPTURE_PENDING_NATIVE.txt', 'RELEASE_CAPTURE_PENDING_NATIVE.txt')]
    pending += [native.check_native(observed(parse(raw['transport/startup-capture-pending.json'])),
                                   job=CAPTURE, state='PENDING', capture=True, parent=PARENT, request_sha=REQUEST_SHA)]
    need(all(x.get('Dependency') == 'afterany:' + PARENT + '(unfulfilled)' for x in pending), 'Observer was not bound to the exact unresolved afterany')
    running = native.check_native(observed(parse(raw['transport/parent-running.json'])), job=PARENT, state='RUNNING', request_sha=REQUEST_SHA)
    native.same_attempt(*held, running)
    observer = native.check_native(observed(parse(raw['transport/capture-running.json']), OLD_ROOT + '/capture-running-' + CAPTURE + '.txt'),
                                   job=CAPTURE, state='RUNNING', capture=True, parent=PARENT, request_sha=REQUEST_SHA)
    terminal = native.check_native(observed(parse(raw['transport/capture-terminal.json'])), job=CAPTURE, state='FAILED', capture=True, parent=PARENT, request_sha=REQUEST_SHA)
    native.same_attempt(*pending, observer)
    need(terminal['SubmitTime'] == observer['SubmitTime'] and terminal['StartTime'] == observer['StartTime']
         and terminal['NodeList'] == 'gpu21' and terminal['NumNodes'] == '1'
         and terminal['ExitCode'] == '1:0' and terminal['Reason'] == 'NonZeroExitCode'
         and native.native_time(terminal['EndTime']) >= native.native_time(observer['StartTime'])
         and {'cpu=2','mem=1G','node=1'} <= set(terminal['AllocTRES'].split(',')), 'Observer failed-terminal continuity/resources differ')
    need(observer.get('Dependency') == terminal.get('Dependency') == '(null)', 'Resolved observer dependency differs')
    parent_observation = parse(raw['transport/parent-running.json'])
    observer_observation = parse(raw['transport/capture-running.json'])
    terminal_observation = parse(raw['transport/capture-terminal.json'])
    host_observation = parse(raw['transport/host-exit.json'])
    need(native.native_time(running['StartTime']) <= observation_interval(parent_observation)[0]
         <= observation_interval(parent_observation)[1] <= native.native_time(observer['StartTime']), 'Parent running observation interval differs')
    need(instant_nanoseconds(native.native_time(running['StartTime'])) <= file_mtime(host_observation) <= instant_nanoseconds(native.native_time(observer['StartTime'])), 'Host failure/afterany interval differs')
    need(instant_nanoseconds(native.native_time(observer['StartTime'])) <= file_mtime(observer_observation) <= instant_nanoseconds(native.native_time(terminal['EndTime']))
         <= instant_nanoseconds(observation_interval(terminal_observation)[0]), 'Observer running/terminal observation interval differs')
    need(all(native.native_time(row['SubmitTime']) <= native.native_time(observer['StartTime']) for row in pending), 'Pending observer interval differs')
    trace_row = parse(raw['transport/traceback.json'])
    trace = observed(trace_row, OLD_ROOT + '/logs/' + NAME + '-' + PARENT + '.err')
    need(trace.count('Traceback (most recent call last):') == 1 and 'line 146, in allocated\n    verify_smoke(root,q,b)' in trace
         and 'line 79, in verify_smoke' in trace and "'--check-completed', '412821']' timed out after 60 seconds\n" in trace
         and 'subprocess.TimeoutExpired:' in trace, 'Not the exact pre-launch timeout location')
    need(observed(parse(raw['transport/host-exit.json']), OLD_ROOT + '/host-exit-' + PARENT + '.txt').strip() == '1', 'Actual host1 missing')
    audit = parse(raw['transport/controlflow.json'])
    need(audit['status'] == 'POSITIVE_PRELAUNCHER_FAILURE_CONTROLFLOW_VERIFIED_PARENT_NATIVE_TERMINAL_UNKNOWN'
         and audit['parent'] == PARENT and audit['observer'] == CAPTURE and audit['request_sha256'] == REQUEST_SHA
         and audit['sealed_manifest_sha256'] == FILES_SHA
         and audit['zero_work_assessment']['training_launcher_reached'] is False
         and audit['zero_work_assessment']['optimizer_updates_possible_through_this_sealed_allocation_entrypoint'] == 0,
         'Exact reviewed positive control-flow conclusion missing')
    shell = raw['original/full.sbatch'].decode()
    gate = '/usr/bin/python3 -B "$root/guard_full.py" --root "$root"'
    launch = 'bash "$root/ops/full.sbatch" "$root/ops" "$request" "$request_sha" "$ops_sha"'
    need('set -euo pipefail' in shell and shell.count(gate) == shell.count(launch) == 1 and shell.index(gate) < shell.index(launch), 'Sealed failed guard does not precede launcher')
    steps = parse(raw['transport/postcapture-steps.json'])
    need(instant(steps['start_utc']) >= native.native_time(terminal['EndTime']), 'Complete step observation predates observer termination')
    stdout = observed(steps)
    for job in ('412823', '412824', PARENT, CAPTURE):
        native.no_owned_steps(stdout, job)
    return terminal


def verify_prelaunch_retry(evidence_root):
    root = Path(evidence_root)
    need(root.is_absolute() and root.resolve(strict=True) == root, 'Physical absolute evidence root required')
    need(isinstance(EVIDENCE_MANIFEST_SHA, str), 'Actual accepted-claim evidence not bound')
    pinbytes = bounded(root / 'EVIDENCE.json')
    need(sha(pinbytes) == EVIDENCE_MANIFEST_SHA, 'Changed reviewed evidence closure')
    pins = parse(pinbytes)
    actual = set()
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs:
            need(not (Path(directory) / name).is_symlink(), 'Symlink evidence directory')
        actual.update(str((Path(directory) / name).relative_to(root)) for name in names)
    need(actual == set(pins) | {'EVIDENCE.json'}, 'Unexpected or missing evidence file')
    raw = {}
    for name, digest in pins.items():
        need(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe evidence path')
        raw[name] = bounded(root / name)
        need(sha(raw[name]) == digest, 'Changed actual evidence: ' + name)
    need(sha(raw['original/FILES.sha256']) == FILES_SHA and sha(raw['original/' + REQUEST_REL]) == REQUEST_SHA, 'Wrong failed request/wrapper')
    original = parse(raw['original/' + REQUEST_REL])
    need(canonical(original) == CANONICAL_SHA, 'Changed original science')
    manifest = dict((line.split('  ', 1)[1], line.split('  ', 1)[0]) for line in raw['original/FILES.sha256'].decode().splitlines())
    for name, digest in pins.items():
        if name.startswith('original/') and name != 'original/FILES.sha256':
            need(manifest.get(name[9:]) == digest, 'Reviewed control file is outside original seal')
    previous = reviewed_parser(raw['original/prelaunch_retry.py'], PRIOR_READER_SHA, '_unchanged_prior_prelaunch')
    prior = previous.verify_prelaunch_retry(root / 'previous-v1')
    previous.verify_retry_request(prior['original_request'], original)
    native = reviewed_parser(raw['original/native_guard.py'], NATIVE_READER_SHA, '_exact_original_native')
    receipt = parse(raw['transport/accepted-claims-readback.json'])
    need(receipt['all_before_after_stats_equal'] is True and receipt['all_remote_small_hashes_match_received'] is True
         and receipt['parent_job_id'] == PARENT and receipt['observer_job_id'] == CAPTURE, 'Actual readback provenance differs')
    need(set(receipt['received']) == {name[len('observed/'):] for name in raw if name.startswith('observed/')}, 'Received file closure differs')
    for name, item in receipt['received'].items():
        need(item['sha256'] == sha(raw['observed/'+name]) and item['bytes'] == len(raw['observed/'+name])
             and item['tar_uid'] == 41003, 'Received actual byte/owner identity differs')
        if 'original/'+name in raw:
            need(raw['observed/'+name] == raw['original/'+name], 'Current readback differs from executed original code')
    for rel,envelope in [('host-exit-'+PARENT+'.txt','host-exit.json'),
                         ('capture-running-'+CAPTURE+'.txt','capture-running.json'),
                         ('logs/'+NAME+'-'+PARENT+'.err','traceback.json')]:
        need(raw['observed/'+rel].decode() == observed(parse(raw['transport/'+envelope]),OLD_ROOT+'/'+rel), 'Independent readbacks disagree')
    need(raw['observed/request-claims/'+NAME+'/REQUEST_SHA256'] == (REQUEST_SHA+'\n').encode(), 'Accepted request-claim marker differs')
    need(raw['observed/native-'+PARENT+'.txt'] == b'' and raw['observed/native-'+PARENT+'.exit'].strip() == b'1'
         and b'Invalid job id' in raw['observed/native-'+PARENT+'.stderr'], 'Parent terminal must remain genuinely unknown')
    terminal = _protocol(raw, native)
    return dict(schema='elmo-exact-prelaunch-timeout-retry-v1', status='EXACT_TWO_PRELAUNCH_FAILURES_VERIFIED',
                failed_parent=PARENT, capture=CAPTURE, prior_failed_parent='412823', original_request=original,
                original_request_sha256=REQUEST_SHA, original_manifest_sha256=FILES_SHA,
                optimizer_updates_possible_by_reviewed_control_flow=0, maintained_training_launcher_reached=False,
                parent_native_state='UNKNOWN', parent_native_exit_code='UNKNOWN', parent_native_end_time='UNKNOWN',
                observer_native_state='FAILED', observer_native_exit_code='1:0', observer_end_time=terminal['EndTime'],
                genuine_parent_terminal_proof=False, checkpoint_recovery_authorized=False, live_launch_admitted=False)


def verify_retry_request(original, current):
    need(canonical(original) == CANONICAL_SHA, 'Unverified full-v2 request')
    def relocate(value):
        if isinstance(value, dict): return {k: relocate(v) for k, v in value.items()}
        if isinstance(value, list): return [relocate(v) for v in value]
        return value.replace(OLD_ROOT, NEW_ROOT) if isinstance(value, str) else value
    expected = relocate(copy.deepcopy(original))
    expected['job_name'] = NEW_NAME
    need(original['local_logs'].endswith('-attempt-v2'), 'Unexpected prior CSV namespace')
    expected['local_logs'] = original['local_logs'][:-len('-attempt-v2')] + '-attempt-v3'
    expected['overrides'] = [s.replace('logger.csv.save_dir=' + original['local_logs'], 'logger.csv.save_dir=' + expected['local_logs']) for s in expected['overrides']]
    need(current == expected and current['output'] == original['output'] and current['target_steps'] == 300000
         and current['continuation'] is None and not current.get('interruption')
         and 'logger.wandb.resume=never' in current['overrides'], 'Retry changed science or exceeded exact fresh-zero route')
    return dict(status='EXACT_FRESH0_FULL_V3_REQUEST_COMPATIBLE', live_launch_admitted=False)


PRIOR_ATTEMPTS = (
    dict(root=TASK+'/full-v1', name=NAME[:-1]+'1', parent='412823', capture='412824',
         request_sha='39d1059efc6292fcaf0a8c2b6b438bb84496ba947ee4b198b70cf3ad28108d37',
         files_sha='c1d356ba049931c479eaf886873a3ca6168611baf67e07f69898ecd3c1439ffb'),
    dict(root=OLD_ROOT, name=NAME, parent=PARENT, capture=CAPTURE,
         request_sha=REQUEST_SHA, files_sha=FILES_SHA),
)
CLAIM_ROOTS = tuple(root+'/'+sub for root in (TASK+'/full-v1',OLD_ROOT,NEW_ROOT)
                    for sub in ('claims','request-claims','ops/claims','ops/request-claims'))


def verify_claim_entries(full_roots, checked_roots, entries, *, current_request_sha=None):
    """Check caller's successful physical, complete direct-child listings.

Caller must additionally reopen both old manifests/requests/accepted claim files
against the verified evidence and check output/CSV absence. This pure comparison
cannot turn caller-provided strings into an actual fresh filesystem observation.
"""
    need(len(full_roots)==3 and set(full_roots)=={TASK+'/full-v1',OLD_ROOT,NEW_ROOT}, 'Unknown or missing full attempt root')
    need(len(checked_roots)==len(CLAIM_ROOTS) and set(checked_roots)==set(CLAIM_ROOTS), 'Incomplete claim roots')
    expected = set()
    for attempt in PRIOR_ATTEMPTS:
        expected |= {attempt['root']+'/claims/'+attempt['request_sha'], attempt['root']+'/request-claims/'+attempt['name']}
    if current_request_sha is not None:
        need(re.fullmatch(r'[0-9a-f]{64}',current_request_sha) is not None, 'Missing current immutable request hash')
        expected |= {NEW_ROOT+'/claims/'+current_request_sha, NEW_ROOT+'/request-claims/'+NEW_NAME}
    need(len(entries)==len(expected) and set(entries)==expected, 'Unknown, partial, or missing wrapper/maintained claims')
    return sorted(expected)


def verify_queue_steps(queue, steps, *, current_job=None, current_capture=None):
    """Validate complete successful command output, supplied by live caller.

Only the caller's separately native-authenticated full-v3 parent/observer may be
excluded. All earlier parent/observer IDs and dotted/array aliases are rejected.
No query success, freshness, allocation, or storage PASS is claimed here.
"""
    need(isinstance(queue,str) and isinstance(steps,str) and max(len(queue.encode()),len(steps.encode()))<=MAX_BYTES, 'Oversized native census')
    prior = {'412823','412824',PARENT,CAPTURE}
    allowed = set()
    if current_job is not None or current_capture is not None:
        need(isinstance(current_job,str) and isinstance(current_capture,str)
             and current_job.isdecimal() and current_capture.isdecimal()
             and len(prior|{current_job,current_capture})==6, 'Wrong current accepted IDs')
        allowed = {current_job,current_capture}
    for line in queue.splitlines():
        row=line.split('|');need(len(row)==5 and row[2]=='paphiwetsa3','Malformed complete owned queue')
        job,name,_,state,_=row
        need(not any(job==old or job.startswith(old+'.') or job.startswith(old+'_') for old in prior), 'Prior allocation remains queued')
        if 'qknorm-swiglu-rmsnorm' in name or job in allowed:
            need(job in allowed and name==(NEW_NAME if job==current_job else 'elmo-native-qknorm-swiglu-rmsnorm-full-v3')
                 and state in {'PENDING','RUNNING'}, 'Another or incorrectly excluded campaign writer')
    for line in steps.splitlines():
        if line=='No job steps exist in the system.':continue
        match=re.search(r'(?:^|\s)StepId=(\S+)',line);need(match is not None,'Malformed complete native step census')
        need(not any(match[1]==old or match[1].startswith(old+'.') or match[1].startswith(old+'_') for old in prior), 'Prior worker or array-alias step remains')
    return dict(prior_jobs_absent_from_supplied_complete_native_outputs=True, live_launch_admitted=False)


def expected_live_file_hashes(evidence_root):
    """Required current readback, including every preserved accepted-claim field.

The caller reads these exact physical remote files anew before claim/release;
this function only returns expectations after validating the complete evidence.
"""
    verify_prelaunch_retry(evidence_root)
    pins = parse(bounded(Path(evidence_root)/'EVIDENCE.json'))
    expected = {OLD_ROOT+'/FILES.sha256':FILES_SHA, OLD_ROOT+'/'+REQUEST_REL:REQUEST_SHA}
    for attempt in PRIOR_ATTEMPTS:
        expected[attempt['root']+'/request-claims/'+attempt['name']+'/REQUEST_SHA256'] = sha((attempt['request_sha']+'\n').encode())
    for relative,digest in pins.items():
        if relative.startswith('observed/claims/'):
            expected[OLD_ROOT+'/'+relative[len('observed/'):]] = digest
        elif relative.startswith('previous-v1/observed/claims/'):
            expected[TASK+'/full-v1/'+relative[len('previous-v1/observed/'):]] = digest
        elif relative in {'previous-v1/original/FILES.sha256',
                          'previous-v1/original/ops/requests/elmo-large314m-qknorm-swiglu-rmsnorm-full-v1.json'}:
            expected[TASK+'/full-v1/'+relative[len('previous-v1/original/'):]] = digest
    return expected


def verify_live_file_hashes(evidence_root, observed_hashes):
    need(observed_hashes == expected_live_file_hashes(evidence_root), 'Current prior request/manifest/accepted claim bytes changed or missing')
    return dict(exact_prior_file_hash_comparison=True, live_launch_admitted=False)


verify = verify_prelaunch_retry
verify_request = verify_retry_request
