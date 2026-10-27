"""Exact four-attempt prelauncher history, not a producer terminal certificate.

The historical readers are service-owned, pure metadata parsers.  A fresh census
reopens only the four old namespaces; current full-v5 scientific output, CSV,
status and checkpoints are deliberately outside its absence checks.  The
original 412831 scientific RUNNING/failure/observer/checkpoint gates are separate.
No query result, boolean, or historical zero-entry proof grants recovery here.
"""
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import socket
import stat

from scripts.train.joint.qk_recovery_profile import (
    RUN_ID, TASK_ROOT, ROOT, REQUEST_PATH, REQUEST_SHA, FILES_SHA, validate_parent,
)
from scripts.train.joint.qk_prelaunch_history_readers import prelaunch_held_retry as proof

SCHEMA = 'qk-four-prelauncher-history-binding-v1'
EVIDENCE_PATH = ROOT + '/proofs/prelaunch-held-v1'
EVIDENCE_SHA = '15ee3a5bb152468d4616c90b9d5756a86926a799b726e877f6fdcc231cc8d7d4'
MAX_LEAF = 2 << 20
MAX_MEMBERS = 280  # 279 actual leaves plus their exact EVIDENCE.json manifest.
MAX_TOTAL = 12 << 20
UID = 41003
CSV_PARENT = '/ephemeral/enroot/paphiwetsa3/elmo-qknorm-swiglu-rmsnorm-imagenet-20261026-v1/logs'
OLD = proof.PRIOR_ATTEMPTS
OLD_IDS = frozenset(a[k] for a in OLD for k in ('parent', 'capture'))
OLD_CLAIMS = {a['root'] + '/claims/' + a['request_sha']: a for a in OLD}
OLD_ROOTS = frozenset(a['root'] for a in OLD)
_CENSUS_TOKEN = object()


class _VerifiedCensus(dict):
    """Transient in-process evidence; re-parsed JSON cannot become admission."""
    def __init__(self, token, **value):
        need(token is _CENSUS_TOKEN, 'Only the live census can construct historical admission')
        super().__init__(value)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_metadata(path, *, owned=False, limit=MAX_LEAF):
    """Physical regular metadata; stat/read/stat equality, before any hashing."""
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts and path.resolve(strict=True) == path,
         'Physical metadata path required')
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and before.st_size <= limit
             and (not owned or before.st_uid == UID), 'Unsafe/oversized/unowned historical metadata')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(limit + 1)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        need(len(raw) <= limit and identity(before) == identity(os.fstat(fd))
             == identity(path.stat(follow_symlinks=False)), 'Historical metadata changed during read')
        return raw
    finally:
        os.close(fd)


def children(path):
    path = Path(path)
    need(path.is_absolute() and path.resolve(strict=True) == path and path.is_dir()
         and path.stat().st_uid == UID, 'Physical owned historical directory required')
    rows = list(path.iterdir())
    need(len(rows) <= 64, 'Excessive historical directory inventory')
    for row in rows:
        need(not row.is_symlink() and row.resolve(strict=True) == row
             and row.stat().st_uid == UID, 'Unowned/aliased historical directory entry')
    return rows


def absent(path):
    # lexists includes broken symlinks; absence cannot hide an alias.
    need(not os.path.lexists(path), 'Prior prelauncher namespace appeared: ' + str(path))


def full_attempt_roots():
    root = Path(TASK_ROOT)
    need(root.resolve(strict=True) == root and root.is_dir() and root.stat().st_uid == UID,
         'Physical owned campaign root required')
    # Other ImageNet/campaign artifacts are outside this exact history scope.
    matches = [p for p in root.iterdir() if p.name.startswith('full-v')]
    need(len(matches) <= 8, 'Unexpected full-attempt directory count')
    for p in matches:
        need(p.resolve(strict=True) == p and p.is_dir() and p.stat().st_uid == UID,
             'Aliased/unowned full-attempt directory')
    return [str(p) for p in matches]


def expected_binding():
    return dict(schema=SCHEMA,
        parent_request=dict(path=REQUEST_PATH, sha256=REQUEST_SHA),
        parent_files=dict(path=ROOT + '/FILES.sha256', sha256=FILES_SHA),
        evidence=dict(path=EVIDENCE_PATH, sha256=EVIDENCE_SHA))


def verify_history(evidence_root):
    """Authenticate the actual retained chain, with no live/native authority."""
    root = Path(evidence_root)
    manifest = read_metadata(root / 'EVIDENCE.json')
    need(sha(manifest) == EVIDENCE_SHA, 'Unreviewed four-attempt historical evidence')
    members = proof.parse(manifest)
    need(len(members) + 1 == MAX_MEMBERS, 'Incomplete historical evidence closure')
    total = len(manifest)
    for name in members:
        p = Path(name)
        need(not p.is_absolute() and '..' not in p.parts, 'Escaping historical evidence member')
        raw = read_metadata(root / p)
        total += len(raw)
        need(total <= MAX_TOTAL and sha(raw) == members[name], 'Historical closure changed or exceeds bounds')
    actual = set()
    for directory, dirs, names in os.walk(root, followlinks=False):
        need(all(not (Path(directory) / d).is_symlink() for d in dirs), 'Historical symlink directory')
        actual.update(str((Path(directory) / n).relative_to(root)) for n in names)
        need(len(actual) <= MAX_MEMBERS, 'Unexpected historical evidence members')
    need(actual == set(members) | {'EVIDENCE.json'}, 'Historical evidence member union differs')
    verified = proof.verify(root)
    need(verified['earlier_412825_parent_terminal'] == 'UNKNOWN_NOT_INFERRED'
         and verified['optimizer_updates_possible_by_reviewed_control_flow'] == 0
         and verified['live_launch_admitted'] is False and verified['checkpoint_recovery_authorized'] is False,
         'Historical proof scope widened')
    return verified


def disposition(attempt):
    """Typed history: UNKNOWN is intentionally not a terminal certificate."""
    job = attempt['parent']
    state, code, terminal = {
        '412823': ('FAILED', '1:0', True),
        '412825': ('UNKNOWN', None, False),
        '412827': ('FAILED', '1:0', True),
        '412829': ('CANCELLED', '0:0', True),
    }[job]
    return dict(job_id=job, observer_job_id=attempt['capture'],
        provenance='REVIEWED_PRELAUNCH_NO_TRAINING_ENTRY',
        parent_native_state=state, parent_native_exit_code=code,
        genuine_parent_terminal_proof=terminal,
        maintained_training_launcher_reached=False,
        optimizer_updates_possible_by_reviewed_control_flow=0,
        checkpoint_recovery_authorized=False, phase_completion_proven=False,
        historical_evidence_manifest_sha256=EVIDENCE_SHA)


def claim_disposition(census, claim, run_id):
    """Only a transient census made in this admission may discharge old claims."""
    if run_id != RUN_ID or str(claim) not in OLD_CLAIMS:
        return None
    need(isinstance(census, _VerifiedCensus) and census['schema'] == 'qk-four-prelauncher-fresh-census-v1'
         and census['historical_evidence_manifest_sha256'] == EVIDENCE_SHA,
         'Fresh complete historical census required')
    at = proof.instant(census['checked_at'])
    age = (datetime.now(timezone.utc) - at).total_seconds()
    need(0 <= age <= 300 and set(census['claims']) == set(OLD_CLAIMS)
         and census['claims'][str(claim)] == disposition(OLD_CLAIMS[str(claim)]),
         'Historical census is stale or its claim context changed')
    return disposition(OLD_CLAIMS[str(claim)])


def accepted_v1_files(raw, attempt):
    """Fresh semantic check of acceptance fields not present in old hash capture."""
    prefix = attempt['root'] + '/claims/' + attempt['request_sha'] + '/'
    get = lambda name: raw[prefix + name]
    prepared = proof.parse(get('PREPARED.json'))
    parent = proof.parse(get('parent-sbatch.argv.json'))
    need(parent == prepared['parent_argv'] and parent[-4:] == [attempt['root'] + '/full.sbatch',
         attempt['root'], attempt['request_sha'], attempt['files_sha']], 'Original parent acceptance argv changed')
    observer = proof.parse(get('capture-sbatch.argv.json'))
    expected = ['/usr/local/bin/sbatch', '--parsable', '--no-requeue', '--account=loaner',
        '--qos=loaner-normal', '--partition=rl2', '--job-name=elmo-native-qknorm-swiglu-rmsnorm-full-v1',
        '--nodelist=gpu21', '--nodes=1', '--ntasks=1', '--cpus-per-task=2', '--mem=1G', '--gres=none',
        '--time=00:10:00', '--export=NONE', '--dependency=afterany:' + attempt['parent'],
        '--comment=elmo-full-observer:' + attempt['request_sha'] + ':' + attempt['parent'],
        '--chdir=' + attempt['root'], '--output=' + attempt['root'] + '/logs/%x-%j.out',
        '--error=' + attempt['root'] + '/logs/%x-%j.err', attempt['root'] + '/capture-native.sbatch',
        attempt['root'], attempt['parent'], attempt['request_sha'], attempt['files_sha']]
    need(observer == expected, 'Original observer acceptance argv changed')
    for label, job in [('parent', attempt['parent']), ('capture', attempt['capture'])]:
        need(get(label + '-sbatch.stdout').strip() == job.encode()
             and get(label + '-sbatch.stderr') == b'' and get(label + '-sbatch.exit').strip() == b'0',
             'Original acceptance outcome changed')
    released = proof.parse(get('RELEASE_ACCEPTED.json'))
    need(released['parent'] == attempt['parent'] and released['capture'] == attempt['capture']
         and released['stdout'] == '' and proof.instant(released['released_at_utc']) <= datetime.now(timezone.utc),
         'Original release outcome changed')


def check_live_native(queue, steps, scope, current_job, current_state):
    """Complete owner outputs; reject old aliases and every other scoped writer."""
    need(len(queue.encode()) <= 1 << 20 and len(steps.encode()) <= 1 << 20,
         'Oversized native history census')
    from scripts.train.joint.original_observer import strict_step_rows
    from scripts.train.joint.phase_observer_proof import check_owned_steps
    strict_step_rows(steps)
    for line in queue.splitlines():
        if not line.strip():
            continue
        parts = [v.strip() for v in line.split('|')]
        need(len(parts) == 4, 'Malformed complete owner queue')
        job, name, state, comment = parts
        need(not any(job == old or job.startswith(old + '.') or job.startswith(old + '_')
                     or job.startswith(old + '+') for old in OLD_IDS), 'Prior accepted job remains in queue')
        scoped = (name in scope['scientific_job_names']
            or any(name.startswith(p) for p in scope['scientific_name_prefixes']) or comment == RUN_ID)
        if scoped:
            need(job == current_job and state == current_state, 'Another scientific writer appeared')
    for job in OLD_IDS:
        check_owned_steps(steps, job)


def csv_parent_identity():
    """The same host directory must be visible inside the exact bind mount."""
    path = Path(CSV_PARENT)
    need(path.resolve(strict=True) == path and path.is_dir(), 'Actual physical CSV parent is unavailable')
    st = path.stat()
    need(st.st_uid == UID, 'CSV parent owner differs')
    return dict(path=CSV_PARENT, device=st.st_dev, inode=st.st_ino, uid=st.st_uid)


def verify_host_csv_context(native_fields, job, csv_parent, location):
    """Reopen the existing host-native record; no new proof schema or writer."""
    need(location in {'CONTAINER_GATE', 'CONTAINER_DISPATCH'}, 'Unknown CSV observation boundary')
    attempt = Path(os.environ.get('ELMO_HISTORY_ATTEMPT', ''))
    need(attempt.is_absolute() and attempt.parent.name == 'status'
         and re.fullmatch(r'(?:capture|inspect|diagnostic|finalize|full)-' + job + r'\.attempt', attempt.name)
         and str(attempt).startswith(TASK_ROOT + '/')
         and Path(native_fields['Command']).parent == attempt.parent.parent,
         'Host CSV proof is outside the actual native wrapper/attempt')
    host_path = attempt / 'host-native.json'
    raw = read_metadata(host_path, owned=True, limit=4 << 20)
    host = proof.parse(raw)
    from scripts.train.joint.native_interruption import fields
    hf = fields(host['native'])
    immutable = ('JobId', 'UserId', 'JobName', 'SubmitTime', 'StartTime', 'Command', 'Comment',
                 'NodeList', 'NumNodes', 'NumCPUs', 'ReqTRES', 'AllocTRES', 'TimeLimit', 'WorkDir')
    need(host['job_id'] == job and all(hf.get(k) == native_fields.get(k) for k in immutable)
         and hf.get('JobState') == 'RUNNING', 'Host CSV record belongs to another actual allocation')
    prior = host['live_admission']['quiescence']['prelaunch_history']['node_context']
    need(prior['job_id'] == job and prior['location'] == 'ALLOCATED_HOST'
         and prior['csv_parent'] == csv_parent, 'Container masks or relocates the host CSV directory')
    if location == 'CONTAINER_DISPATCH':
        container_path = attempt / 'container-native.json'
        container_raw = read_metadata(container_path, owned=True, limit=4 << 20)
        container = proof.parse(container_raw)
        need(container['schema'] == 'native-elmo-container-auth-v1'
             and container['job_id'] == job and container['host_native_sha256'] == sha(raw)
             and container['request_sha256'] == host['request_sha256'],
             'Final container authentication does not bind actual host CSV context')
    return dict(path=str(host_path), sha256=sha(raw))


def node_local_context(query, *, required):
    """Do not interpret gateway /ephemeral absence as gpu21 evidence."""
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        need(not required, 'Allocated gpu21 node-local history check required')
        return None
    need(job.isdecimal() and os.environ.get('SLURMD_NODENAME') == 'gpu21' and os.getuid() == UID,
         'Wrong actual historical CSV allocation namespace')
    from scripts.train.joint.native_interruption import fields
    from scripts.train.joint.original_observer import query_record
    argv = ['/usr/local/bin/scontrol', 'show', 'job', job, '-o', '-dd']
    record = query(argv); query_record(record, argv)
    need(record['returncode'] == 0 and not record['stderr'].strip(), 'Actual node ownership query failed')
    f = fields(record['stdout'])
    need(f.get('JobId') == job and f.get('JobState') == 'RUNNING'
         and f.get('UserId') == 'paphiwetsa3(41003)' and f.get('NodeList') == 'gpu21'
         and f.get('NumNodes') == '1', 'Actual allocation is not exclusively on gpu21')
    node_argv = ['/usr/local/bin/scontrol', 'show', 'node', 'gpu21', '-o']
    node = query(node_argv); query_record(node, node_argv)
    need(node['returncode'] == 0 and not node['stderr'].strip(), 'Actual physical node query failed')
    nf = fields(node['stdout'])
    need(nf.get('NodeName') == 'gpu21' and nf.get('NodeHostName') == socket.gethostname(),
         'Physical host differs from actual gpu21 node')
    location = os.environ.get('ELMO_HISTORY_LOCATION')
    need(location in {'ALLOCATED_HOST', 'CONTAINER_GATE', 'CONTAINER_DISPATCH'},
         'Reviewed host/container CSV observation boundary required')
    csv_parent = csv_parent_identity()
    host_binding = None if location == 'ALLOCATED_HOST' else verify_host_csv_context(f, job, csv_parent, location)
    return dict(job_id=job, native=record, physical_node=node, location=location,
                csv_parent=csv_parent, host_context=host_binding)


def census(scope, run_id, *, current_job=None, current_state='RUNNING', require_node=False):
    """Reopened once per admission; compact evidence, no recursive claim copies."""
    if run_id != RUN_ID:
        need('qk_prelaunch_history' not in scope, 'QK history attached to another scientific identity')
        return None
    need(scope.get('qk_prelaunch_history') == expected_binding(), 'Exact QK history binding is mandatory')
    need(scope['user'] == 'paphiwetsa3', 'Wrong complete owner census')
    root_items = {item['path']: item for item in scope['claim_roots']}
    mandatory = {a['root'] + '/claims' for a in OLD} | {ROOT + '/claims'}
    need(len(root_items) == len(scope['claim_roots']) and mandatory <= set(root_items),
         'Four historical claims or original production claim omitted')
    need(all(root_items[a['root'] + '/claims']['terminal_directory'] == a['root'] for a in OLD),
         'Historical terminal directory moved')
    parent_raw = read_metadata(REQUEST_PATH, owned=True)
    parent = proof.parse(parent_raw)
    need(sha(parent_raw) == REQUEST_SHA
         and sha(read_metadata(ROOT + '/FILES.sha256', owned=True)) == FILES_SHA,
         'Original full-v5 source/request closure changed')
    validate_parent(parent)
    current_marker = ROOT + '/request-claims/' + parent['job_name']
    need({str(p) for p in children(ROOT + '/claims')} == {ROOT + '/claims/' + REQUEST_SHA}
         and {str(p) for p in children(ROOT + '/request-claims')} == {current_marker}
         and {p.name for p in children(current_marker)} == {'REQUEST_SHA256'}
         and read_metadata(current_marker + '/REQUEST_SHA256', owned=True) == (REQUEST_SHA + '\n').encode(),
         'Original full-v5 request marker changed or unknown partial marker appeared')
    verified = verify_history(EVIDENCE_PATH)
    proof.verify_request(verified['original_request'], parent)
    full_roots = full_attempt_roots()
    need(set(full_roots) == OLD_ROOTS | {ROOT}, 'Unknown or missing original full-attempt root')
    claims = {}
    for attempt in OLD:
        base = attempt['root']; claim = base + '/claims/' + attempt['request_sha']
        marker = base + '/request-claims/' + attempt['name']
        need({str(p) for p in children(base + '/claims')} == {claim}
             and {str(p) for p in children(base + '/request-claims')} == {marker}, 'Unknown prior claim namespace')
        need({p.name for p in children(claim)} == set(proof.prior_claim_names(EVIDENCE_PATH, attempt))
             and {p.name for p in children(marker)} == {'REQUEST_SHA256'}, 'Prior accepted file inventory changed')
        for suffix in ('ops/claims', 'ops/request-claims', 'ops/status'):
            need(not children(base + '/' + suffix), 'Prior maintained namespace appeared')
        absent(base + '/ops/requests/' + attempt['name'] + '.attempt.json')
        claims[claim] = disposition(attempt)
    expected = proof.expected_live_file_hashes(EVIDENCE_PATH)
    all_files = set(expected) | {OLD[0]['root'] + '/claims/' + OLD[0]['request_sha'] + '/' + name
                                for name in proof.CLAIM_NAMES}
    need(len(all_files) <= 96, 'Unreviewed prior metadata scope')
    raw = {p: read_metadata(p, owned=True) for p in sorted(all_files)}
    hashes = {p: sha(v) for p, v in raw.items()}
    proof.verify_live_file_hashes(EVIDENCE_PATH, {p: hashes[p] for p in expected})
    accepted_v1_files(raw, OLD[0])
    from scripts.train.joint.original_observer import query, query_record, step_argv
    qargv = ['/usr/local/bin/squeue', '--noheader', '--user', scope['user'], '--format=%i|%j|%T|%k']
    sargv = step_argv(scope['user'])
    queue, steps = query(qargv), query(sargv)
    for record, argv in [(queue, qargv), (steps, sargv)]:
        query_record(record, argv)
        need(record['returncode'] == 0 and not record['stderr'].strip(), 'Fresh complete native census failed')
    check_live_native(queue['stdout'], steps['stdout'], scope, current_job, current_state)
    node = node_local_context(query, required=require_node)
    base_csv = verified['original_request']['local_logs']
    need(base_csv.endswith('-attempt-v4'), 'Historical local log derivation changed')
    base_csv = base_csv[:-len('-attempt-v4')]
    old_csv = [base_csv, base_csv + '-attempt-v2', base_csv + '-attempt-v3', base_csv + '-attempt-v4']
    if node is not None:
        for path in old_csv:
            absent(path)
    return _VerifiedCensus(_CENSUS_TOKEN, schema='qk-four-prelauncher-fresh-census-v1', checked_at=datetime.now(timezone.utc).isoformat(),
        historical_evidence_manifest_sha256=EVIDENCE_SHA, claims=claims, prior_file_hashes=hashes,
        queue=queue, steps=steps, node_context=node,
        old_csv=dict(paths=old_csv, status='ACTUAL_GPU21_ABSENT' if node else 'DEFERRED_TO_ALLOCATED_ENTRY'),
        earlier_412825_parent_terminal='UNKNOWN_NOT_INFERRED', current_scientific_output_absence_required=False,
        checkpoint_recovery_authorized=False, phase_completion_proven=False)
