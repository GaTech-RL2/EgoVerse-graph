"""Actual read-only admission before claim, before release, and inside allocation.

Metadata proofs are re-opened here. Large checkpoint hashing/loading remains in
the allocated maintained producers. No flag or mock receipt grants readiness.
"""
import json
import hashlib
import os
from pathlib import Path
import re
import shutil
from wrapper_common import bound, command, fields, need, physical, read_json, service, sha, now

_LOADED_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()



def context(spec, q):
    parent = bound(spec['parent_request'])
    need(parent['parameters'] == 314187064 and parent['world_size'] == 4
         and parent['kind'] == 'native-subtask-world-model-v1' and parent['target_steps'] == 300000,
         'Exact frozen native campaign parent required')
    runtime = spec['runtime']
    need(runtime['schema'] == 'native-elmo-wrapper-runtime-v1', 'Explicit wrapper runtime contract required')
    need(spec['native_contract']['NodeList'] == parent['lambda_runtime']['node'] == parent['data_cache']['node'],
         'New node requires new actual matching runtime/cache/diagnostic proof')
    admission = service(spec, 'interruption_admission')
    need(q['operational_service'] == spec['operational_service'], 'Producer service differs from execution spec')
    if spec['mode'] not in {'capture', 'full'}:
        for key in ('source', 'source_revision', 'source_manifest', 'source_manifest_sha256',
                    'data_manifest', 'data_manifest_sha256', 'lambda_runtime', 'data_cache'):
            need(q.get(key) == parent.get(key), 'Producer changed parent science/runtime/cache: ' + key)
    if spec['mode'] == 'full':
        admission.compatible(parent, q)
        gate, result = admission.completed_producer(q['interruption']['admission'],
            'native-elmo-interruption-finalized-v1', 'FINALIZED_REQUIRES_HOST_AND_NATIVE_SUCCESS')
        need(result['resume_request_fingerprint'] == admission.fingerprint(q)
             and result['checkpoint'] == q['interruption']['checkpoint']
             and result['run_id'] == parent['run_id'] and result['original_total_target'] == 300000
             and result['actual_updates'] == 6 and result['diagnostic_scope'] == 'production-checkpoint-only'
             and result['resume_native_contract'] == spec['native_contract'],
             'Completed finalizer does not admit this exact fresh full request')
        need(result['parent_request'] == gate['parent_request'] == spec['parent_request'],
             'Completed finalizer belongs to another parent')
        intent_hash = admission.current_intent(result['current_intent_path'], parent['run_id'], result['current_intent_sha256'])
    else:
        gate = q
        need(gate['parent_request'] == spec['parent_request'], 'Producer parent binding differs')
        intent_hash = admission.current_intent(gate['current_intent_path'], parent['run_id'])
    need(runtime['admission'] == {k: gate[k] for k in ('current_intent_path', 'quiescence')},
         'Submission and allocated intent/claim scope differ')
    return parent, gate, admission, intent_hash


def phase_evidence(spec, q, gate, admission):
    native = service(spec, 'native_interruption')
    if spec['mode'] == 'capture':
        expected = q['parent_native']; job = str(spec['original_parent_job_id'])
        need(str(q['parent_job_id']) == job and expected['JobId'] == job,
             'Capture parent native identity differs')
        accepted = bound(q['parent_running_admission'])
        need(accepted['job_id'] == job and accepted['request_sha256'] == spec['parent_request']['sha256'],
             'Actual parent RUNNING admission missing')
        native.native_identity(accepted['native'], expected, 'RUNNING')
        observer = service(spec, 'original_observer')
        query = observer.query(['/usr/local/bin/scontrol', 'show', 'job', job, '-o', '-dd'])
        raw, provenance = observer.resolve_terminal(q, query)
        steps = observer.query(observer.step_argv('paphiwetsa3'))
        observer.validate_steps(steps, job)
        service(spec, 'phase_observer_proof').check_owned_steps(steps['stdout'], job)
        # Classification and finite numeric loss evidence belong to the actual capture producer.
        return {'parent_native': raw, 'parent_terminal_query': query,
                'parent_terminal_provenance': provenance, 'parent_step_query': steps,
                'capture_is_not_recovery_authorization': True}
    failure = admission.validate_failure(gate['failure'], spec['parent_request'])
    result = {'failure_reason': failure['reason'], 'parent_native': bound(failure['evidence']['parent-native.txt'], False),
              'validated_failure_parent': spec['parent_request']}
    if spec['mode'] in {'diagnostic', 'finalize', 'full'}:
        bundle = q['inspection_producer'] if spec['mode'] == 'diagnostic' else gate['inspection']
        iq, ir = admission.completed_producer(bundle, 'native-elmo-checkpoint-inspection-result-v1',
                                             'INSPECTED_REQUIRES_NATIVE_PRODUCER_COMPLETION')
        need(iq['parent_request'] == ir['parent_request'] == spec['parent_request']
             and ir['highest_healthy_known'] is True and ir['selected']['status'] == 'CPU_STRICT_INSPECTION_PASS',
             'Actual highest production checkpoint inspection missing')
        need(ir['selected']['global_step'] == max(x['global_step'] for x in ir['census']['candidates'])
             and all(x['status'] == 'CPU_STRICT_INSPECTION_PASS' for x in ir['results']),
             'Inspection skipped a newer or unhealthy retained checkpoint')
        if spec['mode'] == 'diagnostic':
            need(q['inspection'] == bundle['result'], 'Diagnostic references another inspection')
        policy = spec['runtime']['native_client']
        prior = bound(policy['prior_container_observation'])
        prerequisite = bound(policy['prior_allocated_prerequisites'])
        need(policy['prior_allocated_prerequisites']['path'] == iq['allocated_prerequisites_path']
             and prerequisite['status'] == 'ACTUAL_PREREQUISITES_AND_NATIVE_ALLOCATION_PASS'
             and prerequisite['job_id'] == prior['job_id'] == iq['producer_native']['JobId']
             and prerequisite['request_sha256'] == prior['request_sha256'] == bundle['request']['sha256']
             and prerequisite['container_auth'] == policy['prior_container_observation']
             and prerequisite['native_client_manifest_sha256'] == prior['native_client_manifest_sha256'] == policy['expected_manifest_sha256']
             and prior['parent_request'] == spec['parent_request'],
             'Prior capsule proof is not the completed inspection producer actual prerequisite chain')
        native.native_identity(prior['native'], iq['producer_native'], 'RUNNING')
        result['inspection_result_sha256'] = bundle['result']['sha256']
    if spec['mode'] in {'finalize', 'full'}:
        dq, dr = admission.completed_producer(gate['diagnostic'], 'native-elmo-production-recovery-diagnostic-v1',
                                             'DIAGNOSTIC_COMPLETE_REQUIRES_HOST_AND_NATIVE_SUCCESS')
        need(dq['parent_request'] == spec['parent_request'] and len(dr['rank_proofs']) == 8,
             'Completed actual all-rank diagnostic missing')
        for binding in dr['rank_proofs']:
            bound(binding)
        result['diagnostic_result_sha256'] = gate['diagnostic']['result']['sha256']
    return result


def own_current(spec, job, request_sha, *, running=False):
    """Validate the sole allowed held/current exception through immutable acceptance."""
    from execution_spec import read_binding, held_native, load_execution
    need(str(job).isdecimal() and re.fullmatch('[0-9a-f]{64}', request_sha or ''), 'Exact current job/request required')
    root = physical(spec['root'])
    claim = root / ('scientific-claims' if spec['mode'] in {'diagnostic', 'full'} else 'producer-claims') / request_sha
    prepared = read_json(claim / 'PREPARED.json')
    args = (prepared['spec_path'], prepared['spec_sha256'], spec['mode'], spec['ops_manifest']['sha256'])
    observed = load_execution(*args) if running else read_binding(*args)
    need(observed[0] == spec and observed[3] == job and sha(observed[1]) == request_sha and observed[4] == claim,
         'Own claim is not the accepted exact spec/request')
    raw = command(['/usr/local/bin/scontrol', 'show', 'job', job, '-o', '-dd'])
    if running:
        service(spec, 'native_interruption').native_identity(raw, dict(spec['native_contract'], JobId=job), 'RUNNING')
    else:
        held_native(raw, spec, prepared['spec_sha256'], job)
    return raw, claim


def quiescence(spec, scope, run_id, current_job=None, current_request_sha=None, *, running=False, original_terminal=None):
    """All prior claims require terminal proof; only this exact held/RUNNING job may remain."""
    native = service(spec, 'native_interruption')
    observer = service(spec, 'original_observer')
    need(scope['claim_roots'] and scope['scientific_job_names'] and scope['scientific_name_prefixes']
         and all(len(p) >= 16 for p in scope['scientific_name_prefixes']), 'Complete identity claim/queue scope required')
    own = None
    if current_job is not None:
        _, own = own_current(spec, current_job, current_request_sha, running=running)
    queue = command(['/usr/local/bin/squeue', '--noheader', '--user', scope['user'], '--format=%i|%j|%T|%k'])
    step_query = observer.query(observer.step_argv(scope['user']))
    # A purged exact job query cannot certify release. Read the full owned inventory.
    observer.query_record(step_query, observer.step_argv(scope['user']))
    need(step_query['returncode'] == 0 and not step_query['stderr'].strip(), 'Complete owned step query failed')
    observer.strict_step_rows(step_query['stdout'])
    if current_job is not None and not running:
        service(spec, 'phase_observer_proof').check_owned_steps(step_query['stdout'], current_job)
    for line in queue.splitlines():
        if not line.strip():
            continue
        row = [x.strip() for x in line.split('|')]
        need(len(row) == 4, 'Unparseable actual owner queue')
        job, name, state, comment = row
        scoped = name in scope['scientific_job_names'] or any(name.startswith(p) for p in scope['scientific_name_prefixes']) or comment == run_id
        if scoped:
            need(job == current_job and state == ('RUNNING' if running else 'PENDING'),
                 'Other pending/active scientific writer: ' + job)
    seen = set(); evidence = []; own_seen = False
    for item in scope['claim_roots']:
        root = physical(item['path']); need(root.is_dir(), 'Claim root unavailable')
        for claim in sorted(root.iterdir()):
            physical(claim)
            need(claim.is_dir() and re.fullmatch('[0-9a-f]{64}', claim.name), 'Unknown claim entry')
            p = read_json(claim / 'PREPARED.json'); job = (claim / 'PARENT_JOB_ID').read_text().strip()
            need(p['request_sha256'] == claim.name and job.isdecimal() and job not in seen,
                 'Unknown/duplicate accepted scientific claim')
            seen.add(job)
            if claim == own:
                need(job == current_job and claim.name == current_request_sha, 'Own claim identity changed')
                own_seen = True; continue
            if job == str(spec['original_parent_job_id']):
                need(claim.name == spec['parent_request']['sha256'] and original_terminal is not None,
                     'Original parent needs separately validated original observer/failure evidence')
                actual = read_json(claim / 'RUNNING_ADMISSION.json')
                need(actual['job_id'] == job and actual['request_sha256'] == claim.name,
                     'Original parent lacks actual RUNNING provenance')
                f = fields(actual['native'])
                expected = {k: f.get(k, '') for k in (*native.NATIVE_KEYS, 'NodeList', 'TresPerNode')}
                native.native_identity(actual['native'], expected, 'RUNNING')
                raw = original_terminal; state = fields(raw).get('JobState')
                need(state in native.TERMINAL, 'Original parent is not terminal')
                terminal_fields = native.native_identity(raw, expected, state)
                need(terminal_fields.get('StartTime') == f.get('StartTime')
                     and terminal_fields.get('SubmitTime') == f.get('SubmitTime'),
                     'Original parent terminal differs from actual RUNNING allocation')
                import hashlib
                proof = {'route': 'validated-original-parent', 'terminal_native_sha256': hashlib.sha256(raw.encode()).hexdigest()}
            else:
                from execution_spec import request_claim_terminal
                captured, raw = request_claim_terminal(spec, claim, p, job, item['terminal_directory'])
                proof = {'route': 'maintained-phase-observer', 'observer_spec': captured['observer_spec'],
                         'terminal_native_sha256': captured['producer_native']['sha256']}
            service(spec, 'phase_observer_proof').check_owned_steps(step_query['stdout'], job)
            evidence.append(dict(claim=str(claim), job_id=job, **proof))
    if current_job is not None and spec['mode'] in {'diagnostic', 'full'}:
        need(own_seen, 'Current scientific claim omitted from complete scope')
    return {'queue': queue, 'step_query': step_query, 'prior_claims': evidence,
            'own_current_claim': str(own) if own else None}


def storage_and_node(spec, q, parent):
    output = Path(q['output']); physical(output.parent)
    need(not output.exists() and not output.is_symlink(), 'Output exists or is a symlink; preserve prior attempts')
    if spec['mode'] == 'full':
        need(not Path(q['local_logs']).exists(), 'Full local log namespace already exists')
    reserve = parent['reserved_checkpoint_bytes']
    need(type(reserve) is int and reserve >= 80_000_000_000, 'Complete production retention reserve required')
    free = shutil.disk_usage(output.parent).free
    need(free > reserve, 'Insufficient actual destination free space')
    node = command(['/usr/local/bin/scontrol', 'show', 'node', spec['native_contract']['NodeList'], '-o'])
    f = fields(node)
    need(f.get('NodeName') == spec['native_contract']['NodeList'] and f.get('State') in {'IDLE', 'ALLOCATED', 'MIXED'},
         'Node has an unavailable/drain/failure flag')
    need(int(f['CPUTot']) >= int(spec['native_contract']['NumCPUs'])
         and int(f['RealMemory']) >= int(spec['native_contract']['MinMemoryNode'][:-1]) * 1024,
         'Node cannot supply requested resources')
    return {'node_native': node, 'available_bytes': free, 'reserved_bytes': reserve,
            'write_fsync_certified': False}


def admit_before_submission(spec, template, current_job=None, current_request_sha=None, *, running=False):
    from execution_spec import execution_claim_census
    execution_claims = execution_claim_census(spec, current_job=current_job,
        current_request_sha=current_request_sha, allow_held=current_job is not None and not running)
    parent, gate, admission, intent = context(spec, template)
    evidence = phase_evidence(spec, template, gate, admission)
    live = quiescence(spec, gate['quiescence'], parent['run_id'], current_job, current_request_sha, running=running, original_terminal=evidence['parent_native'])
    health = storage_and_node(spec, template, parent)
    return {'schema': 'native-elmo-readonly-host-admission-v1', 'checked_at': now(),
            'mode': spec['mode'], 'parent_request': spec['parent_request'],
            'current_intent_sha256': intent, 'evidence': evidence, 'quiescence': live,
            'execution_claims': execution_claims, 'node_storage': health, 'checkpoint_content_checked': False}
