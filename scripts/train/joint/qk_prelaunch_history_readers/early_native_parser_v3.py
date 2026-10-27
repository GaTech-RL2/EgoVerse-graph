"""Exact historical full-v3 pure parser; no collection or runtime code loading.

Only unchanged constants, need/digest and verify_record are retained from the
reviewed original. Original source is authenticated as data by reviewed_parsers.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT=Path('/workspace/users/paphiwetsa3/tasks/elmo-qknorm-swiglu-rmsnorm-imagenet-20261026-v1/full-v3')


CLIENT='/usr/local/bin/scontrol'


LIMIT=1<<20


TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def need(value,message):
    if not value:raise ValueError(message)


def digest(raw):return hashlib.sha256(raw.encode()).hexdigest()


def verify_record(record,parent,capture,request_sha,files_sha,read,api):
    need(record['schema']=='early-full-native-observation-v1' and record['status']=='RAW_UNAUTHENTICATED_NO_TERMINAL_AUTHORITY'
         and record['parent_job_id']==parent and record['observer_job_id']==capture
         and record['request_sha256']==request_sha and record['files_manifest_sha256']==files_sha
         and record['uid']==41003 and record['node']=='gpu21','Wrong raw observation identity')
    expected={'parent':[CLIENT,'show','job',parent,'-o','-dd'],
              'steps':[CLIENT,'show','step','-o'],
              'observer':[CLIENT,'show','job',capture,'-o','-dd']}
    need(set(record['queries'])==set(expected),'Incomplete early native query closure')
    for key,argv in expected.items():
        row=record['queries'][key]
        need(row['argv']==argv and row['exit_code']==0 and row['stderr']=='' and row['failure'] is None and row.get('output_truncated') is False,'Actual native query failed/unknown/truncated; raw is not terminal proof')
        start=datetime.fromisoformat(row['start_utc']);end=datetime.fromisoformat(row['end_utc'])
        need(start.tzinfo is not None and end.tzinfo is not None and end>=start
             and end<=datetime.now(timezone.utc)+timedelta(seconds=5),'Invalid/future actual query interval')
    names={f'native-{parent}.txt',f'native-{parent}.stderr',f'native-{parent}.exit',
           f'native-{parent}.steps.json',f'capture-running-{capture}.txt'}
    need(set(record['canonical_sha256'])==names,'Changed canonical early evidence set')
    for name,sha in record['canonical_sha256'].items():need(digest(read(ROOT/name))==sha,'Changed early raw bytes')
    need(read(ROOT/f'native-{parent}.txt')==record['queries']['parent']['stdout']
         and read(ROOT/f'capture-running-{capture}.txt')==record['queries']['observer']['stdout']
         and read(ROOT/f'native-{parent}.stderr')=='' and read(ROOT/f'native-{parent}.exit')=='0\n','Raw native files differ from actual query')
    claim=ROOT/'claims'/request_sha
    need(read(claim/'PARENT_JOB_ID').strip()==parent and read(claim/'CAPTURE_JOB_ID').strip()==capture,'Wrong accepted claim IDs')
    for name,jid in [('parent-sbatch.stdout',parent),('capture-sbatch.stdout',capture),('parent-sbatch.exit','0'),('capture-sbatch.exit','0')]:
        need(read(claim/name).strip()==jid,'Actual accepted query/claim outcome differs')
    prepared=json.loads(read(claim/'PREPARED.json'))
    need(prepared['request_sha256']==request_sha and prepared['manifest_sha256']==files_sha,'Prepared request/code closure changed')
    release=json.loads(read(claim/'RELEASE_ATTEMPT.json'))
    need(release.get('parent')==parent and release.get('capture')==capture and release.get('request_sha256')==request_sha,'Wrong immutable release binding')
    parent_raw=record['queries']['parent']['stdout'];parent_fields=api.fields(parent_raw)
    need(parent_fields.get('JobState') in TERMINAL,'Parent is not genuinely terminal')
    terminal=api.check_native(parent_raw,job=parent,state=parent_fields['JobState'],request_sha=request_sha)
    need(terminal.get('NodeList')=='gpu21' and terminal.get('NumNodes')=='1'
         and api.native_time(terminal['EndTime'])>=api.native_time(terminal['StartTime'])>=api.native_time(terminal['SubmitTime'])
         and re.fullmatch('[0-9]+:[0-9]+',terminal.get('ExitCode','')),'Actual terminal node/interval/exit absent')
    active=api.check_native(record['queries']['observer']['stdout'],job=capture,state='RUNNING',capture=True,parent=parent,request_sha=request_sha)
    parent_end=api.native_time(terminal['EndTime'])
    need(api.native_time(active['StartTime'])>=parent_end,'Observer started before parent terminal; not genuine afterany')
    for row in record['queries'].values():
        need(datetime.fromisoformat(row['start_utc'])>=parent_end,'Native query predates parent terminal')
    need(datetime.fromisoformat(record['queries']['parent']['end_utc'])<=datetime.fromisoformat(record['queries']['steps']['start_utc'])
         and datetime.fromisoformat(record['queries']['steps']['end_utc'])<=datetime.fromisoformat(record['queries']['observer']['start_utc'])
         and datetime.fromisoformat(record['queries']['observer']['end_utc'])>=api.native_time(active['StartTime']),
         'Early native query order/observer interval differs')
    api.same_attempt(*[api.check_native(read(claim/name),job=parent,state='PENDING',request_sha=request_sha) for name in ['HELD_NATIVE.txt','RELEASE_HELD_NATIVE.txt']],terminal)
    api.same_attempt(*[api.check_native(read(claim/name),job=capture,state='PENDING',capture=True,parent=parent,request_sha=request_sha) for name in ['CAPTURE_PENDING_NATIVE.txt','RELEASE_CAPTURE_PENDING_NATIVE.txt']],active)
    steps=json.loads(read(ROOT/f'native-{parent}.steps.json'))
    need(steps['stdout']==record['queries']['steps']['stdout'] and steps['stderr']==record['queries']['steps']['stderr']
         and steps['observed_at_utc']==record['queries']['steps']['end_utc'],'Canonical complete steps differs')
    api.release_observation(steps,parent,request_sha,terminal)
    return dict(schema='authenticated-early-full-native-observation-v1',status='AUTHENTICATED_TERMINAL_EVIDENCE_ONLY',
        parent_job_id=parent,observer_job_id=capture,parent_state=terminal['JobState'],parent_exit_code=terminal['ExitCode'],
        parent_start=terminal['StartTime'],parent_end=terminal['EndTime'],request_sha256=request_sha,
        files_manifest_sha256=files_sha,canonical_sha256=record['canonical_sha256'],
        zero_work_proven=False,production_completion_proven=False,observer_own_terminal='UNKNOWN_NOT_INFERRED')
