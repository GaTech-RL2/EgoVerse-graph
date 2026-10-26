"""Bounded Python-free gateway bootstrap and actual allocated-host controller.

No model/checkpoint loading, remote mirror semantics, cancellation, or retry.
The allocated controller calls maintained producer and observer implementations.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone

PREFIX = '/workspace/users/paphiwetsa3/tasks/elmo-large-continuous-20261026-v1/'
PINS = {
    'plain': ('412789', 'gpu11', '936e9acd30fa517f6f38311c0bcfd2b88544da8d476bd41125c4008de2845073'),
    'fused': ('412798', 'gpu11', 'e34190bfef155942fe216a1e5e81317cfa31b65d4c98ccb025a579c874fd3876'),
    'boundary': ('412800', 'gpu03', '210e6f73c8a1a396716fab2f1e7e17f3b9d0663b099971eb05807181f9f00684'),
}
MODES = {'capture', 'inspect', 'diagnostic', 'finalize', 'full'}
FAILURE_STATES = {'FAILED', 'TIMEOUT', 'PREEMPTED', 'NODE_FAIL', 'BOOT_FAIL', 'CANCELLED'}
MAX_FILE = 4 << 20

_LOADED_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()



def need(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def parse(value):
    def pairs(items):
        need(len(items) == len(dict(items)), 'Duplicate JSON field')
        return dict(items)
    return json.loads(value, object_pairs_hook=pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def path(value):
    p = PurePosixPath(value)
    need(isinstance(value, str) and value.startswith(PREFIX) and '..' not in p.parts
         and str(p) == value and re.fullmatch(r'[A-Za-z0-9_./=-]+', value), 'Unsafe/noncanonical campaign path')
    return p


def fields(raw):
    items = re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9:/_-]*)=(\S+)', raw)
    need(len(items) == len(dict(items)), 'Ambiguous native fields')
    return dict(items)


def binding(value):
    need(set(value) == {'path', 'sha256'} and re.fullmatch('[0-9a-f]{64}', value['sha256']), 'Exact path/hash binding required')
    path(value['path'])
    return value


class OutcomeUnknown(RuntimeError):
    pass


class SSH:
    """Explicit remote transport. Never executes an unquoted caller command."""
    def __init__(self, host='lambda'):
        need(host == 'lambda', 'Only the configured Lambda endpoint is supported')
        self.host = host

    def run(self, script, *, mutation=False):
        try:
            result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2', self.host,
                'bash -c ' + shlex.quote(script)], capture_output=True, timeout=55)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise (OutcomeUnknown if mutation else RuntimeError)(
                'Transport failed; inspect exact existing claim/native state, never retry a mutation') from exc
        need(len(result.stdout) <= MAX_FILE + 65536 and len(result.stderr) <= 65536, 'Unbounded transport output')
        if result.returncode:
            raise (OutcomeUnknown if mutation else RuntimeError)(
                'Remote operation did not certify success; inspect immutable claim; no retry')
        return result.stdout

    def read(self, filename):
        p = shlex.quote(str(path(filename)))
        return self.run('set -euo pipefail; test "$(readlink -f -- ' + p + ')" = ' + p
            + '; test -f ' + p + '; test "$(stat -c %s -- ' + p + ')" -le ' + str(MAX_FILE)
            + '; timeout --kill-after=2s 15s cat -- ' + p)

    def native(self, argv):
        need(argv[0] in {'/usr/local/bin/squeue', '/usr/local/bin/scontrol'}, 'Read-only native client required')
        need(argv[1:2] != ['release'], 'Mutation through native reader prohibited')
        return self.run('set -euo pipefail; timeout --kill-after=2s 20s ' + shlex.join(argv)).decode()

    def census(self, root):
        p = shlex.quote(str(path(root)))
        return self.run('set -euo pipefail; test "$(readlink -f -- ' + p + ')" = ' + p
            + '; test -d ' + p + '; timeout --kill-after=2s 15s find ' + p
            + ' -mindepth 1 -maxdepth 1 -printf "%f|%y\\n" | LC_ALL=C sort').decode()


class Reader:
    def __init__(self, read):
        self.read = read
        self.readset = {}

    def file(self, filename):
        path(filename)
        value = self.read(filename)
        need(isinstance(value, bytes) and len(value) <= MAX_FILE, 'Bounded byte reader required')
        old = self.readset.setdefault(filename, digest(value))
        need(old == digest(value), 'Remote input changed during admission')
        return value

    def bound(self, b, *, parsed=True):
        binding(b)
        value = self.file(b['path'])
        need(digest(value) == b['sha256'], 'Changed bound remote input')
        return parse(value) if parsed else value

    def manifest(self, root, b):
        need(b['path'] == root + '/FILES.sha256', 'Manifest root differs')
        data = self.bound(b, parsed=False).decode()
        names = set()
        for line in data.splitlines():
            sha, name = line.split('  ', 1)
            need(name not in names and not name.startswith('/') and '..' not in PurePosixPath(name).parts,
                 'Unsafe or duplicate manifest member')
            names.add(name)
            self.bound({'path': root + '/' + name, 'sha256': sha}, parsed=False)
        need(names and len(names) <= 128, 'Unbounded operations manifest')
        return names


def validate(reader, spec_path, spec_sha, ops_sha):
    spec = reader.bound({'path': spec_path, 'sha256': spec_sha})
    need(spec['schema'] == 'native-elmo-allocated-controller-v1' and spec['arm'] in PINS, 'Controller schema/arm differs')
    root = str(path(spec['root']))
    need(str(path(spec_path).parent) == root + '/controller-specs', 'Controller spec escaped root')
    need(spec['ops_manifest'] == {'path': root + '/FILES.sha256', 'sha256': ops_sha}, 'Operations manifest differs')
    need(spec['controller_claim_root']==PREFIX+'recovery-controller-claims/'+spec['arm'],
         'Canonical same-arm controller claim root required across operations packets')
    names = reader.manifest(root, spec['ops_manifest'])
    need({'controller.py', 'controller.sbatch', 'execution_spec.py', 'host_guard.py', 'phase_observer.py', 'observe-phase.sbatch'} <= names,
         'Controller/producer/observer immutable closure incomplete')
    need(not any(PurePosixPath(n).parts[0] in {'controller-specs','controller-claims','execution','templates','requests','observer-specs','observer-claims','status','logs','scientific-claims','producer-claims','execution-claims'} for n in names),
         'Dynamic records must not enter immutable executable manifest')
    parent_id, node, parent_sha = PINS[spec['arm']]
    parent = reader.bound(spec['parent_request'])
    need(spec['parent_request']['sha256'] == parent_sha, 'Controller changed original parent identity')
    run_id = 'mot-elmo-pickplace-continuous-large314m-' + spec['arm'] + '-robot-human-camera-subtask-s42-300k-20261026-v1'
    need(parent['run_id'] == run_id and parent['parameters'] == 314187064 and parent['world_size'] == 4
         and parent['target_steps'] == 300000 and parent['phase'] == 'full', 'Wrong original campaign identity')
    intent = reader.bound(spec['current_intent'])
    need(intent['schema'] == 'native-elmo-current-intent-v1' and intent['authority'] == 'human'
         and intent['status'] == 'ACTIVE' and intent['hold_recovery'] is False
         and run_id in intent['authorized_recovery_run_ids'] and run_id not in intent['retired_run_ids'], 'Recovery no longer authorized')
    terminal = fields(reader.bound(spec['original_failed_native'], parsed=False).decode())
    need(terminal.get('JobId') == parent_id and terminal.get('JobName') == parent['job_name']
         and terminal.get('UserId') == 'paphiwetsa3(41003)' and terminal.get('NodeList') == node
         and terminal.get('JobState') in FAILURE_STATES
         and terminal.get('EndTime') not in {None, 'Unknown', 'None'}
         and re.fullmatch(r'\d+:\d+', terminal.get('ExitCode', '')), 'Actual failed original parent required for controller allocation')
    execution = spec['producer_execution']
    need(set(execution) == {'path','sha256','mode','ops_manifest_sha256'} and execution['mode'] in MODES
         and execution['ops_manifest_sha256'] == ops_sha, 'Producer execution binding differs')
    producer = reader.bound({k: execution[k] for k in ('path','sha256')})
    need(producer['root'] == root and producer['arm'] == spec['arm'] and producer['mode'] == execution['mode']
         and producer['parent_request'] == spec['parent_request'] and producer['runtime']['admission']['current_intent_path'] == spec['current_intent']['path'], 'Producer and controller scopes differ')
    queue_scope=producer['runtime']['admission']['quiescence']
    need(spec['queue_scope']=={k:queue_scope[k] for k in ('user','scientific_job_names','scientific_name_prefixes')}
         and spec['queue_scope']['user']=='paphiwetsa3' and spec['queue_scope']['scientific_job_names']
         and spec['queue_scope']['scientific_name_prefixes']
         and all(len(p)>=16 for p in spec['queue_scope']['scientific_name_prefixes']), 'Complete actual producer queue scope required')
    reader.bound(producer['request_template'])
    service = producer['operational_service']
    reader.manifest(service['root'], {'path':service['manifest'],'sha256':service['manifest_sha256']})
    observer = reader.bound(spec['observer_spec'])
    need(observer == {'schema':'native-elmo-phase-observer-spec-v1', 'root':root,
         'producer_execution':execution, 'terminal_directory':spec['terminal_directory']}, 'Observer ID-free scope differs')
    need(str(path(spec['observer_spec']['path']).parent) == root + '/observer-specs'
         and producer['release_observer_path'] == spec['observer_spec']['path'], 'Mandatory release observer differs')
    name = spec['job_name']
    need(re.fullmatch('elmo-recovery-controller-' + spec['arm'] + '-[a-z0-9-]{1,64}', name), 'Controller must have non-scientific scoped name')
    need(spec['interpreter']['path'] == '/usr/bin/python3' and spec['interpreter']['minimum_version'] == [3,10], 'Explicit allocated-host interpreter required')
    evidence = reader.bound(spec['interpreter']['provenance'])
    need(evidence['schema'] == 'native-elmo-host-python-provenance-v1' and evidence['node'] == node
         and evidence['interpreter_path'] == '/usr/bin/python3', 'Interpreter provenance belongs to another node')
    script = reader.bound(evidence['script'], parsed=False).decode()
    native = fields(reader.bound(evidence['native'], parsed=False).decode())
    need('/usr/bin/python3 -B ' in script and native['JobId'] == evidence['job_id'] and native['NodeList'] == node
         and native['JobState'] == 'COMPLETED' and native['ExitCode'] == '0:0'
         and reader.bound(evidence['host_exit'], parsed=False).strip() == b'0', 'No successful allocated-host interpreter route')
    # Prior script proves the route, not current binary identity or GPU/runtime health.
    return spec, producer


def controller_argv(spec, spec_path, spec_sha, ops_sha):
    root=spec['root']; node=PINS[spec['arm']][1]
    return ['/usr/local/bin/sbatch','--parsable','--hold','--no-requeue','--export=NONE',
        '--account=loaner','--qos=loaner-normal','--partition=rl2','--nodes=1','--ntasks=1',
        '--cpus-per-task=2','--mem=1G','--gres=none','--time=00:20:00','--nodelist='+node,
        '--job-name='+spec['job_name'],'--comment=elmo-controller:'+spec_sha,'--chdir='+root,
        '--output='+root+'/logs/%x-%j.out','--error='+root+'/logs/%x-%j.err',
        root+'/controller.sbatch',root,spec_path,spec_sha,ops_sha]


def check_controller(raw,spec,spec_sha,job,state):
    f=fields(raw); root=spec['root']; node=PINS[spec['arm']][1]
    expected={'JobId':job,'JobName':spec['job_name'],'UserId':'paphiwetsa3(41003)',
        'Account':'loaner','QOS':'loaner-normal','Partition':'rl2','Requeue':'0','Restarts':'0',
        'JobState':state,'ReqNodeList':node,'NumCPUs':'2','NumTasks':'1','MinMemoryNode':'1G',
        'TimeLimit':'00:20:00','Command':root+'/controller.sbatch','Comment':'elmo-controller:'+spec_sha}
    need(all(f.get(k)==v for k,v in expected.items()),'Controller actual native identity/resources differ')
    need(f.get('SubmitTime') not in {None,'Unknown','None'} and f.get('NumNodes') in ({'1','1-1'} if state=='PENDING' else {'1'}), 'Controller submit/node contract missing')
    if state=='PENDING':
        need(f.get('Reason')=='JobHeldUser' and f.get('NodeList','') in {'','(null)',node}
             and f.get('AllocTRES','') in {'','(null)'} and f.get('StartTime') in {None,'Unknown','N/A'}, 'Controller is not genuinely held')
    else:
        need(f.get('NodeList')==node and f.get('StartTime') not in {None,'Unknown','None'},'Controller allocation not actual')
    for key in ('ReqTRES', *(['AllocTRES'] if state=='RUNNING' else [])):
        values=set(f.get(key,'').split(','))
        need({'cpu=2','node=1'} <= values and bool(values & {'mem=1G','mem=1024M'}) and not any('gpu' in v.lower() for v in values),'Controller CPU/memory TRES conflict')
    need(f.get('JOB_GRES','') in {'','(null)'} and f.get('GRES','') in {'','(null)'}
         and f.get('TresPerNode','') in {'','(null)'} and f.get('TresPerTask','cpu:2')=='cpu:2', 'Controller has GPU/auxiliary resource claims')
    need(not any('gpu' in k.lower() or ('tresper' in k.lower() and 'gpu' in v.lower()) for k,v in f.items()), 'Controller owns unexpected GPU resources')
    return f


QUEUE=['/usr/local/bin/squeue','--noheader','--user','paphiwetsa3','--format=%i|%j|%T|%k|%u']


def queue_guard(raw,spec,current=None):
    for line in raw.splitlines():
        if not line.strip(): continue
        row=line.split('|')
        need(len(row)==5 and row[0].isdecimal() and row[4]=='paphiwetsa3','Malformed full owner queue')
        scope=spec['queue_scope']
        run_id='mot-elmo-pickplace-continuous-large314m-'+spec['arm']+'-robot-human-camera-subtask-s42-300k-20261026-v1'
        scientific=(row[1] in scope['scientific_job_names'] or any(row[1].startswith(p) for p in scope['scientific_name_prefixes'])
                    or row[3]==run_id)
        controller=row[1].startswith('elmo-recovery-controller-'+spec['arm']+'-')
        need(not scientific and (not controller or row[0]==current and row[2]=='PENDING'), 'Other active/pending scientific writer or controller')


def ledger(spec):
    return spec['controller_claim_root']+'/'+spec['producer_execution']['sha256']


def write_shell(filename,payload):
    # GNU dd exclusive create plus fsync; never overwrites or deletes a record.
    data=base64.b64encode(payload).decode()
    return 'printf %s '+shlex.quote(data)+' | base64 --decode | dd status=none conv=excl,fsync of='+shlex.quote(filename)+'; '


def transaction(reader,remote,spec,queue,census,body,*,held=None):
    root=spec['root']; claims=spec['controller_claim_root']; q=shlex.quote
    lock=claims+'.lock'
    script='set -euo pipefail; umask 077; test "$(readlink -f -- '+q(str(PurePosixPath(claims).parent))+')" = '+q(str(PurePosixPath(claims).parent))+'; test ! -L '+q(lock)+'; exec 9>>'+q(lock)+'; flock -n 9; '
    for filename,sha in sorted(reader.readset.items()):
        script+='test "$(readlink -f -- '+q(filename)+')" = '+q(filename)+'; printf "%s  %s\\n" '+q(sha)+' '+q(filename)+' | sha256sum --check --status; '
    script+='current=$(timeout --kill-after=2s 20s '+shlex.join(QUEUE)+'); test "$(printf %s "$current" | sha256sum | cut -d" " -f1)" = '+q(digest(queue.rstrip('\n').encode()))+'; '
    for directory,observed in sorted(census.items()):
        script+='current=$(find '+q(directory)+' -mindepth 1 -maxdepth 1 -printf "%f|%y\\n" | LC_ALL=C sort); test "$(printf %s "$current" | sha256sum | cut -d" " -f1)" = '+q(digest(observed.rstrip('\n').encode()))+'; '
    if held:
        job,raw=held
        script+='current=$(timeout --kill-after=2s 20s /usr/local/bin/scontrol show job '+q(job)+' -o -dd); test "$(printf %s "$current" | sha256sum | cut -d" " -f1)" = '+q(digest(raw.rstrip('\n').encode()))+'; '
    return remote.run(script+body,mutation=True)


def prior_census(reader,remote,spec,*,own=False):
    census={}; current=spec['producer_execution']['sha256']; seen=set()
    for directory in [spec['controller_claim_root']]:
        raw=remote.census(directory); census[directory]=raw
        for line in raw.splitlines():
            name,kind=line.split('|')
            need(re.fullmatch('[0-9a-f]{64}',name) and kind=='d' and name not in seen,'Unexpected/duplicate controller claim entry')
            seen.add(name)
            if own and name==current: continue
            need(name!=current,'Controller claim already exists; inspect, never retry')
            old=directory+'/'+name
            prepared=parse(reader.file(old+'/PREPARED.json'))
            need(prepared['schema']=='native-elmo-controller-prepared-v1'
                 and prepared['producer_spec_sha256']==name,'Earlier controller intent differs')
            previous=reader.bound({'path':prepared['spec_path'],'sha256':prepared['spec_sha256']})
            need(previous['schema']=='native-elmo-allocated-controller-v1'
                 and previous['arm']==spec['arm'] and previous['parent_request']==spec['parent_request']
                 and previous['controller_claim_root']==spec['controller_claim_root']
                 and str(path(prepared['spec_path']).parent)==previous['root']+'/controller-specs'
                 and previous['ops_manifest']=={'path':previous['root']+'/FILES.sha256','sha256':prepared['ops_manifest_sha256']},
                 'Earlier controller belongs to another root/identity/manifest')
            reader.manifest(previous['root'],previous['ops_manifest'])
            execution=previous['producer_execution']
            need(execution['sha256']==name and execution['ops_manifest_sha256']==prepared['ops_manifest_sha256']
                 and str(path(execution['path']).parent)==previous['root']+'/execution',
                 'Earlier producer execution escaped immutable controller intent')
            earlier=reader.bound({k:execution[k] for k in ('path','sha256')})
            need(earlier['root']==previous['root'] and earlier['arm']==spec['arm']
                 and earlier['parent_request']==spec['parent_request'] and earlier['mode']==execution['mode'],
                 'Earlier producer belongs to another original campaign')
            current_producer=reader.bound({k:spec['producer_execution'][k] for k in ('path','sha256')})
            old_roots=earlier['execution_claim_roots']
            own_root={'path':earlier['root']+'/execution-claims','terminal_directory':previous['terminal_directory']}
            need(own_root in old_roots and old_roots
                 and all(item in current_producer['execution_claim_roots'] for item in old_roots),
                 'Current execution census omits a prior released child root or terminal directory')
            if earlier['mode'] in {'diagnostic','full'}:
                scientific={'path':earlier['root']+'/scientific-claims','terminal_directory':previous['terminal_directory']}
                need(scientific in current_producer['runtime']['admission']['quiescence']['claim_roots'],
                     'Current scientific census omits a prior released writer claim root')
            accepted=parse(reader.file(old+'/acceptance.json')); result=parse(reader.file(old+'/RESULT.json'))
            prior_argv=controller_argv(previous,prepared['spec_path'],prepared['spec_sha256'],prepared['ops_manifest_sha256'])
            need(prepared['argv']==prior_argv and accepted['argv']==prior_argv
                 and accepted['schema']=='native-elmo-controller-acceptance-v1'
                 and accepted['spec_sha256']==prepared['spec_sha256'] and accepted['producer_spec_sha256']==name
                 and str(accepted['job_id']).isdecimal()
                 and reader.file(old+'/PARENT_JOB_ID').strip()==accepted['job_id'].encode()
                 and reader.file(old+'/sbatch.stdout').strip()==accepted['job_id'].encode()
                 and reader.file(old+'/sbatch.exit').strip()==b'0',
                 'Earlier controller acceptance is incomplete or differs from immutable intent')
            held=reader.file(old+'/HELD_NATIVE.txt').decode()
            need(digest(held.encode())==accepted['held_native_sha256'],'Earlier controller held bytes changed')
            check_controller(held,previous,prepared['spec_sha256'],accepted['job_id'],'PENDING')
            need(result['schema']=='native-elmo-controller-result-v1' and result['controller_job_id']==accepted['job_id']
                 and result['producer_spec_sha256']==name and result['status']=='CHILD_RELEASE_ACCEPTED_NOT_COMPLETION',
                 'Earlier controller outcome unknown; inspect child claims/native, never bypass')
            # This result proves no child terminal state. The actual allocated
            # host must reopen every included execution/request claim and its
            # maintained observer final commit before submitting a new child.
    return census


def submit_controller(remote,spec_path,spec_sha,ops_sha):
    reader=Reader(remote.read); spec,_=validate(reader,spec_path,spec_sha,ops_sha)
    census=prior_census(reader,remote,spec); queue=remote.native(QUEUE); queue_guard(queue,spec)
    claim=ledger(spec); argv=controller_argv(spec,spec_path,spec_sha,ops_sha)
    prepared=dict(schema='native-elmo-controller-prepared-v1',spec_path=spec_path,spec_sha256=spec_sha,
        ops_manifest_sha256=ops_sha,producer_spec_sha256=spec['producer_execution']['sha256'],argv=argv)
    body='mkdir '+shlex.quote(claim)+'; sync -f '+shlex.quote(spec['controller_claim_root'])+'; '
    body+=write_shell(claim+'/PREPARED.json',encode(prepared))
    body+='sync -f '+shlex.quote(claim)+'; '
    body+='set +e; (set -o noclobber; timeout --kill-after=3s 25s '+shlex.join(argv)+' > '+shlex.quote(claim+'/sbatch.stdout')+' 2> '+shlex.quote(claim+'/sbatch.stderr')+'); code=$?; set -e; '
    body+='printf "%s\\n" "$code" | dd status=none conv=excl,fsync of='+shlex.quote(claim+'/sbatch.exit')+'; sync -f '+shlex.quote(claim)+'; test "$code" = 0; cat '+shlex.quote(claim+'/sbatch.stdout')+'; '
    job=transaction(reader,remote,spec,queue,census,body).decode().strip()
    need(job.isdecimal(),'Controller acceptance unknown; inspect existing claim, never retry')
    # A failed binding read/write leaves the durable submission claim unresolved.
    raw=remote.native(['/usr/local/bin/scontrol','show','job',job,'-o','-dd'])
    check_controller(raw,spec,spec_sha,job,'PENDING')
    need(reader.file(claim+'/sbatch.stdout').strip()==job.encode() and reader.file(claim+'/sbatch.exit').strip()==b'0','Actual controller acceptance not preserved')
    census=prior_census(reader,remote,spec,own=True); queue=remote.native(QUEUE); queue_guard(queue,spec,job)
    accepted=dict(schema='native-elmo-controller-acceptance-v1',job_id=job,spec_sha256=spec_sha,
        producer_spec_sha256=spec['producer_execution']['sha256'],argv=argv,held_native_sha256=digest(raw.encode()))
    body=write_shell(claim+'/PARENT_JOB_ID',(job+'\n').encode())+write_shell(claim+'/HELD_NATIVE.txt',raw.encode())+write_shell(claim+'/acceptance.json',encode(accepted))
    body+='sync -f '+shlex.quote(claim)+'; '
    transaction(reader,remote,spec,queue,census,body,held=(job,raw))
    return accepted


def release_controller(remote,spec_path,spec_sha,ops_sha):
    reader=Reader(remote.read); spec,_=validate(reader,spec_path,spec_sha,ops_sha); claim=ledger(spec)
    accepted=parse(reader.file(claim+'/acceptance.json')); prepared=parse(reader.file(claim+'/PREPARED.json')); job=accepted['job_id']
    argv=controller_argv(spec,spec_path,spec_sha,ops_sha)
    need(accepted['schema']=='native-elmo-controller-acceptance-v1' and accepted['spec_sha256']==spec_sha
         and accepted['argv']==prepared['argv']==argv and prepared['spec_sha256']==spec_sha
         and reader.file(claim+'/PARENT_JOB_ID').strip()==job.encode()
         and reader.file(claim+'/sbatch.stdout').strip()==job.encode() and reader.file(claim+'/sbatch.exit').strip()==b'0', 'Controller acceptance changed')
    held=reader.file(claim+'/HELD_NATIVE.txt').decode(); old=check_controller(held,spec,spec_sha,job,'PENDING')
    need(digest(held.encode())==accepted['held_native_sha256'],'Held evidence changed')
    raw=remote.native(['/usr/local/bin/scontrol','show','job',job,'-o','-dd']); current=check_controller(raw,spec,spec_sha,job,'PENDING')
    need(current['SubmitTime']==old['SubmitTime'],'Controller allocation replaced')
    census=prior_census(reader,remote,spec,own=True); queue=remote.native(QUEUE); queue_guard(queue,spec,job)
    attempt=dict(job_id=job,spec_sha256=spec_sha,held_native_sha256=digest(raw.encode()),completion_proven=False)
    body=write_shell(claim+'/RELEASE_ATTEMPT.json',encode(attempt))
    body+='sync -f '+shlex.quote(claim)+'; '
    body+='set +e; (set -o noclobber; timeout --kill-after=3s 25s /usr/local/bin/scontrol release '+shlex.quote(job)+' > '+shlex.quote(claim+'/release.stdout')+' 2> '+shlex.quote(claim+'/release.stderr')+'); code=$?; set -e; '
    body+='printf "%s\\n" "$code" | dd status=none conv=excl,fsync of='+shlex.quote(claim+'/release.exit')+'; sync -f '+shlex.quote(claim)+'; test "$code" = 0; '
    transaction(reader,remote,spec,queue,census,body,held=(job,raw))
    return {'job_id':job,'status':'CONTROLLER_RELEASE_ACCEPTED_NOT_CHILD_AUTHORIZATION_OR_COMPLETION'}


def local_read(filename):
    p=Path(str(path(filename)))
    need(p.resolve(strict=True)==p and p.is_file() and p.stat().st_size<=MAX_FILE,'Actual physical allocated-host input required')
    return p.read_bytes()


def write_once(filename,value):
    p=Path(filename); need(p.parent.resolve(strict=True)==p.parent,'Physical output parent required')
    with p.open('xb') as stream:
        stream.write(encode(value)); stream.flush(); os.fsync(stream.fileno())
    fd=os.open(p.parent,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def import_bound(root,name,manifest_sha):
    need(name in {'execution_spec','phase_observer'},'No arbitrary controller import')
    names=Reader(local_read).manifest(root,{'path':str(Path(root)/'FILES.sha256'),'sha256':manifest_sha})
    need(name+'.py' in names,'Controller module absent from immutable closure')
    location=Path(root)/(name+'.py')
    module_spec=importlib.util.spec_from_file_location(name,location)
    module=importlib.util.module_from_spec(module_spec)
    module._verified_ops_manifest_sha256=manifest_sha
    module._verified_ops_member_sha256=digest(location.read_bytes())
    sys.modules[name]=module; module_spec.loader.exec_module(module)
    return module


def advance(execution,observer,execution_args,observer_args,*,before_each):
    """One producer, one observer, one release; no recovery/retry branch."""
    need(getattr(execution,'RELEASE_OBSERVER_REQUIRED',False) is True,'Execution release lacks mandatory observer guard')
    before_each(); accepted=execution.submit_held(*execution_args)
    before_each(); observed=observer.submit_once(*observer_args)
    before_each(); ready=observer.verify_ready_for_release(*observer_args)
    # release_bound must also enforce its mandatory observer hook internally.
    released=execution.release_bound(*execution_args)
    return {'accepted':accepted,'observer':observed,'observer_readiness':ready,'released':released}


def allocated(spec_path,spec_sha,ops_sha):
    reader=Reader(local_read); spec,producer=validate(reader,spec_path,spec_sha,ops_sha); root=spec['root']; claim=ledger(spec)
    job=os.environ.get('SLURM_JOB_ID','')
    need(job.isdecimal() and os.getuid()==41003 and os.environ.get('SLURM_RESTART_COUNT','0')=='0'
         and os.environ.get('SLURMD_NODENAME')==PINS[spec['arm']][1]
         and not os.environ.get('SLURM_JOB_GPUS') and not os.environ.get('SLURM_STEP_GPUS'),'Actual CPU controller process identity missing')
    need(sys.version_info >= (3,10) and Path(sys.executable).resolve()==Path('/usr/bin/python3').resolve(),'Unbound current host Python interpreter')
    native=subprocess.run(['/usr/local/bin/scontrol','show','job',job,'-o','-dd'],capture_output=True,text=True,timeout=20)
    need(native.returncode==0 and not native.stderr,'Controller native authentication failed')
    current=check_controller(native.stdout,spec,spec_sha,job,'RUNNING')
    accepted=parse(reader.file(claim+'/acceptance.json')); held=reader.file(claim+'/HELD_NATIVE.txt').decode()
    prepared=parse(reader.file(claim+'/PREPARED.json')); released=parse(reader.file(claim+'/RELEASE_ATTEMPT.json'))
    previous=check_controller(held,spec,spec_sha,job,'PENDING')
    need(accepted['job_id']==job and accepted['spec_sha256']==spec_sha and accepted['argv']==controller_argv(spec,spec_path,spec_sha,ops_sha)
         and accepted['held_native_sha256']==digest(held.encode()) and current['SubmitTime']==previous['SubmitTime']
         and released['job_id']==job and released['spec_sha256']==spec_sha
         and prepared['spec_sha256']==spec_sha and prepared['argv']==accepted['argv']
         and reader.file(claim+'/PARENT_JOB_ID').strip()==job.encode()
         and reader.file(claim+'/sbatch.stdout').strip()==job.encode()
         and reader.file(claim+'/sbatch.exit').strip()==b'0','Controller not the accepted/released actual allocation')
    write_once(claim+'/RUNNING.json',dict(job_id=job,native=native.stdout,interpreter=sys.executable,
        interpreter_realpath=str(Path(sys.executable).resolve()),interpreter_sha256=digest(Path(sys.executable).read_bytes()),python_version=sys.version))
    sys.path.insert(0,root)
    execution=import_bound(root,'execution_spec',ops_sha); observer=import_bound(root,'phase_observer',ops_sha)
    e=spec['producer_execution']; execution_args=(e['path'],e['sha256'],e['mode'],ops_sha)
    observer_args=(spec['observer_spec']['path'],spec['observer_spec']['sha256'],ops_sha)
    def refresh():
        # Real shared filesystem reread before every phase transition; mutation fails closed.
        fresh=Reader(local_read); actual,_=validate(fresh,spec_path,spec_sha,ops_sha)
        need(actual==spec,'Controller spec changed')
    result=advance(execution,observer,execution_args,observer_args,before_each=refresh)
    record=dict(schema='native-elmo-controller-result-v1',controller_job_id=job,
        producer_spec_sha256=e['sha256'],status='CHILD_RELEASE_ACCEPTED_NOT_COMPLETION',details=result,
        controller_native_completion='UNKNOWN_NOT_USED_TO_CERTIFY_CHILD',created_at=datetime.now(timezone.utc).isoformat())
    write_once(claim+'/RESULT.json',record)
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['plan','submit-controller','release-controller','allocated'])
    parser.add_argument('spec_path'); parser.add_argument('spec_sha'); parser.add_argument('ops_sha')
    args=parser.parse_args(); values=(args.spec_path,args.spec_sha,args.ops_sha)
    if args.operation=='allocated': result=allocated(*values)
    else:
        remote=SSH()
        if args.operation=='plan':
            reader=Reader(remote.read); spec,_=validate(reader,*values)
            result={'argv':controller_argv(spec,*values),'allocates':False,'remote_readset':reader.readset}
        else: result={'submit-controller':submit_controller,'release-controller':release_controller}[args.operation](remote,*values)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
