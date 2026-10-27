"""Once-only native GPU full submitter. Default never contacts the cluster."""
import argparse,json,os,re,shlex,subprocess
from datetime import datetime,timezone
from pathlib import Path
from bounded_census_io import CensusTransportError,MAX_COMMAND_BYTES
from guard_full import REMOTE,REQUEST_NAME,CAPTURE_NAME,check_native,fields,need,sha,validate,verify_smoke,verify_review,same_attempt,retry_evidence


def command_plan(root):
    root=Path(root).resolve();request=root/'ops/requests'/(REQUEST_NAME+'.json')
    q=json.loads(request.read_text());b=json.loads((root/'bindings.json').read_text());template=json.loads((root/'REQUEST_TEMPLATE.json').read_text())
    ready=validate(q,b,template,bound=False);manifest=root/'FILES.sha256';seen=set()
    for line in manifest.read_text().splitlines():
        digest,name=line.split('  ',1);path=root/name
        need(name not in seen and path.resolve()==path and path.is_file() and path.is_relative_to(root) and sha(path)==digest,'Changed static packet closure');seen.add(name)
    required={'bounded_census_io.py','prelaunch_stat_retry.py','host_hashlib_compat.py','prelaunch_retry.py','prelaunch_timeout_retry.py','prelaunch_held_retry.py','live_retry_census.py','early_native_capture.py','completed_smoke_reader.py','guard_full.py','native_guard.py','submit_full_once.py','full.sbatch','capture-native.sbatch','bindings.json','REQUEST_TEMPLATE.json','required-directories.json','inputs/smoke-candidate-files.sha256','inputs/full-resolved.yaml','ops/FILES.sha256','ops/requests/'+REQUEST_NAME+'.json'}
    pending=[root/'ops/FILES.sha256'];visited=set()
    while pending:
        m=pending.pop();need(m not in visited,'Operations manifest cycle');visited.add(m)
        for line in m.read_text().splitlines():
            digest,name=line.split('  ',1);f=m.parent/name
            need(f.resolve()==f and f.is_file() and f.is_relative_to(root) and sha(f)==digest,'Changed maintained operation')
            required.add(str(f.relative_to(root)))
            if f.name=='FILES.sha256':pending.append(f)
    if ready:
        retry_evidence(root,q)
        verify_smoke(root,q,b)
        required.update('proofs/smoke/'+line.split('  ',1)[1] for line in (root/'SMOKE_EVIDENCE.sha256').read_text().splitlines())
        required.add('SMOKE_EVIDENCE.sha256')
        from prelaunch_retry import EVIDENCE_HASHES
        required.update('proofs/prelaunch-v1/'+p for p in EVIDENCE_HASHES)
        required.add('proofs/prelaunch-held-v1/EVIDENCE.json')
        required.update('proofs/prelaunch-held-v1/'+p for p in json.loads((root/'proofs/prelaunch-held-v1/EVIDENCE.json').read_text()))
    need(required<=seen,'Deployment manifest omits required runtime/proof files')
    request_sha=sha(request);remote=str(REMOTE)
    args=['/usr/local/bin/sbatch','--parsable','--hold','--no-requeue','--account=loaner','--qos=loaner-normal','--partition=rl2',
          '--comment=elmo-full:'+request_sha,'--job-name='+REQUEST_NAME,'--nodelist=gpu21','--nodes=1','--ntasks=1',
          '--cpus-per-task=32','--mem=192G','--gres=gpu:h100:4','--time=7-00:00:00','--export=NONE','--chdir='+remote,
          '--output='+remote+'/logs/%x-%j.out','--error='+remote+'/logs/%x-%j.err',remote+'/full.sbatch',remote,request_sha,sha(manifest)] if ready else None
    return dict(status='SMOKE_BOUND_REVIEW_DEPLOYMENT_REQUIRED' if ready else 'NON_EXECUTABLE_UNBOUND',parent_argv=args,
         request_sha256=request_sha,manifest_sha256=sha(manifest),remote_root=remote,local_root=str(root),
         claim=remote+'/claims/'+request_sha,request_claim=remote+'/request-claims/'+REQUEST_NAME,default_allocates=False)

class Remote:
    def __init__(self, host):
        self.host = host
        self.socket='/tmp/elmo-gpu-full-'+str(os.getuid())+'-'+str(os.getpid())+'.sock'
        need(not Path(self.socket).exists(),'Fresh SSH control socket required')
    def run(self, command, *, mutation=False, timeout=55):
        need(len(command.encode())<MAX_COMMAND_BYTES,'Remote command exceeds4KiB')
        need(timeout in {55,120,600} and (not mutation or timeout==55),'Only bounded read-only inspections may extend their deadline')
        try:
            p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10','-o','ControlMaster=auto','-o','ControlPersist=60','-o','ControlPath='+self.socket, self.host, command], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise CensusTransportError(('ACCEPTANCE/RELEASE UNKNOWN' if mutation else 'Transport failed') + '; inspect exact claim/native state; never retry blindly',stdout=exc.stdout or '',stderr=exc.stderr or '',timed_out=True) from exc
        if p.returncode:
            raise CensusTransportError(('ACCEPTANCE/RELEASE UNKNOWN' if mutation else 'Remote inspection failed') + ': ' + p.stderr[-1500:] + p.stdout[-1500:],stdout=p.stdout,stderr=p.stderr,returncode=p.returncode)
        return p.stdout
    def write_once(self, path, value):
        raw=value.encode();need(len(raw)<=8<<20,'Oversized immutable proof payload')
        p=Path(path);need(p.is_absolute() and p.is_relative_to(REMOTE) and p.parts[len(REMOTE.parts)] in {'claims','request-claims'},'Proof write outside fresh claim roots')
        quote=shlex.quote;parent=quote(str(p.parent));target=quote(str(p));digest=__import__('hashlib').sha256(raw).hexdigest()
        script='set -euo pipefail; umask 077; test -d '+parent+'; test "$(readlink -e -- '+parent+')" = '+parent+'; test "$(stat -c %u -- '+parent+')" = 41003; test ! -e '+target+'; test ! -L '+target+'; set -o noclobber; head -c '+str(len(raw)+1)+' > '+target+'; test "$(stat -c %s -- '+target+')" = '+str(len(raw))+'; printf "%s  %s\\n" '+quote(digest)+' '+target+' | sha256sum --check --status; /usr/bin/sync -f '+target+'; printf "%s\\n" '+quote(digest)
        command='bash -c '+quote(script);need(len(command.encode())<4096,'Oversized SSH proof command')
        try:
            result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ControlMaster=auto','-o','ControlPersist=60','-o','ControlPath='+self.socket,'-o','ForwardAgent=no',self.host,command],input=raw,capture_output=True,timeout=55)
        except (subprocess.TimeoutExpired,OSError) as exc:
            raise RuntimeError('STREAM_WRITE_UNKNOWN; inspect exact existing file and claims; never retry') from exc
        need(result.returncode==0 and result.stdout==(digest+'\n').encode() and result.stderr==b'','STREAM_WRITE_UNKNOWN; preserve partial file and inspect exact claim; never retry')

def check_capacity(raw):
    f=fields(raw);need(f.get('NodeName')=='gpu21' and not any(x in f.get('State','') for x in ('DOWN','DRAIN','FAIL','MAINT')),'Node21 unavailable')
    need(int(f['CPUTot'])-int(f['CPUAlloc'])>=32 and int(f['RealMemory'])-int(f['AllocMem'])>=192*1024,'Insufficient32CPU/192GiB')
    def tres(value):return dict(part.split('=',1) for part in value.split(',') if '=' in part)
    cfg,alloc=tres(f['CfgTRES']),tres(f.get('AllocTRES',''))
    # This controller omits GPU TRES accounting. Native Gres/GresUsed are
    # authoritative inventory; reject ambiguity or conflicts when TRES exists.
    total=re.fullmatch(r'gpu:h100:(\d+)',f.get('Gres',''))
    used=re.fullmatch(r'gpu:h100:(\d+)\(IDX:[A-Za-z0-9,/-]+\)',f.get('GresUsed',''))
    need(total is not None and used is not None,'Unambiguous native H100 inventory required')
    total_count,used_count=int(total[1]),int(used[1])
    need(0<=used_count<=total_count and total_count-used_count>=4,'Four free native H100s required')
    for key in ('gres/gpu:h100','gres/gpu'):
        if key in cfg:need(int(cfg[key])==total_count,'Conflicting configured GPU inventory')
        if key in alloc:need(int(alloc[key])==used_count,'Conflicting allocated GPU inventory')
    need('rl2' in f.get('Partitions','').split(','),'Node outside authorized partition')


def check_queue(raw, allowed=()):
    for line in raw.splitlines():
        row=line.split('|');need(len(row)==5,'Malformed complete owned queue')
        job,name,user,state,node=row
        need(user=='paphiwetsa3','Wrong queue owner')
        need('qknorm-swiglu-rmsnorm' not in name or job in allowed,'Existing campaign writer/producer')

def inspect_claims(remote,plan,own=False):
    quote=shlex.quote;root=plan['remote_root']
    raw=remote.run('find '+' '.join(quote(root+'/'+name) for name in ['claims','request-claims','ops/claims','ops/request-claims'])+' -mindepth 1 -maxdepth 1 -print')
    expected={plan['claim'],plan['request_claim']} if own else set()
    need(set(raw.splitlines())==expected,'Prior/partial wrapper or maintained scientific claims; inspect, never retry')

def retry_census(remote,plan,*,phase='PRE_SUBMIT',parent=None,capture=None):
    from live_retry_census import collect
    request=json.loads((Path(plan['local_root'])/'ops/requests'/(REQUEST_NAME+'.json')).read_text())
    return collect(lambda command: remote.run(command,timeout=120),Path(plan['local_root'])/'proofs/prelaunch-held-v1',request,plan['request_sha256'],
                   phase=phase,receipt_dir=Path(plan['local_root'])/'output'/('census-'+phase),actor='CONTROLLER_LOCAL',current_job=parent,current_capture=capture)


def prerelease(remote,plan,parent,capture):
    inspect_claims(remote,plan,own=True);quote=shlex.quote;claim=plan['claim']
    queue=remote.run("/usr/local/bin/squeue -h -u paphiwetsa3 -o '%i|%j|%u|%T|%N'")
    check_queue(queue,allowed=(parent,capture))
    node=remote.run('/usr/local/bin/scontrol show node gpu21 -o -dd');check_capacity(node)
    held=remote.run('/usr/local/bin/scontrol show job '+quote(parent)+' -o -dd')
    current=check_native(held,job=parent,state='PENDING',request_sha=plan['request_sha256'])
    original=check_native(remote.run('cat '+quote(claim+'/HELD_NATIVE.txt')),job=parent,state='PENDING',request_sha=plan['request_sha256'])
    same_attempt(original,current)
    observer=remote.run('/usr/local/bin/scontrol show job '+quote(capture)+' -o -dd')
    current=check_native(observer,job=capture,state='PENDING',capture=True,parent=parent,request_sha=plan['request_sha256'])
    original=check_native(remote.run('cat '+quote(claim+'/CAPTURE_PENDING_NATIVE.txt')),job=capture,state='PENDING',capture=True,parent=parent,request_sha=plan['request_sha256'])
    same_attempt(original,current)
    retry_snapshot=retry_census(remote,plan,phase='PRE_RELEASE',parent=parent,capture=capture)
    remote.write_once(claim+'/RELEASE_HELD_NATIVE.txt',held)
    remote.write_once(claim+'/RELEASE_CAPTURE_PENDING_NATIVE.txt',observer)
    remote.write_once(claim+'/RELEASE_ATTEMPT.json',json.dumps(dict(parent=parent,capture=capture,
         request_sha256=plan['request_sha256'],queue=queue,node=node,prelaunch_retry_census=retry_snapshot,at_utc=datetime.now(timezone.utc).isoformat()))+'\n')

def apply(plan, host):
    need(plan.get('parent_argv') is not None,'UNBOUND request cannot contact remote')
    verify_review(Path(plan['local_root']),plan['request_sha256'],plan['manifest_sha256'])
    remote = Remote(host); q = shlex.quote; root = plan['remote_root']; claim = plan['claim']
    remote.run('bash -c ' + q('set -euo pipefail; cd ' + q(root) + '; printf "%s  %s\\n" ' + q(plan['request_sha256']) + ' ops/requests/'+REQUEST_NAME+'.json | sha256sum --check --status; printf "%s  %s\\n" ' + q(plan['manifest_sha256']) + ' FILES.sha256 | sha256sum --check --status; sha256sum --check --status FILES.sha256'),timeout=600)
    remote.run('bash -c '+q('set -euo pipefail; '+'; '.join('test -d '+q(root+'/'+name) for name in ['logs','claims','request-claims','proofs','output','ops/claims','ops/request-claims','ops/status','ops/logs'])))
    for name in ('root-review.json','deployment-readback.json'):
        digest=sha(Path(plan['local_root'])/name)
        remote.run('bash -c '+q('set -euo pipefail; printf "%s  %s\\n" '+q(digest)+' '+q(root+'/'+name)+' | sha256sum --check --status'))
    queue = remote.run("/usr/local/bin/squeue -h -u paphiwetsa3 -o '%i|%j|%u|%T|%N'")
    check_queue(queue); inspect_claims(remote,plan);retry_snapshot=retry_census(remote,plan)
    need(remote.run('id -u').strip()=='41003','Wrong configured native owner')
    node = remote.run('/usr/local/bin/scontrol show node gpu21 -o -dd'); check_capacity(node)
    # Both exclusive claims must be fresh. A partially created claim is evidence,
    # never a reason to retry this function.
    remote.run('bash -c ' + q('set -euo pipefail; mkdir ' + q(plan['request_claim']) + '; mkdir ' + q(claim)+'; /usr/bin/sync -f '+q(root)),mutation=True)
    remote.write_once(claim + '/PREPARED.json', json.dumps(dict(request_sha256=plan['request_sha256'], manifest_sha256=plan['manifest_sha256'], parent_argv=plan['parent_argv'], prepared_at_utc=datetime.now(timezone.utc).isoformat(), queue=queue, node=node, prelaunch_retry_census=retry_snapshot), indent=2) + '\n')
    remote.write_once(plan['request_claim']+'/REQUEST_SHA256',plan['request_sha256']+'\n')
    def submit(argv, label):
        remote.write_once(claim+'/'+label+'.argv.json',json.dumps(argv)+'\n')
        command = 'set -uo pipefail; set -o noclobber; ' + shlex.join(argv) + ' > ' + q(claim + '/' + label + '.stdout') + ' 2> ' + q(claim + '/' + label + '.stderr') + '; code=$?; printf "%s\\n" "$code" > ' + q(claim + '/' + label + '.exit') + '; /usr/bin/sync -f '+q(claim)+' || exit 74; cat ' + q(claim + '/' + label + '.stdout') + '; exit "$code"'
        raw = remote.run('bash -c ' + q(command), mutation=True).strip()
        need(raw.isdecimal(), 'Submission acceptance unknown; inspect claim/native state')
        return raw
    parent = submit(plan['parent_argv'], 'parent-sbatch')
    remote.write_once(claim + '/PARENT_JOB_ID', parent + '\n')
    held = remote.run('/usr/local/bin/scontrol show job ' + q(parent) + ' -o -dd')
    remote.write_once(claim + '/HELD_NATIVE.txt', held); check_native(held, job=parent, state='PENDING',request_sha=plan['request_sha256'])
    capture_args = ['/usr/local/bin/sbatch', '--parsable', '--no-requeue', '--account=loaner', '--qos=loaner-normal', '--partition=rl2',
                    '--job-name=elmo-native-qknorm-swiglu-rmsnorm-full-v5', '--nodelist=gpu21', '--nodes=1', '--ntasks=1', '--cpus-per-task=2',
                    '--mem=1G', '--gres=none', '--time=00:10:00', '--export=NONE', '--dependency=afterany:' + parent, '--comment=elmo-full-observer:'+plan['request_sha256']+':'+parent,
                    '--chdir=' + root, '--output=' + root + '/logs/%x-%j.out', '--error=' + root + '/logs/%x-%j.err',
                    root + '/capture-native.sbatch', root, parent, plan['request_sha256'], plan['manifest_sha256']]
    capture = submit(capture_args, 'capture-sbatch')
    remote.write_once(claim + '/CAPTURE_JOB_ID', capture + '\n')
    pending = remote.run('/usr/local/bin/scontrol show job ' + q(capture) + ' -o -dd')
    remote.write_once(claim + '/CAPTURE_PENDING_NATIVE.txt', pending)
    check_native(pending, job=capture, state='PENDING', capture=True, parent=parent,request_sha=plan['request_sha256'])
    prerelease(remote,plan,parent,capture)
    # No retry after this operation if SSH outcome is unknown.
    release = remote.run('/usr/local/bin/scontrol release ' + q(parent), mutation=True)
    remote.write_once(claim + '/RELEASE_ACCEPTED.json', json.dumps(dict(parent=parent, capture=capture, stdout=release, released_at_utc=datetime.now(timezone.utc).isoformat()), indent=2) + '\n')
    return dict(parent=parent, capture=capture, claim=claim, status='RELEASE_ACCEPTED_NOT_COMPLETION')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--apply',action='store_true');parser.add_argument('--host',default='lambda');args=parser.parse_args();plan=command_plan(args.root)
    print(json.dumps(apply(plan,args.host) if args.apply else plan,indent=2))
