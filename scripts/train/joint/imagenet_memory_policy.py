"""Skynet CR_CPU memory semantics, actual headroom and inherited cgroup limits."""
from datetime import datetime,timezone
import hashlib
from pathlib import Path,PurePosixPath
import re

POLICY={'schema':'skynet-cr-cpu-requested-memory-policy-v1','select_type':'select/cons_tres','select_type_parameters':'CR_CPU','task_plugins':['task/affinity','task/cgroup'],'proctrack_type':'proctrack/cgroup','memory_semantics':'requested_not_consumable_reserved','ram_enforcement_configured':False,'cgroup_config_path':'/opt/slurm/configs/cgroup.conf','cgroup_config_sha256':'b094e7de582ba3b28003fe0dfe238ee1a6591a9ab59ed114508ce1b8ab1638b4','minimum_memavailable_kib':{'producer':33554432,'observer':2097152}}

def need(ok,message):
    if not ok:raise ValueError(message)

def digest(text):return hashlib.sha256(text.encode()).hexdigest()

def config_fields(raw):
    need(len(raw.encode())<=1<<20,'Oversized native config');out={}
    for line in raw.splitlines():
        if '=' not in line:continue
        k,v=line.split('=',1);k=k.strip();v=v.strip()
        if k in ('SelectType','SelectTypeParameters','TaskPlugin','ProctrackType'):
            need(k not in out,'Duplicate policy field');out[k]=v
    return out

def check_observation(q,observation,*,capture=False,maximum_age_seconds=None):
    need(q.get('memory_policy')==POLICY,'Unreviewed CR_CPU memory policy')
    need(observation['schema']=='skynet-cr-cpu-memory-observation-v1' and observation['node']==q['node'],'Wrong memory observation node')
    config=config_fields(observation['controller_config'])
    need(set(config)=={'SelectType','SelectTypeParameters','TaskPlugin','ProctrackType'},'Incomplete native memory policy')
    need(config['SelectType']==POLICY['select_type'] and config['SelectTypeParameters']==POLICY['select_type_parameters'] and sorted(config['TaskPlugin'].split(','))==POLICY['task_plugins'] and config['ProctrackType']==POLICY['proctrack_type'],'Controller memory/task policy changed')
    need(digest(observation['cgroup_config'])==POLICY['cgroup_config_sha256'],'Cgroup configuration changed; explicit review required')
    match=re.findall(r'^MemAvailable:\s+([0-9]+)\s+kB\s*$',observation['meminfo'],re.M)
    need(len(match)==1,'Actual /proc MemAvailable required, not FreeMem')
    available=int(match[0]);minimum=POLICY['minimum_memavailable_kib']['observer' if capture else 'producer']
    need(available>=minimum,'Insufficient actual MemAvailable headroom')
    at=datetime.fromisoformat(observation['observed_at_utc']);need(at.tzinfo is not None,'Memory observation time must be UTC-aware')
    if maximum_age_seconds is not None:
        age=(datetime.now(timezone.utc)-at).total_seconds();need(0<=age<=maximum_age_seconds,'Fresh memory/config observation required')
    return dict(memory_semantics=POLICY['memory_semantics'],reserved_memory_mib=None,requested_memory_mib=q['capture_resources' if capture else 'resources']['memory_mib'],memavailable_kib=available,minimum_memavailable_kib=minimum,configured_ram_enforcement=False,controller_config_sha256=digest(observation['controller_config']),cgroup_config_sha256=digest(observation['cgroup_config']))

def cgroup_limits(cgroup_raw,mountinfo_raw,read_text):
    """Read this process and every mounted ancestor; never assume leaf is effective."""
    need(len(cgroup_raw)<=65536 and len(mountinfo_raw)<=1<<20,'Oversized cgroup metadata')
    memberships=[]
    for line in cgroup_raw.splitlines():
        hid,controllers,path=line.split(':',2)
        if controllers=='' or 'memory' in controllers.split(','):
            need(path.startswith('/') and '..' not in PurePosixPath(path).parts and '\\' not in path,'Unsafe cgroup membership')
            memberships.append(('v2' if controllers=='' else 'v1',path))
    observations=[];unknown=[]
    for version,member in memberships:
        mounts=[]
        for line in mountinfo_raw.splitlines():
            left,sep,right=line.partition(' - ')
            if not sep:continue
            l=left.split();r=right.split()
            if len(l)<5 or len(r)<3:continue
            if version=='v2' and r[0]!='cgroup2' or version=='v1' and not(r[0]=='cgroup' and 'memory' in r[2].split(',')):continue
            root,mount=l[3:5]
            if '\\' in root+mount:continue
            try:relative=PurePosixPath(member).relative_to(root)
            except ValueError:continue
            mounts.append((Path(mount),Path(mount)/str(relative)))
        if len(mounts)!=1:unknown.append(version+':unresolved_mount');continue
        mount,current=mounts[0];need(current.is_relative_to(mount),'Cgroup path escaped mount')
        for _ in range(64):
            limitfile=current/('memory.max' if version=='v2' else 'memory.limit_in_bytes')
            try:
                raw=read_text(limitfile).strip();need(raw=='max' or raw.isdecimal(),'Unparseable actual cgroup memory limit')
                value=None if raw=='max' or version=='v1' and int(raw)>=1<<60 else int(raw)
                observations.append(dict(version=version,path=str(limitfile),raw=raw,finite_limit_bytes=value))
            except (OSError,PermissionError) as exc:unknown.append(str(limitfile)+':'+type(exc).__name__)
            if current==mount:break
            current=current.parent
        else:raise ValueError('Unbounded cgroup ancestor chain')
    finite=[x['finite_limit_bytes'] for x in observations if x['finite_limit_bytes'] is not None]
    return dict(schema='allocated-cgroup-memory-limits-v1',membership_raw=cgroup_raw,mountinfo_raw=mountinfo_raw,observations=observations,unresolved=unknown,effective_finite_limit_bytes=min(finite) if finite else None,enforcement='FINITE_ANCESTOR_LIMIT_OBSERVED' if finite else 'UNKNOWN_OR_UNLIMITED_NOT_CERTIFIED')

def check_limit(report,requested_mib):
    need(report.get('schema')=='allocated-cgroup-memory-limits-v1','Wrong allocated cgroup observation')
    rows=report['observations'];values={r['path']:r['raw'] for r in rows}
    need(len(values)==len(rows),'Repeated cgroup limit observation')
    def reopen_recorded(path):
        value=str(path)
        if value in values:return values[value]
        errors=[x.split(':')[-1] for x in report['unresolved'] if x.startswith(value+':')]
        need(len(errors)==1,'Missing actual ancestor observation')
        types={'OSError':OSError,'FileNotFoundError':FileNotFoundError,'PermissionError':PermissionError}
        need(errors[0] in types,'Unknown cgroup read failure type')
        raise types[errors[0]]('Preserved actual cgroup read failure')
    replay=cgroup_limits(report['membership_raw'],report['mountinfo_raw'],reopen_recorded)
    need(replay==report,'Cgroup summary differs from complete preserved ancestor reads')
    finite=[x['finite_limit_bytes'] for x in report['observations'] if x['finite_limit_bytes'] is not None]
    expected=min(finite) if finite else None
    need(report['effective_finite_limit_bytes']==expected,'Inherited limit summary differs')
    need(expected is None or expected>=requested_mib*1024**2,'Actual inherited cgroup memory limit below request')
    return report
