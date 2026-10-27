"""Stage an isolated native Slurm client closure inside an actual allocation.

The private host loader/libc are used only by child Slurm clients. They never
replace the pinned container's loader, libc or Torch dependencies.
"""
import argparse
import hashlib
import json
import os
import posixpath
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess

from wrapper_common import bound, need, physical, sha, write_once

LOGICAL = Path('/run/elmo-slurm-client')
INPUT = {'schema': 'native-elmo-client-closure-input-v1',
         'scontrol': '/usr/local/bin/scontrol', 'squeue': '/usr/local/bin/squeue',
         'config': '/etc/slurm/slurm.conf', 'munge_socket': '/run/munge/munge.socket.2'}


def ldd(path):
    result = subprocess.run(['ldd', str(path)], capture_output=True, text=True, timeout=20)
    loader = Path(path).name.startswith('ld-linux-x86-64.so.')
    need((result.returncode == 0 or loader and result.returncode == 1)
         and 'not found' not in result.stdout,
         'Could not close native ELF dependencies: ' + str(path))
    files = []
    for line in result.stdout.splitlines():
        if 'linux-vdso' in line or not line.strip() or loader and line.strip() == 'statically linked':
            continue
        paths = re.findall(r'(?<!\S)(/[^\s()]+)', line)
        need(len(paths) == 1, 'Unrecognized native ELF dependency line')
        files.append(Path(paths[0]).resolve(strict=True))
    return files


def configuration(text, observed=None):
    values = {}
    canonical = {key.casefold(): key for key in ('PluginDir', 'AuthType', 'CredType', 'SelectType')}
    for raw in text.splitlines():
        line = raw.split('#', 1)[0].strip()
        if not line:
            continue
        need(not re.match(r'(?i)^include\b', line), 'Slurm config includes need an explicit reviewed closure')
        match = re.match(r'^([A-Za-z0-9_]+)\s*=\s*([^\s]+)', line)
        if match and match[1].casefold() in canonical:
            key = canonical[match[1].casefold()]
            need(key not in values, 'Duplicate native Slurm client configuration')
            values[key] = match[2]
    observed = {} if observed is None else observed
    for key in ('AuthType', 'CredType', 'SelectType', 'PluginDir'):
        if key not in values:
            need(key in observed, 'Missing actual native default observation: ' + key)
            values[key] = observed[key]
        elif key in observed:
            need(values[key] == observed[key], 'Configured and actual native value differ: ' + key)
    need(values['AuthType'] == 'auth/munge' and values['CredType'] == 'cred/munge'
         and values['SelectType'] in {'select/cons_tres', 'select/cons_res', 'select/linear'}
         and values['PluginDir'] == '/usr/local/lib/slurm',
         'Native client auth/cred/select/plugin layout changed')
    plugins = {value.replace('/', '_') + '.so' for key, value in values.items()
               if key in {'AuthType', 'CredType', 'SelectType'}}
    replaced, count = re.subn(r'(?im)^(\s*PluginDir\s*=\s*)[^\s#]+',
                             lambda m: m[1] + str(LOGICAL / 'plugins'), text)
    need(count <= 1, 'Native private client PluginDir transformation is ambiguous')
    if not count:
        replaced = text.rstrip('\n') + '\nPluginDir=' + str(LOGICAL / 'plugins') + '\n'
    return plugins, replaced


def copy_stable(source, destination, *, executable=False):
    source = Path(source).resolve(strict=True)
    need(source.is_file() and not source.is_symlink()
         and source.parts[1] in {'usr', 'lib', 'lib64', 'opt', 'etc'},
         'Unbounded native client member')
    before = source.stat()
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    need(before.st_size < 256_000_000, 'Unbounded native client member size')
    digest = sha(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as incoming, destination.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing, 8 << 20)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    after = source.stat()
    need(identity == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
         and sha(destination) == digest == sha(source), 'Native client source changed while copying')
    destination.chmod(0o555 if executable else 0o444)
    return {'source': str(source), 'sha256': digest, 'bytes': after.st_size}


def validate_elf_metadata(metadata, aliases, destinations):
    """Reject search paths that could outrank the private copied closure."""
    for member, record in metadata.items():
        need(all('/' not in name and name in aliases for name in record['needed']),
             'Native ELF dependency does not resolve to a copied alias')
        for entry in record['search_paths']:
            need(bool(entry), 'Empty native ELF search entry would use working directory')
            expanded = entry.replace('${ORIGIN}', '$ORIGIN')
            for parent in destinations[member]:
                path = posixpath.normpath(expanded.replace('$ORIGIN', str(parent)))
                need((path == str(LOGICAL) or path.startswith(str(LOGICAL) + '/')) and '$' not in path,
                     'Unreviewed native ELF RPATH/RUNPATH outside private capsule')


def stage(spec, request, job, attempt, *, dependency_reader=ldd):
    policy = spec['runtime']['native_client']
    need({key: policy.get(key) for key in INPUT} == INPUT,
         'Explicit reviewed native client source paths required')
    scratch = physical(request['lambda_runtime']['scratch_root'])
    need(scratch.is_dir() and scratch.stat().st_uid == os.getuid(),
         'Owned current node-local scratch required for native client staging')
    socket = Path(policy['munge_socket'])
    need(socket.is_socket() and socket.parent.resolve() == Path('/run/munge'),
         'Exact native munge socket unavailable; no keys may be mounted')
    config_path = Path(policy['config']).resolve(strict=True)
    need(config_path.stat().st_size < 8_000_000, 'Unbounded native client config')
    before_config = config_path.read_bytes()
    current = subprocess.run([policy['scontrol'], 'show', 'config'],
                             capture_output=True, text=True, timeout=20)
    need(current.returncode == 0, 'Actual native client defaults query failed')
    observed = {}
    for key in ('PluginDir', 'AuthType', 'CredType', 'SelectType'):
        values = re.findall(r'(?im)^\s*' + key + r'\s*=\s*(\S+)', current.stdout)
        need(len(values) == 1, 'Ambiguous actual native default: ' + key)
        observed[key] = values[0]
    plugins, private_config = configuration(before_config.decode(), observed)
    roots = {'bin/scontrol': Path(policy['scontrol']), 'bin/squeue': Path(policy['squeue'])}
    roots.update({'plugins/' + name: Path('/usr/local/lib/slurm') / name for name in plugins})
    for root in roots.values():
        need(root.resolve(strict=True).is_file(), 'Missing exact native executable/plugin')
    dependencies = set()
    pending = [path.resolve(strict=True) for path in roots.values()]
    inspected = set()
    while pending:
        member = pending.pop()
        if member in inspected:
            continue
        inspected.add(member)
        need(len(inspected) <= 128, 'Native client dependency closure exceeded bound')
        for dependency in dependency_reader(member):
            dependency = dependency.resolve(strict=True)
            need(dependency.parts[1] in {'usr', 'lib', 'lib64', 'opt'}, 'Unbounded native dependency')
            dependencies.add(dependency)
            if dependency not in inspected:
                pending.append(dependency)
    # Preserve requested SONAME aliases as well as canonical files. ldd paths are
    # resolved above, so map known SONAMEs from each raw ELF NEEDED entry using
    # readelf rather than guessing a versioned filename.
    aliases = {member.name: member for member in dependencies}
    metadata = {}
    for member in inspected:
        result = subprocess.run(['readelf', '-d', str(member)], capture_output=True, text=True, timeout=20)
        need(result.returncode == 0, 'Native ELF metadata inspection failed')
        metadata[member] = {
            'needed': re.findall(r'\(NEEDED\).*?\[([^]]+)\]', result.stdout),
            'search_paths': [entry for value in re.findall(r'\((?:RPATH|RUNPATH)\).*?\[([^]]*)\]', result.stdout)
                             for entry in value.split(':')],
        }
        sonames = re.findall(r'\(SONAME\).*?\[([^]]+)\]', result.stdout)
        for name in sonames:
            need('/' not in name and name not in {'.', '..'}, 'Unsafe native ELF SONAME')
            need(name not in aliases or aliases[name] == member, 'Conflicting native ELF SONAME')
            aliases[name] = member
    loader = next((path for path in dependencies if path.name.startswith('ld-linux-x86-64.so.')), None)
    need(loader is not None, 'Exact native ELF loader was not discovered')
    destinations = {member: set() for member in inspected}
    for relative, source in roots.items():
        destinations[source.resolve(strict=True)].add((LOGICAL / relative).parent)
    for name, member in aliases.items():
        destinations[member].add(LOGICAL / 'lib')
    validate_elf_metadata(metadata, aliases, destinations)
    need(sum(path.stat().st_size for path in [*roots.values(), *aliases.values()]) < 1_000_000_000,
         'Native client closure exceeds reviewed one-gigabyte bound')
    parent = scratch / 'native-clients'
    parent.mkdir(exist_ok=True)
    need(parent.resolve() == parent and parent.stat().st_uid == os.getuid(), 'Unsafe native capsule parent')
    capsule = parent / job
    capsule.mkdir(exist_ok=False)
    records = {}
    for relative, source in sorted(roots.items()):
        records[relative] = copy_stable(source, capsule / relative, executable=relative.startswith('bin/'))
    # Duplicate aliases are regular byte copies, not links outside the capsule.
    for name, source in sorted(aliases.items()):
        records['lib/' + name] = copy_stable(source, capsule / 'lib' / name)
    if 'lib/ld-linux-x86-64.so.2' not in records:
        records['lib/ld-linux-x86-64.so.2'] = copy_stable(loader, capsule / 'lib/ld-linux-x86-64.so.2', executable=True)
    else:
        (capsule / 'lib/ld-linux-x86-64.so.2').chmod(0o555)
    config = capsule / 'etc/slurm.conf'
    config.parent.mkdir()
    with config.open('xb') as stream:
        stream.write(private_config.encode())
        stream.flush()
        os.fsync(stream.fileno())
    config.chmod(0o444)
    need(config_path.read_bytes() == before_config, 'Native config changed during capsule preparation')
    records['etc/slurm.conf'] = {'source': str(config_path), 'sha256': sha(config),
        'original_sha256': hashlib.sha256(before_config).hexdigest(),
        'only_transformation': 'PluginDir=/run/elmo-slurm-client/plugins'}
    lines = ''.join(f'{sha(capsule / name)}  {name}\n' for name in sorted(records))
    manifest = capsule / 'FILES.sha256'
    manifest.write_text(lines)
    manifest.chmod(0o444)
    digest = sha(manifest)
    expected = policy.get('expected_manifest_sha256')
    prior_binding = policy.get('prior_container_observation')
    if spec['mode'] in ('diagnostic', 'finalize', 'full'):
        need(prior_binding is not None and expected is not None,
             'Actual prior container authentication required after CPU inspection')
        prior = bound(prior_binding)
        need(prior['schema'] == 'native-elmo-container-auth-v1'
             and prior['status'] == 'CONTAINER_NATIVE_AUTHENTICATED'
             and prior['node'] == spec['native_contract']['NodeList']
             and prior['original_parent_job_id'] == str(spec['original_parent_job_id'])
             and prior['parent_request'] == spec['parent_request']
             and prior['native_client_manifest_sha256'] == expected,
             'Prior actual container proof has a different parent/node/client')
    need(expected is None or digest == expected, 'Changed exact native client closure')
    record = {'schema': 'native-elmo-client-stage-v1', 'job_id': job,
        'phase': spec['mode'], 'parent_request': spec['parent_request'],
        'capsule': str(capsule), 'logical_container_root': str(LOGICAL),
        'manifest_sha256': digest, 'members': records,
        'container_native_query_certified': False, 'munge_keys_read_or_copied': False}
    write_once(attempt / 'native-client-stage.json', record)
    root = Path(spec['root'])
    executable_wrappers(root)
    mounts = ','.join((str(capsule) + ':' + str(LOGICAL) + ':ro',
                       str(root / 'native_scontrol.sh') + ':/usr/local/bin/scontrol:ro',
                       str(root / 'native_squeue.sh') + ':/usr/local/bin/squeue:ro',
                       '/run/munge:/run/munge:ro'))
    return {'ELMO_NATIVE_CLIENT_HOST_ROOT': str(capsule), 'ELMO_NATIVE_CLIENT_ROOT': str(LOGICAL),
            'ELMO_NATIVE_CLIENT_MANIFEST_SHA256': digest, 'ELMO_NATIVE_MOUNTS': mounts}


def executable_wrappers(root):
    """File bind mounts preserve mode bits; both native entry points must execute."""
    root = physical(root)
    for name in ('native_scontrol.sh', 'native_squeue.sh'):
        source = physical(root / name)
        need(source.is_file() and source.stat().st_mode & 0o111 and os.access(source, os.X_OK),
             'Native wrapper bind mount is not executable: ' + name)


def main(argv=None):
    from execution_spec import load_execution
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec', type=Path)
    parser.add_argument('spec_sha256')
    parser.add_argument('mode')
    parser.add_argument('manifest_sha256')
    args = parser.parse_args(argv)
    spec, request_path, request, job, claim = load_execution(
        args.spec, args.spec_sha256, args.mode, args.manifest_sha256)
    parent = bound(spec['parent_request'])
    attempt = Path(spec['root']) / 'status' / f'{spec["mode"]}-{job}.attempt'
    need((attempt / 'host-native.json').is_file(), 'Host native authentication required first')
    need(not any('key' in p.name.lower() for p in Path('/run/munge').iterdir()),
         'Socket directory contains key-named entries; refuse any credential-key mount')
    exports = stage(spec, parent, job, attempt)
    print('\n'.join('export ' + key + '=' + shlex.quote(value) for key, value in exports.items()))


if __name__ == '__main__':
    main()
