"""Fail-closed allocation gate for coordinator-selected cross-cluster candidates.

Run inside the allocated batch before invoking the existing training launcher.
The coordinator must persist one exclusive winner and reconcile every ungated
candidate before publishing this job/request-specific grant. This module never
self-elects, launches, releases, cancels, or edits another scheduler job.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def write_once(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def identity(request_path, environment):
    path = Path(request_path)
    raw = path.read_bytes()
    request = json.loads(raw)
    policy = request['candidate_gate']
    if policy['schema'] != 'coordinator-candidate-v1':
        raise ValueError('Unknown candidate gate contract')
    job = environment['SLURM_JOB_ID']
    if not job.isdecimal() or environment.get('SLURM_RESTART_COUNT', '0') != '0':
        raise ValueError('A fresh native allocation is required')
    if environment['USER'] != policy['user']:
        raise ValueError('Candidate owner mismatch')
    directory = Path(policy['directory'])
    if not directory.is_absolute() or directory.is_symlink():
        raise ValueError('Gate directory must be an absolute non-symlink path')
    expected = dict(schema='coordinator-candidate-v1', job_id=job,
                    host=policy['host'], user=policy['user'], cell=policy['cell'],
                    round_id=policy['round_id'], request_sha256=hashlib.sha256(raw).hexdigest())
    return policy, expected


def validate_grant(grant, expected):
    if any(grant.get(k) != v for k, v in expected.items()):
        raise ValueError('Grant does not identify this exact candidate')
    if grant.get('decision') != 'TRAIN' or grant.get('losers_reconciled') is not True:
        raise ValueError('Training is not authorized by this grant')
    winner_sha = grant.get('winner_record_sha256', '')
    if len(winner_sha) != 64 or any(c not in '0123456789abcdef' for c in winner_sha):
        raise ValueError('Missing immutable coordinator winner receipt')


def wait_for_grant(request_path, environment=None):
    environment = os.environ if environment is None else environment
    policy, expected = identity(request_path, environment)
    timeout = policy.get('wait_seconds', 600)
    if type(timeout) is not int or not 1 <= timeout <= 900:
        raise ValueError('Gate wait must be bounded to at most 900 seconds')
    directory = Path(policy['directory'])
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / expected['job_id']
    write_once(str(prefix)+'.ready.json', dict(expected, node=environment['SLURMD_NODENAME'],
                                             optimizer_started=False))
    deadline = time.monotonic() + timeout
    while True:
        reject = Path(str(prefix)+'.reject.json')
        grant_path = Path(str(prefix)+'.grant.json')
        if reject.exists() or reject.is_symlink():
            raise RuntimeError('Candidate was rejected before training')
        if grant_path.is_symlink():
            raise ValueError('Symlink grant rejected')
        if grant_path.exists():
            raw = grant_path.read_bytes()
            validate_grant(json.loads(raw), expected)
            write_once(str(prefix)+'.accepted.json', dict(expected,
                       grant_sha256=hashlib.sha256(raw).hexdigest()))
            return
        if time.monotonic() >= deadline:
            raise TimeoutError('No exact coordinator grant; training never started')
        time.sleep(min(2, max(0, deadline-time.monotonic())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request')
    args = parser.parse_args()
    wait_for_grant(args.request)
