#!/usr/bin/env python3
"""Bind the maintained runner to its audited single-rank trainer-PID transport."""
import hashlib,os,pathlib,runpy,sys
native=pathlib.Path(os.environ['ICE_CANONICAL_REQUEUE_RUNNER']).resolve(strict=True)
repo=pathlib.Path(os.environ['ICE_TRAINER_REPO']).resolve(strict=True)
assert native==repo/'scripts/ice/ice_requeue_runner.py'
assert hashlib.sha256(native.read_bytes()).hexdigest()==os.environ['ICE_NATIVE_RUNNER_SHA256']
adapter=pathlib.Path(__file__).with_name('trainer_pid_signal.py').resolve(strict=True)
assert hashlib.sha256(adapter.read_bytes()).hexdigest()==os.environ['ICE_TRAINER_SIGNAL_ADAPTER_SHA256']
assert '--scancel' not in sys.argv, 'Conflicting signal route rejected'
index=sys.argv.index('--')
sys.argv[index:index]=['--scancel',str(adapter)]
sys.argv[0]=str(native)
runpy.run_path(str(native),run_name='__main__')
