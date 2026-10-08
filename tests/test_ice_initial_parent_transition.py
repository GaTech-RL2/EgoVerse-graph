import importlib.util
import os
from pathlib import Path
import sys
import unittest

path = Path(os.environ.get('RUNNER_UNDER_TEST', str(Path(__file__).parents[1] / 'scripts/ice/ice_requeue_runner.py')))
spec = importlib.util.spec_from_file_location('initial_parent_runner', path)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)

class ParentTransitionTests(unittest.TestCase):
    def setUp(self):
        self.meta = {'run_id':'parent', 'source_commit':'model', 'split_sha256':'split', 'normalization_sha256':'norm'}
        self.initial = runner.CheckpointInfo(Path('/parent/initial.ckpt'), 'a'*64, 10, 100, 0, 1, dict(self.meta))
        self.high = self.initial.as_dict()
        runner.INITIAL_RUN_TRANSITION = (self.initial, 'parent', 'smoke')
    def tearDown(self):
        runner.INITIAL_RUN_TRANSITION = None
    def child(self, **changes):
        meta = dict(self.meta, run_id='smoke'); meta.update(changes)
        return runner.CheckpointInfo(Path('/child/new.ckpt'), 'b'*64, 14, 100, 0, 2, meta)
    def test_declared_child_preserves_parent_identity(self):
        runner.enforce_high_water(self.child(), self.high)
        self.assertEqual(self.high['metadata']['run_id'], 'parent')
    def test_default_refuses_run_change(self):
        runner.INITIAL_RUN_TRANSITION = None
        with self.assertRaises(SystemExit): runner.enforce_high_water(self.child(), self.high)
    def test_unknown_child_refused(self):
        with self.assertRaises(SystemExit): runner.enforce_high_water(self.child(run_id='other'), self.high)
    def test_source_change_refused(self):
        with self.assertRaises(SystemExit): runner.enforce_high_water(self.child(source_commit='other'), self.high)
    def test_split_change_refused(self):
        with self.assertRaises(SystemExit): runner.enforce_high_water(self.child(split_sha256='other'), self.high)
    def test_normalization_change_refused(self):
        with self.assertRaises(SystemExit): runner.enforce_high_water(self.child(normalization_sha256='other'), self.high)
    def test_different_parent_path_refused(self):
        self.high['path'] = '/another/initial.ckpt'
        with self.assertRaises(SystemExit): runner.enforce_high_water(self.child(), self.high)
    def test_same_step_fork_refused(self):
        child = self.child()
        child = runner.CheckpointInfo(child.path, child.sha256, 10, 100, 0, 2, child.metadata)
        with self.assertRaises(SystemExit): runner.enforce_high_water(child, self.high)
    def test_no_second_identity_transition(self):
        high = self.child().as_dict()
        runner.enforce_high_water(self.child(), high)
        with self.assertRaises(SystemExit): runner.enforce_high_water(self.child(run_id='third'), high)

if __name__ == '__main__': unittest.main()
