"""Real temporary-file history counterfactuals; no checkpoint content is loaded."""

import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


from scripts.train.joint.crash_history import collect_milestones, collect_validation, history_segments


METRICS = ["val/libero_gripper/forecast/image_normalized_mse",
           "val/libero_gripper/completion/image_normalized_mse"]


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.parent = self.request("parent")
        self.write_checkpoint(self.parent, 20000)
        self.write_csv(self.parent, 0, [dict(step=19999, **dict.fromkeys(METRICS, .01))])
        parent_binding = self.binding("parent.json", self.parent)
        self.inspection = dict(schema="joint-crash-checkpoint-inspection-v1", status="PASS",
                               parent_request_sha256=parent_binding["sha256"], run_id="same-run",
                               source_revision="source", global_step=20000)
        self.child = self.request("child")
        self.child["crash_recovery"] = dict(
            parent_request=parent_binding, inspection=self.binding("inspection.json", self.inspection))
        self.write_checkpoint(self.child, 40000)
        self.write_csv(self.child, 20000, [dict(step=39999, **dict.fromkeys(METRICS, .02))])

    def tearDown(self):
        self.temp.cleanup()

    def request(self, name):
        q = dict(run_id="same-run", source_revision="source", world_size=4,
                 active_streams=["image"], target_steps=40000,
                 output=str(self.root / name), local_logs=str(self.root / (name + "-logs")))
        (Path(q["output"]) / "checkpoints").mkdir(parents=True)
        return q

    def binding(self, name, data):
        path = self.root / name
        path.write_text(json.dumps(data))
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def write_checkpoint(self, q, step):
        path = Path(q["output"]) / "checkpoints" / f"epoch_epoch=0001-step_step={step:09d}.ckpt"
        path.write_text("synthetic file, never loaded as a checkpoint")
        return path

    def write_csv(self, q, start, rows):
        path = Path(q["local_logs"]) / "csv" / f"step{start:09d}" / "metrics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(dict.fromkeys(["step", *METRICS, *(key for row in rows for key in row)]))
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader(); writer.writerows(rows)
        return path

    def test_merges_two_immutable_checkpoint_and_validation_segments(self):
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(set(collect_milestones(self.child)), {20000, 40000})
        self.assertEqual(set(collect_validation(self.child)), {20000, 40000})
        self.assertEqual([(s["start_step"], s["through_step"])
                          for s in history_segments(self.child)], [(0, 20000), (20000, 40000)])
        self.assertTrue(all(path.read_bytes() == data for path, data in before.items()))

    def test_ignores_preserved_parent_events_beyond_restored_checkpoint(self):
        self.write_csv(self.parent, 0, [dict(step=19999, **dict.fromkeys(METRICS, .01)),
                                       dict(step=39999, **dict.fromkeys(METRICS, .07))])
        self.assertEqual(collect_validation(self.child)[40000][METRICS[0]], .02)

    def test_rejects_unreconciled_newer_parent_checkpoint(self):
        self.write_checkpoint(self.parent, 40000)
        with self.assertRaisesRegex(ValueError, "Unreconciled"):
            collect_milestones(self.child)

    def test_rejects_repeated_boundary_save_in_new_tree(self):
        self.write_checkpoint(self.child, 20000)
        with self.assertRaisesRegex(ValueError, "repeated resume-boundary save"):
            collect_milestones(self.child)

    def test_rejects_checkpoint_symlink(self):
        path = self.write_checkpoint(self.child, 40000)
        target = self.root / "external.ckpt"; target.write_text("retained")
        path.unlink(); path.symlink_to(target)
        with self.assertRaisesRegex(ValueError, "mutable retained"):
            collect_milestones(self.child)

    def test_last_alias_does_not_replace_immutable_milestone(self):
        path = self.write_checkpoint(self.child, 40000)
        path.rename(path.parent / "other-file.bin")
        (path.parent / "last.ckpt").symlink_to(path.parent / "other-file.bin")
        with self.assertRaisesRegex(ValueError, "Missing retained checkpoint"):
            collect_milestones(self.child)

    def test_rejects_repeated_boundary_validation(self):
        self.write_csv(self.child, 20000, [dict(step=19999, **dict.fromkeys(METRICS, .01)),
                                         dict(step=39999, **dict.fromkeys(METRICS, .02))])
        with self.assertRaisesRegex(ValueError, "Repeated validation at"):
            collect_validation(self.child)

    def test_rejects_nonfinite_metric(self):
        self.write_csv(self.child, 20000, [dict(step=39999, **{METRICS[0]: "nan", METRICS[1]: .01})])
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            collect_validation(self.child)

    def test_rejects_missing_stream_metric(self):
        self.write_csv(self.child, 20000, [dict(step=39999, **{METRICS[0]: .01})])
        with self.assertRaisesRegex(ValueError, "Incomplete scheduled"):
            collect_validation(self.child)

    def test_rejects_unscheduled_validation(self):
        self.write_csv(self.child, 20000, [dict(step=29999, **dict.fromkeys(METRICS, .02))])
        with self.assertRaisesRegex(ValueError, "20k optimizer cadence"):
            collect_validation(self.child)

    def test_rejects_duplicate_validation_even_if_metric_equal(self):
        row = dict(step=39999, **dict.fromkeys(METRICS, .02))
        self.write_csv(self.child, 20000, [row, row])
        with self.assertRaisesRegex(ValueError, "Repeated validation metric"):
            collect_validation(self.child)

    def test_merges_disjoint_metric_rows_at_one_event(self):
        self.write_csv(self.child, 20000, [dict(step=39999, **{METRICS[0]: .01}),
                                         dict(step=39999, **{METRICS[1]: .02})])
        self.assertEqual(set(collect_validation(self.child)[40000]), set(METRICS))

    def test_mixed_csv_step_conventions_fail_in_one_event(self):
        self.write_csv(self.child, 20000, [dict(step=39999, **{METRICS[0]: .01}),
                                         dict(step=40000, **{METRICS[1]: .02})])
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            collect_validation(self.child)

    def test_zero_progress_attempt_kept_in_chain(self):
        child, _ = self.with_zero_progress_parent()
        self.assertEqual(len(history_segments(child)), 3)
        self.assertEqual(set(collect_milestones(child)), {20000, 40000})

    def with_zero_progress_parent(self):
        failed = self.request("zero-progress")
        failed["crash_recovery"] = copy.deepcopy(self.child["crash_recovery"])
        self.write_csv(failed, 20000, [])
        child = copy.deepcopy(self.child)
        binding = self.binding("zero.json", failed)
        inspection = {**self.inspection, "parent_request_sha256": binding["sha256"]}
        child["crash_recovery"] = dict(parent_request=binding,
                                       inspection=self.binding("zero-inspect.json", inspection))
        return child, failed

    def test_zero_owned_attempt_may_have_no_checkpoint_directory(self):
        child, failed = self.with_zero_progress_parent()
        (Path(failed["output"]) / "checkpoints").rmdir()
        self.assertEqual(set(collect_milestones(child)), {20000, 40000})

    def test_zero_owned_attempt_may_have_no_csv_or_checkpoint_directory(self):
        child, failed = self.with_zero_progress_parent()
        (Path(failed["output"]) / "checkpoints").rmdir()
        (Path(failed["local_logs"]) / "csv" / "step000020000" / "metrics.csv").unlink()
        self.assertEqual(set(collect_milestones(child)), {20000, 40000})
        self.assertEqual(set(collect_validation(child)), {20000, 40000})

    def test_zero_owned_existing_checkpoint_is_not_silently_ignored(self):
        child, failed = self.with_zero_progress_parent()
        self.write_checkpoint(failed, 20000)
        with self.assertRaisesRegex(ValueError, "repeated resume-boundary save"):
            collect_milestones(child)

    def test_zero_owned_existing_validation_is_not_silently_ignored(self):
        child, failed = self.with_zero_progress_parent()
        self.write_csv(failed, 20000, [dict(step=19999, **dict.fromkeys(METRICS, .01))])
        with self.assertRaisesRegex(ValueError, "Repeated validation at"):
            collect_validation(child)

    def test_rejects_overlapping_attempt_output_and_wrong_identity(self):
        repeated = copy.deepcopy(self.child); repeated["output"] = self.parent["output"]
        with self.assertRaisesRegex(ValueError, "Repeated history"):
            history_segments(repeated)
        repeated = copy.deepcopy(self.child); repeated["run_id"] = "another-run"
        with self.assertRaisesRegex(ValueError, "Unbound crash"):
            history_segments(repeated)

    def test_rejects_nested_or_aliased_attempt_output(self):
        repeated = copy.deepcopy(self.child)
        repeated["output"] = str(Path(self.parent["output"]) / "nested-attempt")
        with self.assertRaisesRegex(ValueError, "nested attempt"):
            history_segments(repeated)
        alias = self.root / "alias"
        alias.symlink_to(Path(self.parent["output"]), target_is_directory=True)
        repeated["output"] = str(alias)
        with self.assertRaisesRegex(ValueError, "Repeated history"):
            history_segments(repeated)

    def test_rejects_ambiguous_continuation_and_corrupt_binding(self):
        ambiguous = copy.deepcopy(self.child); ambiguous["continuation"] = {}
        with self.assertRaisesRegex(ValueError, "both crash recovery"):
            history_segments(ambiguous)
        binding = self.child["crash_recovery"]["inspection"]
        Path(binding["path"]).write_text("{}")
        with self.assertRaises(ValueError):
            history_segments(self.child)


if __name__ == "__main__":
    unittest.main(verbosity=2)
