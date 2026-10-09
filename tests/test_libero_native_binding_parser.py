"""Real Hydra grammar regression; no model construction or smoke claims."""

import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "native_schema",
    Path(__file__).parents[1] / "scripts/train/libero_native_schema_contract.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
KEYS = (
    "path",
    "file_sha256",
    "normalizer_module_sha256",
    "source_commit",
    "dataset_receipt_path",
    "dataset_receipt_sha256",
    "split_receipt_path",
    "split_receipt_sha256",
    "data_root",
    "normalizer_target",
    "physical_proof_path",
    "physical_proof_sha256",
)


class BindingParserTests(unittest.TestCase):
    def test_real_twelve_field_hydra_mapping(self):
        expected = {
            k: (
                "/workspace/" + k
                if k.endswith("path") or k == "data_root"
                else "abc123"
            )
            for k in KEYS
        }
        arg = (
            "++norm_stats.native_saved_state_binding={"
            + ",".join(k + ":" + v for k, v in expected.items())
            + "}"
        )
        self.assertEqual(module.parse_native_saved_state_binding([arg]), expected)

    def test_malformed_and_incomplete(self):
        for value in ("{path:/workspace/x}", "{path:/workspace/x", "null", "[x,y]"):
            with self.subTest(value=value), self.assertRaises(Exception):
                module.parse_native_saved_state_binding(
                    ["++norm_stats.native_saved_state_binding=" + value]
                )

    def test_sweep_and_delete(self):
        arg = (
            "++norm_stats.native_saved_state_binding={"
            + ",".join(k + ":abc" for k in KEYS)
            + "}"
        )
        for anomaly in ("trainer.max_steps=2,3", "~trainer.max_steps"):
            with self.subTest(anomaly=anomaly), self.assertRaises(ValueError):
                module.parse_native_saved_state_binding([arg, anomaly])

    def test_missing_and_duplicate(self):
        arg = (
            "++norm_stats.native_saved_state_binding={"
            + ",".join(k + ":abc" for k in KEYS)
            + "}"
        )
        for argv in ([], [arg, arg], [arg, arg.replace("++", "", 1)]):
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                module.parse_native_saved_state_binding(argv)


if __name__ == "__main__":
    unittest.main()
