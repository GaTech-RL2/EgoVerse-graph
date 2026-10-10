"""Focused pure metadata tests; never a full-model smoke proof."""

import unittest

from scripts.train.verify_libero_native_action_flow_smoke import (
    PARAMETER_COUNT,
    PROFILE,
    validate_dit_counts,
    validate_preflight,
)


class Contract(unittest.TestCase):
    def test_missing_preflight(self):
        with self.assertRaises(KeyError):
            validate_preflight({}, {})

    def test_tiny_count_not_accepted(self):
        self.assertEqual(PARAMETER_COUNT, 39750391)
        self.assertEqual(
            PROFILE, "libero_historical/action_flow_libero10_h240_euler50_dithalf_80k_s42"
        )

    def test_preparation_rejected(self):
        with self.assertRaisesRegex(ValueError, "did not pass"):
            validate_preflight(
                {"schema": "libero-native-launch-preflight/v1", "status": "PREPARED"},
                {},
            )

    def test_wrong_profile_rejected(self):
        with self.assertRaisesRegex(ValueError, "profile"):
            validate_preflight(
                {
                    "schema": "libero-native-launch-preflight/v1",
                    "status": "PASS",
                    "profile": "invented",
                },
                {},
            )

    def test_dit_inactive_rejected(self):
        class Shared:
            @staticmethod
            def _complete_row(rows, required, **kw):
                return 1, {name: 0 for name in required}

        with self.assertRaisesRegex(ValueError, "inactive"):
            validate_dit_counts({}, Shared)


if __name__ == "__main__":
    unittest.main()
