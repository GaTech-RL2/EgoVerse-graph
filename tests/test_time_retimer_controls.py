"""The baseline (time) recipe's deployed decoder and its dashboard speed controls."""

import numpy as np
from hydra.utils import instantiate

from egomimic.pipeline.inference_config import build_inference_config
from scripts.audit_hydra_configs import CONFIGS, compose_for_audit


def test_time_recipe_speed_controls_bind_to_the_deployed_retimer_and_act():
    with compose_for_audit(CONFIGS / "experiment/real/earbuds_time.yaml") as cfg:
        profile = build_inference_config(cfg)["inference_graph"]["profiles"]["default"]
        decoder = instantiate(profile["adapter"]["decoder"])
        assert type(decoder).__name__ == "TimeChunkRetimer"
        targets = {
            name: spec["target"]["attribute_path"]
            for name, spec in profile["overrides"].items()
            if spec["target"]["kind"] == "decoder_attribute"
        }
        assert set(targets) == {"arc_speed_percent", "arc_hold_speed_percent"}
        for name, path in targets.items():
            assert hasattr(decoder, path), (name, path)
        chunk = np.zeros((1, 100, 14))
        chunk[0, :, 0] = np.linspace(0, 0.3, 100)  # 9 cm/s, well above hold_threshold
        np.testing.assert_array_equal(decoder(chunk), chunk)  # 100 % is byte-identical
        setattr(decoder, targets["arc_speed_percent"], 200)
        setattr(decoder, targets["arc_hold_speed_percent"], 200)
        out = decoder(chunk)[0]
        np.testing.assert_allclose(out[:50, 0], chunk[0, ::2, 0], atol=1e-9)
        np.testing.assert_allclose(
            out[50:, 0], 0.3
        )  # endpoint held, never extrapolated
        assert decoder.last_stats["valid_steps"] == 50
