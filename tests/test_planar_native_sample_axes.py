import torch
from egomimic.eval.planar_action_eval import PlanarActionEval
from egomimic.pipeline.pushshapes import PlanarCommon5NativeDecoder


def test_native_sample_batch_axes_match_serial_decoding():
    class AffineNormalizer:
        def unnormalize(self, values, embodiment_id):
            assert embodiment_id == 20
            return {key: value * 2 + value.new_tensor([1, -3, 0, 0, 0.5]) for key, value in values.items()}

    evaluator = PlanarActionEval(energy_score_enabled=False)
    evaluator.bind_data_context(normalizer=AffineNormalizer())
    decoder = PlanarCommon5NativeDecoder(action_horizon=16, native_action_dim=4)
    generator = torch.Generator().manual_seed(42)
    predictions = torch.randn(32, 16, 16, 5, generator=generator)
    unnormalized = AffineNormalizer().unnormalize({"actions": predictions}, 20)["actions"]
    expected = torch.stack([decoder.decode(sample) for sample in unnormalized])
    actual = evaluator._native(predictions, 20, decoder)
    assert actual.shape == (32, 16, 16, 4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(evaluator._native(predictions[0], 20, decoder), expected[0], rtol=0, atol=0)
