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


def test_chain_es32_native_artifact_preserves_axes_and_exact_identity(tmp_path):
    import hashlib, json
    from pathlib import Path
    from types import SimpleNamespace
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    config = tmp_path / '.hydra/config.yaml'
    config.parent.mkdir(); config.write_text('native_chain_dp: true\n')
    content = tmp_path / 'content.json'
    content.write_text(json.dumps({'aggregate_sha256': 'd'*64}))
    seedbank = Path(__file__).parents[1] / 'egomimic/hydra_configs/evaluator/energy_score_seed_bank_k32_v1.json'
    evaluator = PlanarActionEval(
        artifact_root=str(tmp_path/'artifacts'), seed_bank_path=str(seedbank),
        seed_bank_sha256=sha(seedbank),
        native_decoder=PlanarCommon5NativeDecoder(16, 4),
        energy_score_validation_view={'split_manifest_sha256': 'c'*64},
        energy_score_provenance={
            'source_commit':'a'*40, 'normalization_sha256':'b'*64,
            'split_manifest_sha256':'c'*64, 'resolved_config_path':str(config),
            'wandb':{'entity':'rl2-group','project':'fixture','run_id':'fixture-smoke'},
            'dataset_content':{'manifest_path':str(content),'manifest_sha256':sha(content),'aggregate_sha256':'d'*64},
        },
    )
    class Normalizer:
        def unnormalize(self, values, _embodiment):
            return {key: value*2+value.new_tensor([1,-3,0,0,.5]) for key,value in values.items()}
    evaluator.bind_data_context(normalizer=Normalizer())
    evaluator.trainer = SimpleNamespace(global_step=2,current_epoch=0,global_rank=0,precision='bf16')
    predictions=torch.randn(32,16,16,5,generator=torch.Generator().manual_seed(42))
    batch={'chain':{'actions':predictions[0], 'embodiment':torch.full((16,),20),
                    'episode_hash':torch.arange(16)+100,'frame_index':torch.arange(16)}}
    scores={'chain':{key:torch.zeros(16) for key in ['accuracy_by_condition','diversity_by_condition','score_by_condition']}}
    evaluator._save_artifact(0,{'chain':predictions},scores,batch)
    artifact=torch.load(next((tmp_path/'artifacts').rglob('*.pt')),weights_only=False)
    domain=artifact['domains']['pushshapes_sim_chain_gripper']
    expected=torch.stack([evaluator.native_decoder.decode(sample) for sample in Normalizer().unnormalize({'actions':predictions},20)['actions']])
    torch.testing.assert_close(domain['native_predictions'],expected,rtol=0,atol=0)
    assert domain['native_predictions'].shape==(32,16,16,4)
    assert artifact['seed_bank']==evaluator.seeds and len(artifact['seed_bank'])==32
    assert artifact['identity']['normalization_sha256']=='b'*64
    assert len(artifact['identity']['validation_conditions']['pushshapes_sim_chain_gripper'])==16
