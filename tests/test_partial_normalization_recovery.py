"""Recovery uses native provenance checks and computes only missing embodiments."""
import json
import numpy as np
import pytest
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset


def fixture(tmp_path):
    path = tmp_path / 'partial.json'
    path.write_text(json.dumps({'stats': {'3': {'actions': {'min': [0.], 'max': [1.]}}, '7': {}}, 'provenance': {'norm_mode': 'minmax'}}))
    normalizer = MultiDataset(state={}, norm_mode='minmax')
    normalizer.key_types = {3: {'actions': 'action_keys'}, 7: {'actions': 'action_keys'}}
    return normalizer, path


def test_recovery_reuses_completed_and_computes_missing(tmp_path, monkeypatch):
    normalizer, path = fixture(tmp_path)
    calls = []
    def collect(loader, keys, embodiment, *args):
        calls.append(embodiment)
        return {'actions': [np.array([[0.], [1.]], dtype=np.float32)]}
    monkeypatch.setattr(normalizer, '_collect_norm_samples', collect)
    normalizer.infer_norm_from_dataset([0, 1], 3, sample_frac=1., num_workers=0, resume_partial_norm_path=str(path))
    assert calls == []
    assert normalizer.norm_stats[3]['actions']['max'] == [1.]
    normalizer.infer_norm_from_dataset([0, 1], 7, sample_frac=1., num_workers=0, resume_partial_norm_path=str(path))
    assert calls == [7]
    assert normalizer.norm_stats[7]['actions']['max'].tolist() == [1.]


def test_recovery_rejects_conflict_and_wrong_provenance(tmp_path):
    normalizer, path = fixture(tmp_path)
    with pytest.raises(ValueError, match='conflicts'):
        normalizer.infer_norm_from_dataset([0, 1], 3, precomputed_norm_path=str(path), resume_partial_norm_path=str(path))
    payload = json.loads(path.read_text()); payload['provenance']['norm_mode'] = 'zscore'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='norm_mode'):
        normalizer.infer_norm_from_dataset([0, 1], 3, resume_partial_norm_path=str(path))
