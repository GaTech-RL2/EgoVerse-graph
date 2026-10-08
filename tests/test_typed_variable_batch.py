"""CPU-only guards for the typed native fallback; no model proof is fabricated."""
import unittest
import sys
import types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/train/homogeneous_pipeline_runtime"))
class Tensor:
    def __init__(self, rows): self.shape=(rows,1)
def stub(name, **attrs):
    module=types.ModuleType(name)
    for key,value in attrs.items(): setattr(module,key,value)
    sys.modules[name]=module
stub("lightning", Callback=object)
stub("homogeneous_dithalf_training",batch_shape_signature=lambda batch:tuple((key,val["embodiment"].shape) for key,val in batch.items()))
stub("noaug_homogeneous_native_rows_v4",HomogeneousNativeRows=object)
stub("optimization_parity_v3",policy=lambda *args:None)
from homogeneous_typed_native_training import validate_variable_source_batches
from typed_execution_contract_v1 import validate_source_batches
class Tests(unittest.TestCase):
    contract={"source_rows":{"u":32,"c":32}}
    def batch(self,u=32,c=32): return {"u":{"embodiment":Tensor(u)},"c":{"embodiment":Tensor(c)}}
    def test_full_stays_regular(self):
        self.assertEqual(validate_variable_source_batches(self.batch(),self.contract),self.contract["source_rows"])
        validate_source_batches(self.batch(),self.contract)
    def test_short_u_allowed_native_only(self):
        self.assertEqual(validate_variable_source_batches(self.batch(13),self.contract),{"u":13,"c":32})
        with self.assertRaises(ValueError): validate_source_batches(self.batch(13),self.contract)
    def test_short_chain_allowed_native_only(self): self.assertEqual(validate_variable_source_batches(self.batch(c=7),self.contract),{"u":32,"c":7})
    def test_no_source_sampling(self):
        with self.assertRaises(ValueError): validate_variable_source_batches({"u":self.batch()["u"]},self.contract)
    def test_empty_rejected(self):
        with self.assertRaises(ValueError): validate_variable_source_batches(self.batch(0),self.contract)
    def test_oversize_rejected(self):
        with self.assertRaises(ValueError): validate_variable_source_batches(self.batch(33),self.contract)
    def test_extra_source_rejected(self):
        batch=self.batch();batch["other"]={"embodiment":Tensor(1)}
        with self.assertRaises(ValueError):validate_variable_source_batches(batch,self.contract)
    def test_invalid_rank_rejected(self):
        batch=self.batch();batch["u"]["embodiment"].shape=(32,1,1)
        with self.assertRaises(ValueError):validate_variable_source_batches(batch,self.contract)
if __name__=="__main__":unittest.main()
