import hashlib,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from egomimic.rldb.zarr.libero_saved_state import validate_receipt_contract
class Guards(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(); root=Path(self.tmp.name).resolve()
  data=dict(replay_path=str(root),suite="libero10",split_seed=42,valid_ratio=.01,dataset_logical_sha256="d"*64,train_episode_indices=[0,1],valid_episode_indices=[2],episodes=3,frames=30)
  split={k:data[k] for k in ("train_episode_indices","valid_episode_indices","split_seed","valid_ratio")}
  split.update(schema_version=1,artifact_kind="libero_native_episode_split",historical_split_manifest_sha256="h"*64,serialization_replacement_requires_parent_review=True,dataset_logical_sha256="d"*64,replay_path=str(root),suite="libero10",episodes=3,recipe="test fixture",train_episode_ids=["episode_000000","episode_000001"],valid_episode_ids=["episode_000002"],physical_frame_range_proof="OPEN")
  split["dataset_receipt_sha256"]=hashlib.sha256(json.dumps(data).encode()).hexdigest()
  self.binding=dict(path="unneeded",file_sha256="n"*64,normalizer_module_sha256="m"*64,source_commit="s"*40,data_root=str(root),normalizer_target="egomimic.rldb.zarr.libero_action_flow.LiberoActionFlowNormalizer")
  for name,value in (("dataset",data),("split",split)):
   file=root/(name+".json"); file.write_text(json.dumps(value)); self.binding[name+"_receipt_path"]=str(file); self.binding[name+"_receipt_sha256"]=hashlib.sha256(file.read_bytes()).hexdigest()
  proof=dict(schema_version=1,status="PASS",split_manifest_sha256=self.binding["split_receipt_sha256"],dataset_receipt_sha256=self.binding["dataset_receipt_sha256"],data_root=str(root),episode_ends_sha256="e"*64,episode_count=3,frames=30,zero_frame_overlap=True,complete_frame_coverage=True,actual_selected_train_ids=["episode_000000","episode_000001"],train_ranges=[[0,0,10],[1,10,20]],valid_ranges=[[2,20,30]])
  pf=root/"physical.json";pf.write_text(json.dumps(proof));self.binding["physical_proof_path"]=str(pf);self.binding["physical_proof_sha256"]=hashlib.sha256(pf.read_bytes()).hexdigest()
  self.prov=dict(historical_split_manifest_sha256="h"*64,source_commit="s"*40,normalization_sha256="n"*64,split_manifest_sha256=self.binding["split_receipt_sha256"],dataset_sha256="d"*64,train_episode_count=2,valid_episode_count=1,union_episode_count=3)
  self.ds=SimpleNamespace(resolver=SimpleNamespace(folder_path=str(root),suite="libero10"),split_seed=42,val_ratio=.01,datasets={"episode_000000":None,"episode_000001":None})
 def tearDown(self): self.tmp.cleanup()
 def call(self): return validate_receipt_contract(self.binding,self.prov,self.ds,source_head="s"*40)
 def test_good(self): self.assertEqual(self.call()["training_episode_count"],2)
 def test_source(self):
  self.prov["source_commit"]="x"
  with self.assertRaisesRegex(ValueError,"source"): self.call()
 def test_receipt_hash(self):
  self.binding["dataset_receipt_sha256"]="0"*64
  with self.assertRaisesRegex(ValueError,"identity"): self.call()
 def test_actual_selection(self):
  self.ds.datasets={"episode_000002":None}
  with self.assertRaisesRegex(ValueError,"actual selected"): self.call()
 def test_path(self):
  self.ds.resolver.folder_path="/wrong"
  with self.assertRaisesRegex(ValueError,"path"): self.call()
 def test_missing_field(self):
  del self.binding["data_root"]
  with self.assertRaisesRegex(ValueError,"schema"): self.call()
 def test_open_physical_proof(self):
  pf=Path(self.binding["physical_proof_path"]);q=json.loads(pf.read_text());q["status"]="OPEN";pf.write_text(json.dumps(q));self.binding["physical_proof_sha256"]=hashlib.sha256(pf.read_bytes()).hexdigest()
  with self.assertRaisesRegex(ValueError,"nonpassing"): self.call()
 def test_missing_physical_proof(self):
  del self.binding["physical_proof_path"]
  with self.assertRaisesRegex(ValueError,"schema"): self.call()
if __name__=="__main__": unittest.main()
