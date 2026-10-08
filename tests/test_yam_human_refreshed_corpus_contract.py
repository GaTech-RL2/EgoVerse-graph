"""Refreshed corpus must retain native deterministic splits and complete bytes."""
import copy
import hashlib
import json
import random
import pytest
from egomimic.rldb.zarr.episode_split import split_dataset_names
from scripts.train.verify_yam_human_action_flow_launch import DOMAINS, corpus_domains, split_identities, aggregate_content


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, value):
 path.write_text(json.dumps(value)); return sha(path)
def fixture(tmp_path):
 contract={"schema_version":1,"complete_training_corpus":True,"split_seed":42,"valid_ratio":0.01,"domains":{}}
 split={"split_seed":42,"valid_ratio":0.01}
 manifests={}
 total=0
 for domain,count in (("yam",378),("human",38)):
  names=[f"{domain}-{i:04}" for i in range(count)]
  train,valid=split_dataset_names(names,0.01,42)
  prefix=DOMAINS[domain][-1]
  rows=[{"episode_hash":n,"zarr_processed_path":f"{prefix}{n}.zarr","num_frames":101+i,"split":"train" if n in train else "valid"} for i,n in enumerate(names)]
  summary={"episodes":count,"train_episodes":len(train),"valid_episodes":len(valid)}
  for group,ns in (("train",train),("valid",valid)):
   summary[group+"_frame_windows"]=sum(r["num_frames"] for r in rows if r["split"]==group)
   summary[group+"_ids_sha256"]=hashlib.sha256("".join(n+"\n" for n in sorted(ns)).encode()).hexdigest()
  total+=summary["train_frame_windows"]
  inv=sorted((r["episode_hash"],r["zarr_processed_path"],r["num_frames"]) for r in rows)
  inv_sha=hashlib.sha256(json.dumps(inv,separators=(",",":")).encode()).hexdigest()
  content={"complete_training_corpus":True,"episode_count":count,"object_count":count*10,"content_bytes":count*1000,"episode_inventory_sha256":inv_sha,"aggregate_content_sha256":hashlib.sha256(domain.encode()).hexdigest()}
  path=tmp_path/(domain+"-content.json");manifest_sha=save(path,content);manifests[domain]=path
  contract["domains"][domain]={**summary,"episode_inventory_sha256":inv_sha,"complete_content_manifest_sha256":manifest_sha,"aggregate_content_sha256":content["aggregate_content_sha256"],"object_count":content["object_count"],"content_bytes":content["content_bytes"]}
  split[domain]={"summary":summary,"episodes":rows}
 split["train_frame_windows_total"]=total
 split["proportional_train_window_probabilities"]={d:split[d]["summary"]["train_frame_windows"]/total for d in DOMAINS}
 cp=tmp_path/"contract.json";sp=tmp_path/"split.json";save(cp,contract);save(sp,split)
 return cp,sp,contract,split,manifests


def test_refreshed_complete_corpus_and_deterministic_split(tmp_path):
 cp,sp,c,s,content=fixture(tmp_path)
 assert corpus_domains(cp,sha(cp))["yam"][1:3]==(375,3)
 assert corpus_domains(cp,sha(cp))["human"][1:3]==(37,1)
 assert set(split_identities(sp,sha(sp),cp,sha(cp)))=={"yam_train","yam_valid","human_train","human_valid"}
 assert len(aggregate_content(content["yam"],sha(content["yam"]),content["human"],sha(content["human"]),cp,sha(cp)))==64

@pytest.mark.parametrize("mutation",["sparse","ratio","domain","holdout","bad_hash","bool_count"])
def test_contract_rejects_bad_complete_identity(tmp_path,mutation):
 cp,sp,c,s,_=fixture(tmp_path)
 if mutation=="sparse": c["complete_training_corpus"]=False
 elif mutation=="ratio": c["valid_ratio"]=0.02
 elif mutation=="domain": del c["domains"]["human"]
 elif mutation=="holdout": c["domains"]["yam"]["valid_episodes"]=2
 elif mutation=="bad_hash": c["domains"]["yam"]["aggregate_content_sha256"]="z"*64
 elif mutation=="bool_count": c["domains"]["yam"]["content_bytes"]=True
 save(cp,c)
 with pytest.raises(ValueError): corpus_domains(cp,sha(cp))

@pytest.mark.parametrize("mutation",["duplicate","wrong_inventory","balanced_sampling","swapped_membership"])
def test_split_rejects_changed_corpus_or_native_membership(tmp_path,mutation):
 cp,sp,c,s,_=fixture(tmp_path)
 if mutation=="duplicate": s["yam"]["episodes"][1]=s["yam"]["episodes"][0].copy()
 elif mutation=="wrong_inventory": c["domains"]["yam"]["episode_inventory_sha256"]="0"*64;save(cp,c)
 elif mutation=="balanced_sampling": s["proportional_train_window_probabilities"]={"yam":.5,"human":.5}
 else:
  rows=s["yam"]["episodes"]
  a=next(r for r in rows if r["split"]=="train");b=next(r for r in rows if r["split"]=="valid")
  a["split"],b["split"]=b["split"],a["split"]
 save(sp,s)
 with pytest.raises(ValueError): split_identities(sp,sha(sp),cp,sha(cp))


def test_sparse_audit_cannot_become_training_content(tmp_path):
 cp,sp,c,s,content=fixture(tmp_path)
 p=content["yam"];item=json.loads(p.read_text());item["sparse_audit_cache_not_training_data"]=True
 c["domains"]["yam"]["complete_content_manifest_sha256"]=save(p,item);save(cp,c)
 with pytest.raises(ValueError): aggregate_content(p,sha(p),content["human"],sha(content["human"]),cp,sha(cp))


def test_native_split_preserves_original_algorithm():
 for count in (1,37,38,229,378):
  names=[str(i) for i in range(count)];reference=sorted(names);random.Random(42).shuffle(reference)
  n=max(1,int(count*.01))
  assert split_dataset_names(names,.01,42)==(set(reference[n:]),set(reference[:n]))
