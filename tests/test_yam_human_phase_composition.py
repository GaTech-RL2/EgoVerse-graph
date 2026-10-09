"""Compose the maintained real argv for all phases before any corpus traversal.
Synthetic identities are schema regression fixtures, never launch receipts.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict
from scripts.train.verify_yam_human_action_flow_launch import verify_config, split_identities, aggregate_content

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("corpus_fixture",ROOT/"tests/test_yam_human_refreshed_corpus_contract.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

@pytest.mark.parametrize("phase",["norm","smoke","full"])
def test_real_shared_argv_composes_and_verifies(tmp_path,monkeypatch,phase):
 cp,sp,c,s,content=module.fixture(tmp_path)
 ids=split_identities(sp,module.sha(sp),cp,module.sha(cp))
 aggregate=aggregate_content(content["yam"],module.sha(content["yam"]),content["human"],module.sha(content["human"]),cp,module.sha(cp))
 norm=tmp_path/"norm.json";norm.write_text("{}")
 env={"AF_YAM_RECIPE":"yam_human_keypoints","AF_YAM_PHASE":phase,"AF_YAM_ACCUMULATION":"1","AF_YAM_CHECKPOINT_POLICY":"dit_half","AF_YAM_INFERENCE_METHOD":"euler","AF_YAM_DATA_ROOT":str(tmp_path/"yam"),"AF_HUMAN_DATA_ROOT":str(tmp_path/"human"),"AF_YAM_NORM_JSON":str(norm),"AF_COTRAIN_NORM":str(norm),"AF_YAM_NORM_CACHE_DIR":str(tmp_path/"norm-cache"),"AF_YAM_RUN_DIR":str(tmp_path/phase),"AF_YAM_SOURCE_COMMIT":"0"*40,"AF_YAM_NORM_SHA":module.sha(norm),"AF_YAM_SPLIT_SHA":module.sha(sp),"AF_YAM_WANDB_ENTITY":"schema_fixture","AF_YAM_WANDB_PROJECT":"schema_fixture","AF_YAM_WANDB_ID":"schema_fixture_"+phase,"content_sha":aggregate}
 for d in ("yam","human"):
  for group in ("train","valid"):
   env[d+"_"+group+"_count"]=str(s[d]["summary"][group+"_episodes"])
   env[d+"_"+group+"_windows"]=str(s[d]["summary"][group+"_frame_windows"])
   env[d.upper()+"_"+group.upper()+"_IDS_SHA256"]=ids[d+"_"+group]
 for k,v in env.items():monkeypatch.setenv(k,v)
 command=r'source "$1"; yam_action_flow_arguments; printf "%s\0" "${overrides[@]}"'
 assert chr(0) not in command, "shell executable argv must never embed a NUL"
 argv=subprocess.check_output(["bash","-c",command,"argv",str(ROOT/"scripts/train/yam_action_flow_arguments.sh")],env={**os.environ,**env}).decode().rstrip("\0").split("\0")
 with initialize_config_dir(version_base=None,config_dir=str(ROOT/"egomimic/hydra_configs")):
  cfg=compose(config_name="train_zarr_cartesian",overrides=argv,return_hydra_config=True)
  cfg.hydra.runtime.output_dir=str(tmp_path/phase)
  HydraConfig.instance().set_config(cfg)
  with open_dict(cfg): del cfg["hydra"]
  OmegaConf.resolve(cfg)
 resolved=tmp_path/(phase+".yaml");OmegaConf.save(cfg,resolved)
 args=argparse.Namespace(phase=phase,yam_root=tmp_path/"yam",human_root=tmp_path/"human",norm_json=norm,norm_sha=module.sha(norm),norm_cache_dir=tmp_path/"norm-cache",source_commit="0"*40,split_sha=module.sha(sp),wandb_entity="schema_fixture",wandb_project="schema_fixture",wandb_id="schema_fixture_"+phase,run_dir=tmp_path/phase,inference_method="euler",checkpoint_policy="dit_half",corpus_contract=cp,corpus_contract_sha=module.sha(cp))
 verify_config(resolved,args,ids,aggregate)
 assert cfg.model.pipeline.stages[6].num_inference_steps==50
 for provenance in ("energy_score_provenance","action_flow_diagnostics.provenance"):
  path="evaluator."+provenance+".sampler"
  OmegaConf.update(cfg,path,"dopri5")
  OmegaConf.save(cfg,resolved)
  with pytest.raises(ValueError):verify_config(resolved,args,ids,aggregate)
  OmegaConf.update(cfg,path,"euler")
 OmegaConf.save(cfg,resolved)
 assert not any("cuda" in value for value in argv)
 if phase=="norm":
  cfg.norm_stats.precomputed_norm_path=str(norm)
  OmegaConf.save(cfg,resolved)
  with pytest.raises(ValueError):verify_config(resolved,args,ids,aggregate)
