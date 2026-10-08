"""Load an immutable native task snapshot in its recorded Hydra context.

Only the suite replay-root environment is restored, from its hash-bound receipt.
No current-working-directory defaults, model construction or dataset reads.
"""
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from .native_launch_profiles import profile_for_config

def require(condition,message):
    if not condition:raise ValueError(message)

def checked_json(path,digest):
    raw=Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==digest,'saved context receipt SHA mismatch')
    return json.loads(raw)

def load_saved_native_config(run_dir,repo=None):
    from omegaconf import OmegaConf,open_dict
    from hydra import __version__ as hydra_version
    from hydra.conf import HydraConf
    from hydra.core.hydra_config import HydraConfig
    from hydra.core.utils import setup_globals
    root=Path(run_dir).resolve(strict=True)
    repo=Path(repo or Path(__file__).resolve().parents[3]).resolve(strict=True)
    config=OmegaConf.load(root/'.hydra/config.yaml')
    saved=OmegaConf.load(root/'.hydra/hydra.yaml')
    literal=OmegaConf.to_container(config,resolve=False)
    profile=profile_for_config(config)
    require(str(saved.hydra.runtime.cwd)==str(repo),'saved Hydra cwd/source mismatch')
    require(str(saved.hydra.runtime.output_dir)==str(root),'saved Hydra output mismatch')
    require(str(saved.hydra.runtime.version)==hydra_version,'saved Hydra version mismatch')
    head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    require(literal['run_provenance']['source_commit']==head,'saved task/source HEAD mismatch')
    binding=literal['norm_stats']['native_saved_state_binding']
    require(binding['source_commit']==head,'saved binding/source HEAD mismatch')
    replay=binding['data_root']
    require(isinstance(replay,str) and Path(replay).is_absolute() and '${' not in replay,'literal bound replay root required')
    receipt=checked_json(binding['dataset_receipt_path'],binding['dataset_receipt_sha256'])
    require(receipt['replay_path']==replay and receipt['suite']==profile.suite,'saved suite/replay receipt mismatch')
    require(binding['dataset_receipt_sha256']==literal['run_provenance']['dataset_sha256'],'saved dataset provenance mismatch')
    checked_json(binding['physical_proof_path'],binding['physical_proof_sha256'])
    references=set(re.findall(r'\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)',OmegaConf.to_yaml(config)))
    require(references=={profile.replay_environment},'unexpected saved oc.env reference inventory')
    existing=os.environ.get(profile.replay_environment)
    require(existing is None or existing==replay,'conflicting saved replay-root environment')
    typed=OmegaConf.structured(HydraConf)
    with open_dict(typed):typed.merge_with(saved.hydra)
    before=OmegaConf.to_container(saved.hydra,resolve=False,enum_to_str=True)
    after=OmegaConf.to_container(typed,resolve=False,enum_to_str=True)
    require(all(after.get(k)==v for k,v in before.items()),'saved Hydra fields changed')
    os.environ[profile.replay_environment]=replay
    context=OmegaConf.merge(config,OmegaConf.create({'hydra':typed}))
    setup_globals();HydraConfig.instance().set_config(context)
    payload=OmegaConf.to_container(config,resolve=True)
    require(payload['benchmark']['dataset']==replay,'resolved benchmark root mismatch')
    for group in ('train_datasets','valid_datasets'):
        require(payload['data'][group]['libero_panda']['resolver']['folder_path']==replay,'resolved dataset root mismatch')
    require(payload['evaluator']['energy_seed_bank_path']==str(repo/'egomimic/hydra_configs/evaluator/energy_score_seed_bank_k32_v1.json'),'saved seed-bank context mismatch')
    return config
