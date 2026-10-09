"""Shared stochastic chunk validation for raw-action LIBERO comparisons."""
import hashlib
import json
from pathlib import Path
import torch
from egomimic.eval.energy_score import energy_score, semantic_chunk_distance
from egomimic.eval.libero_eval import LiberoActionEvaluator, action_metrics
from egomimic.rldb.zarr.libero_dataset import EMBODIMENT

SEEDS = tuple(730001 + i * 7919 for i in range(32))
SEED_SHA = hashlib.sha256(json.dumps(SEEDS).encode()).hexdigest()

def distance(a, b):
    return semantic_chunk_distance(a, b, blocks=((0,3),(3,6),(6,7)))

class LiberoComparisonEvaluator(LiberoActionEvaluator):
    """32 fixed samples on the first ordered batch; native metrics on all batches."""
    def __init__(self, artifact_root, source_commit, config_sha256):
        super().__init__()
        self.artifact_root=Path(artifact_root)
        self.identity={'source_commit':source_commit,'config_sha256':config_sha256,
                       'seed_bank_sha256':SEED_SHA,'seeds':SEEDS,
                       'sample_count':32,'distance':'equal-weight normalized translation/rotation-command/gripper RMS over complete H32 chunk',
                       'action_contract':'normalized raw OSC commands (7D), not absolute rotations'}
        self.action_flow_diagnostics_enabled=False
        self._completed_validation_step=None

    @torch.inference_mode()
    def on_validation_step(self,batch,batch_idx,dataloader_idx=0):
        del dataloader_idx
        devices=sorted({v.device.index for b in batch.values() for v in b.values() if torch.is_tensor(v) and v.is_cuda})
        predictions=[]
        for seed in (SEEDS if batch_idx == 0 else SEEDS[:1]):
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(seed)
                predictions.append(self.model.forward_eval(batch))
        prefix='Valid' if self.group == 'valid' else f'Valid_{self.group}'
        for source,values in batch.items():
            target=values['actions'];pred=predictions[0][source]['pred_action']
            assert pred.shape == target.shape and target.shape[-2:] == (32,7)
            native_target=self.normalizer.unnormalize({'actions':target},EMBODIMENT)['actions']
            native_pred=self.normalizer.unnormalize({'actions':pred},EMBODIMENT)['actions']
            metrics=action_metrics(native_pred,native_target)
            metrics['normalized_reconst_mse']=(pred-target).square().mean()
            if batch_idx == 0:
                samples=torch.stack([p[source]['pred_action'] for p in predictions])
                scores=energy_score(samples,target,distance_fn=distance)
                for key,name in [('score','energy_score32'),('accuracy','energy_accuracy32'),('diversity','energy_diversity32')]:
                    metrics[name]=scores[key]
                step=int(self.trainer.global_step)
                path=self.artifact_root/f'step-{step:09d}-{self.group}-{source}.pt'
                path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists(): raise FileExistsError(path)
                torch.save({'identity':self.identity,'global_step':step,'predictions':samples.cpu(),
                            'targets':target.cpu(),'normalizer_state':self.normalizer.to_state()},path)
                path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')
            for key,value in metrics.items():
                assert torch.isfinite(value).all(), key
                self.model.log(f'{prefix}/{key}',value,batch_size=len(target),on_step=False,on_epoch=True,sync_dist=True)
        self._completed_validation_step=int(self.trainer.global_step)
