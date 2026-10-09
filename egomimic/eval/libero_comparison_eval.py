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
    """32 fixed samples on an ordered eight-condition panel shared by both families."""
    def __init__(self, artifact_root, source_commit, config_sha256, diagnostic_config=None, panel_size=8):
        super().__init__()
        self.artifact_root=Path(artifact_root)
        self.panel_size=int(panel_size)
        if self.panel_size != 8: raise ValueError('paired validation requires the same ordered eight-condition panel')
        self.shared_diagnostics=None
        if diagnostic_config is not None:
            from egomimic.eval.action_flow_diagnostics import ActionFlowDiagnostics
            self.shared_diagnostics=ActionFlowDiagnostics(diagnostic_config)
        self.identity={'source_commit':source_commit,'config_sha256':config_sha256,
                       'seed_bank_sha256':SEED_SHA,'seeds':SEEDS,
                       'sample_count':32,'panel_size':self.panel_size,'distance':'equal-weight normalized translation/rotation-command/gripper RMS over complete H32 chunk',
                       'action_contract':'normalized raw OSC commands (7D), not absolute rotations'}
        self.action_flow_diagnostics_enabled=self.shared_diagnostics is not None
        self._completed_validation_step=None

    @torch.inference_mode()
    def on_validation_step(self,batch,batch_idx,dataloader_idx=0):
        del dataloader_idx
        devices=sorted({v.device.index for b in batch.values() for v in b.values() if torch.is_tensor(v) and v.is_cuda})
        def subset(values):
            return {key: value[:self.panel_size] if torch.is_tensor(value) or isinstance(value,(list,tuple)) else value
                    for key,value in values.items()}
        panel={source:subset(values) for source,values in batch.items()}
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(SEEDS[0])
            first=self.model.forward_eval(batch)
        predictions=[]
        if batch_idx == 0:
            for seed in SEEDS:
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(seed)
                    predictions.append(self.model.forward_eval(panel))
        prefix='Valid' if self.group == 'valid' else f'Valid_{self.group}'
        for source,values in batch.items():
            target=values['actions'];pred=first[source]['pred_action']
            assert pred.shape == target.shape and target.shape[-2:] == (32,7)
            native_target=self.normalizer.unnormalize({'actions':target},EMBODIMENT)['actions']
            native_pred=self.normalizer.unnormalize({'actions':pred},EMBODIMENT)['actions']
            metrics=action_metrics(native_pred,native_target)
            metrics['normalized_reconst_mse']=(pred-target).square().mean()
            if batch_idx == 0:
                samples=torch.stack([p[source]['pred_action'] for p in predictions])
                panel_target=panel[source]['actions']
                assert len(panel_target) == self.panel_size
                scores=energy_score(samples,panel_target,distance_fn=distance)
                for key,name in [('score','energy_score32'),('accuracy','energy_accuracy32'),('diversity','energy_diversity32')]:
                    metrics[name]=scores[key]
                step=int(self.trainer.global_step)
                path=self.artifact_root/f'step-{step:09d}-{self.group}-{source}.pt'
                path.parent.mkdir(parents=True,exist_ok=True)
                if path.exists(): raise FileExistsError(path)
                torch.save({'identity':self.identity,'global_step':step,'predictions':samples.cpu(),
                            'targets':panel_target.cpu(),'alignment':{k:v for k,v in panel[source].items() if k in ('episode_hash','frame_index','task_uid')},'normalizer_state':self.normalizer.to_state()},path)
                path.with_suffix('.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')
                if self.shared_diagnostics is not None:
                    diagnostics=self.model.run_diagnostic('action_flow',panel,
                        raw_noise_levels=self.shared_diagnostics.noise_levels,
                        noise_seed=SEEDS[0],max_samples=8,jacobian_samples=2,capture_activations=True)
                    self.shared_diagnostics.reset()
                    def native_error(left,right):
                        left=self.normalizer.unnormalize({'actions':left},EMBODIMENT)['actions']
                        right=self.normalizer.unnormalize({'actions':right},EMBODIMENT)['actions']
                        return sum((left[...,a:b]-right[...,a:b]).square().mean(dim=(-2,-1))
                                   for a,b in ((0,3),(3,6),(6,7))) / 3
                    analysis=self.shared_diagnostics.analyze_precomputed(
                        diagnostics={source:diagnostics[source]},batch={source:panel[source]},
                        batch_idx=int(batch_idx),rank=int(self.model.global_rank),epoch=int(self.model.current_epoch),
                        global_step=int(self.model.global_step),precision=self.model.trainer.precision,
                        source_labels={source:source},native_error_fns={source:native_error})
                    for key,value in analysis['metrics'].items():
                        assert torch.isfinite(value).all(),key
                        self.model.log(key,value,batch_size=8,on_step=False,on_epoch=True,sync_dist=True)
            for key,value in metrics.items():
                assert torch.isfinite(value).all(), key
                self.model.log(f'{prefix}/{key}',value,batch_size=len(target),on_step=False,on_epoch=True,sync_dist=True)
        self._completed_validation_step=int(self.trainer.global_step)
