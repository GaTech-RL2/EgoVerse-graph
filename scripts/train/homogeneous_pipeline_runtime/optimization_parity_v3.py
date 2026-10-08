"""Isolated checkpoint-policy A/B tests. Never modify the shared checkout."""
import argparse, gc, hashlib, json, os, statistics, time
from pathlib import Path
VARIANTS = ("baseline-control", "dit-half", "dit-off", "decoder-off")
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def policy(original, mode, family, counters):
    identities = {}
    def wrapped(fn, *args, **kw):
        assert kw.get("use_reentrant") is False
        if fn not in identities: identities[fn] = len(identities)
        bypass = (mode == "dit-off" and family == "dit") or (mode == "decoder-off" and family == "decoder") or (mode == "dit-half" and family == "dit" and identities[fn] % 2 == 0)
        counters[family + ("/direct" if bypass else "/checkpoint")] = counters.get(family + ("/direct" if bypass else "/checkpoint"), 0) + 1
        if not bypass: return original(fn, *args, **kw)
        assert set(kw) <= {"use_reentrant"}, "do not silently drop checkpoint options"
        return fn(*args)
    return wrapped
def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["prepare","run"]);p.add_argument("--task",required=True);p.add_argument("--variant",choices=VARIANTS);a=p.parse_args()
    assert os.environ.get("SLURM_STEP_ID")
    task=Path(a.task);root=task/"optimization-parity-v3"
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    import torch
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cudnn.deterministic=True
    from omegaconf import OmegaConf,open_dict
    from torch.utils.checkpoint import checkpoint
    if a.mode=="prepare":
        assert not root.exists()
        parent=task/"training-profile-v3"
        receipt=json.loads((parent/"CPU_ENVELOPE.json").read_text())
        assert sha(parent/"profile-config.yaml")==receipt["profile_config_sha256"]
        assert sha(task/"full-v1/.hydra/config.yaml")==receipt["original_config_sha256"]
        cfg=OmegaConf.load(parent/"profile-config.yaml")
        assert cfg.trainer.max_steps==8 and cfg.logger is None and not cfg.trainer.enable_checkpointing
        control=json.loads((task/"repro-trace-v3/RESULT.json").read_text())
        assert control["status"]=="UNCHANGED_REPRO_32_PASS" and control["backend"]["deterministic"]
        assert control["identity"]["parent"]["parent"]==receipt
        with open_dict(cfg):
            cfg.trainer.max_steps=1000
            cfg.trainer.deterministic=True
        # Small executable parity regression exercises all policy branches and RNG replay.
        for variant in VARIANTS:
            for family in ("dit","decoder"):
                def f(x): return torch.nn.functional.dropout(x.sin(),p=.2,training=True).square()
                def evaluate(adapter):
                    torch.manual_seed(42)
                    x=torch.linspace(-1,1,32,requires_grad=True)
                    y=adapter(f,x,use_reentrant=False);y.sum().backward()
                    return y.detach(),x.grad
                expected=evaluate(checkpoint); counts={};got=evaluate(policy(checkpoint,variant,family,counts))
                for x,y in zip(expected,got): torch.testing.assert_close(x,y,rtol=0,atol=0)
                assert counts
        root.mkdir()
        OmegaConf.save(cfg,root/"profile-config.yaml",resolve=True)
        env={"status":"OPTIMIZATION_CPU_PASS","parent":receipt,"deterministic_control":control,"execution_delta":"strict deterministic backend; not old-run bit-equivalence","script_sha256":sha(__file__),"config_sha256":sha(root/"profile-config.yaml"),"variants":VARIANTS,"parity_updates":[1,2,100,500,1000],"total_updates":1000,"timing_warmup":20,"exact_loss_every_update":True,"rtol":1e-5,"atol":1e-6,"limits":"1000-step identical-seed A/B tests; no validation/rollout or full 80k quality claim"}
        (root/"CPU_ENVELOPE.json").write_text(json.dumps(env,indent=2))
        print(json.dumps(env),flush=True);return
    import egomimic.trainHydra as entry
    from egomimic.pipeline import stages_action_flow as stages
    from egomimic.models import unite_dit,unite_action_decoder
    from lightning import Callback
    from lightning.pytorch.callbacks import ModelCheckpoint,LearningRateMonitor
    assert a.variant
    env=json.loads((root/"CPU_ENVELOPE.json").read_text())
    assert sha(__file__)==env["script_sha256"] and sha(root/"profile-config.yaml")==env["config_sha256"]
    assert torch.cuda.is_available()
    out=root/a.variant;assert not out.exists();out.mkdir()
    originals=(unite_dit.checkpoint,unite_action_decoder.checkpoint,entry.instantiate_callbacks,stages.ActionFlowObjectiveStage.forward)
    refs={};summaries={};checks={"tensors":0,"max_abs_error":0.0};phase_state={};exact_records={}
    def cpu_copy(value):
        if torch.is_tensor(value): return value.detach().cpu().clone()
        if isinstance(value,dict): return {k:cpu_copy(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)): return [cpu_copy(v) for v in value]
        return value
    def compare(got,expected,path):
        if torch.is_tensor(expected):
            assert torch.is_tensor(got) and got.shape==expected.shape and got.dtype==expected.dtype,path
            assert torch.isfinite(got).all() and torch.isfinite(expected).all(),path
            torch.testing.assert_close(got,expected,rtol=env["rtol"],atol=env["atol"],msg=lambda m:path+" "+m)
            checks["tensors"]+=1
            if got.numel(): checks["max_abs_error"]=max(checks["max_abs_error"],float((got-expected).abs().max()))
        elif isinstance(expected,dict):
            assert got.keys()==expected.keys(),path
            for k in expected:compare(got[k],expected[k],path+"/"+str(k))
        elif isinstance(expected,list):
            assert len(got)==len(expected),path
            for i in range(len(expected)): compare(got[i],expected[i],path+"/"+str(i))
        else: assert got==expected,path
    def check(key,value,exact=False):
        value=cpu_copy(value)
        if exact:
            assert torch.isfinite(value).all(), key
            exact_records.setdefault(phase_state["name"],{})[key]={"dtype":str(value.dtype),"shape":list(value.shape),"sha256":hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()}
        if phase_state["name"]=="baseline":refs[key]=value
        else:
            if exact:
                if not torch.equal(value,refs[key]):
                    (out/"EXACT_MISMATCH.json").write_text(json.dumps({"update":phase_state["update"],"key":key,"baseline":exact_records["baseline"][key],"candidate":exact_records[phase_state["name"]][key],"status":"REJECTED_EXACT_MISMATCH"},indent=2))
                    raise AssertionError("EXACT_LOSS_MISMATCH "+key)
                checks["exact_loss_checks"]=checks.get("exact_loss_checks",0)+1
            else:compare(value,refs[key],key)
    for phase in ("baseline",a.variant):
        phase_state.update(name=phase,update=1,calls=0)
        counts={};rows=[]
        unite_dit.checkpoint=policy(originals[0],phase,"dit",counts)
        unite_action_decoder.checkpoint=policy(originals[1],phase,"decoder",counts)
        def objective(self,batch):
            result=originals[3](self,batch)
            phase_state["calls"]+=1
            if phase_state["update"] in env["parity_updates"]:
                keys=[self.target_key,self.residual_key,self.reconstruction_key,self.decoded_residual_key,self.loss_key,self.flow_log_key,self.reconstruction_log_key,self.action_velocity_log_key]
                check("objective/"+str(phase_state["update"])+"/"+str(phase_state["calls"]),{k:result[k] for k in keys})
            for label in ("loss_key","total_log_key","flow_log_key","reconstruction_log_key","reconstruction_l1_log_key","action_velocity_log_key"):
                key=getattr(self,label)
                check("exact/"+str(phase_state["update"])+"/"+str(phase_state["calls"])+"/"+label,result[key],exact=True)
            return result
        stages.ActionFlowObjectiveStage.forward=objective
        class Audit(Callback):
            def on_fit_start(self,trainer,module):
                check("initial_parameters",{n:p for n,p in module.named_parameters()})
                self.parameters=list(module.named_parameters())
            def on_train_batch_start(self,trainer,module,batch,batch_idx):
                phase_state["update"]=int(trainer.global_step)+1;phase_state["calls"]=0
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();self.start=time.perf_counter()
            def on_after_backward(self,trainer,module):
                if phase_state["update"] in env["parity_updates"]:
                    check("gradients/"+str(phase_state["update"]),{n:p.grad for n,p in self.parameters})
            def on_train_batch_end(self,trainer,module,outputs,batch,batch_idx):
                torch.cuda.synchronize();elapsed=time.perf_counter()-self.start
                loss=outputs["loss"] if isinstance(outputs,dict) else outputs
                assert torch.isfinite(loss).all() and phase_state["calls"]==2
                check("exact_total/"+str(phase_state["update"]),loss,exact=True)
                if phase_state["update"] in env["parity_updates"]:
                    check("loss/"+str(phase_state["update"]),loss)
                    check("parameters/"+str(phase_state["update"]),{n:p for n,p in self.parameters})
                    # Complete optimizer state, including both Muon and AdamW families.
                    check("optimizer/"+str(phase_state["update"]),[o.state_dict() for o in trainer.optimizers])
                rows.append({"update":int(trainer.global_step),"loss":float(loss.detach()),"seconds":elapsed,"peak_allocated":torch.cuda.max_memory_allocated(),"peak_reserved":torch.cuda.max_memory_reserved()})
                (out/(phase+"-PROGRESS.json")).write_text(json.dumps({"rows":rows,"parity":checks,"policy_counts":counts},indent=2))
                (out/"EXACT_LOSS_HASHES.json").write_text(json.dumps(exact_records,indent=2))
        def callbacks(config):
            return [c for c in originals[2](config) if not isinstance(c,(ModelCheckpoint,LearningRateMonitor))]+[Audit()]
        entry.instantiate_callbacks=callbacks
        cfg=OmegaConf.load(root/"profile-config.yaml")
        with open_dict(cfg):
            cfg.paths.output_dir=str(out/phase);cfg.paths.log_dir=str(out/phase);cfg.trainer.default_root_dir=str(out/phase)
        for key in ("ICE_REQUEUE_OWNER","ICE_CHILD_REQUEUE_DISABLED","ICE_RESUME_CHECKPOINT","ICE_RESUME_CHECKPOINT_METADATA_JSON"):os.environ.pop(key,None)
        _,objects=entry.train(cfg)
        assert objects["trainer"].global_step==1000 and len(rows)==1000
        summaries[phase]={"rows":rows,"steady_median_seconds":statistics.median(r["seconds"] for r in rows[env["timing_warmup"]:] if r["update"] not in env["parity_updates"]),"peak_allocated":max(r["peak_allocated"] for r in rows),"policy_counts":counts}
        del objects
        entry.instantiate_callbacks=originals[2];stages.ActionFlowObjectiveStage.forward=originals[3]
        gc.collect();torch.cuda.empty_cache()
    required={"baseline-control":"dit/checkpoint","dit-half":"dit/direct","dit-off":"dit/direct","decoder-off":"decoder/direct"}[a.variant]
    assert summaries[a.variant]["policy_counts"].get(required,0)>0 and checks["tensors"]>0 and checks["exact_loss_checks"]==13000
    result={"status":"PARITY_AND_TIMING_PASS","identity":env,"variant":a.variant,"gpu":torch.cuda.get_device_name(),"deterministic":torch.are_deterministic_algorithms_enabled(),"cublas_workspace_config":os.environ.get("CUBLAS_WORKSPACE_CONFIG"),"parity":checks,"runs":summaries,"speedup":summaries["baseline"]["steady_median_seconds"]/summaries[a.variant]["steady_median_seconds"],"limitations":env["limits"]+"; snapshots1/2/100/500/1000 include parity copies; exclude them and warmup20 from timing; no adoption into main training"}
    (out/"RESULT.json").write_text(json.dumps(result,indent=2));print("OPTIMIZATION_RESULT "+json.dumps(result),flush=True)
if __name__=="__main__":main()
