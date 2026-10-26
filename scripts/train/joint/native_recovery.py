"""Operational proof helpers for the native conditional Elmo model.

These reuse scientific checkpoint and sampler implementations. They neither
train nor authorize an interrupted production allocation by themselves.
"""
import hashlib
import json
import math
from contextlib import contextmanager

import numpy as np
import torch

from scripts.train.joint.contract import need, sha


def checkpoint_science(saved_config, current_config):
    """Loss and sampling policy must match even when tensor shapes are unchanged."""
    from omegaconf import OmegaConf

    saved = OmegaConf.to_container(OmegaConf.create(saved_config), resolve=True)
    current = OmegaConf.to_container(OmegaConf.create(current_config), resolve=True)
    need(all(saved[key] == current[key] for key in ("model", "data")),
         "Recovery changed checkpoint-bound model/loss/data/sampling semantics")


def state_digest(value):
    """Stable digest of RNG/container state without drawing random numbers."""
    h = hashlib.sha256()

    def add(item):
        h.update(type(item).__name__.encode() + b"\0")
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            h.update(str(tensor.dtype).encode())
            h.update(str(tuple(tensor.shape)).encode())
            h.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            h.update(str(item.dtype).encode())
            h.update(str(item.shape).encode())
            h.update(item.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item):
                add(key)
                add(item[key])
        elif isinstance(item, (list, tuple)):
            for entry in item:
                add(entry)
        else:
            h.update(json.dumps(item, allow_nan=False).encode())
        h.update(b"\xff")

    add(value)
    return h.hexdigest()


def conditional_record(optimizer, named_parameters, counts, step):
    """Bind counters to the real optimizer parameter order and moment shapes."""
    from egomimic.pl_utils.joint_smoke import check_conditional_adam_state

    named = list(named_parameters)
    names = [name for name, _ in named]
    parameters = [parameter for _, parameter in named]
    ordered = [p for group in optimizer.param_groups for p in group["params"]]
    need(len(ordered) == len(parameters) == len({id(p) for p in ordered})
         and all(a is b for a, b in zip(ordered, parameters)),
         "Conditional optimizer parameter identity/order changed")
    record = dict(schema="native-conditional-optimizer-usage-v1", global_step=step,
                  parameter_names=names, parameter_update_counts=dict(counts))
    state = optimizer.state_dict()
    ids = [key for group in state["param_groups"] for key in group["params"]]
    need(set(state["state"]) <= set(ids), "Unexpected extra optimizer state")
    check_conditional_adam_state(state, record, step)
    for parameter in parameters:
        moments = optimizer.state.get(parameter)
        if moments is not None:
            need(all(moments[key].shape == parameter.shape
                     for key in ("exp_avg", "exp_avg_sq")),
                 "Conditional Adam moment shape changed")
    return record


def restored_conditional_counts(trainer, module, checkpoint):
    from egomimic.pl_utils.joint_smoke import ConditionalOptimizerUsage

    record = checkpoint.get("native_conditional_optimizer_usage")
    need(isinstance(record, dict), "Missing native conditional checkpoint counts")
    actual = conditional_record(trainer.optimizers[0], module.named_parameters(),
                                record["parameter_update_counts"], trainer.global_step)
    need(actual == record, "Saved conditional parameter names/order changed")
    callbacks = [c for c in trainer.callbacks if isinstance(c, ConditionalOptimizerUsage)]
    need(len(callbacks) == 1 and callbacks[0].counts == record["parameter_update_counts"]
         and callbacks[0].restored == record,
         "Maintained conditional usage callback was not strictly restored")
    return dict(record["parameter_update_counts"]), callbacks[0]


def install_rng_observer(module, saved, equal_state):
    """Observe the existing restore at the first default training-step boundary.

    WorldModelTrainingBehavior restores RNG and acknowledges the sampler before
    this call. The observer draws no randomness and never restores state itself.
    """
    from egomimic.pl_utils.world_model_behavior import _capture_rng_state, _rank_rng_state

    expected = _rank_rng_state(saved)
    need(expected.get("cuda_scope") == "current_device"
         and len(expected.get("torch_cuda", [])) == 1,
         "Recovery requires exactly one saved CUDA RNG state per rank")
    previous = module._default_training_step
    record = dict(observed=False, expected_sha256=state_digest(expected))

    def observe(*args, **kwargs):
        need(module.training_behavior._pending_rng is None,
             "Scientific RNG restore did not execute before the training step")
        actual = _capture_rng_state(current_cuda_only=True)
        need(equal_state(actual, expected), "Actual first-update rank RNG differs from checkpoint")
        record.update(observed=True, actual_sha256=state_digest(actual))
        module._default_training_step = previous
        return previous(*args, **kwargs)

    module._default_training_step = observe

    def cleanup():
        module._default_training_step = previous

    return record, cleanup


def expected_metrics(network):
    return {f"val/{emb}/{kind}/{('action_command_mse' if stream == 'action' else stream + '_normalized_mse')}"
            for emb in network.pose_dims for kind in ("forecast", "completion")
            for stream in network.active_streams(emb)}


@contextmanager
def preserved_rng():
    from egomimic.pl_utils.world_model_behavior import _capture_rng_state, _restore_rng_state

    state = _capture_rng_state(current_cuda_only=True)
    try:
        yield
    finally:
        _restore_rng_state(state)


def validate_hidden_sample(sample, output, streams):
    need(set(output) == set(streams), "Unexpected native sampled streams")
    need("language_features" in sample and "pixel_metadata" in sample,
         "Missing sampled subtask/language provenance")
    for name in streams:
        source, value = sample["world"][name], output[name]
        need(torch.isnan(source["value"][~source["observed"]]).all(),
             "Withheld labels were not poisoned before recovery sampling")
        need(value.shape == source["value"].shape and value.dtype == torch.float32
             and torch.isfinite(value).all(), "Nonfinite or malformed recovery sample")
        need(torch.equal(value[source["observed"]], source["value"][source["observed"]]),
             "Recovery sampling changed clean observations")
        need(not value[~source["valid"]].count_nonzero(), "Recovery sampling emitted padded values")


def sample_both_embodiments(module, network, output):
    """Use the configured maintained 20-step sampler on harvested held-out data."""
    sampler = module.model.pipeline.stage_by_id("joint_sampler").sampler
    need(sampler.state_steps == 20 and sampler.action_steps == 20,
         "Recovery hidden-label sample must retain the exact20-step schedule")
    was_training = module.model.training
    rows = {}
    try:
        module.model.eval()
        module.model.device = next(module.parameters()).device
        need(module.model.device.type == "cuda", "Actual native recovery sampling requires CUDA")
        with preserved_rng(), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for emb in sorted(network.pose_dims):
                path = output / f"sample-input-{emb}.pt"
                need(path.is_file() and not path.is_symlink(), "Missing actual validation sample")
                sample = torch.load(path, map_location="cpu", weights_only=False)
                groups = module.model.process_batch_for_training({emb: sample})
                # Check poisoning before invoking the maintained sampler too.
                for stream in groups[emb]["world"].values():
                    need(torch.isnan(stream["value"][~stream["observed"]]).all(),
                         "Recovery sample contains withheld clean targets")
                result = module.model.forward_eval(groups)[emb]["image_proprio_prediction"]
                validate_hidden_sample(groups[emb], result, network.active_streams(emb))
                rows[emb] = dict(sample_input_sha256=sha(path), state_steps=20,
                                 action_steps=20 if emb in network.action_dims else 0,
                                 streams=list(result), finite=True, hidden_targets_nan=True)
    finally:
        module.model.train(was_training)
    return rows


def summarize_native_phases(phases):
    """Validate all-rank conditional evidence for the actual2+4 diagnostic."""
    expected_streams = {"eva_bimanual": ["image", "proprio", "action", "camera"],
                        "human_bimanual": ["image", "proprio", "camera"]}
    expected = {f"val/{emb}/{kind}/{('action_command_mse' if s == 'action' else s + '_normalized_mse')}"
                for emb, streams in expected_streams.items()
                for kind in ("forecast", "completion") for s in streams}
    need(len(phases) == 2 and len(phases[0]) == len(phases[1]) in (2, 4),
         "Native recovery requires both diagnostic phases and all ranks")
    for phase, (start, stop) in zip(phases, ((50, 52), (52, 56))):
        first = phase[0]
        for row in phase:
            need(row.get("multimodal") is True
                 and row.get("private_bank_optimizer_steps_verified") is True
                 and row.get("human_action_absent") is True
                 and row.get("active_streams_by_embodiment") == expected_streams,
                 "Missing native conditional recovery proof")
            record = row["conditional_optimizer_record"]
            initial, final = row["initial_parameter_usage"], record["parameter_update_counts"]
            names = record["parameter_names"]
            need(record["schema"] == "native-conditional-optimizer-usage-v1"
                 and record["global_step"] == stop
                 and set(initial) == set(final) == set(names) and len(names) == len(set(names))
                 and all(type(initial[n]) is type(final[n]) is int
                         and 0 <= initial[n] <= start and initial[n] <= final[n] <= stop
                         and final[n] - initial[n] <= stop - start for n in names),
                 "Invalid restored conditional counter progression")
            need(initial == first["initial_parameter_usage"]
                 and record == first["conditional_optimizer_record"],
                 "Conditional counter restoration differs across ranks")
            batches = row["embodiment_optimizer_batches"]
            need(set(batches) == set(expected_streams)
                 and all(type(v) is int and v >= 0 for v in batches.values())
                 and sum(batches.values()) == stop - start
                 and batches == first["embodiment_optimizer_batches"],
                 "Missing actual native embodiment updates")
            rng = row["actual_rank_rng_restore"]
            need(rng.get("observed") is True and isinstance(rng.get("actual_sha256"), str)
                 and len(rng["actual_sha256"]) == 64
                 and rng["actual_sha256"] == rng["expected_sha256"],
                 "Actual rank RNG restore is not proven")
            metrics = row["validation_metrics"]
            need(set(metrics) == expected and all(math.isfinite(v) for v in metrics.values())
                 and metrics == first["validation_metrics"],
                 "Missing finite native validation metrics on all ranks")
            samples = row["hidden_label_samples"]
            need(set(samples) == set(expected_streams), "Missing hidden-label native samples")
            for emb, sample in samples.items():
                need(sample.get("state_steps") == 20
                     and sample.get("action_steps") == (20 if emb == "eva_bimanual" else 0)
                     and sample.get("streams") == expected_streams[emb]
                     and sample.get("finite") is True and sample.get("hidden_targets_nan") is True
                     and isinstance(sample.get("sample_input_sha256"), str)
                     and len(sample["sample_input_sha256"]) == 64,
                     "Missing finite20-step hidden-label recovery sample")
    for a, b in zip(phases[0], phases[1]):
        need(a["conditional_optimizer_record"]["parameter_update_counts"] == b["initial_parameter_usage"],
             "Second process did not restore conditional optimizer counts")
    need(all(sum(phase[0]["embodiment_optimizer_batches"][emb] for phase in phases) > 0
             for emb in expected_streams), "Actual2+4 recovery did not exercise both embodiments")
    return dict(native_multimodal=True, private_bank_optimizer_steps_verified=True,
                actual_all_rank_rng_restore_observed=True, both_embodiments_optimized=True,
                both_embodiments_hidden_nan_20step_sample=True,
                diagnostic_scope="scheduled-smoke-checkpoint-only")
