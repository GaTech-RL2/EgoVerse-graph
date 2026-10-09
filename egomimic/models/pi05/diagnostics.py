"""OpenPI-specific attention capture behind the shared diagnostic interface."""

import inspect
from collections import OrderedDict
from contextlib import contextmanager

import torch

from egomimic.eval.diagnostic_provider import DiagnosticProvider


@contextmanager
def capture_attention(policy, *, emit_combined=True, mask_padding=False):
    """Capture prefix keys and the first action-expert denoising call.

    Hooks and the prefix wrapper are scoped to one inference and restored even
    on errors. Layer layout knowledge belongs here, never in the evaluator.
    """
    if getattr(policy, "_attention_capture_active", False):
        raise RuntimeError("Concurrent attention capture on one policy is unsupported")
    prefix = policy.paligemma_with_expert.paligemma.language_model.layers
    expert = policy.paligemma_with_expert.gemma_expert.model.layers
    layers = [
        (f"paligemma_layer_{i:02d}", layer.self_attn.k_proj, True)
        for i, layer in enumerate(prefix)
    ]
    layers += [
        (f"expert_layer_{i:02d}", layer.self_attn.k_proj, False)
        for i, layer in enumerate(expert)
    ]
    if not layers:
        raise ValueError("The configured OpenPI backend exposes no attention layers")
    captured, boundary, handles = OrderedDict(), {}, []
    original = policy.embed_prefix
    instance_override = policy.__dict__.get("embed_prefix")
    sampler = getattr(policy, "sample_actions", None)
    sampler_override = policy.__dict__.get("sample_actions")
    eager_sampler = inspect.unwrap(sampler) if callable(sampler) else None
    policy._attention_capture_active = True

    def wrapped(images, image_masks, language_tokens, language_masks):
        embeddings, padding, attention = original(
            images, image_masks, language_tokens, language_masks
        )
        boundary["images"] = embeddings.shape[1] - language_masks.shape[1]
        boundary["valid"] = padding.bool()
        return embeddings, padding, attention

    def hook_for(name, is_prefix):
        def capture(_module, _inputs, output):
            if name in boundary:
                return
            if not torch.is_tensor(output) or output.ndim != 3:
                raise ValueError(
                    f"{name} must emit (batch, tokens, width) attention keys"
                )
            if is_prefix and "images" not in boundary:
                raise RuntimeError(
                    "Attention capture ran before the declared prefix boundary"
                )
            boundary[name] = True
            slices = [(name, slice(None))]
            if is_prefix:
                cut = boundary["images"]
                slices = [
                    (f"{name}_img", slice(0, cut)),
                    (f"{name}_lang", slice(cut, None)),
                ]
                if emit_combined:
                    slices.append((f"{name}_combined", slice(None)))
            for label, selection in slices:
                keys = output[:, selection].detach().float().cpu()
                mask = (
                    boundary["valid"][:, selection].detach().cpu()
                    if is_prefix and mask_padding
                    else torch.ones(keys.shape[:2], dtype=torch.bool)
                )
                if mask.shape != keys.shape[:2]:
                    raise ValueError(
                        f"{label}: prefix mask and attention token shapes differ"
                    )
                captured[label] = {"tokens": keys, "mask": mask}

        return capture

    try:
        policy.embed_prefix = wrapped
        if eager_sampler is not None and eager_sampler is not sampler:
            # Hooks inspect Python tensor boundaries; do not capture their
            # temporary state in the deployment sampler's compiled graph.
            policy.sample_actions = eager_sampler
        for name, module, is_prefix in layers:
            handles.append(module.register_forward_hook(hook_for(name, is_prefix)))
        yield captured
        if len([name for name, _, _ in layers if name in boundary]) != len(layers):
            raise RuntimeError(
                "OpenPI inference did not visit every configured attention layer"
            )
    finally:
        for handle in handles:
            handle.remove()
        if instance_override is None:
            del policy.embed_prefix
        else:
            policy.embed_prefix = instance_override
        if eager_sampler is not None and eager_sampler is not sampler:
            if sampler_override is None:
                del policy.sample_actions
            else:
                policy.sample_actions = sampler_override
        del policy._attention_capture_active


class PI05AttentionProvider(DiagnosticProvider):
    capability = "token_activations"

    def __init__(self, stage_id="sampler", emit_combined=True, mask_padding=False):
        self.stage_id = str(stage_id)
        self.emit_combined = bool(emit_combined)
        self.mask_padding = bool(mask_padding)

    def run(self, model, batch, *, already_processed, **kwargs):
        if kwargs:
            raise TypeError(
                f"Unsupported attention diagnostic arguments: {sorted(kwargs)}"
            )
        values = batch if already_processed else model.process_batch_for_training(batch)
        stage = model.pipeline.stage_by_id(self.stage_id)
        if stage.backend is None:
            raise RuntimeError("Bind PI data context before attention diagnostics")
        policy = stage.policy_nets["policy"]
        output = OrderedDict()
        for source, source_batch in values.items():
            with capture_attention(
                policy, emit_combined=self.emit_combined, mask_padding=self.mask_padding
            ) as activations:
                prediction = model.forward_eval({source: source_batch})[source]
            output[source] = {"activations": activations, "predictions": prediction}
        return output
