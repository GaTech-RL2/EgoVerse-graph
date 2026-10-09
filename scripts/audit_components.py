"""Offline constructor gate; meta tensors do not constitute a forward/training test.

External parameters are suppressed through the same scoped restore capability
used by strict checkpoint loading. Qwen's pinned architecture JSON is local;
tokenizers are explicit placeholders that cannot tokenize. PI remains unbound,
so an unselected backend is never imported. No resolver or hardware is opened.
"""

import argparse
import gc
import hashlib
import json
import socket
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from egomimic.pipeline.construction import checkpoint_construction
from egomimic.pipeline.inference_config import build_inference_config
from egomimic.trainHydra import _instantiate_model_wrapper
from scripts.audit_constructor_inputs import evaluator_inputs, model_inputs
from scripts.audit_hydra_configs import CONFIGS, ROOT, compose_for_audit

# This is a shipped, explicitly non-runnable fragment, not ignored migration debt.
FRAGMENTS = {
    "model/pi0.5_base.yaml": "Select a concrete domains/ac_keys recipe; owner: graph integration"
}
QWEN_METADATA = ROOT / "tests/fixtures/qwen3_embedding_06b_config.json"
QWEN_METADATA_SHA256 = (
    "b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9"
)


class NoTokenizer:
    def __call__(self, *args, **kwargs):
        raise AssertionError("Constructor audit cannot tokenize or execute a model")


def offline_model_config(name, **kwargs):
    from transformers import AutoConfig

    if name != "Qwen/Qwen3-Embedding-0.6B":
        raise ValueError(f"No pinned offline architecture fixture for {name!r}")
    raw = QWEN_METADATA.read_bytes()
    if hashlib.sha256(raw).hexdigest() != QWEN_METADATA_SHA256:
        raise ValueError("Offline architecture metadata hash mismatch")
    data = json.loads(raw)
    return AutoConfig.for_model(data.pop("model_type"), **data)


def _fingerprint(config):
    return json.dumps(OmegaConf.to_container(config, resolve=True), sort_keys=True)


@contextmanager
def offline_static_buffer_construction():
    """Materialize only deterministic scalar-derived FSQ buffers on CPU.

    FSQ has no learned parameters; its product.item()/codebook and OAT's scalar
    drop-path schedule cannot be materialized on meta. Keep production source
    untouched and retain every requested-device parameter shape and dtype.
    """
    from egomimic.models.oat.tokenizer.oat.model.transformer import Transformer
    from egomimic.models.oat.tokenizer.oat.quantizer.fsq import FSQ
    from egomimic.pipeline.stages_libero_arc import LiberoArcStage

    original = FSQ.__init__
    linspace = torch.linspace
    representation_context = LiberoArcStage.representation_context

    def codec_metadata(module):
        # The maintained method derives units from scalar codec attributes;
        # its temporary ones/new_tensor/tolist is metadata, not model weights.
        with torch.device("cpu"):
            return representation_context(module)

    def scalar_schedule(*args, **kwargs):
        caller = sys._getframe(1)
        if caller.f_code is Transformer.__init__.__code__:
            expected = (0, caller.f_locals["drop_path_rate"], caller.f_locals["depth"])
            if args != expected or kwargs:
                raise AssertionError("OAT scalar schedule constructor contract changed")
            return linspace(*args, device="cpu")
        kwargs.setdefault("device", torch.empty(0).device)
        return linspace(*args, **kwargs)

    def construct(module, *args, **kwargs):
        device = torch.empty(0).device
        with torch.device("cpu"):
            original(module, *args, **kwargs)
        if tuple(module.parameters()):
            raise AssertionError(
                "FSQ static constructor must not own learned parameters"
            )
        module.to(device)

    with (
        patch.object(FSQ, "__init__", construct),
        patch.object(torch, "linspace", scalar_schedule),
        patch.object(LiberoArcStage, "representation_context", codec_metadata),
    ):
        yield


def audit_components(paths=None):
    from transformers import AutoConfig, AutoTokenizer

    records, cached = [], {}
    with (
        checkpoint_construction(),
        offline_static_buffer_construction(),
        patch.object(
            socket.socket,
            "connect",
            side_effect=AssertionError(
                "Constructor preflight cannot access the network"
            ),
        ),
        patch.object(AutoConfig, "from_pretrained", side_effect=offline_model_config),
        patch.object(AutoTokenizer, "from_pretrained", return_value=NoTokenizer()),
    ):
        for path in sorted(CONFIGS.rglob("*.yaml") if paths is None else paths):
            name = path.relative_to(CONFIGS).as_posix()
            row = {"path": name, "components": {}}
            try:
                with compose_for_audit(path) as cfg:
                    for key in ("data", "evaluator", "model"):
                        config = cfg.get(key)
                        if not config or "_target_" not in config:
                            continue
                        cache_key = key, _fingerprint(config)
                        if key == "model" and cfg.get("benchmark"):
                            cache_key += (_fingerprint(cfg.benchmark),)
                        if cache_key not in cached:
                            if key == "data":
                                dm = instantiate(config, _recursive_=False)
                                cached[cache_key] = dm.preflight_configuration()
                            elif key == "evaluator":
                                inputs, evidence = evaluator_inputs(config)
                                evaluator = instantiate(inputs)
                                evaluator.data_requirements()
                                evaluator.trainer_overrides()
                                cached[cache_key] = {
                                    "constructor": "passed",
                                    **evidence,
                                }
                            else:
                                artifact = build_inference_config(cfg)
                                if name in FRAGMENTS:
                                    assert artifact["status"] == "unsupported", artifact
                                    cached[cache_key] = {
                                        "fragment": FRAGMENTS[name],
                                        "reason": artifact["reason"],
                                    }
                                else:
                                    audit_cfg = OmegaConf.create(
                                        OmegaConf.to_container(cfg, resolve=True)
                                    )
                                    OmegaConf.update(
                                        audit_cfg,
                                        "model.pipeline.device",
                                        "meta",
                                        force_add=True,
                                    )
                                    with (
                                        torch.device("meta"),
                                        model_inputs(
                                            audit_cfg, CONFIGS
                                        ) as input_evidence,
                                    ):
                                        wrapper = _instantiate_model_wrapper(audit_cfg)
                                    graph = wrapper.model
                                    # Honor configured TrainingBehavior parameter binding
                                    # (including named/composite optimizers). Late-bound
                                    # backends have no parameters until data is bound.
                                    placeholder = not any(
                                        True for _ in wrapper.parameters()
                                    )
                                    if placeholder:
                                        wrapper.nets.register_parameter(
                                            "_audit_placeholder",
                                            torch.nn.Parameter(
                                                torch.zeros(1, device="cpu")
                                            ),
                                        )
                                    wrapper.trainer = SimpleNamespace(model=wrapper)
                                    wrapper.configure_optimizers()
                                    if artifact["status"] == "ready":
                                        declaration = artifact["inference_graph"]
                                        runnable, excluded = graph.pipeline.plan(
                                            declaration["input"]["keys"],
                                            mode="inference",
                                        )
                                        blocked = [
                                            missing
                                            for _, missing in excluded
                                            if missing != ["<train-only>"]
                                        ]
                                        if blocked:
                                            raise ValueError(
                                                f"Declared observations leave blocked stages: {blocked}"
                                            )
                                        if declaration["native_output"]["key"] not in {
                                            key
                                            for stage in runnable
                                            for key in stage.contract("inference")[1]
                                        }:
                                            raise ValueError(
                                                "Ready declaration does not produce its native output"
                                            )
                                    cached[cache_key] = {
                                        "stages": len(graph.pipeline.stages),
                                        "inference": artifact["status"],
                                        "reason": artifact.get("reason"),
                                        "optimizer_scheduler": "passed",
                                        "late_bound_parameter_placeholder": placeholder,
                                        **input_evidence,
                                    }
                                    del graph, wrapper
                                    gc.collect()
                        row["components"][key] = cached[cache_key]
                row["status"] = "passed"
            except Exception as error:
                row.update(status="failed", error=f"{type(error).__name__}: {error}")
            records.append(row)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--path",
        action="append",
        help="Exact config-relative path; default audits every YAML",
    )
    args = parser.parse_args()
    selected = None if args.path is None else [CONFIGS / path for path in args.path]
    if selected is not None and any(
        not path.is_file() or not path.resolve().is_relative_to(CONFIGS.resolve())
        for path in selected
    ):
        parser.error("Selected audit paths must be existing YAML files under CONFIGS")
    results = audit_components(selected)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    failures = [row for row in results if row["status"] != "passed"]
    print(f"{len(results)-len(failures)}/{len(results)} constructor contexts passed")
    for row in failures:
        print(row["path"], row["error"])
    raise SystemExit(bool(failures))
