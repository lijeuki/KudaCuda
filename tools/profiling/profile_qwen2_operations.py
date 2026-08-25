"""Measure per-operation CUDA time for a Qwen2 decode step.

This is an eager-model ceiling diagnostic. It deliberately does not claim graph
or native-kernel speedups; it identifies which decoder operation families must
be covered before a whole-model throughput target can be evaluated.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from KudaCuda.harness.compare import write_json_evidence


def operation_family(module_name: str) -> str:
    """Map a concrete Qwen2 module name to a stable profiling family."""

    if module_name.endswith((".q_proj", ".k_proj", ".v_proj")):
        return "qkv_projection"
    if module_name.endswith(".o_proj"):
        return "output_projection"
    if module_name.endswith(".gate_proj"):
        return "mlp_gate_projection"
    if module_name.endswith(".up_proj"):
        return "mlp_up_projection"
    if module_name.endswith(".down_proj"):
        return "mlp_down_projection"
    if module_name.endswith(".self_attn"):
        return "attention"
    if module_name.endswith(".mlp"):
        return "mlp_gate_up_down"
    if module_name.endswith(".input_layernorm"):
        return "input_rmsnorm"
    if module_name.endswith(".post_attention_layernorm"):
        return "post_attention_rmsnorm"
    return "other"


def aggregate_operation_timings(samples: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Aggregate event durations without inventing missing operation data."""

    grouped: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        family = str(sample["family"])
        grouped[family].append(float(sample["ms"]))
    return {
        family: {
            "calls": float(len(values)),
            "total_ms": float(sum(values)),
            "mean_ms": float(sum(values) / len(values)),
            "max_ms": float(max(values)),
        }
        for family, values in sorted(grouped.items())
    }


def _dtype(torch: Any, name: str) -> Any:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _event_pair(torch: Any) -> tuple[Any, Any]:
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Explain the difference between a CPU and a GPU in three clear bullet points.")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    payload: dict[str, Any] = {
        "status": "blocked",
        "model": args.model,
        "measurement": "CUDA events around decoder module calls during eager single-token decode",
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.steps,
    }

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as error:
        payload["reason"] = f"Required dependency unavailable: {error}"
        write_json_evidence(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        payload["reason"] = "A CUDA device is required for operation profiling"
        write_json_evidence(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        dtype = _dtype(torch, args.dtype)
        load_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if args.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = {"": str(device)}
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        if not args.load_in_4bit:
            model = model.to(device)
        model.eval()
        inputs = tokenizer(args.prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
    except Exception as error:
        payload["reason"] = f"Model setup failed: {type(error).__name__}: {error}"
        write_json_evidence(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    samples: list[dict[str, Any]] = []
    handles: list[Any] = []
    modules = getattr(getattr(model, "model", None), "layers", [])
    for index, layer in enumerate(modules):
        layer_modules = (
            ("input_layernorm", getattr(layer, "input_layernorm", None)),
            ("self_attn", getattr(layer, "self_attn", None)),
            ("self_attn.q_proj", getattr(getattr(layer, "self_attn", None), "q_proj", None)),
            ("self_attn.k_proj", getattr(getattr(layer, "self_attn", None), "k_proj", None)),
            ("self_attn.v_proj", getattr(getattr(layer, "self_attn", None), "v_proj", None)),
            ("self_attn.o_proj", getattr(getattr(layer, "self_attn", None), "o_proj", None)),
            ("post_attention_layernorm", getattr(layer, "post_attention_layernorm", None)),
            ("mlp", getattr(layer, "mlp", None)),
            ("mlp.gate_proj", getattr(getattr(layer, "mlp", None), "gate_proj", None)),
            ("mlp.up_proj", getattr(getattr(layer, "mlp", None), "up_proj", None)),
            ("mlp.down_proj", getattr(getattr(layer, "mlp", None), "down_proj", None)),
        )
        for suffix, module in layer_modules:
            if module is None:
                continue
            name = f"model.layers.{index}.{suffix}"

            def before(_module: Any, _inputs: Any, *, _name: str = name) -> None:
                start, end = _event_pair(torch)
                start.record()
                setattr(_module, "_kudacuda_profile_events", (start, end, _name))

            def after(_module: Any, _inputs: Any, _output: Any) -> None:
                start, end, name_value = getattr(_module, "_kudacuda_profile_events")
                end.record()
                samples.append({"family": operation_family(name_value), "module": name_value, "events": (start, end)})

            handles.extend((module.register_forward_pre_hook(before), module.register_forward_hook(after)))

    try:
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            current_mask = attention_mask

            for _ in range(max(0, args.warmup_steps)):
                next_mask = torch.ones((1, 1), dtype=current_mask.dtype, device=device)
                current_mask = torch.cat((current_mask, next_mask), dim=-1)
                outputs = model(
                    input_ids=next_token,
                    attention_mask=current_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

            samples.clear()
            torch.cuda.reset_peak_memory_stats(device)
            decode_events: list[tuple[Any, Any]] = []
            for _ in range(max(1, args.steps)):
                next_mask = torch.ones((1, 1), dtype=current_mask.dtype, device=device)
                current_mask = torch.cat((current_mask, next_mask), dim=-1)
                decode_start, decode_end = _event_pair(torch)
                decode_start.record()
                outputs = model(
                    input_ids=next_token,
                    attention_mask=current_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                decode_end.record()
                decode_events.append((decode_start, decode_end))
                past_key_values = outputs.past_key_values
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

            torch.cuda.synchronize(device)
    except Exception as error:
        payload["reason"] = f"Profile failed: {type(error).__name__}: {error}"
        write_json_evidence(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    finally:
        for handle in handles:
            handle.remove()

    measured_samples: list[dict[str, Any]] = []
    for sample in samples:
        start, end = sample.pop("events")
        sample["ms"] = float(start.elapsed_time(end))
        measured_samples.append(sample)
    decode_ms = [float(start.elapsed_time(end)) for start, end in decode_events]
    payload.update(
        {
            "status": "completed",
            "gpu": torch.cuda.get_device_name(device),
            "architecture_hint": "sm_86 on the RTX 3050; compile target is external to this profiler",
            "vram_capacity_mib": float(torch.cuda.get_device_properties(device).total_memory / (1024**2)),
            "peak_vram_mib": float(torch.cuda.max_memory_allocated(device) / (1024**2)),
            "decode_step_ms": decode_ms,
            "decode_step_median_ms": sorted(decode_ms)[len(decode_ms) // 2],
            "operations": aggregate_operation_timings(measured_samples),
            "samples": measured_samples,
            "scope": "eager baseline only; graph/native dispatch is not inferred",
        }
    )
    write_json_evidence(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
