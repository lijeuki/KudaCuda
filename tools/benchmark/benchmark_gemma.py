"""Run a non-fabricating Gemma baseline/sidecar generation benchmark.

The sidecar leg is optional and is reported as skipped or fallback-only when a
native library or an implemented attention dispatcher is unavailable. It never
substitutes synthetic speed, memory, or output values.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Mapping

# Keep direct file execution usable from the repository checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from KudaCuda.harness.compare import write_json_evidence
from KudaCuda.python.kudacuda_sidecar import (
    KudaCudaBridge,
    patch_llama_attention,
    patch_qwen2_attention,
    patch_qwen2_fused_qkv,
)
from KudaCuda.python.kudacuda_graph_plan import GraphContractError
from KudaCuda.python.kudacuda_static_decode import GraphStaticCache, StaticDecodeGraphRunner


def summarize_generation(
    *,
    prefill_ms: float,
    decode_ms: list[float],
    generated_tokens: int,
    peak_vram_mib: float | None,
) -> dict[str, Any]:
    """Summarize measured device timings without filling missing values."""

    total_ms = float(prefill_ms) + sum(float(value) for value in decode_ms)
    itl_ms = sum(decode_ms) / len(decode_ms) if decode_ms else None
    tokens_per_second = (
        float(generated_tokens) / (total_ms / 1000.0)
        if generated_tokens > 0 and total_ms > 0
        else None
    )
    return {
        "ttft_ms": float(prefill_ms),
        "itl_ms": itl_ms,
        "total_ms": total_ms,
        "tokens_per_second": tokens_per_second,
        "generated_tokens": int(generated_tokens),
        "peak_vram_mib": peak_vram_mib,
    }


def compare_generation_outputs(
    baseline: Mapping[str, Any],
    sidecar: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact generated token IDs and decoded text parity."""

    token_match = baseline.get("token_ids") == sidecar.get("token_ids")
    text_match = baseline.get("text") == sidecar.get("text")
    return {
        "status": "completed" if token_match and text_match else "failed",
        "token_match": token_match,
        "text_match": text_match,
    }


def _timed_call(torch: Any, device: Any, callback: Callable[[], Any]) -> tuple[Any, float]:
    if getattr(device, "type", str(device)) == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = callback()
        end.record()
        end.synchronize()
        return result, float(start.elapsed_time(end))

    start_time = time.perf_counter()
    result = callback()
    return result, (time.perf_counter() - start_time) * 1000.0


def _synchronize(torch: Any, device: Any) -> None:
    if getattr(device, "type", str(device)) == "cuda":
        torch.cuda.synchronize(device)


def _is_eos(token: Any, eos_token_id: int | list[int] | None) -> bool:
    if eos_token_id is None:
        return False
    values = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
    return int(token) in {int(value) for value in values}


def greedy_decode(
    *,
    torch: Any,
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    max_new_tokens: int,
    eos_token_id: int | list[int] | None,
    device: Any,
) -> dict[str, Any]:
    """Decode greedily and measure prefill/decode CUDA-event durations."""

    current_mask = attention_mask
    generated = input_ids.clone()
    wall_start = time.perf_counter()

    def prefill() -> Any:
        return model(
            input_ids=input_ids,
            attention_mask=current_mask,
            use_cache=True,
            return_dict=True,
        )

    outputs, prefill_ms = _timed_call(torch, device, prefill)
    past_key_values = outputs.past_key_values
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
    generated = torch.cat((generated, next_token), dim=-1)
    decode_ms: list[float] = []

    for _ in range(max(0, max_new_tokens - 1)):
        if _is_eos(next_token[0, 0].item(), eos_token_id):
            break
        next_mask = torch.ones(
            (current_mask.shape[0], 1),
            dtype=current_mask.dtype,
            device=current_mask.device,
        )
        current_mask = torch.cat((current_mask, next_mask), dim=-1)

        def decode_step() -> Any:
            return model(
                input_ids=next_token,
                attention_mask=current_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        outputs, elapsed_ms = _timed_call(torch, device, decode_step)
        decode_ms.append(elapsed_ms)
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=-1)

    _synchronize(torch, device)
    generated_tokens = int(generated.shape[-1] - input_ids.shape[-1])
    return {
        "token_ids": generated.detach().cpu().tolist()[0],
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "wall_total_ms": (time.perf_counter() - wall_start) * 1000.0,
        "generated_tokens": generated_tokens,
    }


def run_generation(
    *,
    torch: Any,
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    tokenizer: Any,
    max_new_tokens: int,
    device: Any,
) -> dict[str, Any]:
    """Warm up, measure, and return text plus memory/timing evidence."""

    with torch.inference_mode():
        greedy_decode(
            torch=torch,
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=min(2, max_new_tokens),
            eos_token_id=tokenizer.eos_token_id,
            device=device,
        )
        _synchronize(torch, device)
        if getattr(device, "type", str(device)) == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        measured = greedy_decode(
            torch=torch,
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            device=device,
        )

    peak_vram_mib = None
    if getattr(device, "type", str(device)) == "cuda":
        peak_vram_mib = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    summary = summarize_generation(
        prefill_ms=measured["prefill_ms"],
        decode_ms=measured["decode_ms"],
        generated_tokens=measured["generated_tokens"],
        peak_vram_mib=peak_vram_mib,
    )
    summary["wall_total_ms"] = measured["wall_total_ms"]
    summary["text"] = tokenizer.decode(measured["token_ids"], skip_special_tokens=True)
    summary["token_ids"] = measured["token_ids"]
    return summary


def run_graph_generation(
    *,
    torch: Any,
    model: Any,
    bridge: KudaCudaBridge,
    input_ids: Any,
    attention_mask: Any,
    tokenizer: Any,
    max_new_tokens: int,
    device: Any,
    tactic_name: str,
    kv_bucket: int,
    workspace_gqa: bool = False,
) -> dict[str, Any]:
    """Capture and replay one fixed-shape model decode forward."""

    if max_new_tokens < 2:
        raise GraphContractError("graph decode requires at least two requested new tokens")
    wall_start = time.perf_counter()
    with torch.inference_mode():
        prefill_start = torch.cuda.Event(enable_timing=True)
        prefill_end = torch.cuda.Event(enable_timing=True)
        prefill_start.record()
        prefill_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        prefill_end.record()
        prefill_end.synchronize()
        prefill_ms = float(prefill_start.elapsed_time(prefill_end))

        dynamic_cache = prefill_outputs.past_key_values
        prompt_length = int(input_ids.shape[-1])
        graph_cache = GraphStaticCache.from_dynamic_cache(
            torch,
            dynamic_cache,
            max_cache_len=int(kv_bucket),
        )
        graph_attention_mask = {
            "full_attention": torch.zeros(
                (1, 1, 1, int(kv_bucket)),
                dtype=attention_mask.dtype,
                device=device,
            )
        }
        next_token = torch.argmax(prefill_outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat((input_ids, next_token), dim=-1)
        runner = StaticDecodeGraphRunner(bridge, model, torch)
        workspace = (
            torch.empty((16 * 8 * (2 + 128),), dtype=torch.float32, device=device)
            if workspace_gqa
            else None
        )
        position_ids = torch.tensor(
            [[prompt_length]],
            dtype=torch.long,
            device=device,
        )
        graph_stream = torch.cuda.Stream(device=device)
        graph_stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(graph_stream):
            graph_stream_ptr = int(graph_stream.cuda_stream)
            capture_host_start = time.perf_counter()
            logits = runner.capture(
                input_ids=next_token,
                position_ids=position_ids,
                attention_mask=graph_attention_mask,
                cache_position=graph_cache.cache_position,
                past_key_values=graph_cache,
                logical_kv_len=prompt_length,
                kv_bucket=int(kv_bucket),
                tactic_name=tactic_name,
                stream_ptr=graph_stream_ptr,
                stream=graph_stream,
                workspace=workspace,
            )
            graph_stream.synchronize()
            capture_ms = (time.perf_counter() - capture_host_start) * 1000.0
            decode_ms = [capture_ms]
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=-1)
            logical_kv_len = prompt_length + 1

            for _ in range(max(0, max_new_tokens - 2)):
                if _is_eos(next_token[0, 0].item(), tokenizer.eos_token_id):
                    break
                position_ids = torch.tensor(
                    [[logical_kv_len]],
                    dtype=torch.long,
                    device=device,
                )

                def replay_step() -> Any:
                    return runner.replay(
                        next_token=next_token,
                        position_ids=position_ids,
                        cache_position=logical_kv_len,
                        logical_kv_len=logical_kv_len,
                        stream_ptr=graph_stream_ptr,
                    )

                replayed, elapsed_ms = _timed_call(torch, device, replay_step)
                decode_ms.append(elapsed_ms)
                next_token = torch.argmax(replayed[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat((generated, next_token), dim=-1)
                logical_kv_len += 1

    _synchronize(torch, device)
    generated_tokens = int(generated.shape[-1] - input_ids.shape[-1])
    token_ids = generated.detach().cpu().tolist()[0]
    try:
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
    except Exception as error:
        minimum = min(token_ids) if token_ids else None
        maximum = max(token_ids) if token_ids else None
        raise GraphContractError(
            f"graph output token decode failed: {type(error).__name__}: {error}; "
            f"token_id_range={minimum}..{maximum}; token_ids={token_ids}"
        ) from error
    summary = summarize_generation(
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        generated_tokens=generated_tokens,
        peak_vram_mib=float(torch.cuda.max_memory_allocated(device) / (1024**2)),
    )
    summary.update(
        {
            "wall_total_ms": (time.perf_counter() - wall_start) * 1000.0,
            "text": text,
            "token_ids": token_ids,
            "graph": {
                "status": "replayed" if runner.replay_count > 0 else "captured",
                "backend": runner.backend,
                "capture_ms": capture_ms,
                "replay_count": runner.replay_count,
                "kv_bucket": int(kv_bucket),
                "logical_kv_len_start": prompt_length,
                "native_dispatches_during_capture": int(bridge.native_attention_dispatches),
                "workspace_gqa": bool(workspace_gqa),
            },
        }
    )
    return summary


def _dtype(torch: Any, name: str) -> Any:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def _install_attention_patch(
    model: Any,
    bridge: KudaCudaBridge,
    *,
    qwen_tactic: str | None = None,
    fused_qkv: bool = False,
) -> tuple[Callable[[], None], str]:
    """Patch the model's concrete attention class with its exact tactic contract."""

    attention_cls = type(model.model.layers[0].self_attn)
    model_type = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    if model_type == "qwen2" or attention_cls.__name__ == "Qwen2Attention":
        arch = str(getattr(bridge, "manifest", {}).get("target_arch", "sm_86")).replace("sm_", "sm")
        tactic_name = qwen_tactic or bridge.manifest.get(
            "preferred_qwen_tactic",
            f"qwen2_gqa_decode_{arch}_fp16",
        )
        if fused_qkv:
            fused_tactic = bridge.manifest.get(
                "preferred_qwen_fused_qkv_tactic",
                f"qwen2_fused_rmsnorm_qkv_{arch}_nf4_v1",
            )
            return (
                patch_qwen2_fused_qkv(
                    model,
                    bridge,
                    tactic_name=fused_tactic,
                    attention_tactic_name=tactic_name,
                    attention_cls=attention_cls,
                ),
                tactic_name,
            )
        return (
            patch_qwen2_attention(
                bridge,
                attention_cls=attention_cls,
                tactic_name=tactic_name,
            ),
            tactic_name,
        )
    return (
        patch_llama_attention(
            bridge,
            attention_cls=attention_cls,
            tactic_name="gqa8_attn",
        ),
        "gqa8_attn",
    )


def _base_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "status": "blocked",
        "model_id": args.model,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "dtype": args.dtype,
        "sidecar": {"status": "not-run"},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-2-2b-it")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "KudaCuda" / "manifests" / "rtx3050-sm86.json",
    )
    parser.add_argument("--library", type=Path)
    parser.add_argument("--prompt", default="Explain the difference between a CPU and a GPU in three clear bullet points.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--qwen-tactic", default=None)
    parser.add_argument(
        "--fused-qkv",
        action="store_true",
        help="Opt into the experimental Qwen2 NF4 fused RMSNorm-QKV decoder hook.",
    )
    parser.add_argument("--graph-decode", action="store_true")
    parser.add_argument("--graph-kv-bucket", type=int, default=512)
    parser.add_argument(
        "--workspace-gqa",
        action="store_true",
        help="Use the experimental fixed-workspace two-stage Qwen2 GQA graph tactic.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = _base_payload(args)

    # This must happen before importing torch so allocator behavior is fixed.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        payload["reason"] = f"Required dependency unavailable: {error}"
        write_json_evidence(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        payload["reason"] = "CUDA is unavailable to PyTorch"
        write_json_evidence(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        dtype = _dtype(torch, args.dtype)
        load_kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if args.load_in_4bit:
            from transformers import BitsAndBytesConfig

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
    except Exception as error:  # model access, license, OOM, or load incompatibility
        payload["reason"] = f"Model setup failed: {type(error).__name__}: {error}"
        write_json_evidence(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    payload.update(
        {
            "status": "completed",
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "vram_capacity_mib": float(torch.cuda.get_device_properties(device).total_memory / (1024**2))
            if device.type == "cuda"
            else None,
            "measurement": "CUDA events for prefill/decode kernels; synchronized wall time recorded separately",
        }
    )

    try:
        allocated_after_load = float(torch.cuda.memory_allocated(device) / (1024**2)) if device.type == "cuda" else None
        payload["baseline"] = {
            "status": "completed",
            "allocated_after_load_mib": allocated_after_load,
            **run_generation(
                torch=torch,
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
                device=device,
            ),
        }
    except Exception as error:
        payload["status"] = "failed"
        payload["baseline"] = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
        write_json_evidence(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    if args.library is None:
        payload["sidecar"] = {
            "status": "skipped",
            "reason": "No --library supplied; baseline was run without claiming sidecar acceleration",
        }
    else:
        bridge = None
        unpatch = None
        try:
            bridge = KudaCudaBridge(manifest=args.manifest, lib_path=args.library)
            unpatch, tactic_name = _install_attention_patch(
                model,
                bridge,
                fused_qkv=args.fused_qkv,
                qwen_tactic=(
                    args.qwen_tactic
                    or (
                        f"qwen2_gqa_decode_{str(bridge.manifest.get('target_arch', 'sm_86')).replace('sm_', 'sm')}_fp16_workspace_v3"
                        if args.graph_decode and args.workspace_gqa
                        else f"qwen2_gqa_decode_{str(bridge.manifest.get('target_arch', 'sm_86')).replace('sm_', 'sm')}_fp16_graph_v1"
                        if args.graph_decode
                        else None
                    )
                ),
            )
            dispatcher = bool(getattr(bridge, "native_attention_dispatcher", False))
            tactic_status = bridge.tactic_status(tactic_name)
            allocated_after_load = float(torch.cuda.memory_allocated(device) / (1024**2)) if device.type == "cuda" else None
            graph_evidence: dict[str, Any] | None = None
            if args.graph_decode and str(getattr(getattr(model, "config", None), "model_type", "")).lower() == "qwen2":
                try:
                    generation = run_graph_generation(
                        torch=torch,
                        model=model,
                        bridge=bridge,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        tokenizer=tokenizer,
                        max_new_tokens=args.max_new_tokens,
                        device=device,
                        tactic_name=tactic_name,
                        kv_bucket=args.graph_kv_bucket,
                        workspace_gqa=args.workspace_gqa,
                    )
                    graph_evidence = generation.pop("graph")
                except Exception as error:
                    graph_evidence = {
                        "status": "fallback",
                        "reason": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc().splitlines()[-8:],
                    }
                    if unpatch is not None:
                        unpatch()
                    bridge.clear_qwen2_graph_dispatch()
                    unpatch, tactic_name = _install_attention_patch(
                        model,
                        bridge,
                        fused_qkv=args.fused_qkv,
                        qwen_tactic=f"qwen2_gqa_decode_{str(bridge.manifest.get('target_arch', 'sm_86')).replace('sm_', 'sm')}_fp16",
                    )
                    generation = run_generation(
                        torch=torch,
                        model=model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        tokenizer=tokenizer,
                        max_new_tokens=args.max_new_tokens,
                        device=device,
                    )
            else:
                generation = run_generation(
                    torch=torch,
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    tokenizer=tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    device=device,
                )
            native_dispatches = int(getattr(bridge, "native_attention_dispatches", 0))
            native_fused_qkv_dispatches = int(getattr(bridge, "native_fused_qkv_dispatches", 0))
            fused_qkv_hook_calls = int(getattr(bridge, "native_fused_qkv_hook_calls", 0))
            dispatch_reason = getattr(bridge, "last_qwen2_dispatch_reason", None)
            fused_dispatch_reason = getattr(bridge, "last_qwen2_fused_qkv_dispatch_reason", None)
            sidecar_status = (
                "completed"
                if dispatcher
                and tactic_status in {"implemented", "experimental"}
                and native_dispatches > 0
                and (not args.fused_qkv or native_fused_qkv_dispatches > 0)
                else "fallback-only"
            )
            payload["sidecar"] = {
                "status": sidecar_status,
                "tactic_name": tactic_name,
                "tactic_status": tactic_status,
                "native_attention_dispatcher": dispatcher,
                "native_attention_dispatches": native_dispatches,
                "native_fused_qkv_dispatches": native_fused_qkv_dispatches,
                "fused_qkv_hook_calls": fused_qkv_hook_calls,
                "last_qwen2_dispatch_reason": dispatch_reason,
                "last_qwen2_fused_qkv_dispatch_reason": fused_dispatch_reason,
                "fused_qkv_requested": bool(args.fused_qkv),
                "fused_qkv_tactic": bridge.manifest.get(
                    "preferred_qwen_fused_qkv_tactic",
                    f"qwen2_fused_rmsnorm_qkv_{str(bridge.manifest.get('target_arch', 'sm_86')).replace('sm_', 'sm')}_nf4_v1",
                ) if args.fused_qkv else None,
                "allocated_after_load_mib": allocated_after_load,
                **generation,
            }
            if graph_evidence is not None:
                payload["sidecar"]["graph"] = graph_evidence
        except Exception as error:
            payload["sidecar"] = {
                "status": "failed",
                "reason": f"{type(error).__name__}: {error}",
            }
        finally:
            if unpatch is not None:
                unpatch()
            if bridge is not None:
                bridge.shutdown()

    sidecar = payload["sidecar"]
    if "token_ids" in sidecar:
        payload["comparison"] = compare_generation_outputs(payload["baseline"], sidecar)
        token_match = bool(payload["comparison"]["token_match"])
        text_match = bool(payload["comparison"]["text_match"])
        if sidecar.get("status") == "completed" and not (token_match and text_match):
            # A native or graph path that changes generated tokens is not a
            # usable acceleration result, even if it completed without a CUDA
            # error. Preserve the evidence and fail closed.
            sidecar["status"] = "failed-parity"
            graph = sidecar.get("graph")
            if isinstance(graph, dict) and graph.get("status") == "replayed":
                graph["status"] = "failed-parity"
            payload["status"] = "failed"
    else:
        payload["comparison"] = {"status": "skipped", "reason": "Sidecar generation did not produce output"}

    write_json_evidence(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
