"""Interactive Qwen2 Transformers chat with KudaCuda activation telemetry.

The default path enables the validated direct decode attention tactic. Prompt
prefill and unsupported contracts remain on the Transformers implementation.
Every assistant response reports end-to-end waiting time, generated-token
throughput, and the number of native KudaCuda attention dispatches.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_TACTIC = "qwen2_gqa_decode_sm86_fp16"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def normalize_model_reference(value: str) -> str:
    """Remove terminal line wrapping accidentally pasted into a local path."""

    return re.sub(r"[\r\n]+\s*", "", str(value).strip())


def resolve_sidecar_python_dir(root: str | Path) -> Path:
    """Resolve the sidecar module directory, tolerating a stale root override."""

    candidates = [Path(root), Path(__file__).resolve().parents[1]]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        python_dir = candidate / "python"
        if (python_dir / "kudacuda_sidecar.py").is_file():
            return python_dir
    raise ModuleNotFoundError(
        "Could not find KudaCuda/python/kudacuda_sidecar.py. "
        "Pass --root pointing to the KudaCuda repository."
    )


def build_messages(
    history: Sequence[Mapping[str, str]],
    user_text: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Build a chat-template message list without mutating conversation history."""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in history
    )
    messages.append({"role": "user", "content": user_text})
    return messages


def activation_snapshot(bridge: Any | None, tactic_name: str = DEFAULT_TACTIC) -> dict[str, Any]:
    """Return observable evidence that the native path is available."""

    if bridge is None:
        return {
            "native_attention_dispatcher": False,
            "tactic_status": "disabled",
            "native_attention_dispatches": 0,
            "last_dispatch_reason": "sidecar_disabled",
        }
    return {
        "native_attention_dispatcher": bool(
            getattr(bridge, "native_attention_dispatcher", False)
        ),
        "tactic_status": str(bridge.tactic_status(tactic_name)),
        "native_attention_dispatches": int(
            getattr(bridge, "native_attention_dispatches", 0)
        ),
        "last_dispatch_reason": str(
            getattr(bridge, "last_qwen2_dispatch_reason", "not-started")
        ),
    }


def summarize_turn(
    bridge: Any | None,
    *,
    before_dispatches: int,
    generated_tokens: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Summarize one response, including waiting time and token throughput."""

    current_dispatches = int(
        getattr(bridge, "native_attention_dispatches", 0)
        if bridge is not None
        else 0
    )
    native_dispatches = max(0, current_dispatches - int(before_dispatches))
    elapsed_seconds = max(0.0, float(elapsed_seconds))
    tokens_per_second = (
        float(generated_tokens) / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
    )
    return {
        "waiting_seconds": elapsed_seconds,
        "waiting_ms": elapsed_seconds * 1000.0,
        "generated_tokens": int(generated_tokens),
        "tokens_per_second": tokens_per_second,
        "native_dispatches": native_dispatches,
        "last_dispatch_reason": str(
            getattr(bridge, "last_qwen2_dispatch_reason", "sidecar_disabled")
            if bridge is not None
            else "sidecar_disabled"
        ),
    }


def native_kv_preflight(
    *,
    prompt_tokens: int,
    requested_tokens: int,
    native_limit: int,
) -> dict[str, Any]:
    """Describe how much of a generation can stay inside the native KV profile."""

    prompt_tokens = max(0, int(prompt_tokens))
    requested_tokens = max(0, int(requested_tokens))
    native_limit = max(0, int(native_limit))
    native_budget = max(0, native_limit - prompt_tokens)
    will_fallback = prompt_tokens + requested_tokens > native_limit
    return {
        "prompt_tokens": prompt_tokens,
        "requested_tokens": requested_tokens,
        "native_limit": native_limit,
        "native_budget": native_budget,
        "will_fallback": will_fallback,
        "fallback_after_tokens": native_budget if will_fallback else None,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("KUDACUDA_MODEL"),
        help="Local Hugging Face model directory or model ID.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("KUDACUDA_ROOT", default_root)),
        help="KudaCuda repository root.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Target manifest; defaults to manifests/rtx3050-sm86.json.",
    )
    parser.add_argument(
        "--library",
        type=Path,
        help="Native library; defaults to artifacts/sm86/kudacuda.dll.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        default=True,
        help="Wait for the complete response before printing it.",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        help="Optional JSON file for session metrics.",
    )
    parser.add_argument(
        "--no-sidecar",
        action="store_true",
        help="Run the same interactive chat without KudaCuda for comparison.",
    )
    args = parser.parse_args(argv)
    if not args.model:
        parser.error("--model is required unless KUDACUDA_MODEL is set")
    args.model = normalize_model_reference(args.model)
    args.root = args.root.resolve()
    args.manifest = args.manifest or args.root / "manifests" / "rtx3050-sm86.json"
    args.library = args.library or args.root / "artifacts" / "sm86" / "kudacuda.dll"
    return args


def _load_inputs(tokenizer: Any, messages: Sequence[Mapping[str, str]], device: Any) -> Any:
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    return inputs.to(device) if hasattr(inputs, "to") else {
        key: value.to(device) for key, value in inputs.items()
    }


def run_chat(args: argparse.Namespace) -> int:
    """Load the model and run the interactive chat loop."""

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TextIteratorStreamer,
    )

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for the KudaCuda chat tool")
    if not args.no_sidecar and not args.library.exists():
        raise FileNotFoundError(f"KudaCuda library was not found: {args.library}")
    if not args.manifest.exists() and not args.no_sidecar:
        raise FileNotFoundError(f"KudaCuda manifest was not found: {args.manifest}")
    model_reference = str(args.model)
    looks_like_local_path = (
        Path(model_reference).is_absolute()
        or "\\" in model_reference
        or "/" in model_reference
    )
    if looks_like_local_path and not Path(model_reference).is_dir():
        raise FileNotFoundError(
            "Local model directory was not found after path normalization: "
            f"{model_reference!r}. Keep the --model path on one logical line."
        )

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        torch_dtype=torch.float16,
        device_map={"": str(device)},
    )
    model.eval()

    bridge = None
    unpatch = None
    if not args.no_sidecar:
        model_type = str(getattr(getattr(model, "config", None), "model_type", ""))
        if model_type.lower() != "qwen2":
            raise ValueError(
                f"The active KudaCuda attention contract requires model_type=qwen2; got {model_type!r}"
            )
        bridge_module = str(resolve_sidecar_python_dir(args.root))
        if bridge_module not in sys.path:
            sys.path.insert(0, bridge_module)
        from kudacuda_sidecar import KudaCudaBridge, patch_qwen2_attention

        bridge = KudaCudaBridge(
            manifest=args.manifest,
            lib_path=args.library,
            device_id=device.index or 0,
        )
        attention_cls = type(model.model.layers[0].self_attn)
        unpatch = patch_qwen2_attention(
            bridge,
            attention_cls=attention_cls,
            tactic_name=DEFAULT_TACTIC,
        )

    initial_activation = activation_snapshot(bridge)
    native_kv_limit = 0
    if bridge is not None:
        contract = getattr(bridge, "qwen_contracts", {}).get(DEFAULT_TACTIC)
        if contract is not None:
            native_kv_limit = int(getattr(contract, "kv_len_max", 0))
    print("KudaCuda chat status")
    print(f"  sidecar: {'enabled' if bridge is not None else 'disabled'}")
    if bridge is not None:
        print(f"  python: {bridge_module}")
    print(f"  dispatcher: {initial_activation['native_attention_dispatcher']}")
    print(f"  tactic: {initial_activation['tactic_status']}")
    if bridge is not None:
        print(f"  direct_decode_kv_limit: {native_kv_limit} tokens")
    print(f"  device: {torch.cuda.get_device_name(device)}")
    print("Type /reset to clear conversation, /exit or /quit to leave. Every response reports waiting time and tok/s.\n")

    history: list[dict[str, str]] = []
    turns: list[dict[str, Any]] = []
    try:
        while True:
            try:
                user_text = input("You: ").strip()
            except EOFError:
                print()
                break
            if user_text.lower() in {"/exit", "/quit"}:
                break
            if user_text.lower() == "/reset":
                history.clear()
                print("[KudaCuda] conversation reset.\n")
                continue
            if not user_text:
                continue

            messages = build_messages(
                history,
                user_text,
                system_prompt=args.system_prompt,
            )
            torch.cuda.synchronize(device)
            turn_start = time.perf_counter()
            inputs = _load_inputs(tokenizer, messages, device)
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            preflight = native_kv_preflight(
                prompt_tokens=prompt_tokens,
                requested_tokens=int(args.max_new_tokens),
                native_limit=native_kv_limit,
            )
            if bridge is not None:
                print(
                    "[KudaCuda] "
                    f"kv_preflight=prompt:{preflight['prompt_tokens']} "
                    f"limit:{preflight['native_limit']} "
                    f"native_budget:{preflight['native_budget']}"
                )
                if preflight["will_fallback"]:
                    print(
                        "[KudaCuda] warning: this turn exceeds the direct KV profile; "
                        f"framework attention is expected after {preflight['fallback_after_tokens']} "
                        "generated token(s). Use /reset or a shorter prompt to stay native."
                    )
            before_dispatches = int(
                getattr(bridge, "native_attention_dispatches", 0)
                if bridge is not None
                else 0
            )

            generation_kwargs: dict[str, Any] = {
                **inputs,
                "max_new_tokens": int(args.max_new_tokens),
                "do_sample": False,
                "use_cache": True,
                "pad_token_id": (
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            }
            streamer = (
                TextIteratorStreamer(
                    tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )
                if args.stream
                else None
            )
            if streamer is not None:
                generation_kwargs["streamer"] = streamer

            generation_result: dict[str, Any] = {}
            generation_errors: list[BaseException] = []

            def generate_response() -> None:
                try:
                    with torch.inference_mode():
                        generation_result["output"] = model.generate(**generation_kwargs)
                except BaseException as error:
                    generation_errors.append(error)
                    if streamer is not None:
                        streamer.on_finalized_text("", stream_end=True)

            generation_thread = threading.Thread(
                target=generate_response,
                name="kudacuda-chat-generation",
                daemon=True,
            )
            if streamer is not None:
                print("Assistant: ", end="", flush=True)
            generation_thread.start()
            if streamer is not None:
                for text_chunk in streamer:
                    print(text_chunk, end="", flush=True)
                print()
            generation_thread.join()
            if generation_errors:
                raise generation_errors[0]
            output = generation_result["output"]
            torch.cuda.synchronize(device)
            elapsed_seconds = time.perf_counter() - turn_start

            generated_ids = output[0, prompt_tokens:]
            reply = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            summary = summarize_turn(
                bridge,
                before_dispatches=before_dispatches,
                generated_tokens=int(generated_ids.numel()),
                elapsed_seconds=elapsed_seconds,
            )
            summary["prompt_tokens"] = prompt_tokens
            summary["native_kv_limit"] = preflight["native_limit"]
            summary["native_kv_budget"] = preflight["native_budget"]
            summary["native_fallback_expected"] = preflight["will_fallback"]
            summary["tactic_status"] = initial_activation["tactic_status"]
            summary["native_attention_dispatcher"] = initial_activation[
                "native_attention_dispatcher"
            ]
            turns.append({"user": user_text, "reply": reply, **summary})

            if streamer is None:
                print(f"Assistant: {reply}")
            print(
                "[KudaCuda] "
                f"waiting={summary['waiting_seconds']:.3f}s "
                f"({summary['waiting_ms']:.1f}ms) | "
                f"response_tok/s={summary['tokens_per_second']:.2f} | "
                f"generated={summary['generated_tokens']} | "
                f"native_dispatches={summary['native_dispatches']} | "
                f"reason={summary['last_dispatch_reason']}\n"
            )

            history.extend(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": reply},
                ]
            )
    finally:
        if args.metrics_output:
            args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
            args.metrics_output.write_text(
                json.dumps(
                    {
                        "model": str(args.model),
                        "device": str(device),
                        "tactic": DEFAULT_TACTIC,
                        "activation": initial_activation,
                        "turns": turns,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        if unpatch is not None:
            unpatch()
        if bridge is not None:
            bridge.shutdown()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_chat(args)


if __name__ == "__main__":
    raise SystemExit(main())
