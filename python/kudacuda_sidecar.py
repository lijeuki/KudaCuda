"""Safe Python bindings and framework guard for the KudaCuda sidecar.

The module intentionally imports only the Python standard library at import time.
PyTorch and Transformers are optional integration dependencies and are loaded only
when a caller requests the corresponding integration path.
"""

from __future__ import annotations

import ctypes
import json
import math
import sys
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence


class NativeCudaError(RuntimeError):
    """Raised when the native CUDA sidecar rejects a request or launch."""


def select_library_name(platform_name: Optional[str] = None) -> str:
    """Return the native library filename for a platform name."""

    platform_name = platform_name or sys.platform
    if platform_name.startswith("win"):
        return "kudacuda.dll"
    if platform_name == "darwin":
        return "libkudacuda.dylib"
    return "libkudacuda.so"


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a KudaCuda JSON manifest."""

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)

    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must contain a JSON object: {manifest_path}")
    for required_key in ("target_arch", "implemented_tactics", "contracts"):
        if required_key not in manifest:
            raise ValueError(f"Manifest missing required key '{required_key}': {manifest_path}")
    if not isinstance(manifest["implemented_tactics"], list):
        raise ValueError("Manifest 'implemented_tactics' must be a list")
    if not isinstance(manifest["contracts"], dict):
        raise ValueError("Manifest 'contracts' must be an object")
    return manifest


def _normalize_dtype(dtype: Any) -> str:
    value = str(dtype)
    return value.rsplit(".", 1)[-1].lower()


@dataclass(frozen=True)
class TensorContract:
    """A small immutable shape/layout/dtype contract for one tactic."""

    dtype: Optional[str] = None
    layout: Optional[str] = None
    rank: Optional[int] = None
    batch_size: Optional[int] = None
    seq_len: Optional[int] = None
    count_min: Optional[int] = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "TensorContract":
        return cls(
            dtype=_normalize_dtype(mapping["dtype"]) if mapping.get("dtype") else None,
            layout=str(mapping["layout"]).lower() if mapping.get("layout") else None,
            rank=int(mapping["rank"]) if mapping.get("rank") is not None else None,
            batch_size=int(mapping["batch_size"]) if mapping.get("batch_size") is not None else None,
            seq_len=int(mapping["seq_len"]) if mapping.get("seq_len") is not None else None,
            count_min=int(mapping["count_min"]) if mapping.get("count_min") is not None else None,
        )

    def matches_shape(self, batch_size: int, seq_len: int) -> bool:
        if self.batch_size is not None and batch_size != self.batch_size:
            return False
        if self.seq_len is not None and seq_len != self.seq_len:
            return False
        return batch_size > 0 and seq_len > 0

    def matches_tensor(self, tensor: Any) -> bool:
        if tensor is None:
            return True
        if not bool(getattr(tensor, "is_cuda", False)):
            return False
        if self.layout == "contiguous":
            is_contiguous = getattr(tensor, "is_contiguous", None)
            if not callable(is_contiguous) or not is_contiguous():
                return False
        if self.dtype is not None and _normalize_dtype(getattr(tensor, "dtype", "")) != self.dtype:
            return False
        if self.rank is not None:
            dimension = getattr(tensor, "dim", None)
            if not callable(dimension) or int(dimension()) != self.rank:
                return False
        if self.count_min is not None:
            number_of_elements = getattr(tensor, "numel", None)
            if not callable(number_of_elements) or int(number_of_elements()) < self.count_min:
                return False
        return True


@dataclass(frozen=True)
class Qwen2GqaContract:
    """Exact decode-only Qwen2 grouped-query attention contract."""

    dtype: str
    layout: str
    batch_size: int
    query_len: int
    kv_len_min: int
    kv_len_max: int
    query_heads: int
    kv_heads: int
    head_dim: int

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "Qwen2GqaContract":
        return cls(
            dtype=_normalize_dtype(mapping["dtype"]),
            layout=str(mapping["layout"]).lower(),
            batch_size=int(mapping["batch_size"]),
            query_len=int(mapping["query_len"]),
            kv_len_min=int(mapping["kv_len_min"]),
            kv_len_max=int(mapping["kv_len_max"]),
            query_heads=int(mapping["query_heads"]),
            kv_heads=int(mapping["kv_heads"]),
            head_dim=int(mapping["head_dim"]),
        )

    def matches(self, batch_size: int, query_len: int, kv_len: int) -> bool:
        return (
            int(batch_size) == self.batch_size
            and int(query_len) == self.query_len
            and self.kv_len_min <= int(kv_len) <= self.kv_len_max
            and self.query_heads == 16
            and self.kv_heads == 2
            and self.head_dim == 128
        )

    def matches_tensor(self, tensor: Any, expected_shape: tuple[int, ...]) -> bool:
        if not bool(getattr(tensor, "is_cuda", False)):
            return False
        if self.layout == "contiguous":
            contiguous = getattr(tensor, "is_contiguous", None)
            if not callable(contiguous) or not contiguous():
                return False
        if _normalize_dtype(getattr(tensor, "dtype", "")) != self.dtype:
            return False
        shape = tuple(int(value) for value in getattr(tensor, "shape", ()))
        return shape == expected_shape


def _qwen2_attention_dimensions(module: Any) -> tuple[int, int, int]:
    """Read Qwen2 attention dimensions across supported Transformers layouts."""

    config = getattr(module, "config", None)
    query_heads = getattr(module, "num_heads", None)
    if query_heads is None:
        query_heads = getattr(config, "num_attention_heads", 0)
    kv_heads = getattr(module, "num_key_value_heads", None)
    if kv_heads is None:
        kv_heads = getattr(config, "num_key_value_heads", 0)
    head_dim = getattr(module, "head_dim", None)
    if head_dim is None:
        hidden_size = int(getattr(config, "hidden_size", 0))
        head_dim = hidden_size // int(query_heads) if query_heads else 0
    return int(query_heads), int(kv_heads), int(head_dim)


class KudaCudaBridge:
    """Own a native KudaCuda handle and expose guarded direct-pointer launches."""

    def __init__(
        self,
        *,
        manifest: Mapping[str, Any] | str | Path,
        lib_path: str | Path | None = None,
        device_id: int = 0,
        native_lib: Any = None,
    ) -> None:
        self.manifest = load_manifest(manifest) if isinstance(manifest, (str, Path)) else dict(manifest)
        self.implemented_tactics = frozenset(self.manifest.get("implemented_tactics", ()))
        self.planned_tactics = frozenset(self.manifest.get("planned_tactics", ()))
        self.contracts = {
            name: TensorContract.from_mapping(contract)
            for name, contract in self.manifest.get("contracts", {}).items()
        }
        self.qwen_contracts = {
            name: Qwen2GqaContract.from_mapping(contract)
            for name, contract in self.manifest.get("contracts", {}).items()
            if "query_heads" in contract
        }
        self.library_path = Path(lib_path) if lib_path else None
        self.lib = native_lib or self._load_library(lib_path)
        self._configure_abi()
        self.native_attention_dispatcher = self._qwen2_gqa_decode is not None
        self.native_attention_dispatches = 0
        self.engine_handle = self.lib.kudacuda_create(int(device_id))
        if not self.engine_handle:
            raise NativeCudaError("kudacuda_create returned a null handle")
        self._closed = False
        self.freeze()

    def _load_library(self, lib_path: str | Path | None) -> Any:
        if lib_path is None:
            project_root = Path(__file__).resolve().parents[1]
            library_name = select_library_name()
            candidates = [
                project_root / library_name,
                project_root / "build" / library_name,
                project_root / "build" / "Release" / library_name,
                project_root / "build-sm86" / library_name,
                project_root / "build-sm86" / "Release" / library_name,
            ]
            path = next((candidate for candidate in candidates if candidate.exists()), None)
            if path is None:
                searched = ", ".join(str(candidate) for candidate in candidates)
                raise FileNotFoundError(f"KudaCuda native library not found; searched: {searched}")
        else:
            path = Path(lib_path)
        if not path.exists():
            raise FileNotFoundError(f"KudaCuda native library not found: {path}")
        self.library_path = path
        return ctypes.CDLL(str(path))

    def _configure_abi(self) -> None:
        self.lib.kudacuda_create.argtypes = [ctypes.c_int]
        self.lib.kudacuda_create.restype = ctypes.c_void_p
        self.lib.kudacuda_freeze.argtypes = [ctypes.c_void_p]
        self.lib.kudacuda_freeze.restype = ctypes.c_int
        self.lib.kudacuda_can_dispatch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        self.lib.kudacuda_can_dispatch.restype = ctypes.c_int
        self.lib.kudacuda_vector_add.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int64,
            ctypes.c_uint64,
        ]
        self.lib.kudacuda_vector_add.restype = ctypes.c_int
        self.lib.kudacuda_rmsnorm.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_float,
            ctypes.c_uint64,
        ]
        self.lib.kudacuda_rmsnorm.restype = ctypes.c_int
        self._qwen2_gqa_decode = getattr(self.lib, "kudacuda_qwen2_gqa_decode", None)
        if self._qwen2_gqa_decode is not None:
            self._qwen2_gqa_decode.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_float,
                ctypes.c_uint64,
            ]
            self._qwen2_gqa_decode.restype = ctypes.c_int
        self.lib.kudacuda_begin_capture.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self.lib.kudacuda_begin_capture.restype = ctypes.c_int
        self.lib.kudacuda_end_capture.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self.lib.kudacuda_end_capture.restype = ctypes.c_int
        self.lib.kudacuda_replay_graph.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self.lib.kudacuda_replay_graph.restype = ctypes.c_int
        self.lib.kudacuda_last_error.argtypes = [ctypes.c_void_p]
        self.lib.kudacuda_last_error.restype = ctypes.c_char_p
        self.lib.kudacuda_destroy.argtypes = [ctypes.c_void_p]
        self.lib.kudacuda_destroy.restype = None

    def _native_error(self) -> str:
        raw_error = self.lib.kudacuda_last_error(self.engine_handle)
        if isinstance(raw_error, bytes):
            return raw_error.decode("utf-8", errors="replace")
        if raw_error is None:
            return "unknown native CUDA error"
        return str(raw_error)

    def _raise_if_failed(self, status: int, operation: str) -> None:
        if int(status) != 0:
            raise NativeCudaError(f"{operation} failed: {self._native_error()}")

    def freeze(self) -> None:
        self._raise_if_failed(self.lib.kudacuda_freeze(self.engine_handle), "kudacuda_freeze")

    def matches_contract(
        self,
        tactic_name: str,
        batch_size: int,
        seq_len: int,
        *,
        tensor: Any = None,
    ) -> bool:
        if self._closed or tactic_name not in self.implemented_tactics:
            return False
        contract = self.contracts.get(tactic_name)
        if contract is None or not contract.matches_shape(int(batch_size), int(seq_len)):
            return False
        if not contract.matches_tensor(tensor):
            return False
        return bool(
            self.lib.kudacuda_can_dispatch(
                self.engine_handle,
                tactic_name.encode("utf-8"),
                int(batch_size),
                int(seq_len),
            )
        )

    def tactic_status(self, tactic_name: str) -> str:
        if tactic_name.startswith("qwen2_gqa_decode_") and not self.native_attention_dispatcher:
            return "unavailable"
        if tactic_name in self.implemented_tactics:
            return "implemented"
        if tactic_name in self.planned_tactics:
            return "planned"
        return "unknown"

    @staticmethod
    def _tensor_pointer(tensor: Any, label: str) -> int:
        pointer = getattr(tensor, "data_ptr", None)
        if not callable(pointer):
            raise ValueError(f"{label} does not expose data_ptr()")
        value = int(pointer())
        if value <= 0:
            raise ValueError(f"{label} has an invalid device pointer")
        return value

    def vector_add(self, a: Any, b: Any, output: Any, *, stream_ptr: int) -> None:
        if self._closed:
            raise NativeCudaError("KudaCuda bridge is closed")
        if self._qwen2_gqa_decode is None:
            raise NativeCudaError("native library does not export kudacuda_qwen2_gqa_decode")
        contract = self.contracts.get("vector_add")
        if "vector_add" not in self.implemented_tactics or contract is None:
            raise NativeCudaError("vector_add is not implemented by this manifest")
        tensors = (("a", a), ("b", b), ("output", output))
        for label, tensor in tensors:
            if not contract.matches_tensor(tensor):
                raise ValueError(f"{label} does not satisfy the vector_add CUDA contract")
        sizes = [int(tensor.numel()) for _, tensor in tensors]
        if len(set(sizes)) != 1:
            raise ValueError(f"vector_add tensors must have equal sizes, got {sizes}")
        stream = int(stream_ptr)
        if stream < 0:
            raise ValueError("stream_ptr must be a non-negative CUDA stream handle")
        status = self.lib.kudacuda_vector_add(
            self.engine_handle,
            self._tensor_pointer(a, "a"),
            self._tensor_pointer(b, "b"),
            self._tensor_pointer(output, "output"),
            sizes[0],
            stream,
        )
        self._raise_if_failed(status, "kudacuda_vector_add")

    def rmsnorm(
        self,
        input_tensor: Any,
        weight: Any,
        output: Any,
        *,
        epsilon: float = 1.0e-5,
        stream_ptr: int,
    ) -> None:
        if self._closed:
            raise NativeCudaError("KudaCuda bridge is closed")
        contract = self.contracts.get("rmsnorm")
        if "rmsnorm" not in self.implemented_tactics or contract is None:
            raise NativeCudaError("rmsnorm is not implemented by this manifest")
        if not contract.matches_tensor(input_tensor) or not contract.matches_tensor(output):
            raise ValueError("input and output do not satisfy the RMSNorm CUDA contract")
        if not self._tensor_contract_matches(weight, dtype=contract.dtype, rank=1):
            raise ValueError("weight does not satisfy the RMSNorm CUDA contract")
        input_shape = tuple(int(value) for value in input_tensor.shape)
        output_shape = tuple(int(value) for value in output.shape)
        weight_shape = tuple(int(value) for value in weight.shape)
        if len(input_shape) != 2 or output_shape != input_shape:
            raise ValueError("RMSNorm input and output must have the same 2D shape")
        if weight_shape != (input_shape[1],):
            raise ValueError("RMSNorm weight must match the input hidden dimension")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        status = self.lib.kudacuda_rmsnorm(
            self.engine_handle,
            self._tensor_pointer(input_tensor, "input"),
            self._tensor_pointer(weight, "weight"),
            self._tensor_pointer(output, "output"),
            input_shape[0],
            input_shape[1],
            float(epsilon),
            int(stream_ptr),
        )
        self._raise_if_failed(status, "kudacuda_rmsnorm")

    def dispatch_qwen2_gqa_decode(
        self,
        query: Any,
        key_cache: Any,
        value_cache: Any,
        output: Any,
        *,
        kv_len: int,
        stream_ptr: int,
        tactic_name: str = "qwen2_gqa_decode_sm86_fp16",
    ) -> None:
        if self._closed:
            raise NativeCudaError("KudaCuda bridge is closed")
        contract = self.qwen_contracts.get(tactic_name)
        if tactic_name not in self.implemented_tactics or contract is None:
            raise NativeCudaError(f"{tactic_name} is not implemented by this manifest")
        if not contract.matches(1, 1, kv_len):
            raise ValueError(f"Qwen2 GQA decode contract does not accept kv_len={kv_len}")
        expected_shapes = (
            (1, contract.query_heads, 1, contract.head_dim),
            (1, contract.kv_heads, int(kv_len), contract.head_dim),
            (1, contract.kv_heads, int(kv_len), contract.head_dim),
            (1, contract.query_heads, 1, contract.head_dim),
        )
        for label, tensor, shape in zip(
            ("query", "key_cache", "value_cache", "output"),
            (query, key_cache, value_cache, output),
            expected_shapes,
        ):
            if not contract.matches_tensor(tensor, shape):
                raise ValueError(f"{label} does not satisfy the Qwen2 GQA CUDA contract")
        if int(stream_ptr) < 0:
            raise ValueError("stream_ptr must be a non-negative CUDA stream handle")
        status = self._qwen2_gqa_decode(
            self.engine_handle,
            self._tensor_pointer(query, "query"),
            self._tensor_pointer(key_cache, "key_cache"),
            self._tensor_pointer(value_cache, "value_cache"),
            self._tensor_pointer(output, "output"),
            1,
            1,
            int(kv_len),
            contract.query_heads,
            contract.kv_heads,
            contract.head_dim,
            1.0 / math.sqrt(contract.head_dim),
            int(stream_ptr),
        )
        self._raise_if_failed(status, "kudacuda_qwen2_gqa_decode")
        self.native_attention_dispatches += 1

    def dispatch_qwen2_attention(
        self,
        module: Any,
        hidden_states: Any,
        *args: Any,
        tactic_name: str = "qwen2_gqa_decode_sm86_fp16",
        **kwargs: Any,
    ) -> Any:
        """Run the exact Qwen2 decode path, returning ``None`` for safe fallback.

        Transformers remains responsible for projections, rotary embeddings, cache
        updates, output projection, and all unsupported modes. KudaCuda owns only
        the FP16 decode attention over an already-materialized contiguous cache.
        """

        import torch

        self.last_qwen2_dispatch_reason = "started"

        def fallback(reason: str) -> None:
            self.last_qwen2_dispatch_reason = reason
            return None

        shape = tuple(int(value) for value in hidden_states.shape)
        if len(shape) != 3 or shape[0] != 1 or shape[1] != 1:
            return fallback(f"shape:{shape}")
        if bool(kwargs.get("output_attentions", False)):
            return fallback("output_attentions")
        query_heads, kv_heads, head_dim = _qwen2_attention_dimensions(module)
        if query_heads != 16:
            return fallback(f"query_heads:{query_heads}")
        if kv_heads != 2:
            return fallback(f"kv_heads:{kv_heads}")
        if head_dim != 128:
            return fallback(f"head_dim:{head_dim}")

        position_embeddings = kwargs.get("position_embeddings")
        attention_mask = kwargs.get("attention_mask")
        past_key_values = kwargs.get("past_key_values")
        positional = list(args)
        if position_embeddings is None and positional:
            position_embeddings = positional.pop(0)
        if attention_mask is None and positional:
            attention_mask = positional.pop(0)
        if past_key_values is None and positional:
            past_key_values = positional.pop(0)
        if position_embeddings is None:
            return fallback("missing_position_embeddings")

        cache_updated = False

        try:
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, module.head_dim)
            query_states = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_values is not None:
                cache_position = kwargs.get("cache_position")
                cache_kwargs = {"sin": sin, "cos": cos}
                if cache_position is not None:
                    cache_kwargs["cache_position"] = cache_position
                try:
                    key_states, value_states = past_key_values.update(
                        key_states,
                        value_states,
                        module.layer_idx,
                        cache_kwargs,
                    )
                except TypeError:
                    key_states, value_states = past_key_values.update(
                        key_states,
                        value_states,
                        module.layer_idx,
                    )
                cache_updated = True

            if key_states.dim() != 4 or value_states.shape != key_states.shape:
                return fallback(f"cache_shape:{tuple(key_states.shape)}:{tuple(value_states.shape)}")
            kv_len = int(key_states.shape[-2])

            def eager_fallback(reason: str) -> Any:
                self.last_qwen2_dispatch_reason = reason
                repeat_count = query_heads // kv_heads
                expanded_key = key_states.repeat_interleave(repeat_count, dim=1)
                expanded_value = value_states.repeat_interleave(repeat_count, dim=1)
                scores = torch.matmul(
                    query_states,
                    expanded_key.transpose(-1, -2),
                ) * float(getattr(module, "scaling", 1.0 / math.sqrt(head_dim)))
                if attention_mask is not None:
                    scores = scores + attention_mask
                probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
                context = torch.matmul(probabilities, expanded_value)
                context = context.transpose(1, 2).reshape(*input_shape, -1).contiguous()
                return module.o_proj(context), None

            if attention_mask is not None:
                if attention_mask.dim() != 4 or int(attention_mask.shape[-1]) != kv_len:
                    return fallback(f"attention_mask_shape:{tuple(attention_mask.shape)}:kv={kv_len}")
                if not bool(torch.all(attention_mask == 0)):
                    return eager_fallback("framework_fallback:attention_mask_values")
            if not bool(query_states.is_contiguous()) or not bool(key_states.is_contiguous()):
                return eager_fallback(
                    f"non_contiguous:q={query_states.stride()}:k={key_states.stride()}"
                )

            output = torch.empty_like(query_states)
            stream_ptr = int(torch.cuda.current_stream(hidden_states.device).cuda_stream)
            try:
                self.dispatch_qwen2_gqa_decode(
                    query_states,
                    key_states,
                    value_states,
                    output,
                    kv_len=kv_len,
                    stream_ptr=stream_ptr,
                    tactic_name=tactic_name,
                )
            except Exception as error:
                return eager_fallback(
                    f"framework_fallback:native_exception:{type(error).__name__}:{error}"
                )
            attn_output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
            self.last_qwen2_dispatch_reason = "native"
            return module.o_proj(attn_output), None
        except Exception as error:
            if cache_updated and "query_states" in locals() and "key_states" in locals():
                try:
                    return eager_fallback(
                        f"framework_fallback:exception:{type(error).__name__}:{error}"
                    )
                except Exception:
                    pass
            return fallback(f"exception:{type(error).__name__}:{error}")

    @staticmethod
    def _tensor_contract_matches(tensor: Any, *, dtype: Optional[str], rank: int) -> bool:
        if not bool(getattr(tensor, "is_cuda", False)):
            return False
        contiguous = getattr(tensor, "is_contiguous", None)
        dimension = getattr(tensor, "dim", None)
        return (
            callable(contiguous)
            and contiguous()
            and callable(dimension)
            and int(dimension()) == rank
            and (dtype is None or _normalize_dtype(getattr(tensor, "dtype", "")) == dtype)
        )

    def begin_capture(self, *, stream_ptr: int) -> None:
        self._raise_if_failed(
            self.lib.kudacuda_begin_capture(self.engine_handle, int(stream_ptr)),
            "kudacuda_begin_capture",
        )

    def end_capture(self, *, stream_ptr: int) -> None:
        self._raise_if_failed(
            self.lib.kudacuda_end_capture(self.engine_handle, int(stream_ptr)),
            "kudacuda_end_capture",
        )

    def replay_graph(self, *, stream_ptr: int) -> None:
        self._raise_if_failed(
            self.lib.kudacuda_replay_graph(self.engine_handle, int(stream_ptr)),
            "kudacuda_replay_graph",
        )

    def shutdown(self) -> None:
        if not self._closed:
            self.lib.kudacuda_destroy(self.engine_handle)
            self._closed = True
            self.engine_handle = None

    def __enter__(self) -> "KudaCudaBridge":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.shutdown()

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass


def patch_llama_attention(
    bridge: Any,
    *,
    attention_cls: Any = None,
    tactic_name: str = "gqa8_attn",
) -> Callable[[], None]:
    """Install a guarded attention hook and return a function that restores it.

    A native attention dispatcher must be supplied by the bridge. The base bridge
    intentionally has no attention dispatcher, so the default behavior is the
    original Transformers implementation.
    """

    if attention_cls is None:
        from transformers.models.llama.modeling_llama import LlamaAttention

        attention_cls = LlamaAttention

    original_forward = attention_cls.forward

    @wraps(original_forward)
    def guarded_forward(self: Any, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            shape = tuple(int(value) for value in hidden_states.shape)
            if len(shape) < 2:
                return original_forward(self, hidden_states, *args, **kwargs)
            batch_size, seq_len = shape[0], shape[1]
            if not bridge.matches_contract(
                tactic_name,
                batch_size,
                seq_len,
                tensor=hidden_states,
            ):
                return original_forward(self, hidden_states, *args, **kwargs)
            dispatcher = getattr(bridge, "dispatch_attention", None)
            if not callable(dispatcher):
                return original_forward(self, hidden_states, *args, **kwargs)
            return dispatcher(self, hidden_states, *args, **kwargs)
        except Exception:
            return original_forward(self, hidden_states, *args, **kwargs)

    attention_cls.forward = guarded_forward
    restored = False

    def unpatch() -> None:
        nonlocal restored
        if not restored:
            attention_cls.forward = original_forward
            restored = True

    return unpatch


def patch_qwen2_attention(
    bridge: Any,
    *,
    attention_cls: Any = None,
    tactic_name: str = "qwen2_gqa_decode_sm86_fp16",
) -> Callable[[], None]:
    """Install a guarded Qwen2 attention hook with framework fallback."""

    if attention_cls is None:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

        attention_cls = Qwen2Attention

    original_forward = attention_cls.forward

    @wraps(original_forward)
    def guarded_forward(self: Any, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            shape = tuple(int(value) for value in hidden_states.shape)
            dispatcher = getattr(bridge, "dispatch_qwen2_attention", None)
            if len(shape) >= 2 and callable(dispatcher):
                result = dispatcher(self, hidden_states, *args, tactic_name=tactic_name, **kwargs)
                if result is not None:
                    return result
        except Exception:
            pass
        return original_forward(self, hidden_states, *args, **kwargs)

    attention_cls.forward = guarded_forward
    restored = False

    def unpatch() -> None:
        nonlocal restored
        if not restored:
            attention_cls.forward = original_forward
            restored = True

    return unpatch
