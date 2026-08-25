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
from types import MethodType
from typing import Any, Callable, Mapping, Optional, Sequence


class NativeCudaError(RuntimeError):
    """Raised when the native CUDA sidecar rejects a request or launch."""


def _normalize_cuda_handle(value: int) -> int:
    """Convert PyTorch's signed stream handle representation to uint64."""

    return int(value) & ((1 << 64) - 1)


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
        self.experimental_tactics = frozenset(self.manifest.get("experimental_tactics", ()))
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
        self.native_attention_dispatcher = any(
            dispatcher is not None
            for dispatcher in (
                self._qwen2_gqa_decode,
                self._qwen2_gqa_decode_group_tiled,
                self._qwen2_gqa_decode_vectorized,
                self._qwen2_gqa_decode_graph,
            )
        )
        self.native_attention_dispatches = 0
        self.native_fused_qkv_dispatches = 0
        self.native_fused_qkv_v2_dispatches = 0
        self._graph_kv_len_device: Any = None
        self._graph_kv_capacity: int | None = None
        self._graph_workspace: Any = None
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
        qwen_abi = [
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

        def configure_qwen_symbol(symbol_name: str) -> Any:
            symbol = getattr(self.lib, symbol_name, None)
            if symbol is not None:
                symbol.argtypes = qwen_abi
                symbol.restype = ctypes.c_int
            return symbol

        self._qwen2_gqa_decode = configure_qwen_symbol("kudacuda_qwen2_gqa_decode")
        self._qwen2_gqa_decode_group_tiled = configure_qwen_symbol(
            "kudacuda_qwen2_gqa_decode_group_tiled"
        )
        self._qwen2_gqa_decode_vectorized = configure_qwen_symbol(
            "kudacuda_qwen2_gqa_decode_vectorized"
        )
        self._qwen2_gqa_decode_graph = getattr(self.lib, "kudacuda_qwen2_gqa_decode_graph", None)
        if self._qwen2_gqa_decode_graph is not None:
            self._qwen2_gqa_decode_graph.argtypes = qwen_abi[:-1] + [ctypes.c_uint64, ctypes.c_uint64]
            self._qwen2_gqa_decode_graph.restype = ctypes.c_int
        self._qwen2_gqa_decode_workspace = getattr(
            self.lib, "kudacuda_qwen2_gqa_decode_workspace_v3", None
        )
        if self._qwen2_gqa_decode_workspace is not None:
            self._qwen2_gqa_decode_workspace.argtypes = (
                [ctypes.c_void_p]
                + [ctypes.c_uint64] * 6
                + [ctypes.c_int64] * 6
                + [ctypes.c_float, ctypes.c_uint64]
            )
            self._qwen2_gqa_decode_workspace.restype = ctypes.c_int
        self._fused_rmsnorm_qkv = getattr(self.lib, "kudacuda_fused_rmsnorm_qkv", None)
        if self._fused_rmsnorm_qkv is not None:
            self._fused_rmsnorm_qkv.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_uint64,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_float,
                ctypes.c_uint64,
            ]
            self._fused_rmsnorm_qkv.restype = ctypes.c_int
        self._qwen2_fused_rmsnorm_qkv_nf4 = getattr(
            self.lib,
            "kudacuda_qwen2_fused_rmsnorm_qkv_nf4",
            None,
        )
        if self._qwen2_fused_rmsnorm_qkv_nf4 is not None:
            self._qwen2_fused_rmsnorm_qkv_nf4.argtypes = (
                [ctypes.c_void_p]
                + [ctypes.c_uint64] * 14
                + [ctypes.c_int64] * 5
                + [ctypes.c_float, ctypes.c_uint64]
            )
            self._qwen2_fused_rmsnorm_qkv_nf4.restype = ctypes.c_int
        self._qwen2_fused_rmsnorm_qkv_nf4_v2 = getattr(
            self.lib,
            "kudacuda_qwen2_fused_rmsnorm_qkv_nf4_v2",
            None,
        )
        if self._qwen2_fused_rmsnorm_qkv_nf4_v2 is not None:
            self._qwen2_fused_rmsnorm_qkv_nf4_v2.argtypes = (
                [ctypes.c_void_p]
                + [ctypes.c_uint64] * 15
                + [ctypes.c_int64] * 5
                + [ctypes.c_float, ctypes.c_uint64]
            )
            self._qwen2_fused_rmsnorm_qkv_nf4_v2.restype = ctypes.c_int
        self.lib.kudacuda_begin_capture.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self.lib.kudacuda_begin_capture.restype = ctypes.c_int
        self.lib.kudacuda_end_capture.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self.lib.kudacuda_end_capture.restype = ctypes.c_int
        self.lib.kudacuda_replay_graph.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        self.lib.kudacuda_replay_graph.restype = ctypes.c_int
        self.lib.kudacuda_reset_graph.argtypes = [ctypes.c_void_p]
        self.lib.kudacuda_reset_graph.restype = ctypes.c_int
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
        if tactic_name.startswith("qwen2_gqa_decode_") and self._qwen_symbol(tactic_name) is None:
            return "unavailable"
        if tactic_name.startswith("qwen2_fused_rmsnorm_qkv_"):
            if tactic_name.endswith("_v2") and self._qwen2_fused_rmsnorm_qkv_nf4_v2 is None:
                return "unavailable"
            if not tactic_name.endswith("_v2") and self._qwen2_fused_rmsnorm_qkv_nf4 is None:
                return "unavailable"
        if tactic_name in self.implemented_tactics:
            return "implemented"
        if tactic_name in self.experimental_tactics:
            return "experimental"
        if tactic_name in self.planned_tactics:
            return "planned"
        return "unknown"

    def _qwen_symbol(self, tactic_name: str) -> Any:
        if "graph_v1" in tactic_name:
            return self._qwen2_gqa_decode_graph
        if "group_tiled_v2" in tactic_name:
            return self._qwen2_gqa_decode_group_tiled
        if "vectorized_v2" in tactic_name:
            return self._qwen2_gqa_decode_vectorized
        if "workspace_v3" in tactic_name:
            return self._qwen2_gqa_decode_workspace
        return self._qwen2_gqa_decode

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
        stream = _normalize_cuda_handle(stream_ptr)
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
            _normalize_cuda_handle(stream_ptr),
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
        if "graph_v1" in tactic_name:
            raise NativeCudaError("graph tactic requires an active static decode graph context")
        contract = self.qwen_contracts.get(tactic_name)
        if (
            tactic_name not in self.implemented_tactics
            and tactic_name not in self.experimental_tactics
        ) or contract is None:
            raise NativeCudaError(f"{tactic_name} is not implemented by this manifest")
        native_dispatch = self._qwen_symbol(tactic_name)
        if native_dispatch is None:
            raise NativeCudaError(f"native library does not export the {tactic_name} dispatcher")
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
        status = native_dispatch(
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
            _normalize_cuda_handle(stream_ptr),
        )
        self._raise_if_failed(status, tactic_name)
        self.native_attention_dispatches += 1

    def configure_qwen2_graph_dispatch(
        self,
        *,
        kv_len_device: Any,
        kv_capacity: int,
        workspace: Any = None,
    ) -> None:
        if self._qwen2_gqa_decode_graph is None:
            raise NativeCudaError("native library does not export kudacuda_qwen2_gqa_decode_graph")
        if int(kv_capacity) < 1 or int(kv_capacity) > 512:
            raise ValueError("kv_capacity must be in the range 1..512")
        self._graph_kv_len_device = kv_len_device
        self._graph_kv_capacity = int(kv_capacity)
        self._graph_workspace = workspace

    def clear_qwen2_graph_dispatch(self) -> None:
        self._graph_kv_len_device = None
        self._graph_kv_capacity = None
        self._graph_workspace = None

    def dispatch_qwen2_gqa_decode_workspace(
        self,
        query: Any,
        key_cache: Any,
        value_cache: Any,
        output: Any,
        *,
        kv_capacity: int,
        kv_len_device: Any,
        workspace: Any,
        stream_ptr: int,
        tactic_name: str = "qwen2_gqa_decode_sm86_fp16_workspace_v3",
    ) -> None:
        if self._closed:
            raise NativeCudaError("KudaCuda bridge is closed")
        contract = self.qwen_contracts.get(tactic_name)
        if contract is None or self._qwen2_gqa_decode_workspace is None:
            raise NativeCudaError(f"{tactic_name} workspace contract or dispatcher is unavailable")
        if tactic_name not in self.implemented_tactics and tactic_name not in self.experimental_tactics:
            raise NativeCudaError(f"{tactic_name} is not enabled by this manifest")
        if int(kv_capacity) != 512:
            raise ValueError("workspace GQA requires kv_capacity=512")
        expected_shapes = (
            (1, contract.query_heads, 1, contract.head_dim),
            (1, contract.kv_heads, int(kv_capacity), contract.head_dim),
            (1, contract.kv_heads, int(kv_capacity), contract.head_dim),
            (1, contract.query_heads, 1, contract.head_dim),
        )
        for label, tensor, shape in zip(
            ("query", "key_cache", "value_cache", "output"),
            (query, key_cache, value_cache, output),
            expected_shapes,
        ):
            if not contract.matches_tensor(tensor, shape):
                raise ValueError(f"{label} does not satisfy the Qwen2 workspace CUDA contract")
        if not self._tensor_contract_matches(kv_len_device, dtype="int32", rank=1):
            raise ValueError("kv_len_device does not satisfy the Qwen2 workspace CUDA contract")
        if tuple(int(value) for value in kv_len_device.shape) != (1,):
            raise ValueError("kv_len_device must have shape (1,)")
        expected_workspace = contract.query_heads * 8 * (2 + contract.head_dim)
        if not self._tensor_contract_matches(workspace, dtype="float32", rank=1):
            raise ValueError("workspace does not satisfy the Qwen2 workspace CUDA contract")
        if int(workspace.numel()) < expected_workspace:
            raise ValueError(f"workspace must contain at least {expected_workspace} float32 values")
        status = self._qwen2_gqa_decode_workspace(
            self.engine_handle,
            self._tensor_pointer(query, "query"),
            self._tensor_pointer(key_cache, "key_cache"),
            self._tensor_pointer(value_cache, "value_cache"),
            self._tensor_pointer(output, "output"),
            self._tensor_pointer(workspace, "workspace"),
            self._tensor_pointer(kv_len_device, "kv_len_device"),
            1,
            1,
            int(kv_capacity),
            contract.query_heads,
            contract.kv_heads,
            contract.head_dim,
            1.0 / math.sqrt(contract.head_dim),
            _normalize_cuda_handle(stream_ptr),
        )
        self._raise_if_failed(status, tactic_name)
        self.native_attention_dispatches += 1

    def dispatch_qwen2_gqa_decode_graph(
        self,
        query: Any,
        key_cache: Any,
        value_cache: Any,
        output: Any,
        *,
        kv_capacity: int,
        kv_len_device: Any,
        stream_ptr: int,
        tactic_name: str = "qwen2_gqa_decode_sm86_fp16_graph_v1",
    ) -> None:
        if self._closed:
            raise NativeCudaError("KudaCuda bridge is closed")
        contract = self.qwen_contracts.get(tactic_name)
        if contract is None or self._qwen2_gqa_decode_graph is None:
            raise NativeCudaError(f"{tactic_name} graph contract or dispatcher is unavailable")
        expected_shapes = (
            (1, contract.query_heads, 1, contract.head_dim),
            (1, contract.kv_heads, int(kv_capacity), contract.head_dim),
            (1, contract.kv_heads, int(kv_capacity), contract.head_dim),
            (1, contract.query_heads, 1, contract.head_dim),
        )
        for label, tensor, shape in zip(
            ("query", "key_cache", "value_cache", "output"),
            (query, key_cache, value_cache, output),
            expected_shapes,
        ):
            if not contract.matches_tensor(tensor, shape):
                raise ValueError(f"{label} does not satisfy the Qwen2 graph CUDA contract")
        if not self._tensor_contract_matches(kv_len_device, dtype="int32", rank=1):
            raise ValueError("kv_len_device does not satisfy the Qwen2 graph CUDA contract")
        if tuple(int(value) for value in kv_len_device.shape) != (1,):
            raise ValueError("kv_len_device must have shape (1,)")
        native_args = (
            self.engine_handle,
            self._tensor_pointer(query, "query"),
            self._tensor_pointer(key_cache, "key_cache"),
            self._tensor_pointer(value_cache, "value_cache"),
            self._tensor_pointer(output, "output"),
            1,
            1,
            int(kv_capacity),
            contract.query_heads,
            contract.kv_heads,
            contract.head_dim,
            1.0 / math.sqrt(contract.head_dim),
            self._tensor_pointer(kv_len_device, "kv_len_device"),
            _normalize_cuda_handle(stream_ptr),
        )
        try:
            status = self._qwen2_gqa_decode_graph(*native_args)
        except OverflowError as error:
            raise NativeCudaError(f"graph dispatcher argument conversion failed: {native_args}") from error
        self._raise_if_failed(status, tactic_name)
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
            prefused_qkv = getattr(module, "_kudacuda_prefused_qkv", None)
            if prefused_qkv is not None:
                query_flat, key_flat, value_flat = prefused_qkv
                query_states = query_flat.view(hidden_shape).transpose(1, 2)
                kv_shape = (*input_shape, -1, module.head_dim)
                key_states = key_flat.view(kv_shape).transpose(1, 2)
                value_states = value_flat.view(kv_shape).transpose(1, 2)
            else:
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
            graph_mode = (
                ("graph_v1" in tactic_name or "workspace_v3" in tactic_name)
                and self._graph_kv_len_device is not None
                and self._graph_kv_capacity is not None
            )
            kv_len = int(key_states.shape[-2])
            if graph_mode:
                if kv_len != self._graph_kv_capacity:
                    return fallback(f"graph_cache_capacity:{kv_len}:{self._graph_kv_capacity}")
                if getattr(module, "sliding_window", None) is not None:
                    return fallback("graph_sliding_window_unsupported")

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

            if attention_mask is not None and not graph_mode:
                if attention_mask.dim() != 4 or int(attention_mask.shape[-1]) != kv_len:
                    return fallback(f"attention_mask_shape:{tuple(attention_mask.shape)}:kv={kv_len}")
                if not bool(torch.all(attention_mask == 0)):
                    return eager_fallback("framework_fallback:attention_mask_values")
            if graph_mode:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()
            elif not bool(query_states.is_contiguous()) or not bool(key_states.is_contiguous()):
                return eager_fallback(
                    f"non_contiguous:q={query_states.stride()}:k={key_states.stride()}"
                )

            output = torch.empty_like(query_states)
            stream_ptr = int(torch.cuda.current_stream(hidden_states.device).cuda_stream)
            try:
                if "workspace_v3" in tactic_name:
                    if self._graph_workspace is None:
                        raise NativeCudaError("workspace GQA tactic requires a configured workspace")
                    self.dispatch_qwen2_gqa_decode_workspace(
                        query_states,
                        key_states,
                        value_states,
                        output,
                        kv_capacity=int(self._graph_kv_capacity),
                        kv_len_device=self._graph_kv_len_device,
                        workspace=self._graph_workspace,
                        stream_ptr=stream_ptr,
                        tactic_name=tactic_name,
                    )
                elif graph_mode:
                    self.dispatch_qwen2_gqa_decode_graph(
                        query_states,
                        key_states,
                        value_states,
                        output,
                        kv_capacity=int(self._graph_kv_capacity),
                        kv_len_device=self._graph_kv_len_device,
                        stream_ptr=stream_ptr,
                        tactic_name=tactic_name,
                    )
                else:
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
                if graph_mode:
                    raise
                return eager_fallback(
                    f"framework_fallback:native_exception:{type(error).__name__}:{error}"
                )
            attn_output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
            self.last_qwen2_dispatch_reason = "native"
            return module.o_proj(attn_output), None
        except Exception as error:
            if "graph_mode" in locals() and graph_mode:
                raise
            if cache_updated and "query_states" in locals() and "key_states" in locals():
                try:
                    return eager_fallback(
                        f"framework_fallback:exception:{type(error).__name__}:{error}"
                    )
                except Exception:
                    pass
            return fallback(f"exception:{type(error).__name__}:{error}")

    def fused_rmsnorm_qkv(
        self,
        input_tensor: Any,
        norm_weight: Any,
        qkv_weight: Any,
        bias: Any,
        output: Any,
        *,
        epsilon: float = 1.0e-5,
        stream_ptr: int,
    ) -> None:
        """Launch the FP16 dense-weight fused RMSNorm-QKV foundation kernel."""

        if self._closed:
            raise NativeCudaError("KudaCuda bridge is closed")
        if self._fused_rmsnorm_qkv is None:
            raise NativeCudaError("native library does not export kudacuda_fused_rmsnorm_qkv")
        tensors = (
            ("input", input_tensor, 2),
            ("norm_weight", norm_weight, 1),
            ("qkv_weight", qkv_weight, 2),
            ("bias", bias, 1),
            ("output", output, 2),
        )
        for label, tensor, rank in tensors:
            if not self._tensor_contract_matches(tensor, dtype="float16", rank=rank):
                raise ValueError(f"{label} does not satisfy the fused RMSNorm-QKV CUDA contract")
        input_shape = tuple(int(value) for value in input_tensor.shape)
        norm_shape = tuple(int(value) for value in norm_weight.shape)
        qkv_shape = tuple(int(value) for value in qkv_weight.shape)
        bias_shape = tuple(int(value) for value in bias.shape)
        output_shape = tuple(int(value) for value in output.shape)
        rows, hidden = input_shape
        if norm_shape != (hidden,):
            raise ValueError("norm_weight must match the input hidden dimension")
        if qkv_shape != (hidden * 3, hidden):
            raise ValueError("qkv_weight must have shape (3 * hidden, hidden)")
        if bias_shape != (hidden * 3,):
            raise ValueError("bias must have shape (3 * hidden,)")
        if output_shape != (rows, hidden * 3):
            raise ValueError("output must have shape (rows, 3 * hidden)")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        status = self._fused_rmsnorm_qkv(
            self.engine_handle,
            self._tensor_pointer(input_tensor, "input"),
            self._tensor_pointer(norm_weight, "norm_weight"),
            self._tensor_pointer(qkv_weight, "qkv_weight"),
            self._tensor_pointer(bias, "bias"),
            self._tensor_pointer(output, "output"),
            rows,
            hidden,
            float(epsilon),
            _normalize_cuda_handle(stream_ptr),
        )
        self._raise_if_failed(status, "kudacuda_fused_rmsnorm_qkv")

    @staticmethod
    def _validate_nf4_weight(
        weight: Any,
        absmax: Any,
        *,
        expected_shape: tuple[int, int],
        label: str,
    ) -> int:
        state = getattr(weight, "quant_state", None)
        if state is None:
            raise ValueError(f"{label} is missing quant_state")
        if str(getattr(state, "quant_type", "")) != "nf4":
            raise ValueError(f"{label} quant_type must be nf4")
        blocksize = int(getattr(state, "blocksize", 0))
        if blocksize != 64:
            raise ValueError(f"{label} blocksize must be 64")
        if bool(getattr(state, "nested", False)):
            raise ValueError(f"{label} nested quantization is unsupported")
        logical_shape = tuple(int(value) for value in getattr(state, "shape", ()))
        if logical_shape != expected_shape:
            raise ValueError(f"{label} logical shape must be {expected_shape}")
        if not KudaCudaBridge._tensor_contract_matches(weight, dtype="uint8", rank=2):
            raise ValueError(f"{label} packed storage must be contiguous CUDA uint8")
        packed_shape = tuple(int(value) for value in weight.shape)
        packed_count = (expected_shape[0] * expected_shape[1] + 1) // 2
        if packed_shape != (packed_count, 1):
            raise ValueError(f"{label} packed shape must be {(packed_count, 1)}")
        if not KudaCudaBridge._tensor_contract_matches(absmax, dtype="float32", rank=1):
            raise ValueError(f"{label} absmax must be contiguous CUDA float32")
        expected_absmax = (expected_shape[0] * expected_shape[1] + blocksize - 1) // blocksize
        if int(absmax.numel()) != expected_absmax:
            raise ValueError(f"{label} absmax must contain {expected_absmax} blocks")
        return blocksize

    def dispatch_qwen2_fused_rmsnorm_qkv_nf4(
        self,
        input_tensor: Any,
        norm_weight: Any,
        q_weight: Any,
        q_absmax: Any,
        q_bias: Any,
        k_weight: Any,
        k_absmax: Any,
        k_bias: Any,
        v_weight: Any,
        v_absmax: Any,
        v_bias: Any,
        q_output: Any,
        k_output: Any,
        v_output: Any,
        *,
        normalized_workspace: Any = None,
        epsilon: float = 1.0e-5,
        stream_ptr: int,
        tactic_name: str = "qwen2_fused_rmsnorm_qkv_sm86_nf4_v1",
    ) -> None:
        """Launch the exact Qwen2.5-3B non-nested NF4 fused QKV contract."""

        if self._closed:
            raise NativeCudaError("KudaCuda bridge is closed")
        is_v2 = tactic_name.endswith("_v2")
        native_function = (
            self._qwen2_fused_rmsnorm_qkv_nf4_v2
            if is_v2
            else self._qwen2_fused_rmsnorm_qkv_nf4
        )
        if native_function is None:
            raise NativeCudaError(
                "native library does not export the requested NF4 fused QKV ABI"
            )
        if tactic_name not in self.implemented_tactics and tactic_name not in self.experimental_tactics:
            raise NativeCudaError(f"{tactic_name} is not enabled by this manifest")
        if not self._tensor_contract_matches(input_tensor, dtype="float16", rank=2):
            raise ValueError("input must be contiguous CUDA float16 rank-2")
        if not self._tensor_contract_matches(norm_weight, dtype="float16", rank=1):
            raise ValueError("norm_weight must be contiguous CUDA float16 rank-1")
        input_shape = tuple(int(value) for value in input_tensor.shape)
        norm_shape = tuple(int(value) for value in norm_weight.shape)
        rows, hidden = input_shape
        q_shape = (2048, 2048)
        kv_shape = (256, 2048)
        if input_shape != (1, 2048) or norm_shape != (2048,):
            raise ValueError("NF4 fused QKV requires input (1, 2048) and norm_weight (2048,)")
        q_blocksize = self._validate_nf4_weight(
            q_weight,
            q_absmax,
            expected_shape=q_shape,
            label="q_weight",
        )
        for label, bias, expected_shape in (
            ("q_bias", q_bias, (2048,)),
            ("k_bias", k_bias, (256,)),
            ("v_bias", v_bias, (256,)),
        ):
            if not self._tensor_contract_matches(bias, dtype="float16", rank=1):
                raise ValueError(f"{label} must be contiguous CUDA float16 rank-1")
            if tuple(int(value) for value in bias.shape) != expected_shape:
                raise ValueError(f"{label} must have shape {expected_shape}")
        k_blocksize = self._validate_nf4_weight(
            k_weight,
            k_absmax,
            expected_shape=kv_shape,
            label="k_weight",
        )
        v_blocksize = self._validate_nf4_weight(
            v_weight,
            v_absmax,
            expected_shape=kv_shape,
            label="v_weight",
        )
        if (q_blocksize, k_blocksize, v_blocksize) != (64, 64, 64):
            raise ValueError("all NF4 projection block sizes must be 64")
        for label, tensor, shape in (
            ("q_output", q_output, (rows, 2048)),
            ("k_output", k_output, (rows, 256)),
            ("v_output", v_output, (rows, 256)),
        ):
            if not self._tensor_contract_matches(tensor, dtype="float16", rank=2):
                raise ValueError(f"{label} must be contiguous CUDA float16 rank-2")
            if tuple(int(value) for value in tensor.shape) != shape:
                raise ValueError(f"{label} must have shape {shape}")
        if is_v2:
            if not self._tensor_contract_matches(normalized_workspace, dtype="float32", rank=2):
                raise ValueError("normalized_workspace must be contiguous CUDA float32 rank-2")
            if tuple(int(value) for value in normalized_workspace.shape) != (rows, hidden):
                raise ValueError(f"normalized_workspace must have shape {(rows, hidden)}")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        native_args = [
            self.engine_handle,
            self._tensor_pointer(input_tensor, "input"),
            self._tensor_pointer(norm_weight, "norm_weight"),
            self._tensor_pointer(q_weight, "q_weight"),
            self._tensor_pointer(q_absmax, "q_absmax"),
            self._tensor_pointer(q_bias, "q_bias"),
            self._tensor_pointer(k_weight, "k_weight"),
            self._tensor_pointer(k_absmax, "k_absmax"),
            self._tensor_pointer(k_bias, "k_bias"),
            self._tensor_pointer(v_weight, "v_weight"),
            self._tensor_pointer(v_absmax, "v_absmax"),
            self._tensor_pointer(v_bias, "v_bias"),
        ]
        if is_v2:
            native_args.append(self._tensor_pointer(normalized_workspace, "normalized_workspace"))
        native_args.extend(
            [
                self._tensor_pointer(q_output, "q_output"),
                self._tensor_pointer(k_output, "k_output"),
                self._tensor_pointer(v_output, "v_output"),
                rows,
                hidden,
                2048,
                256,
                64,
                float(epsilon),
                _normalize_cuda_handle(stream_ptr),
            ]
        )
        status = native_function(*native_args)
        self._raise_if_failed(status, tactic_name)
        self.native_fused_qkv_dispatches += 1
        if is_v2:
            self.native_fused_qkv_v2_dispatches += 1

    def dispatch_qwen2_fused_rmsnorm_qkv_nf4_v2(
        self,
        input_tensor: Any,
        norm_weight: Any,
        q_weight: Any,
        q_absmax: Any,
        q_bias: Any,
        k_weight: Any,
        k_absmax: Any,
        k_bias: Any,
        v_weight: Any,
        v_absmax: Any,
        v_bias: Any,
        q_output: Any,
        k_output: Any,
        v_output: Any,
        *,
        normalized_workspace: Any,
        epsilon: float = 1.0e-5,
        stream_ptr: int,
        tactic_name: str = "qwen2_fused_rmsnorm_qkv_sm86_nf4_v2",
    ) -> None:
        self.dispatch_qwen2_fused_rmsnorm_qkv_nf4(
            input_tensor,
            norm_weight,
            q_weight,
            q_absmax,
            q_bias,
            k_weight,
            k_absmax,
            k_bias,
            v_weight,
            v_absmax,
            v_bias,
            q_output,
            k_output,
            v_output,
            normalized_workspace=normalized_workspace,
            epsilon=epsilon,
            stream_ptr=stream_ptr,
            tactic_name=tactic_name,
        )

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
            self.lib.kudacuda_begin_capture(self.engine_handle, _normalize_cuda_handle(stream_ptr)),
            "kudacuda_begin_capture",
        )

    def end_capture(self, *, stream_ptr: int) -> None:
        self._raise_if_failed(
            self.lib.kudacuda_end_capture(self.engine_handle, _normalize_cuda_handle(stream_ptr)),
            "kudacuda_end_capture",
        )

    def replay_graph(self, *, stream_ptr: int) -> None:
        self._raise_if_failed(
            self.lib.kudacuda_replay_graph(self.engine_handle, _normalize_cuda_handle(stream_ptr)),
            "kudacuda_replay_graph",
        )

    def reset_graph(self) -> None:
        self._raise_if_failed(
            self.lib.kudacuda_reset_graph(self.engine_handle),
            "kudacuda_reset_graph",
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


def prepare_qwen2_fused_qkv(
    model: Any,
    *,
    torch_module: Any = None,
) -> None:
    """Allocate one stable FP16 Q/K/V workspace per Qwen2 decoder layer.

    The buffers are created before CUDA Graph capture. The fused decoder hook
    only writes into these buffers during decode, so graph replay observes the
    same tensor addresses on every iteration.
    """

    if torch_module is None:
        import torch as torch_module

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("Qwen2 model does not expose model.layers")
    for layer in layers:
        attention = getattr(layer, "self_attn", None)
        norm = getattr(layer, "input_layernorm", None)
        norm_weight = getattr(norm, "weight", None)
        if attention is None or norm_weight is None:
            raise ValueError("Qwen2 decoder layer is missing self_attn or input_layernorm")
        device = getattr(norm_weight, "device", None)
        dtype = getattr(norm_weight, "dtype", None)
        if device is None or dtype is None:
            raise ValueError("Qwen2 input layernorm weight has no device/dtype")
        attention._kudacuda_fused_qkv_state = {
            "normalized": torch_module.empty((1, 2048), device=device, dtype=torch_module.float32),
            "query": torch_module.empty((1, 2048), device=device, dtype=dtype),
            "key": torch_module.empty((1, 256), device=device, dtype=dtype),
            "value": torch_module.empty((1, 256), device=device, dtype=dtype),
        }


def patch_qwen2_fused_qkv(
    model: Any,
    bridge: Any,
    *,
    tactic_name: str = "qwen2_fused_rmsnorm_qkv_sm86_nf4_v1",
    attention_tactic_name: str = "qwen2_gqa_decode_sm86_fp16",
    prepare: bool = True,
    torch_module: Any = None,
    attention_cls: Any = None,
) -> Callable[[], None]:
    """Patch Qwen2 decoder layers to launch the native NF4 fused QKV path.

    The successful path performs the input RMSNorm and all three NF4
    projections natively, then reuses the existing attention hook for RoPE,
    cache update, GQA, and output projection. Any unsupported shape, dtype, or
    native error clears the temporary marker and calls the original decoder
    layer unchanged.
    """

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or not list(layers):
        raise ValueError("Qwen2 model does not expose decoder layers")
    layers = list(layers)
    if attention_cls is None:
        attention_cls = type(layers[0].self_attn)
    if prepare:
        prepare_qwen2_fused_qkv(model, torch_module=torch_module)

    attention_unpatch = patch_qwen2_attention(
        bridge,
        attention_cls=attention_cls,
        tactic_name=attention_tactic_name,
    )
    originals: list[tuple[Any, Any, Any, Any]] = []
    marker = object()

    for layer in layers:
        attention = layer.self_attn
        original_forward = layer.forward
        previous_state = getattr(attention, "_kudacuda_fused_qkv_state", marker)
        previous_prefused = getattr(attention, "_kudacuda_prefused_qkv", marker)
        originals.append((layer, original_forward, previous_state, previous_prefused))

        def fused_forward(
            self: Any,
            hidden_states: Any,
            *args: Any,
            _layer: Any = layer,
            _attention: Any = attention,
            _original_forward: Any = original_forward,
            **kwargs: Any,
        ) -> Any:
            bridge.native_fused_qkv_hook_calls = int(
                getattr(bridge, "native_fused_qkv_hook_calls", 0)
            ) + 1
            shape = tuple(int(value) for value in getattr(hidden_states, "shape", ()))
            if shape != (1, 1, 2048):
                if int(getattr(bridge, "native_fused_qkv_dispatches", 0)) == 0:
                    bridge.last_qwen2_fused_qkv_dispatch_reason = f"shape:{shape}"
                return _original_forward(hidden_states, *args, **kwargs)
            state = getattr(_attention, "_kudacuda_fused_qkv_state", None)
            if not isinstance(state, dict) or not all(key in state for key in ("query", "key", "value")):
                bridge.last_qwen2_fused_qkv_dispatch_reason = "missing_workspace"
                return _original_forward(hidden_states, *args, **kwargs)

            prefused_marker = getattr(_attention, "_kudacuda_prefused_qkv", marker)
            try:
                input_tensor = hidden_states.reshape(1, 2048)
                if not bool(input_tensor.is_contiguous()):
                    bridge.last_qwen2_fused_qkv_dispatch_reason = "non_contiguous_input"
                    return _original_forward(hidden_states, *args, **kwargs)
                norm = _layer.input_layernorm
                norm_weight = norm.weight
                q_weight = _attention.q_proj.weight
                k_weight = _attention.k_proj.weight
                v_weight = _attention.v_proj.weight
                q_state = getattr(q_weight, "quant_state", None)
                k_state = getattr(k_weight, "quant_state", None)
                v_state = getattr(v_weight, "quant_state", None)
                if q_state is None or k_state is None or v_state is None:
                    bridge.last_qwen2_fused_qkv_dispatch_reason = "missing_quant_state"
                    return _original_forward(hidden_states, *args, **kwargs)
                q_bias = getattr(_attention.q_proj, "bias", None)
                k_bias = getattr(_attention.k_proj, "bias", None)
                v_bias = getattr(_attention.v_proj, "bias", None)
                if q_bias is None or k_bias is None or v_bias is None:
                    bridge.last_qwen2_fused_qkv_dispatch_reason = "missing_projection_bias"
                    return _original_forward(hidden_states, *args, **kwargs)
                stream_ptr = int(torch_module.cuda.current_stream(hidden_states.device).cuda_stream) if torch_module is not None else None
                if stream_ptr is None:
                    import torch as runtime_torch

                    stream_ptr = int(runtime_torch.cuda.current_stream(hidden_states.device).cuda_stream)
                bridge.dispatch_qwen2_fused_rmsnorm_qkv_nf4(
                    input_tensor,
                    norm_weight,
                    q_weight,
                    q_state.absmax,
                    q_bias,
                    k_weight,
                    k_state.absmax,
                    k_bias,
                    v_weight,
                    v_state.absmax,
                    v_bias,
                    state["query"],
                    state["key"],
                    state["value"],
                    normalized_workspace=state.get("normalized"),
                    epsilon=float(getattr(norm, "variance_epsilon", 1.0e-6)),
                    stream_ptr=stream_ptr,
                    tactic_name=tactic_name,
                )
                bridge.last_qwen2_fused_qkv_dispatch_reason = "native"
                _attention._kudacuda_prefused_qkv = (
                    state["query"],
                    state["key"],
                    state["value"],
                )
                residual = hidden_states
                attention_outputs = _attention(hidden_states, *args, **kwargs)
                attention_output = attention_outputs[0] if isinstance(attention_outputs, tuple) else attention_outputs
                hidden_states = residual + attention_output
                residual = hidden_states
                hidden_states = _layer.post_attention_layernorm(hidden_states)
                hidden_states = _layer.mlp(hidden_states)
                return residual + hidden_states
            except Exception as error:
                setattr(bridge, "last_qwen2_fused_qkv_dispatch_reason", f"fallback:{type(error).__name__}:{error}")
                return _original_forward(hidden_states, *args, **kwargs)
            finally:
                if prefused_marker is marker:
                    try:
                        delattr(_attention, "_kudacuda_prefused_qkv")
                    except AttributeError:
                        pass
                else:
                    _attention._kudacuda_prefused_qkv = prefused_marker

        layer.forward = MethodType(fused_forward, layer)

    restored = False

    def unpatch() -> None:
        nonlocal restored
        if restored:
            return
        for layer, original_forward, previous_state, previous_prefused in originals:
            layer.forward = original_forward
            attention = layer.self_attn
            if previous_state is marker:
                try:
                    delattr(attention, "_kudacuda_fused_qkv_state")
                except AttributeError:
                    pass
            else:
                attention._kudacuda_fused_qkv_state = previous_state
            if previous_prefused is marker:
                try:
                    delattr(attention, "_kudacuda_prefused_qkv")
                except AttributeError:
                    pass
            else:
                attention._kudacuda_prefused_qkv = previous_prefused
        attention_unpatch()
        restored = True

    return unpatch
