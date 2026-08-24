"""Safe Python bindings and framework guard for the KudaCuda sidecar.

The module intentionally imports only the Python standard library at import time.
PyTorch and Transformers are optional integration dependencies and are loaded only
when a caller requests the corresponding integration path.
"""

from __future__ import annotations

import ctypes
import json
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
        self.library_path = Path(lib_path) if lib_path else None
        self.lib = native_lib or self._load_library(lib_path)
        self._configure_abi()
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
