"""Fixed-shape model-forward CUDA Graph capture for Qwen2 decode."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Mapping

try:
    from .kudacuda_graph_plan import DecodeGraphKey, GraphContractError
except ImportError:  # direct checkout imports used by the benchmark tools
    from kudacuda_graph_plan import DecodeGraphKey, GraphContractError


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in getattr(value, "shape", ()))


def _pointer(value: Any, label: str) -> int:
    data_ptr = getattr(value, "data_ptr", None)
    if not callable(data_ptr):
        raise GraphContractError(f"{label} does not expose data_ptr()")
    result = int(data_ptr())
    if result <= 0:
        raise GraphContractError(f"{label} has an invalid device pointer")
    return result


def _clone_value(torch_module: Any, value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _clone_value(torch_module, item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        cloned = [_clone_value(torch_module, item) for item in value]
        return tuple(cloned) if isinstance(value, tuple) else cloned
    clone = getattr(value, "clone", None)
    return clone() if callable(clone) else value


def _tensor_records(value: Any, records: list[tuple[int, tuple[int, ...]]]) -> None:
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            _tensor_records(value[key], records)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _tensor_records(item, records)
        return
    data_ptr = getattr(value, "data_ptr", None)
    shape = getattr(value, "shape", None)
    if callable(data_ptr) and shape is not None:
        records.append((_pointer(value, "graph tensor"), _shape(value)))


def _cache_layers(cache: Any) -> list[Any]:
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return list(layers)
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        return list(zip(key_cache, value_cache))
    return []


def _layer_tensors(cache: Any, layer_index: int) -> tuple[Any, Any]:
    layer = _cache_layers(cache)[layer_index]
    if isinstance(layer, tuple) and len(layer) == 2:
        return layer
    key = getattr(layer, "keys", None)
    value = getattr(layer, "values", None)
    if key is None:
        key = getattr(layer, "key_cache", None)
    if value is None:
        value = getattr(layer, "value_cache", None)
    if key is None or value is None:
        raise GraphContractError(f"cache layer {layer_index} has no key/value tensors")
    return key, value


class GraphStaticCache:
    """Minimal cache with device-resident positions suitable for graph replay."""

    def __init__(
        self,
        *,
        key_cache: list[Any],
        value_cache: list[Any],
        cache_position: Any,
        sequence_length: int,
        max_cache_len: int,
    ) -> None:
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.cache_position = cache_position
        self._sequence_length = int(sequence_length)
        self.max_cache_len = int(max_cache_len)
        self.layers = list(zip(self.key_cache, self.value_cache))

    @classmethod
    def from_dynamic_cache(cls, torch_module: Any, dynamic_cache: Any, *, max_cache_len: int) -> "GraphStaticCache":
        source_layers = _cache_layers(dynamic_cache)
        if not source_layers:
            raise GraphContractError("dynamic cache does not expose cache layers")
        key_cache: list[Any] = []
        value_cache: list[Any] = []
        sequence_length = int(dynamic_cache.get_seq_length())
        for layer_index in range(len(source_layers)):
            source_key, source_value = _layer_tensors(dynamic_cache, layer_index)
            if source_key.dim() != 4 or source_value.shape != source_key.shape:
                raise GraphContractError(f"cache layer {layer_index} is not a rank-4 KV pair")
            if int(source_key.shape[-2]) > int(max_cache_len):
                raise GraphContractError("max_cache_len is smaller than the prefetched cache")
            target_shape = (
                int(source_key.shape[0]),
                int(source_key.shape[1]),
                int(max_cache_len),
                int(source_key.shape[3]),
            )
            target_key = torch_module.zeros(
                target_shape,
                dtype=source_key.dtype,
                device=source_key.device,
            )
            target_value = torch_module.zeros(
                target_shape,
                dtype=source_value.dtype,
                device=source_value.device,
            )
            target_key[..., : int(source_key.shape[-2]), :].copy_(source_key)
            target_value[..., : int(source_value.shape[-2]), :].copy_(source_value)
            key_cache.append(target_key)
            value_cache.append(target_value)
        device = key_cache[0].device
        cache_position = torch_module.tensor(
            [sequence_length],
            dtype=torch_module.long,
            device=device,
        )
        return cls(
            key_cache=key_cache,
            value_cache=value_cache,
            cache_position=cache_position,
            sequence_length=sequence_length,
            max_cache_len=max_cache_len,
        )

    def update(self, key_states: Any, value_states: Any, layer_idx: int, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
        del args, kwargs
        if int(key_states.shape[-2]) != int(self.cache_position.numel()):
            raise GraphContractError("graph cache position count does not match key update length")
        self.key_cache[layer_idx].index_copy_(2, self.cache_position, key_states)
        self.value_cache[layer_idx].index_copy_(2, self.cache_position, value_states)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        return self._sequence_length

    def get_max_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        return self.max_cache_len

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return self.get_max_length(layer_idx)

    def __len__(self) -> int:
        return len(self.key_cache)


@dataclass(frozen=True)
class StaticDecodeContract:
    key: DecodeGraphKey
    stream_ptr: int
    logical_kv_len: int


class StaticDecodeGraphRunner:
    """Capture one complete model decode forward and replay it by pointer identity."""

    def __init__(self, bridge: Any, model: Any, torch_module: Any) -> None:
        self.bridge = bridge
        self.model = model
        self.torch = torch_module
        self.contract: StaticDecodeContract | None = None
        self.static_inputs: dict[str, Any] = {}
        self.cache: Any = None
        self.kv_len_device: Any = None
        self.logits: Any = None
        self.status = "not-requested"
        self.reason: str | None = None
        self.capture_ms: float | None = None
        self.replay_count = 0
        self.backend = "native"
        self.cuda_graph: Any = None
        self.workspace: Any = None

    @property
    def captured(self) -> bool:
        return self.contract is not None and self.status == "captured"

    @staticmethod
    def model_fingerprint(model: Any) -> str:
        config = getattr(model, "config", None)
        values = (
            type(model).__name__,
            getattr(config, "model_type", ""),
            getattr(config, "num_hidden_layers", ""),
            getattr(config, "hidden_size", ""),
            getattr(config, "vocab_size", ""),
            str(getattr(config, "torch_dtype", "")),
        )
        return "|".join(str(value) for value in values)

    def _layer_count(self) -> int:
        config = getattr(self.model, "config", None)
        value = getattr(config, "num_hidden_layers", None)
        if value is not None:
            return int(value)
        layers = getattr(getattr(self.model, "model", None), "layers", None)
        return len(layers) if layers is not None else 0

    def _key(self, *, tactic_name: str, device_id: int, kv_bucket: int, logical_kv_len: int) -> DecodeGraphKey:
        records: list[tuple[int, tuple[int, ...]]] = []
        _tensor_records(self.static_inputs, records)
        _tensor_records(self.cache.key_cache, records)
        _tensor_records(self.cache.value_cache, records)
        records.append((_pointer(self.kv_len_device, "kv_len_device"), _shape(self.kv_len_device)))
        return DecodeGraphKey(
            device_id=int(device_id),
            tactic_name=tactic_name,
            layer_count=self._layer_count(),
            query_shape=_shape(self.static_inputs["input_ids"]),
            key_shape=_shape(self.cache.key_cache[0]),
            value_shape=_shape(self.cache.value_cache[0]),
            output_shape=_shape(self.logits) if self.logits is not None else (),
            kv_bucket=int(kv_bucket),
            pointers=tuple(pointer for pointer, _ in records),
            model_fingerprint=self.model_fingerprint(self.model),
            dtype=str(getattr(self.static_inputs["input_ids"], "dtype", "")),
            quantization=str(getattr(getattr(self.model, "config", None), "quantization_config", "")),
        )

    def capture(
        self,
        *,
        input_ids: Any,
        position_ids: Any,
        attention_mask: Any,
        cache_position: Any,
        past_key_values: GraphStaticCache,
        logical_kv_len: int,
        kv_bucket: int,
        tactic_name: str,
        stream_ptr: int,
        stream: Any = None,
        device_id: int = 0,
        model_kwargs: Mapping[str, Any] | None = None,
        workspace: Any = None,
    ) -> Any:
        if self.captured:
            raise GraphContractError("graph runner already has a captured forward")
        self.static_inputs = {
            "input_ids": _clone_value(self.torch, input_ids),
            "position_ids": _clone_value(self.torch, position_ids),
            "attention_mask": _clone_value(self.torch, attention_mask),
            "cache_position": cache_position,
        }
        self.cache = past_key_values
        self.kv_len_device = self.torch.tensor(
            [int(logical_kv_len) + 1],
            dtype=self.torch.int32,
            device=input_ids.device,
        )
        if hasattr(self.bridge, "configure_qwen2_graph_dispatch"):
            self.bridge.configure_qwen2_graph_dispatch(
                kv_len_device=self.kv_len_device,
                kv_capacity=int(kv_bucket),
                workspace=workspace,
            )
        self.workspace = workspace
        self.cache.cache_position.copy_(
            self.torch.tensor(
                [int(logical_kv_len)],
                dtype=self.torch.long,
                device=self.cache.cache_position.device,
            )
        )
        use_torch_graph = (
            stream is not None
            and hasattr(getattr(self.torch, "cuda", None), "CUDAGraph")
            and callable(getattr(getattr(self.torch, "cuda", None), "graph", None))
        )
        self.backend = "torch_cuda_graph" if use_torch_graph else "native"
        self.bridge.reset_graph()
        try:
            kwargs = dict(model_kwargs or {})
            kwargs.update(
                {
                    "input_ids": self.static_inputs["input_ids"],
                    "position_ids": self.static_inputs["position_ids"],
                    "attention_mask": self.static_inputs["attention_mask"],
                    "past_key_values": self.cache,
                    "cache_position": self.static_inputs["cache_position"],
                    "use_cache": True,
                    "return_dict": True,
                }
            )
            inference_mode = getattr(self.torch, "inference_mode", None)
            context = inference_mode() if callable(inference_mode) else nullcontext()
            if use_torch_graph:
                self.cuda_graph = self.torch.cuda.CUDAGraph()
                with self.torch.cuda.graph(self.cuda_graph, stream=stream):
                    with context:
                        outputs = self.model(**kwargs)
                # CUDA graph capture records work; it does not execute the
                # recorded forward. Replay once to materialize the captured
                # logits before the caller samples the first decode token.
                self.cuda_graph.replay()
                self.replay_count = 1
            else:
                self.bridge.begin_capture(stream_ptr=int(stream_ptr))
                with context:
                    outputs = self.model(**kwargs)
            self.logits = getattr(outputs, "logits", None)
            if self.logits is None:
                raise GraphContractError("model decode output does not expose logits")
            if not use_torch_graph:
                self.bridge.end_capture(stream_ptr=int(stream_ptr))
                # Native end_capture produces an executable graph but, like
                # torch.cuda.graph, does not run the captured forward.
                self.bridge.replay_graph(stream_ptr=int(stream_ptr))
                self.replay_count = 1
        except Exception as error:
            try:
                self.bridge.reset_graph()
            finally:
                if hasattr(self.bridge, "clear_qwen2_graph_dispatch"):
                    self.bridge.clear_qwen2_graph_dispatch()
            self.status = "fallback"
            self.reason = f"capture_failed:{type(error).__name__}:{error}"
            raise GraphContractError(self.reason) from error
        finally:
            if hasattr(self.bridge, "clear_qwen2_graph_dispatch"):
                self.bridge.clear_qwen2_graph_dispatch()
        self.contract = StaticDecodeContract(
            key=self._key(
                tactic_name=tactic_name,
                device_id=device_id,
                kv_bucket=kv_bucket,
                logical_kv_len=int(logical_kv_len) + 1,
            ),
            stream_ptr=int(stream_ptr),
            logical_kv_len=int(logical_kv_len) + 1,
        )
        self.status = "captured"
        self.reason = None
        return self.logits

    def replay(
        self,
        *,
        next_token: Any,
        position_ids: Any,
        cache_position: int,
        logical_kv_len: int,
        stream_ptr: int,
    ) -> Any:
        if not self.captured or self.contract is None:
            raise GraphContractError("graph runner has no captured forward")
        if int(stream_ptr) != self.contract.stream_ptr:
            raise GraphContractError("graph replay stream changed")
        if int(logical_kv_len) + 1 > self.contract.key.kv_bucket:
            raise GraphContractError("graph replay KV length exceeds the captured bucket")
        if _shape(next_token) != _shape(self.static_inputs["input_ids"]):
            raise GraphContractError("graph replay token shape changed")
        if _shape(position_ids) != _shape(self.static_inputs["position_ids"]):
            raise GraphContractError("graph replay position shape changed")
        current_key = self._key(
            tactic_name=self.contract.key.tactic_name,
            device_id=self.contract.key.device_id,
            kv_bucket=self.contract.key.kv_bucket,
            logical_kv_len=int(logical_kv_len) + 1,
        )
        if current_key != self.contract.key:
            raise GraphContractError("graph replay pointer, shape, or model contract changed")
        self.static_inputs["input_ids"].copy_(next_token)
        self.static_inputs["position_ids"].copy_(position_ids)
        self.cache.cache_position.copy_(
            self.torch.tensor(
                [int(cache_position)],
                dtype=self.torch.long,
                device=self.cache.cache_position.device,
            )
        )
        self.kv_len_device.copy_(
            self.torch.tensor(
                [int(logical_kv_len) + 1],
                dtype=self.torch.int32,
                device=self.kv_len_device.device,
            )
        )
        if self.backend == "torch_cuda_graph" and self.cuda_graph is not None:
            self.cuda_graph.replay()
        else:
            self.bridge.replay_graph(stream_ptr=int(stream_ptr))
        self.replay_count += 1
        return self.logits

    def invalidate(self, reason: str) -> None:
        try:
            self.bridge.reset_graph()
        finally:
            if hasattr(self.bridge, "clear_qwen2_graph_dispatch"):
                self.bridge.clear_qwen2_graph_dispatch()
        self.contract = None
        self.status = "fallback"
        self.reason = str(reason)
