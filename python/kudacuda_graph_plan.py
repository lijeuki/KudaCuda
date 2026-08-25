"""Static-contract planning for opt-in CUDA Graph decode replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class GraphContractError(RuntimeError):
    """Raised when a graph capture or replay violates its static contract."""


@dataclass(frozen=True)
class DecodeGraphKey:
    """All state that must remain stable between capture and replay."""

    device_id: int
    tactic_name: str
    layer_count: int
    query_shape: tuple[int, ...]
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    kv_bucket: int
    pointers: tuple[int, ...]
    model_fingerprint: str = ""
    dtype: str = ""
    quantization: str = ""


class KudaCudaGraphPlan:
    """Guard existing bridge graph primitives with an exact static key."""

    def __init__(self, bridge: Any) -> None:
        self.bridge = bridge
        self._capturing_key: DecodeGraphKey | None = None
        self._capturing_stream: int | None = None
        self._captured_key: DecodeGraphKey | None = None
        self._captured_stream: int | None = None

    @property
    def captured(self) -> bool:
        return self._captured_key is not None

    @property
    def capturing(self) -> bool:
        return self._capturing_key is not None

    def begin_capture(self, key: DecodeGraphKey, *, stream_ptr: int) -> None:
        if self.capturing:
            raise GraphContractError("graph plan is already capturing")
        if self.captured:
            raise GraphContractError("graph plan already has a captured key; invalidate it first")
        self.bridge.begin_capture(stream_ptr=int(stream_ptr))
        self._capturing_key = key
        self._capturing_stream = int(stream_ptr)

    def end_capture(self, key: DecodeGraphKey, *, stream_ptr: int) -> None:
        if not self.capturing or self._capturing_key is None:
            raise GraphContractError("graph plan is not capturing")
        if key != self._capturing_key or int(stream_ptr) != self._capturing_stream:
            raise GraphContractError("graph capture key or stream changed")
        try:
            self.bridge.end_capture(stream_ptr=int(stream_ptr))
        except Exception:
            self._capturing_key = None
            self._capturing_stream = None
            raise
        self._captured_key = key
        self._captured_stream = int(stream_ptr)
        self._capturing_key = None
        self._capturing_stream = None

    def can_replay(self, key: DecodeGraphKey, *, stream_ptr: int) -> bool:
        return (
            self.captured
            and self._captured_key == key
            and self._captured_stream == int(stream_ptr)
        )

    def replay(self, key: DecodeGraphKey, *, stream_ptr: int) -> None:
        if not self.captured:
            raise GraphContractError("graph plan has no captured graph")
        if not self.can_replay(key, stream_ptr=stream_ptr):
            raise GraphContractError("graph replay key or stream changed")
        self.bridge.replay_graph(stream_ptr=int(stream_ptr))

    def invalidate(self) -> None:
        self._capturing_key = None
        self._capturing_stream = None
        self._captured_key = None
        self._captured_stream = None
