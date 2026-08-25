import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1] / "python"))

from kudacuda_static_decode import StaticDecodeGraphRunner  # noqa: E402


class FakeTensor:
    next_pointer = 1000

    def __init__(self, shape, *, pointer=None, dtype="float16", device="cuda"):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        if pointer is None:
            pointer = FakeTensor.next_pointer
            FakeTensor.next_pointer += 1
        self.pointer = pointer
        self.copy_calls = 0
        self.fill_values = []

    def data_ptr(self):
        return self.pointer

    def clone(self):
        return FakeTensor(self.shape, dtype=self.dtype, device=self.device)

    def copy_(self, other):
        self.copy_calls += 1
        return self

    def __getitem__(self, key):
        return self

    def fill_(self, value):
        self.fill_values.append(value)
        return self


class FakeTorch:
    int32 = "int32"
    long = "int64"
    float16 = "float16"
    bfloat16 = "bfloat16"
    float32 = "float32"

    class _Context:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    @classmethod
    def inference_mode(cls):
        return cls._Context()

    @staticmethod
    def tensor(values, *, dtype, device):
        return FakeTensor((len(values),), dtype=dtype, device=device)

    @staticmethod
    def finfo(dtype):
        return SimpleNamespace(min=-65504.0 if dtype == "float16" else -3.4e38)


class FakeBridge:
    def __init__(self):
        self.calls = []

    def configure_qwen2_graph_dispatch(self, *, kv_len_device, kv_capacity, workspace=None):
        self.calls.append(("configure", kv_len_device.data_ptr(), kv_capacity))

    def clear_qwen2_graph_dispatch(self):
        self.calls.append(("clear",))

    def reset_graph(self):
        self.calls.append(("reset",))

    def begin_capture(self, *, stream_ptr):
        self.calls.append(("begin", stream_ptr))

    def end_capture(self, *, stream_ptr):
        self.calls.append(("end", stream_ptr))

    def replay_graph(self, *, stream_ptr):
        self.calls.append(("replay", stream_ptr))


class FakeCache:
    def __init__(self):
        self.key_cache = [FakeTensor((1, 2, 128, 128), pointer=200)]
        self.value_cache = [FakeTensor((1, 2, 128, 128), pointer=300)]
        self.cache_position = FakeTensor((1,), pointer=400, dtype="int64")


class FakeModel:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail
        self.config = SimpleNamespace(
            model_type="qwen2",
            num_hidden_layers=1,
            hidden_size=256,
            vocab_size=1024,
            torch_dtype="float16",
            quantization_config="nf4",
        )

    def __call__(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("capture boom")
        return SimpleNamespace(logits=FakeTensor((1, 1, 1024), pointer=500))


def make_runner(model=None):
    bridge = FakeBridge()
    model = model or FakeModel()
    runner = StaticDecodeGraphRunner(bridge, model, FakeTorch)
    common = {
        "input_ids": FakeTensor((1, 1), dtype="int64", pointer=10),
        "position_ids": FakeTensor((1, 1), dtype="int64", pointer=11),
        "attention_mask": {"full_attention": FakeTensor((1, 1, 1, 128), pointer=12)},
        "cache_position": FakeTensor((1,), dtype="int64", pointer=13),
        "past_key_values": FakeCache(),
        "logical_kv_len": 32,
        "kv_bucket": 128,
        "tactic_name": "qwen2_gqa_decode_sm86_fp16_graph_v1",
        "stream_ptr": 77,
    }
    return runner, bridge, model, common


class StaticDecodeTests(unittest.TestCase):
    def test_capture_replay_does_not_call_model_again(self):
        runner, bridge, model, common = make_runner()
        logits = runner.capture(**common)
        replayed = runner.replay(
            next_token=FakeTensor((1, 1), dtype="int64", pointer=20),
            position_ids=FakeTensor((1, 1), dtype="int64", pointer=21),
            cache_position=33,
            logical_kv_len=33,
            stream_ptr=77,
        )

        self.assertIs(logits, replayed)
        self.assertEqual(model.calls, 1)
        self.assertEqual(runner.status, "captured")
        self.assertIn(("replay", 77), bridge.calls)

    def test_capture_and_replay_update_floating_graph_mask_to_current_kv_length(self):
        runner, _bridge, _model, common = make_runner()
        runner.capture(**common)
        mask = runner.static_inputs["attention_mask"]["full_attention"]
        self.assertIn(-65504.0, mask.fill_values)

        runner.replay(
            next_token=FakeTensor((1, 1), dtype="int64", pointer=20),
            position_ids=FakeTensor((1, 1), dtype="int64", pointer=21),
            cache_position=33,
            logical_kv_len=33,
            stream_ptr=77,
        )
        self.assertGreaterEqual(mask.fill_values.count(-65504.0), 2)

    def test_changed_stream_is_rejected_before_replay(self):
        runner, _bridge, _model, common = make_runner()
        runner.capture(**common)
        with self.assertRaisesRegex(RuntimeError, "stream changed"):
            runner.replay(
                next_token=FakeTensor((1, 1), dtype="int64"),
                position_ids=FakeTensor((1, 1), dtype="int64"),
                cache_position=33,
                logical_kv_len=33,
                stream_ptr=88,
            )

    def test_capture_failure_resets_graph_and_records_fallback(self):
        runner, bridge, _model, common = make_runner(FakeModel(fail=True))
        with self.assertRaisesRegex(RuntimeError, "capture_failed"):
            runner.capture(**common)
        self.assertEqual(runner.status, "fallback")
        self.assertIn(("reset",), bridge.calls)


if __name__ == "__main__":
    unittest.main()
