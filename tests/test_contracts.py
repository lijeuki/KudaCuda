import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

PYTHON_ROOT = Path(__file__).parents[1] / "python"
sys.path.insert(0, str(PYTHON_ROOT))

from kudacuda_sidecar import (  # noqa: E402
    KudaCudaBridge,
    NativeCudaError,
    Qwen2GqaContract,
    TensorContract,
    load_manifest,
    patch_qwen2_attention,
    patch_llama_attention,
    _qwen2_attention_dimensions,
    select_library_name,
)
from KudaCuda.harness.reference import rmsnorm_reference  # noqa: E402


class FakeNativeFunction:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class FakeNative:
    def __init__(self, vector_status=0, rmsnorm_status=0, qwen_status=0):
        self.calls = []
        self.vector_status = vector_status
        self.rmsnorm_status = rmsnorm_status
        self.qwen_status = qwen_status
        self.kudacuda_create = FakeNativeFunction(lambda device: 101)
        self.kudacuda_freeze = FakeNativeFunction(lambda handle: 0)
        self.kudacuda_can_dispatch = FakeNativeFunction(lambda handle, tactic, batch, seq: 1)
        self.kudacuda_vector_add = FakeNativeFunction(self._vector_add)
        self.kudacuda_rmsnorm = FakeNativeFunction(self._rmsnorm)
        self.kudacuda_qwen2_gqa_decode = FakeNativeFunction(self._qwen2_gqa_decode)
        self.kudacuda_begin_capture = FakeNativeFunction(lambda handle, stream: 0)
        self.kudacuda_end_capture = FakeNativeFunction(lambda handle, stream: 0)
        self.kudacuda_replay_graph = FakeNativeFunction(lambda handle, stream: 0)
        self.kudacuda_last_error = FakeNativeFunction(lambda handle: b"fake native error")
        self.kudacuda_destroy = FakeNativeFunction(lambda handle: self.calls.append(("destroy", handle)))

    def _vector_add(self, handle, a, b, output, count, stream):
        self.calls.append(("vector_add", handle, a, b, output, count, stream))
        return self.vector_status

    def _rmsnorm(self, handle, input_ptr, weight_ptr, output_ptr, rows, hidden, epsilon, stream):
        self.calls.append(("rmsnorm", handle, input_ptr, weight_ptr, output_ptr, rows, hidden, epsilon, stream))
        return self.rmsnorm_status

    def _qwen2_gqa_decode(
        self,
        handle,
        query_ptr,
        key_ptr,
        value_ptr,
        output_ptr,
        batch,
        query_len,
        kv_len,
        query_heads,
        kv_heads,
        head_dim,
        scale,
        stream,
    ):
        self.calls.append(
            (
                "qwen2_gqa_decode",
                handle,
                query_ptr,
                key_ptr,
                value_ptr,
                output_ptr,
                batch,
                query_len,
                kv_len,
                query_heads,
                kv_heads,
                head_dim,
                scale,
                stream,
            )
        )
        return self.qwen_status


class FakeTensor:
    def __init__(self, *, device="cuda", dtype="float32", shape=(4,), contiguous=True, pointer=100):
        self.device = device
        self.dtype = dtype
        self.shape = shape
        self._contiguous = contiguous
        self._pointer = pointer

    @property
    def is_cuda(self):
        return self.device == "cuda"

    def is_contiguous(self):
        return self._contiguous

    def dim(self):
        return len(self.shape)

    def numel(self):
        result = 1
        for value in self.shape:
            result *= value
        return result

    def data_ptr(self):
        return self._pointer


def make_bridge(native=None, manifest=None):
    return KudaCudaBridge(
        manifest=manifest
        or {
            "project": "KudaCuda",
            "target_arch": "sm_86",
            "implemented_tactics": ["vector_add", "rmsnorm", "fixed_test"],
            "contracts": {
                "vector_add": {
                    "dtype": "float32",
                    "layout": "contiguous",
                    "rank": 1,
                    "count_min": 1,
                },
                "fixed_test": {
                    "dtype": "float32",
                    "layout": "contiguous",
                    "rank": 3,
                    "batch_size": 2,
                    "seq_len": 4,
                },
                "rmsnorm": {
                    "dtype": "float32",
                    "layout": "contiguous",
                    "rank": 2,
                    "count_min": 1,
                },
            },
        },
        native_lib=native or FakeNative(),
    )


def qwen_manifest():
    return {
        "project": "KudaCuda",
        "target_arch": "sm_86",
        "implemented_tactics": ["qwen2_gqa_decode_sm86_fp16"],
        "planned_tactics": [],
        "contracts": {
            "qwen2_gqa_decode_sm86_fp16": {
                "dtype": "float16",
                "layout": "contiguous",
                "batch_size": 1,
                "query_len": 1,
                "kv_len_min": 1,
                "kv_len_max": 512,
                "query_heads": 16,
                "kv_heads": 2,
                "head_dim": 128,
            }
        },
    }


class ContractTests(unittest.TestCase):
    def test_qwen2_attention_dimensions_read_current_transformers_config_layout(self):
        module = SimpleNamespace(
            config=SimpleNamespace(num_attention_heads=16, num_key_value_heads=2),
            head_dim=128,
        )

        self.assertEqual(_qwen2_attention_dimensions(module), (16, 2, 128))

    def test_qwen2_contract_requires_decode_profile_and_fp16_contiguous_tensors(self):
        contract = Qwen2GqaContract.from_mapping(
            {
                "dtype": "float16",
                "layout": "contiguous",
                "batch_size": 1,
                "query_len": 1,
                "kv_len_min": 1,
                "kv_len_max": 512,
                "query_heads": 16,
                "kv_heads": 2,
                "head_dim": 128,
            }
        )

        self.assertTrue(contract.matches(1, 1, 128))
        self.assertFalse(contract.matches(2, 1, 128))
        self.assertFalse(contract.matches(1, 2, 128))
        self.assertFalse(contract.matches(1, 513, 128))

    def test_qwen2_bridge_passes_pointers_dimensions_and_stream(self):
        native = FakeNative()
        bridge = make_bridge(native=native, manifest=qwen_manifest())
        bridge.dispatch_qwen2_gqa_decode(
            FakeTensor(dtype="float16", shape=(1, 16, 1, 128), pointer=11),
            FakeTensor(dtype="float16", shape=(1, 2, 128, 128), pointer=22),
            FakeTensor(dtype="float16", shape=(1, 2, 128, 128), pointer=33),
            FakeTensor(dtype="float16", shape=(1, 16, 1, 128), pointer=44),
            kv_len=128,
            stream_ptr=77,
        )

        call = native.calls[-1]
        self.assertEqual(call[0], "qwen2_gqa_decode")
        self.assertEqual(call[1:6], (101, 11, 22, 33, 44))
        self.assertEqual(call[6:12], (1, 1, 128, 16, 2, 128))
        self.assertAlmostEqual(call[12], 1.0 / (128.0 ** 0.5))
        self.assertEqual(call[13], 77)

    def test_qwen2_bridge_surfaces_native_error(self):
        bridge = make_bridge(native=FakeNative(qwen_status=9), manifest=qwen_manifest())
        tensors = [
            FakeTensor(dtype="float16", shape=(1, 16, 1, 128), pointer=11),
            FakeTensor(dtype="float16", shape=(1, 2, 128, 128), pointer=22),
            FakeTensor(dtype="float16", shape=(1, 2, 128, 128), pointer=33),
            FakeTensor(dtype="float16", shape=(1, 16, 1, 128), pointer=44),
        ]

        with self.assertRaisesRegex(NativeCudaError, "fake native error"):
            bridge.dispatch_qwen2_gqa_decode(*tensors, kv_len=128, stream_ptr=77)
    def test_tensor_contract_requires_exact_batch_and_sequence_when_declared(self):
        contract = TensorContract.from_mapping(
            {
                "dtype": "float32",
                "layout": "contiguous",
                "rank": 3,
                "batch_size": 2,
                "seq_len": 4,
            }
        )

        self.assertTrue(contract.matches_shape(2, 4))
        self.assertFalse(contract.matches_shape(1, 4))
        self.assertFalse(contract.matches_shape(2, 8))

    def test_bridge_rejects_unimplemented_or_wrong_shape_contracts(self):
        bridge = make_bridge()

        self.assertTrue(bridge.matches_contract("fixed_test", 2, 4))
        self.assertFalse(bridge.matches_contract("fixed_test", 1, 4))
        self.assertFalse(bridge.matches_contract("gqa8_attn", 1, 16))

    def test_manifest_status_keeps_planned_attention_on_framework_fallback(self):
        manifest_path = Path(__file__).parents[1] / "manifests" / "rtx3050-sm86.json"
        bridge = KudaCudaBridge(manifest=manifest_path, native_lib=FakeNative())

        self.assertEqual(bridge.tactic_status("rmsnorm"), "implemented")
        self.assertEqual(bridge.tactic_status("gqa8_attn"), "planned")
        self.assertFalse(bridge.matches_contract("gqa8_attn", 1, 16))

    def test_bridge_validates_cuda_contiguous_float_tensor_metadata(self):
        bridge = make_bridge()
        valid = FakeTensor()

        self.assertTrue(bridge.matches_contract("vector_add", 1, 4, tensor=valid))
        self.assertFalse(bridge.matches_contract("vector_add", 1, 4, tensor=FakeTensor(device="cpu")))
        self.assertFalse(bridge.matches_contract("vector_add", 1, 4, tensor=FakeTensor(contiguous=False)))
        self.assertFalse(bridge.matches_contract("vector_add", 1, 4, tensor=FakeTensor(dtype="float16")))

    def test_vector_add_passes_raw_pointers_and_stream_to_native_bridge(self):
        native = FakeNative()
        bridge = make_bridge(native=native)
        tensors = [FakeTensor(pointer=11), FakeTensor(pointer=22), FakeTensor(pointer=33)]

        bridge.vector_add(*tensors, stream_ptr=77)

        self.assertEqual(native.calls, [("vector_add", 101, 11, 22, 33, 4, 77)])

    def test_vector_add_surfaces_native_error(self):
        bridge = make_bridge(native=FakeNative(vector_status=7))
        tensors = [FakeTensor(pointer=11), FakeTensor(pointer=22), FakeTensor(pointer=33)]

        with self.assertRaisesRegex(NativeCudaError, "fake native error"):
            bridge.vector_add(*tensors, stream_ptr=77)

    def test_rmsnorm_reference_is_independent_and_numerically_stable(self):
        result = rmsnorm_reference(
            [[3.0, 4.0], [0.0, 1.0]],
            [1.0, 0.5],
            epsilon=1.0e-5,
        )

        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0][0], 0.848528, places=5)
        self.assertAlmostEqual(result[0][1], 0.565685, places=5)
        self.assertAlmostEqual(result[1][1], 0.7071, places=4)

    def test_rmsnorm_passes_raw_pointers_shape_and_stream_to_native_bridge(self):
        native = FakeNative()
        bridge = make_bridge(native=native)
        input_tensor = FakeTensor(shape=(2, 4), pointer=11)
        weight_tensor = FakeTensor(shape=(4,), pointer=22)
        output_tensor = FakeTensor(shape=(2, 4), pointer=33)

        bridge.rmsnorm(input_tensor, weight_tensor, output_tensor, epsilon=1.0e-5, stream_ptr=77)

        self.assertEqual(
            native.calls,
            [("rmsnorm", 101, 11, 22, 33, 2, 4, 1.0e-5, 77)],
        )

    def test_rmsnorm_rejects_weight_with_wrong_hidden_dimension(self):
        bridge = make_bridge()
        input_tensor = FakeTensor(shape=(2, 4), pointer=11)
        weight_tensor = FakeTensor(shape=(3,), pointer=22)
        output_tensor = FakeTensor(shape=(2, 4), pointer=33)

        with self.assertRaisesRegex(ValueError, "weight"):
            bridge.rmsnorm(input_tensor, weight_tensor, output_tensor, epsilon=1.0e-5, stream_ptr=77)

    def test_manifest_loader_and_platform_library_names(self):
        manifest_path = Path(__file__).parents[1] / "manifests" / "rtx3050-sm86.json"
        manifest = load_manifest(manifest_path)

        self.assertEqual(manifest["target_arch"], "sm_86")
        self.assertEqual(
            manifest["implemented_tactics"],
            ["vector_add", "rmsnorm", "qwen2_gqa_decode_sm86_fp16"],
        )
        self.assertEqual(select_library_name("win32"), "kudacuda.dll")
        self.assertEqual(select_library_name("linux"), "libkudacuda.so")


class FakeAttention:
    def forward(self, hidden_states, *args, **kwargs):
        return ("fallback", hidden_states, args, kwargs)


class FakeAttentionBridge:
    def __init__(self, enabled):
        self.enabled = enabled
        self.calls = []

    def matches_contract(self, tactic_name, batch_size, seq_len, *, tensor=None):
        return self.enabled and tactic_name == "gqa8_attn"

    def dispatch_attention(self, module, hidden_states, *args, **kwargs):
        self.calls.append((module, hidden_states, args, kwargs))
        return ("hot", hidden_states)


class AttentionPatchTests(unittest.TestCase):
    def test_attention_patch_falls_back_and_is_reversible(self):
        original = FakeAttention.forward
        bridge = FakeAttentionBridge(enabled=False)
        unpatch = patch_llama_attention(bridge, attention_cls=FakeAttention)

        instance = FakeAttention()
        self.assertEqual(instance.forward(FakeTensor(shape=(1, 2, 8)), flag=True)[0], "fallback")
        unpatch()
        self.assertIs(FakeAttention.forward, original)

    def test_attention_patch_dispatches_only_when_real_dispatcher_is_available(self):
        bridge = FakeAttentionBridge(enabled=True)
        unpatch = patch_llama_attention(bridge, attention_cls=FakeAttention)

        instance = FakeAttention()
        self.assertEqual(instance.forward(FakeTensor(shape=(1, 2, 8)))[0], "hot")
        self.assertEqual(len(bridge.calls), 1)
        unpatch()

    def test_qwen2_attention_patch_falls_back_and_is_reversible(self):
        original = FakeAttention.forward
        bridge = FakeAttentionBridge(enabled=False)
        unpatch = patch_qwen2_attention(bridge, attention_cls=FakeAttention)

        instance = FakeAttention()
        self.assertEqual(instance.forward(FakeTensor(shape=(1, 1, 2048)))[0], "fallback")
        unpatch()
        self.assertIs(FakeAttention.forward, original)


if __name__ == "__main__":
    unittest.main()
