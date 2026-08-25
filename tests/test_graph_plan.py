import unittest

from KudaCuda.python.kudacuda_graph_plan import (
    DecodeGraphKey,
    GraphContractError,
    KudaCudaGraphPlan,
)


class FakeBridge:
    def __init__(self):
        self.calls = []

    def begin_capture(self, *, stream_ptr):
        self.calls.append(("begin", stream_ptr))

    def end_capture(self, *, stream_ptr):
        self.calls.append(("end", stream_ptr))

    def replay_graph(self, *, stream_ptr):
        self.calls.append(("replay", stream_ptr))


def graph_key(pointer_seed=100, kv_bucket=128):
    return DecodeGraphKey(
        device_id=0,
        tactic_name="qwen2_gqa_decode_sm86_fp16",
        layer_count=36,
        query_shape=(1, 16, 1, 128),
        key_shape=(1, 2, kv_bucket, 128),
        value_shape=(1, 2, kv_bucket, 128),
        output_shape=(1, 16, 1, 128),
        kv_bucket=kv_bucket,
        pointers=(pointer_seed, pointer_seed + 1, pointer_seed + 2, pointer_seed + 3),
    )


class GraphPlanTests(unittest.TestCase):
    def test_capture_then_replay_requires_the_same_static_key(self):
        bridge = FakeBridge()
        plan = KudaCudaGraphPlan(bridge)
        key = graph_key()

        plan.begin_capture(key, stream_ptr=77)
        plan.end_capture(key, stream_ptr=77)
        plan.replay(key, stream_ptr=77)

        self.assertEqual(bridge.calls, [("begin", 77), ("end", 77), ("replay", 77)])

    def test_changed_pointer_or_kv_bucket_is_a_safe_contract_failure(self):
        bridge = FakeBridge()
        plan = KudaCudaGraphPlan(bridge)
        key = graph_key()
        plan.begin_capture(key, stream_ptr=77)
        plan.end_capture(key, stream_ptr=77)

        with self.assertRaisesRegex(GraphContractError, "key"):
            plan.replay(graph_key(pointer_seed=200), stream_ptr=77)
        with self.assertRaisesRegex(GraphContractError, "key"):
            plan.replay(graph_key(kv_bucket=512), stream_ptr=77)

    def test_capture_state_rejects_nested_capture_and_invalidates(self):
        bridge = FakeBridge()
        plan = KudaCudaGraphPlan(bridge)
        key = graph_key()
        plan.begin_capture(key, stream_ptr=77)
        with self.assertRaisesRegex(GraphContractError, "capturing"):
            plan.begin_capture(key, stream_ptr=77)
        plan.invalidate()
        with self.assertRaisesRegex(GraphContractError, "captured"):
            plan.replay(key, stream_ptr=77)


if __name__ == "__main__":
    unittest.main()
