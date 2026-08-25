from tools.profiling.profile_qwen2_operations import aggregate_operation_timings, operation_family


def test_operation_family_and_aggregation_are_stable_without_torch():
    assert operation_family("model.layers.0.self_attn") == "attention"
    assert operation_family("model.layers.0.self_attn.q_proj") == "qkv_projection"
    assert operation_family("model.layers.0.self_attn.o_proj") == "output_projection"
    assert operation_family("model.layers.0.mlp") == "mlp_gate_up_down"
    assert operation_family("model.layers.0.mlp.down_proj") == "mlp_down_projection"
    assert operation_family("model.layers.0.input_layernorm") == "input_rmsnorm"

    result = aggregate_operation_timings(
        [
            {"family": "attention", "ms": 2.0},
            {"family": "attention", "ms": 4.0},
            {"family": "mlp_gate_up_down", "ms": 3.0},
        ]
    )
    assert result["attention"] == {"calls": 2.0, "total_ms": 6.0, "mean_ms": 3.0, "max_ms": 4.0}
    assert result["mlp_gate_up_down"]["total_ms"] == 3.0
