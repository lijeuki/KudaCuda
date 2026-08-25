#include "../common/qwen2_gqa_workspace.cuh"

extern "C" cudaError_t kudacuda_launch_qwen2_gqa_decode_workspace_v3(
    const __half* query,
    const __half* key_cache,
    const __half* value_cache,
    __half* output,
    float* workspace,
    const int32_t* kv_len_device,
    int64_t batch,
    int64_t query_len,
    int64_t kv_capacity,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    cudaStream_t stream
) {
    return kudacuda::launch_qwen2_gqa_workspace(
        query, key_cache, value_cache, output, workspace, kv_len_device,
        batch, query_len, kv_capacity, query_heads, kv_heads, head_dim, scale, stream
    );
}
