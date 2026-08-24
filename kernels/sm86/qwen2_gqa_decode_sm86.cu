#include "../common/qwen2_gqa_decode.cuh"

extern "C" cudaError_t kudacuda_launch_qwen2_gqa_decode(
    const __half* query,
    const __half* key_cache,
    const __half* value_cache,
    __half* output,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_len,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    cudaStream_t stream
) {
    return kudacuda::launch_qwen2_gqa_decode(
        query,
        key_cache,
        value_cache,
        output,
        batch_size,
        query_len,
        kv_len,
        query_heads,
        kv_heads,
        head_dim,
        scale,
        stream
    );
}
