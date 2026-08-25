#pragma once

#include <stdint.h>

#if defined(_WIN32)
#  if defined(kudacuda_EXPORTS)
#    define KUDACUDA_API __declspec(dllexport)
#  else
#    define KUDACUDA_API __declspec(dllimport)
#  endif
#else
#  define KUDACUDA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

KUDACUDA_API void* kudacuda_create(int device_id);

KUDACUDA_API int kudacuda_freeze(void* handle);

KUDACUDA_API int kudacuda_can_dispatch(
    void* handle,
    const char* tactic_name,
    int64_t batch_size,
    int64_t seq_len
);

KUDACUDA_API int kudacuda_vector_add(
    void* handle,
    uint64_t a,
    uint64_t b,
    uint64_t output,
    int64_t count,
    uint64_t stream
);

KUDACUDA_API int kudacuda_rmsnorm(
    void* handle,
    uint64_t input,
    uint64_t weight,
    uint64_t output,
    int64_t rows,
    int64_t hidden,
    float epsilon,
    uint64_t stream
);

KUDACUDA_API int kudacuda_fused_rmsnorm_qkv(
    void* handle,
    uint64_t input,
    uint64_t norm_weight,
    uint64_t qkv_weight,
    uint64_t bias,
    uint64_t output,
    int64_t rows,
    int64_t hidden,
    float epsilon,
    uint64_t stream
);

KUDACUDA_API int kudacuda_qwen2_fused_rmsnorm_qkv_nf4(
    void* handle,
    uint64_t input,
    uint64_t norm_weight,
    uint64_t q_weight_packed,
    uint64_t q_absmax,
    uint64_t q_bias,
    uint64_t k_weight_packed,
    uint64_t k_absmax,
    uint64_t k_bias,
    uint64_t v_weight_packed,
    uint64_t v_absmax,
    uint64_t v_bias,
    uint64_t q_output,
    uint64_t k_output,
    uint64_t v_output,
    int64_t rows,
    int64_t hidden,
    int64_t q_out,
    int64_t kv_out,
    int64_t blocksize,
    float epsilon,
    uint64_t stream
);

KUDACUDA_API int kudacuda_qwen2_gqa_decode_workspace_v3(
    void* handle,
    uint64_t query,
    uint64_t key_cache,
    uint64_t value_cache,
    uint64_t output,
    uint64_t workspace,
    uint64_t kv_len_device,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_capacity,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    uint64_t stream
);

KUDACUDA_API int kudacuda_qwen2_gqa_decode(
    void* handle,
    uint64_t query,
    uint64_t key_cache,
    uint64_t value_cache,
    uint64_t output,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_len,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    uint64_t stream
);

KUDACUDA_API int kudacuda_qwen2_gqa_decode_graph(
    void* handle,
    uint64_t query,
    uint64_t key_cache,
    uint64_t value_cache,
    uint64_t output,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_capacity,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    uint64_t kv_len_device,
    uint64_t stream
);

KUDACUDA_API int kudacuda_qwen2_gqa_decode_group_tiled(
    void* handle,
    uint64_t query,
    uint64_t key_cache,
    uint64_t value_cache,
    uint64_t output,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_len,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    uint64_t stream
);

KUDACUDA_API int kudacuda_qwen2_gqa_decode_vectorized(
    void* handle,
    uint64_t query,
    uint64_t key_cache,
    uint64_t value_cache,
    uint64_t output,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_len,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    uint64_t stream
);

KUDACUDA_API int kudacuda_begin_capture(void* handle, uint64_t stream);

KUDACUDA_API int kudacuda_end_capture(void* handle, uint64_t stream);

KUDACUDA_API int kudacuda_replay_graph(void* handle, uint64_t stream);

KUDACUDA_API int kudacuda_reset_graph(void* handle);

KUDACUDA_API const char* kudacuda_last_error(void* handle);

KUDACUDA_API void kudacuda_destroy(void* handle);

#ifdef __cplusplus
}
#endif
