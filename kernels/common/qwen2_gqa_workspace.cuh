#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace kudacuda {

constexpr int kQwen2WorkspaceTile = 64;

__global__ void qwen2_gqa_workspace_partial_kernel(
    const __half* query,
    const __half* key_cache,
    const __half* value_cache,
    float* workspace,
    const int32_t* kv_len_device,
    int64_t kv_capacity,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale
) {
    if (threadIdx.x != 0) {
        return;
    }
    const int64_t tiles = (kv_capacity + kQwen2WorkspaceTile - 1) / kQwen2WorkspaceTile;
    const int64_t work_id = static_cast<int64_t>(blockIdx.x);
    const int64_t query_head = work_id / tiles;
    const int64_t tile = work_id % tiles;
    if (query_head >= query_heads || kv_heads <= 0 || head_dim <= 0) {
        return;
    }
    int64_t kv_len = static_cast<int64_t>(*kv_len_device);
    kv_len = kv_len < 0 ? 0 : (kv_len > kv_capacity ? kv_capacity : kv_len);
    const int64_t kv_head = query_head / (query_heads / kv_heads);
    const int64_t tile_start = tile * kQwen2WorkspaceTile;
    const int64_t workspace_stride = 2 + head_dim;
    float* partial = workspace + work_id * workspace_stride;
    const __half* query_head_ptr = query + query_head * head_dim;
    constexpr float kNegativeInfinity = -3.402823466e+38F;
    const int64_t tile_end = (tile_start + kQwen2WorkspaceTile < kv_len) ? tile_start + kQwen2WorkspaceTile : kv_len;
    float tile_max = kNegativeInfinity;
    for (int64_t position = tile_start; position < tile_end; ++position) {
        const __half* key_ptr = key_cache + (kv_head * kv_capacity + position) * head_dim;
        float score = 0.0F;
        for (int64_t dimension = 0; dimension < head_dim; ++dimension) {
            score += __half2float(query_head_ptr[dimension]) * __half2float(key_ptr[dimension]);
        }
        tile_max = fmaxf(tile_max, score * scale);
    }
    partial[0] = tile_max;
    if (!isfinite(tile_max)) {
        partial[1] = 0.0F;
        for (int64_t dimension = 0; dimension < head_dim; ++dimension) {
            partial[2 + dimension] = 0.0F;
        }
        return;
    }
    float tile_sum = 0.0F;
    for (int64_t dimension = 0; dimension < head_dim; ++dimension) {
        partial[2 + dimension] = 0.0F;
    }
    for (int64_t position = tile_start; position < tile_end; ++position) {
        const __half* key_ptr = key_cache + (kv_head * kv_capacity + position) * head_dim;
        const __half* value_ptr = value_cache + (kv_head * kv_capacity + position) * head_dim;
        float score = 0.0F;
        for (int64_t dimension = 0; dimension < head_dim; ++dimension) {
            score += __half2float(query_head_ptr[dimension]) * __half2float(key_ptr[dimension]);
        }
        const float probability = expf(score * scale - tile_max);
        tile_sum += probability;
        for (int64_t dimension = 0; dimension < head_dim; ++dimension) {
            partial[2 + dimension] += probability * __half2float(value_ptr[dimension]);
        }
    }
    partial[1] = tile_sum;
}

__global__ void qwen2_gqa_workspace_reduce_kernel(
    float* workspace,
    __half* output,
    int64_t query_heads,
    int64_t head_dim,
    int64_t tiles
) {
    if (threadIdx.x != 0) {
        return;
    }
    const int64_t query_head = static_cast<int64_t>(blockIdx.x);
    if (query_head >= query_heads) {
        return;
    }
    const int64_t workspace_stride = 2 + head_dim;
    constexpr float kNegativeInfinity = -3.402823466e+38F;
    float global_max = kNegativeInfinity;
    for (int64_t tile = 0; tile < tiles; ++tile) {
        global_max = fmaxf(global_max, workspace[(query_head * tiles + tile) * workspace_stride]);
    }
    float total = 0.0F;
    for (int64_t tile = 0; tile < tiles; ++tile) {
        float* partial = workspace + (query_head * tiles + tile) * workspace_stride;
        if (!isfinite(partial[0])) {
            continue;
        }
        const float factor = expf(partial[0] - global_max);
        total += partial[1] * factor;
        for (int64_t dimension = 0; dimension < head_dim; ++dimension) {
            partial[2 + dimension] *= factor;
        }
    }
    for (int64_t dimension = 0; dimension < head_dim; ++dimension) {
        float weighted = 0.0F;
        for (int64_t tile = 0; tile < tiles; ++tile) {
            weighted += workspace[(query_head * tiles + tile) * workspace_stride + 2 + dimension];
        }
        output[query_head * head_dim + dimension] = __float2half(total > 0.0F ? weighted / total : 0.0F);
    }
}

inline cudaError_t launch_qwen2_gqa_workspace(
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
    if (batch != 1 || query_len != 1 || kv_capacity != 512 || query_heads != 16 || kv_heads != 2 || head_dim != 128) {
        return cudaErrorInvalidValue;
    }
    const int64_t tiles = (kv_capacity + kQwen2WorkspaceTile - 1) / kQwen2WorkspaceTile;
    qwen2_gqa_workspace_partial_kernel<<<static_cast<unsigned int>(query_heads * tiles), 1, 0, stream>>>(
        query, key_cache, value_cache, workspace, kv_len_device, kv_capacity, query_heads, kv_heads, head_dim, scale
    );
    qwen2_gqa_workspace_reduce_kernel<<<static_cast<unsigned int>(query_heads), 1, 0, stream>>>(
        workspace, output, query_heads, head_dim, tiles
    );
    return cudaGetLastError();
}

}  // namespace kudacuda
