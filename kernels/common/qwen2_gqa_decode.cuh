#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace kudacuda {

constexpr int kQwen2GqaThreads = 128;
// The direct decode profile supports the long-context path used by the
// interactive Transformers adapter. Graph and experimental kernels retain
// their own 512-token guards in their entry points until separately tuned.
constexpr int kQwen2GqaMaxKvLen = 2048;
constexpr int kQwen2GqaWarpCount = kQwen2GqaThreads / 32;

template <bool kReadKvLenFromDevice>
__global__ void qwen2_gqa_decode_kernel(
    const __half* query,
    const __half* key_cache,
    const __half* value_cache,
    __half* output,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_len_argument,
    int64_t kv_stride,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    const int32_t* kv_len_device
) {
    const int64_t kv_len = kReadKvLenFromDevice
        ? static_cast<int64_t>(*kv_len_device)
        : kv_len_argument;
    const int64_t head_index = static_cast<int64_t>(blockIdx.x);
    const int64_t total_heads = batch_size * query_heads;
    if (head_index >= total_heads || query_len != 1 || head_dim != kQwen2GqaThreads ||
        kv_len < 1 || kv_len > kQwen2GqaMaxKvLen || kv_stride < kv_len) {
        return;
    }

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int64_t batch = head_index / query_heads;
    const int64_t query_head = head_index % query_heads;
    const int64_t group_size = query_heads / kv_heads;
    const int64_t kv_head = query_head / group_size;

    const __half* query_head_data = query + (batch * query_heads + query_head) * head_dim;
    const __half* key_head_data = key_cache + (batch * kv_heads + kv_head) * kv_stride * head_dim;
    const __half* value_head_data = value_cache + (batch * kv_heads + kv_head) * kv_stride * head_dim;
    __half* output_head_data = output + (batch * query_heads + query_head) * head_dim;

    __shared__ float scores[kQwen2GqaMaxKvLen];
    __shared__ float query_values[kQwen2GqaThreads];
    __shared__ float reduction[kQwen2GqaThreads];
    __shared__ float max_score;
    __shared__ float inverse_sum;

    query_values[tid] = __half2float(query_head_data[tid]);
    __syncthreads();

    // Each warp computes one token score at a time. Four warps therefore
    // produce four independent dot products without a block-wide barrier per
    // token; the previous pilot serialized every score through thread zero.
    for (int64_t token_base = 0; token_base < kv_len; token_base += kQwen2GqaWarpCount) {
        const int64_t token = token_base + warp;
        float partial = 0.0F;
        if (token < kv_len) {
            for (int dimension = lane; dimension < kQwen2GqaThreads; dimension += 32) {
                partial += query_values[dimension] *
                           __half2float(key_head_data[token * head_dim + dimension]);
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(0xffffffffU, partial, offset);
        }
        if (lane == 0) {
            if (token < kv_len) {
                scores[token] = partial * scale;
            }
        }
    }
    __syncthreads();

    float local_maximum = -3.402823466e+38F;
    for (int64_t token = tid; token < kv_len; token += blockDim.x) {
        local_maximum = fmaxf(local_maximum, scores[token]);
    }
    reduction[tid] = local_maximum;
    __syncthreads();
    for (int stride = kQwen2GqaThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduction[tid] = fmaxf(reduction[tid], reduction[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        max_score = reduction[0];
    }
    __syncthreads();

    float local_sum = 0.0F;
    for (int64_t token = tid; token < kv_len; token += blockDim.x) {
        local_sum += expf(scores[token] - max_score);
    }
    reduction[tid] = local_sum;
    __syncthreads();
    for (int stride = kQwen2GqaThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduction[tid] += reduction[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        inverse_sum = 1.0F / reduction[0];
    }
    __syncthreads();

    float result = 0.0F;
    for (int64_t token = 0; token < kv_len; ++token) {
        const float probability = expf(scores[token] - max_score) * inverse_sum;
        result += probability * __half2float(value_head_data[token * head_dim + tid]);
    }
    output_head_data[tid] = __float2half(result);
}

inline cudaError_t launch_qwen2_gqa_decode(
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
    qwen2_gqa_decode_kernel<false><<<
        static_cast<unsigned int>(batch_size * query_heads),
        kQwen2GqaThreads,
        0,
        stream
    >>>(
        query,
        key_cache,
        value_cache,
        output,
        batch_size,
        query_len,
        kv_len,
        kv_len,
        query_heads,
        kv_heads,
        head_dim,
        scale,
        nullptr
    );
    return cudaGetLastError();
}

inline cudaError_t launch_qwen2_gqa_decode_graph(
    const __half* query,
    const __half* key_cache,
    const __half* value_cache,
    __half* output,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_capacity,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    const int32_t* kv_len_device,
    cudaStream_t stream
) {
    qwen2_gqa_decode_kernel<true><<<
        static_cast<unsigned int>(batch_size * query_heads),
        kQwen2GqaThreads,
        0,
        stream
    >>>(
        query,
        key_cache,
        value_cache,
        output,
        batch_size,
        query_len,
        0,
        kv_capacity,
        query_heads,
        kv_heads,
        head_dim,
        scale,
        kv_len_device
    );
    return cudaGetLastError();
}

}  // namespace kudacuda
