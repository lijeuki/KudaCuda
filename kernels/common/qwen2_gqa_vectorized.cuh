#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>

namespace kudacuda {

constexpr int kQwen2VectorizedThreads = 128;
constexpr int kQwen2VectorizedMaxKvLen = 512;
constexpr int kQwen2VectorizedHeadDim = 128;

__device__ __forceinline__ __half2 load_vectorized_half2(const __half* source) {
    const auto address = reinterpret_cast<unsigned long long>(source);
    if ((address & 0x3ULL) == 0ULL) {
        return *reinterpret_cast<const __half2*>(source);
    }
    return __halves2half2(source[0], source[1]);
}

__global__ void qwen2_gqa_vectorized_kernel(
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
    float scale
) {
    const int64_t head_index = static_cast<int64_t>(blockIdx.x);
    const int64_t total_heads = batch_size * query_heads;
    if (head_index >= total_heads || query_len != 1 || head_dim != kQwen2VectorizedHeadDim) {
        return;
    }

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int64_t batch = head_index / query_heads;
    const int64_t query_head = head_index % query_heads;
    const int64_t group_size = query_heads / kv_heads;
    const int64_t kv_head = query_head / group_size;

    const __half* query_head_data =
        query + (batch * query_heads + query_head) * head_dim;
    const __half* key_head_data =
        key_cache + (batch * kv_heads + kv_head) * kv_len * head_dim;
    const __half* value_head_data =
        value_cache + (batch * kv_heads + kv_head) * kv_len * head_dim;
    __half* output_head_data =
        output + (batch * query_heads + query_head) * head_dim;

    __shared__ float scores[kQwen2VectorizedMaxKvLen];
    __shared__ float query_values[kQwen2VectorizedThreads];
    __shared__ float reduction[kQwen2VectorizedThreads];
    __shared__ float max_score;
    __shared__ float inverse_sum;

    query_values[tid] = __half2float(query_head_data[tid]);
    __syncthreads();

    for (int64_t token_base = 0; token_base < kv_len; token_base += 4) {
        const int64_t token = token_base + warp;
        float partial = 0.0F;
        if (token < kv_len) {
            const int dimension = lane * 4;
            const __half2 key_pair0 = load_vectorized_half2(
                key_head_data + token * head_dim + dimension
            );
            const __half2 key_pair1 = load_vectorized_half2(
                key_head_data + token * head_dim + dimension + 2
            );
            partial += query_values[dimension] * __low2float(key_pair0);
            partial += query_values[dimension + 1] * __high2float(key_pair0);
            partial += query_values[dimension + 2] * __low2float(key_pair1);
            partial += query_values[dimension + 3] * __high2float(key_pair1);
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial += __shfl_down_sync(0xffffffffU, partial, offset);
        }
        if (lane == 0 && token < kv_len) {
            scores[token] = partial * scale;
        }
    }
    __syncthreads();

    float local_maximum = -FLT_MAX;
    for (int64_t token = tid; token < kv_len; token += blockDim.x) {
        local_maximum = fmaxf(local_maximum, scores[token]);
    }
    reduction[tid] = local_maximum;
    __syncthreads();
    for (int stride = kQwen2VectorizedThreads / 2; stride > 0; stride >>= 1) {
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
    for (int stride = kQwen2VectorizedThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduction[tid] += reduction[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        inverse_sum = 1.0F / reduction[0];
    }
    __syncthreads();

    const int dimension = lane * 4;
    float result0 = 0.0F;
    float result1 = 0.0F;
    float result2 = 0.0F;
    float result3 = 0.0F;
    for (int64_t token = 0; token < kv_len; ++token) {
        const float probability = expf(scores[token] - max_score) * inverse_sum;
        const __half2 value_pair0 = load_vectorized_half2(
            value_head_data + token * head_dim + dimension
        );
        const __half2 value_pair1 = load_vectorized_half2(
            value_head_data + token * head_dim + dimension + 2
        );
        result0 += probability * __low2float(value_pair0);
        result1 += probability * __high2float(value_pair0);
        result2 += probability * __low2float(value_pair1);
        result3 += probability * __high2float(value_pair1);
    }
    output_head_data[dimension] = __float2half(result0);
    output_head_data[dimension + 1] = __float2half(result1);
    output_head_data[dimension + 2] = __float2half(result2);
    output_head_data[dimension + 3] = __float2half(result3);
}

inline cudaError_t launch_qwen2_gqa_vectorized(
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
    qwen2_gqa_vectorized_kernel<<<
        static_cast<unsigned int>(batch_size * query_heads),
        kQwen2VectorizedThreads,
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
        query_heads,
        kv_heads,
        head_dim,
        scale
    );
    return cudaGetLastError();
}

}  // namespace kudacuda
