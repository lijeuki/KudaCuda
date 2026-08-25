#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cfloat>

namespace kudacuda {

constexpr int kQwen2GroupTileThreads = 256;
constexpr int kQwen2GroupTileWarps = 8;
constexpr int kQwen2GroupTileQueryHeadsPerKv = 8;
constexpr int kQwen2GroupTileTokens = 32;
constexpr int kQwen2GroupTileHeadDim = 128;
constexpr int kQwen2GroupTilePairsPerToken = kQwen2GroupTileHeadDim / 2;
constexpr int kQwen2GroupTilePairs = kQwen2GroupTileTokens * kQwen2GroupTilePairsPerToken;

__device__ __forceinline__ __half2 load_aligned_half2(const __half* source) {
    const auto address = reinterpret_cast<unsigned long long>(source);
    if ((address & 0x3ULL) == 0ULL) {
        return *reinterpret_cast<const __half2*>(source);
    }
    return __halves2half2(source[0], source[1]);
}

__global__ void qwen2_gqa_group_tiled_kernel(
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
    const int64_t group_index = static_cast<int64_t>(blockIdx.x);
    const int64_t total_groups = batch_size * kv_heads;
    if (group_index >= total_groups || query_len != 1 || query_heads != 16 ||
        kv_heads != 2 || head_dim != kQwen2GroupTileHeadDim || kv_len < 1 || kv_len > 512) {
        return;
    }

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (warp >= kQwen2GroupTileWarps) {
        return;
    }

    const int64_t batch = group_index / kv_heads;
    const int64_t kv_head = group_index % kv_heads;
    const int64_t group_size = query_heads / kv_heads;
    const int64_t query_head = kv_head * group_size + warp;

    const __half* query_head_data =
        query + (batch * query_heads + query_head) * head_dim;
    const __half* key_head_data =
        key_cache + (batch * kv_heads + kv_head) * kv_len * head_dim;
    const __half* value_head_data =
        value_cache + (batch * kv_heads + kv_head) * kv_len * head_dim;
    __half* output_head_data =
        output + (batch * query_heads + query_head) * head_dim;

    __shared__ __half key_tile[kQwen2GroupTileTokens * kQwen2GroupTileHeadDim];
    __shared__ __half value_tile[kQwen2GroupTileTokens * kQwen2GroupTileHeadDim];

    const int query_dim = lane * 4;
    const __half2 query_pair0 = load_aligned_half2(query_head_data + query_dim);
    const __half2 query_pair1 = load_aligned_half2(query_head_data + query_dim + 2);
    const float query0 = __low2float(query_pair0);
    const float query1 = __high2float(query_pair0);
    const float query2 = __low2float(query_pair1);
    const float query3 = __high2float(query_pair1);

    float running_maximum = -FLT_MAX;
    float running_sum = 0.0F;
    float accumulated0 = 0.0F;
    float accumulated1 = 0.0F;
    float accumulated2 = 0.0F;
    float accumulated3 = 0.0F;

    for (int64_t tile_start = 0; tile_start < kv_len; tile_start += kQwen2GroupTileTokens) {
        const int tile_length = static_cast<int>(
            min(static_cast<int64_t>(kQwen2GroupTileTokens), kv_len - tile_start)
        );

        for (int pair_index = tid; pair_index < kQwen2GroupTilePairs; pair_index += blockDim.x) {
            const int tile_token = pair_index / kQwen2GroupTilePairsPerToken;
            const int pair_dim = (pair_index % kQwen2GroupTilePairsPerToken) * 2;
            __half2 key_pair = __float2half2_rn(0.0F);
            __half2 value_pair = __float2half2_rn(0.0F);
            if (tile_token < tile_length) {
                const __half* key_source =
                    key_head_data + (tile_start + tile_token) * head_dim + pair_dim;
                const __half* value_source =
                    value_head_data + (tile_start + tile_token) * head_dim + pair_dim;
                key_pair = load_aligned_half2(key_source);
                value_pair = load_aligned_half2(value_source);
            }
            reinterpret_cast<__half2*>(key_tile)[pair_index] = key_pair;
            reinterpret_cast<__half2*>(value_tile)[pair_index] = value_pair;
        }
        __syncthreads();

        for (int tile_token = 0; tile_token < tile_length; ++tile_token) {
            const int pair_index = tile_token * kQwen2GroupTilePairsPerToken + lane * 2;
            const __half2 key_pair0 = reinterpret_cast<const __half2*>(key_tile)[pair_index];
            const __half2 key_pair1 = reinterpret_cast<const __half2*>(key_tile)[pair_index + 1];
            float partial = query0 * __low2float(key_pair0);
            partial += query1 * __high2float(key_pair0);
            partial += query2 * __low2float(key_pair1);
            partial += query3 * __high2float(key_pair1);
            for (int offset = 16; offset > 0; offset >>= 1) {
                partial += __shfl_down_sync(0xffffffffU, partial, offset);
            }
            const float score = __shfl_sync(0xffffffffU, partial, 0) * scale;

            const float next_maximum = fmaxf(running_maximum, score);
            const float previous_scale = running_sum > 0.0F
                ? expf(running_maximum - next_maximum)
                : 0.0F;
            const float token_scale = expf(score - next_maximum);
            running_sum = running_sum * previous_scale + token_scale;

            const __half2 value_pair0 = reinterpret_cast<const __half2*>(value_tile)[pair_index];
            const __half2 value_pair1 = reinterpret_cast<const __half2*>(value_tile)[pair_index + 1];
            accumulated0 = accumulated0 * previous_scale + token_scale * __half2float(__low2half(value_pair0));
            accumulated1 = accumulated1 * previous_scale + token_scale * __half2float(__high2half(value_pair0));
            accumulated2 = accumulated2 * previous_scale + token_scale * __half2float(__low2half(value_pair1));
            accumulated3 = accumulated3 * previous_scale + token_scale * __half2float(__high2half(value_pair1));
            running_maximum = next_maximum;
        }
        __syncthreads();
    }

    const float inverse_sum = 1.0F / running_sum;
    output_head_data[query_dim] = __float2half(accumulated0 * inverse_sum);
    output_head_data[query_dim + 1] = __float2half(accumulated1 * inverse_sum);
    output_head_data[query_dim + 2] = __float2half(accumulated2 * inverse_sum);
    output_head_data[query_dim + 3] = __float2half(accumulated3 * inverse_sum);
}

inline cudaError_t launch_qwen2_gqa_group_tiled(
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
    qwen2_gqa_group_tiled_kernel<<<
        static_cast<unsigned int>(batch_size * kv_heads),
        kQwen2GroupTileThreads,
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
