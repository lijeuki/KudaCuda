#pragma once

#include "nf4_device.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace kudacuda {

constexpr int kNf4TiledThreads = 64;
constexpr int kNf4ValuesPerThread = 32;

__device__ __forceinline__ float nf4_dot_word(
    uint32_t packed_word,
    int64_t logical_base,
    int64_t weight_base,
    const float* normalized,
    const float* absmax,
    int64_t blocksize
) {
    float result = 0.0F;
#pragma unroll
    for (int byte_index = 0; byte_index < 4; ++byte_index) {
        const uint8_t packed_byte = static_cast<uint8_t>(packed_word >> (byte_index * 8));
        const int64_t even_index = logical_base + byte_index * 2;
        const int64_t block = (weight_base + even_index) / blocksize;
        const float scale = absmax[block];
        result += normalized[even_index] * kNf4Codebook[packed_byte >> 4] * scale;
        result += normalized[even_index + 1] * kNf4Codebook[packed_byte & 0x0f] * scale;
    }
    return result;
}

__device__ __forceinline__ float nf4_dot_uint4(
    const uint4 packed,
    int64_t output_index,
    int64_t hidden,
    const float* normalized,
    const float* absmax,
    int64_t blocksize
) {
    const int64_t weight_base = output_index * hidden;
    const int64_t logical_base = static_cast<int64_t>(threadIdx.x) * kNf4ValuesPerThread;
    float result = 0.0F;
    result += nf4_dot_word(packed.x, logical_base, weight_base, normalized, absmax, blocksize);
    result += nf4_dot_word(packed.y, logical_base + 8, weight_base, normalized, absmax, blocksize);
    result += nf4_dot_word(packed.z, logical_base + 16, weight_base, normalized, absmax, blocksize);
    result += nf4_dot_word(packed.w, logical_base + 24, weight_base, normalized, absmax, blocksize);
    return result;
}

__device__ __forceinline__ float nf4_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void nf4_tiled_rmsnorm_kernel(
    const __half* input,
    const __half* norm_weight,
    float* normalized,
    int64_t rows,
    int64_t hidden,
    float epsilon
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    __shared__ float warp_sums[kNf4TiledThreads / 32];
    const __half* input_row = input + row * hidden;
    float sum_squares = 0.0F;
    for (int64_t dimension = tid; dimension < hidden; dimension += blockDim.x) {
        const float value = __half2float(input_row[dimension]);
        sum_squares += value * value;
    }
    sum_squares = nf4_warp_sum(sum_squares);
    if ((tid & 31) == 0) {
        warp_sums[tid >> 5] = sum_squares;
    }
    __syncthreads();
    float total = tid < (kNf4TiledThreads / 32) ? warp_sums[tid] : 0.0F;
    if (tid < 32) {
        total = nf4_warp_sum(total);
    }
    if (tid == 0) {
        warp_sums[0] = rsqrtf(total / static_cast<float>(hidden) + epsilon);
    }
    __syncthreads();
    const float inverse_rms = warp_sums[0];
    for (int64_t dimension = tid; dimension < hidden; dimension += blockDim.x) {
        normalized[row * hidden + dimension] =
            __half2float(input_row[dimension]) * inverse_rms * __half2float(norm_weight[dimension]);
    }
}

__global__ void nf4_tiled_qkv_gemv_kernel(
    const float* normalized,
    const uint8_t* q_weight,
    const float* q_absmax,
    const __half* q_bias,
    const uint8_t* k_weight,
    const float* k_absmax,
    const __half* k_bias,
    const uint8_t* v_weight,
    const float* v_absmax,
    const __half* v_bias,
    __half* q_output,
    __half* k_output,
    __half* v_output,
    int64_t rows,
    int64_t hidden,
    int64_t q_out,
    int64_t kv_out,
    int64_t blocksize
) {
    const int64_t output_block = static_cast<int64_t>(blockIdx.x);
    const int64_t row = static_cast<int64_t>(blockIdx.y);
    const int tid = threadIdx.x;
    if (row >= rows || output_block >= q_out + kv_out || hidden != 2048 || blocksize != 64) {
        return;
    }

    const float* normalized_row = normalized + row * hidden;
    const bool is_kv = output_block >= q_out;
    const int64_t output_index = is_kv ? output_block - q_out : output_block;
    float key_sum = 0.0F;
    float value_sum = 0.0F;
    if (!is_kv) {
        const uint4 packed = reinterpret_cast<const uint4*>(q_weight + output_index * (hidden / 2))[tid];
        key_sum = nf4_dot_uint4(packed, output_index, hidden, normalized_row, q_absmax, blocksize);
    } else {
        const uint4 packed_key = reinterpret_cast<const uint4*>(k_weight + output_index * (hidden / 2))[tid];
        const uint4 packed_value = reinterpret_cast<const uint4*>(v_weight + output_index * (hidden / 2))[tid];
        key_sum = nf4_dot_uint4(packed_key, output_index, hidden, normalized_row, k_absmax, blocksize);
        value_sum = nf4_dot_uint4(packed_value, output_index, hidden, normalized_row, v_absmax, blocksize);
    }

    __shared__ float key_warp_sums[kNf4TiledThreads / 32];
    __shared__ float value_warp_sums[kNf4TiledThreads / 32];
    key_sum = nf4_warp_sum(key_sum);
    value_sum = nf4_warp_sum(value_sum);
    if ((tid & 31) == 0) {
        key_warp_sums[tid >> 5] = key_sum;
        value_warp_sums[tid >> 5] = value_sum;
    }
    __syncthreads();
    if (tid < 32) {
        key_sum = nf4_warp_sum(tid < (kNf4TiledThreads / 32) ? key_warp_sums[tid] : 0.0F);
        value_sum = nf4_warp_sum(tid < (kNf4TiledThreads / 32) ? value_warp_sums[tid] : 0.0F);
    }
    if (tid == 0) {
        if (!is_kv) {
            q_output[row * q_out + output_index] =
                __float2half(key_sum + __half2float(q_bias[output_index]));
        } else {
            k_output[row * kv_out + output_index] =
                __float2half(key_sum + __half2float(k_bias[output_index]));
            v_output[row * kv_out + output_index] =
                __float2half(value_sum + __half2float(v_bias[output_index]));
        }
    }
}

inline cudaError_t launch_nf4_tiled_rmsnorm_qkv(
    const __half* input,
    const __half* norm_weight,
    const uint8_t* q_weight,
    const float* q_absmax,
    const __half* q_bias,
    const uint8_t* k_weight,
    const float* k_absmax,
    const __half* k_bias,
    const uint8_t* v_weight,
    const float* v_absmax,
    const __half* v_bias,
    float* normalized,
    __half* q_output,
    __half* k_output,
    __half* v_output,
    int64_t rows,
    int64_t hidden,
    int64_t q_out,
    int64_t kv_out,
    int64_t blocksize,
    float epsilon,
    cudaStream_t stream
) {
    if (rows != 1 || hidden != 2048 || q_out != 2048 || kv_out != 256 || blocksize != 64 || epsilon <= 0.0F) {
        return cudaErrorInvalidValue;
    }
    nf4_tiled_rmsnorm_kernel<<<static_cast<unsigned int>(rows), kNf4TiledThreads, 0, stream>>>(
        input, norm_weight, normalized, rows, hidden, epsilon
    );
    auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        return status;
    }
    nf4_tiled_qkv_gemv_kernel<<<dim3(static_cast<unsigned int>(q_out + kv_out), static_cast<unsigned int>(rows)), kNf4TiledThreads, 0, stream>>>(
        normalized,
        q_weight,
        q_absmax,
        q_bias,
        k_weight,
        k_absmax,
        k_bias,
        v_weight,
        v_absmax,
        v_bias,
        q_output,
        k_output,
        v_output,
        rows,
        hidden,
        q_out,
        kv_out,
        blocksize
    );
    return cudaGetLastError();
}

}  // namespace kudacuda
