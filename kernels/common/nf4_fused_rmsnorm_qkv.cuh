#pragma once

#include "nf4_device.cuh"
#include "rmsnorm_decode.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace kudacuda {

constexpr int kNf4FusedQkvThreads = 256;

__global__ void nf4_fused_rmsnorm_qkv_kernel(
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
    __half* q_output,
    __half* k_output,
    __half* v_output,
    int64_t rows,
    int64_t hidden,
    int64_t q_out,
    int64_t kv_out,
    int64_t blocksize,
    float epsilon
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows || hidden <= 0 || q_out <= 0 || kv_out <= 0 || blocksize != 64) {
        return;
    }

    extern __shared__ float shared[];
    float* normalized = shared;
    float* reduction = shared + hidden;
    const int tid = threadIdx.x;
    const __half* input_row = input + row * hidden;

    float sum_squares = 0.0F;
    for (int64_t dimension = tid; dimension < hidden; dimension += blockDim.x) {
        const float value = __half2float(input_row[dimension]);
        sum_squares += value * value;
    }
    reduction[tid] = sum_squares;
    __syncthreads();
    for (int stride = kNf4FusedQkvThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduction[tid] += reduction[tid + stride];
        }
        __syncthreads();
    }
    const float inverse_rms = rsqrtf(reduction[0] / static_cast<float>(hidden) + epsilon);
    for (int64_t dimension = tid; dimension < hidden; dimension += blockDim.x) {
        normalized[dimension] =
            __half2float(input_row[dimension]) * inverse_rms *
            __half2float(norm_weight[dimension]);
    }
    __syncthreads();

    for (int64_t output_index = tid; output_index < q_out; output_index += blockDim.x) {
        float value = 0.0F;
        for (int64_t dimension = 0; dimension < hidden; ++dimension) {
            value += normalized[dimension] * nf4_value(
                q_weight,
                q_absmax,
                output_index * hidden + dimension,
                blocksize
            );
        }
        q_output[row * q_out + output_index] = __float2half(value + __half2float(q_bias[output_index]));
    }
    for (int64_t output_index = tid; output_index < kv_out; output_index += blockDim.x) {
        float key_value = 0.0F;
        float value_value = 0.0F;
        for (int64_t dimension = 0; dimension < hidden; ++dimension) {
            const int64_t weight_index = output_index * hidden + dimension;
            key_value += normalized[dimension] * nf4_value(
                k_weight,
                k_absmax,
                weight_index,
                blocksize
            );
            value_value += normalized[dimension] * nf4_value(
                v_weight,
                v_absmax,
                weight_index,
                blocksize
            );
        }
        k_output[row * kv_out + output_index] = __float2half(key_value + __half2float(k_bias[output_index]));
        v_output[row * kv_out + output_index] = __float2half(value_value + __half2float(v_bias[output_index]));
    }
}

inline cudaError_t launch_nf4_fused_rmsnorm_qkv(
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
    const std::size_t shared_bytes = static_cast<std::size_t>(hidden + kNf4FusedQkvThreads) * sizeof(float);
    nf4_fused_rmsnorm_qkv_kernel<<<
        static_cast<unsigned int>(rows),
        kNf4FusedQkvThreads,
        shared_bytes,
        stream
    >>>(
        input,
        norm_weight,
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
        blocksize,
        epsilon
    );
    return cudaGetLastError();
}

}  // namespace kudacuda
