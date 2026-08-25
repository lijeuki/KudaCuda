#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace kudacuda {

constexpr int kRmsNormQkvThreads = 256;

__global__ void fused_rmsnorm_qkv_kernel(
    const __half* input,
    const __half* norm_weight,
    const __half* qkv_weight,
    const __half* bias,
    __half* output,
    int64_t rows,
    int64_t hidden,
    float epsilon
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows || hidden <= 0) {
        return;
    }

    extern __shared__ float shared[];
    float* normalized = shared;
    float* reduction = shared + hidden;
    const int tid = threadIdx.x;
    const __half* input_row = input + row * hidden;
    __half* output_row = output + row * hidden * 3;

    float sum_squares = 0.0F;
    for (int64_t dimension = tid; dimension < hidden; dimension += blockDim.x) {
        const float value = __half2float(input_row[dimension]);
        sum_squares += value * value;
    }
    reduction[tid] = sum_squares;
    __syncthreads();
    for (int stride = kRmsNormQkvThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduction[tid] += reduction[tid + stride];
        }
        __syncthreads();
    }
    const float inverse_rms = rsqrtf(reduction[0] / static_cast<float>(hidden) + epsilon);

    for (int64_t dimension = tid; dimension < hidden; dimension += blockDim.x) {
        normalized[dimension] =
            __half2float(input_row[dimension]) * inverse_rms * __half2float(norm_weight[dimension]);
    }
    __syncthreads();

    for (int64_t output_index = tid; output_index < hidden * 3; output_index += blockDim.x) {
        float value = 0.0F;
        const __half* weight_row = qkv_weight + output_index * hidden;
        for (int64_t dimension = 0; dimension < hidden; ++dimension) {
            value += normalized[dimension] * __half2float(weight_row[dimension]);
        }
        if (bias != nullptr) {
            value += __half2float(bias[output_index]);
        }
        output_row[output_index] = __float2half(value);
    }
}

inline cudaError_t launch_fused_rmsnorm_qkv(
    const __half* input,
    const __half* norm_weight,
    const __half* qkv_weight,
    const __half* bias,
    __half* output,
    int64_t rows,
    int64_t hidden,
    float epsilon,
    cudaStream_t stream
) {
    const std::size_t shared_bytes =
        static_cast<std::size_t>(hidden + kRmsNormQkvThreads) * sizeof(float);
    fused_rmsnorm_qkv_kernel<<<
        static_cast<unsigned int>(rows),
        kRmsNormQkvThreads,
        shared_bytes,
        stream
    >>>(
        input,
        norm_weight,
        qkv_weight,
        bias,
        output,
        rows,
        hidden,
        epsilon
    );
    return cudaGetLastError();
}

}  // namespace kudacuda
