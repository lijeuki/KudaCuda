#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace kudacuda {

constexpr int kRmsNormThreads = 256;

__global__ void rmsnorm_kernel(
    const float* input,
    const float* weight,
    float* output,
    int64_t rows,
    int64_t hidden,
    float epsilon
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    const int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    extern __shared__ float partial[];
    const float* row_input = input + row * hidden;
    float* row_output = output + row * hidden;
    float sum_squares = 0.0F;
    for (int64_t index = tid; index < hidden; index += blockDim.x) {
        const float value = row_input[index];
        sum_squares += value * value;
    }
    partial[tid] = sum_squares;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }

    const float inverse_rms = rsqrtf(partial[0] / static_cast<float>(hidden) + epsilon);
    for (int64_t index = tid; index < hidden; index += blockDim.x) {
        row_output[index] = row_input[index] * inverse_rms * weight[index];
    }
}

inline cudaError_t launch_rmsnorm(
    const float* input,
    const float* weight,
    float* output,
    int64_t rows,
    int64_t hidden,
    float epsilon,
    cudaStream_t stream
) {
    rmsnorm_kernel<<<
        static_cast<unsigned int>(rows),
        kRmsNormThreads,
        kRmsNormThreads * sizeof(float),
        stream
    >>>(input, weight, output, rows, hidden, epsilon);
    return cudaGetLastError();
}

}  // namespace kudacuda
