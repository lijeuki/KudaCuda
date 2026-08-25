#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace kudacuda {

constexpr int kDecodeNormThreads = 256;

template <typename T>
__device__ __forceinline__ float decode_norm_to_float(T value);

template <>
__device__ __forceinline__ float decode_norm_to_float<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ __forceinline__ float decode_norm_to_float<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T decode_norm_from_float(float value);

template <>
__device__ __forceinline__ __half decode_norm_from_float<__half>(float value) {
    return __float2half(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 decode_norm_from_float<__nv_bfloat16>(float value) {
    return __float2bfloat16(value);
}

template <typename T>
__global__ void rmsnorm_decode_kernel(
    const T* input,
    const T* weight,
    T* output,
    int64_t rows,
    int64_t hidden,
    float epsilon
) {
    const int64_t row = static_cast<int64_t>(blockIdx.x);
    if (row >= rows || hidden <= 0) {
        return;
    }

    __shared__ float reduction[kDecodeNormThreads];
    const int tid = threadIdx.x;
    const T* input_row = input + row * hidden;
    T* output_row = output + row * hidden;

    float sum_squares = 0.0F;
    for (int64_t dimension = tid; dimension < hidden; dimension += blockDim.x) {
        const float value = decode_norm_to_float(input_row[dimension]);
        sum_squares += value * value;
    }
    reduction[tid] = sum_squares;
    __syncthreads();
    for (int stride = kDecodeNormThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduction[tid] += reduction[tid + stride];
        }
        __syncthreads();
    }
    const float inverse_rms = rsqrtf(reduction[0] / static_cast<float>(hidden) + epsilon);
    for (int64_t dimension = tid; dimension < hidden; dimension += blockDim.x) {
        output_row[dimension] = decode_norm_from_float<T>(
            decode_norm_to_float(input_row[dimension]) * inverse_rms *
            decode_norm_to_float(weight[dimension])
        );
    }
}

template <typename T>
inline cudaError_t launch_rmsnorm_decode(
    const T* input,
    const T* weight,
    T* output,
    int64_t rows,
    int64_t hidden,
    float epsilon,
    cudaStream_t stream
) {
    rmsnorm_decode_kernel<T><<<
        static_cast<unsigned int>(rows),
        kDecodeNormThreads,
        0,
        stream
    >>>(input, weight, output, rows, hidden, epsilon);
    return cudaGetLastError();
}

}  // namespace kudacuda
