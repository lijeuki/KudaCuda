#pragma once

#include "nf4_device.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace kudacuda {

constexpr int kNf4SwiGluThreads = 64;
constexpr int kNf4SwiGluRowsPerBlock = 4;

__device__ __forceinline__ float nf4_swiglu_dot_word(
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

__device__ __forceinline__ float nf4_swiglu_dot_uint4(
    const uint4 packed,
    int64_t output_index,
    int64_t hidden,
    const float* normalized,
    const float* absmax,
    int64_t blocksize
) {
    const int64_t weight_base = output_index * hidden;
    const int64_t logical_base = static_cast<int64_t>(threadIdx.x) * 32;
    float result = 0.0F;
    result += nf4_swiglu_dot_word(packed.x, logical_base, weight_base, normalized, absmax, blocksize);
    result += nf4_swiglu_dot_word(packed.y, logical_base + 8, weight_base, normalized, absmax, blocksize);
    result += nf4_swiglu_dot_word(packed.z, logical_base + 16, weight_base, normalized, absmax, blocksize);
    result += nf4_swiglu_dot_word(packed.w, logical_base + 24, weight_base, normalized, absmax, blocksize);
    return result;
}

__device__ __forceinline__ float nf4_swiglu_warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ __forceinline__ float nf4_silu(float value) {
    return value / (1.0F + expf(-value));
}

__global__ void nf4_swiglu_kernel(
    const __half* input,
    const __half* norm_weight,
    const uint8_t* gate_weight,
    const float* gate_absmax,
    const uint8_t* up_weight,
    const float* up_absmax,
    __half* output,
    int64_t rows,
    int64_t hidden,
    int64_t intermediate,
    int64_t blocksize,
    float epsilon
) {
    const int64_t output_group = static_cast<int64_t>(blockIdx.x);
    const int64_t row = static_cast<int64_t>(blockIdx.y);
    const int tid = threadIdx.x;
    const int64_t first_output = output_group * kNf4SwiGluRowsPerBlock;
    if (row >= rows || first_output >= intermediate || hidden != 2048 || intermediate != 11008 || blocksize != 64) {
        return;
    }

    __shared__ float normalized_tile[2048];
    __shared__ float rms_warp_sums[kNf4SwiGluThreads / 32];
    __shared__ float gate_warp_sums[kNf4SwiGluRowsPerBlock][kNf4SwiGluThreads / 32];
    __shared__ float up_warp_sums[kNf4SwiGluRowsPerBlock][kNf4SwiGluThreads / 32];

    const __half* input_row = input + row * hidden;
    float sum_squares = 0.0F;
    for (int64_t dimension = tid; dimension < hidden; dimension += blockDim.x) {
        const float value = __half2float(input_row[dimension]);
        sum_squares += value * value;
    }
    sum_squares = nf4_swiglu_warp_sum(sum_squares);
    if ((tid & 31) == 0) {
        rms_warp_sums[tid >> 5] = sum_squares;
    }
    __syncthreads();
    float total = tid < (kNf4SwiGluThreads / 32) ? rms_warp_sums[tid] : 0.0F;
    if (tid < 32) {
        total = nf4_swiglu_warp_sum(total);
    }
    if (tid == 0) {
        rms_warp_sums[0] = rsqrtf(total / static_cast<float>(hidden) + epsilon);
    }
    __syncthreads();
    const float inverse_rms = rms_warp_sums[0];
    for (int64_t dimension = tid; dimension < hidden; dimension += blockDim.x) {
        normalized_tile[dimension] =
            __half2float(input_row[dimension]) * inverse_rms * __half2float(norm_weight[dimension]);
    }
    __syncthreads();

    float gate_sums[kNf4SwiGluRowsPerBlock] = {0.0F, 0.0F, 0.0F, 0.0F};
    float up_sums[kNf4SwiGluRowsPerBlock] = {0.0F, 0.0F, 0.0F, 0.0F};
    const int64_t packed_row_bytes = hidden / 2;
    const int warp = tid >> 5;
    for (int output_offset = 0; output_offset < kNf4SwiGluRowsPerBlock; ++output_offset) {
        const int64_t output_index = first_output + output_offset;
        if (output_index < intermediate) {
            const uint4 gate_packed = reinterpret_cast<const uint4*>(
                gate_weight + output_index * packed_row_bytes
            )[tid];
            const uint4 up_packed = reinterpret_cast<const uint4*>(
                up_weight + output_index * packed_row_bytes
            )[tid];
            gate_sums[output_offset] = nf4_swiglu_dot_uint4(
                gate_packed,
                output_index,
                hidden,
                normalized_tile,
                gate_absmax,
                blocksize
            );
            up_sums[output_offset] = nf4_swiglu_dot_uint4(
                up_packed,
                output_index,
                hidden,
                normalized_tile,
                up_absmax,
                blocksize
            );
        }
        gate_sums[output_offset] = nf4_swiglu_warp_sum(gate_sums[output_offset]);
        up_sums[output_offset] = nf4_swiglu_warp_sum(up_sums[output_offset]);
        if ((tid & 31) == 0) {
            gate_warp_sums[output_offset][warp] = gate_sums[output_offset];
            up_warp_sums[output_offset][warp] = up_sums[output_offset];
        }
    }
    __syncthreads();
    if (tid < kNf4SwiGluRowsPerBlock) {
        const int output_offset = tid;
        const int64_t output_index = first_output + output_offset;
        if (output_index < intermediate) {
            const float gate_sum = gate_warp_sums[output_offset][0] + gate_warp_sums[output_offset][1];
            const float up_sum = up_warp_sums[output_offset][0] + up_warp_sums[output_offset][1];
            output[row * intermediate + output_index] = __float2half(nf4_silu(gate_sum) * up_sum);
        }
    }
}

inline cudaError_t launch_nf4_swiglu(
    const __half* input,
    const __half* norm_weight,
    const uint8_t* gate_weight,
    const float* gate_absmax,
    const uint8_t* up_weight,
    const float* up_absmax,
    __half* output,
    int64_t rows,
    int64_t hidden,
    int64_t intermediate,
    int64_t blocksize,
    float epsilon,
    cudaStream_t stream
) {
    if (rows != 1 || hidden != 2048 || intermediate != 11008 || blocksize != 64 || epsilon <= 0.0F) {
        return cudaErrorInvalidValue;
    }
    nf4_swiglu_kernel<<<
        dim3(
            static_cast<unsigned int>((intermediate + kNf4SwiGluRowsPerBlock - 1) / kNf4SwiGluRowsPerBlock),
            static_cast<unsigned int>(rows)
        ),
        kNf4SwiGluThreads,
        0,
        stream
    >>>(
        input,
        norm_weight,
        gate_weight,
        gate_absmax,
        up_weight,
        up_absmax,
        output,
        rows,
        hidden,
        intermediate,
        blocksize,
        epsilon
    );
    return cudaGetLastError();
}

}  // namespace kudacuda
