#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace kudacuda {

// bitsandbytes' canonical NF4 codebook (0.50.x).
static __device__ __constant__ float kNf4Codebook[16] = {
    -1.0F,
    -0.6961928009986877F,
    -0.5250730514526367F,
    -0.39491748809814453F,
    -0.28444138169288635F,
    -0.18477343022823334F,
    -0.09105003625154495F,
    0.0F,
    0.07958029955625534F,
    0.16093020141124725F,
    0.24611230194568634F,
    0.33791524171829224F,
    0.44070982933044434F,
    0.5626170039176941F,
    0.7229568362236023F,
    1.0F,
};

__device__ __forceinline__ float nf4_value(
    const uint8_t* packed,
    const float* absmax,
    int64_t logical_index,
    int64_t blocksize
) {
    const uint8_t byte = packed[logical_index >> 1];
    // bitsandbytes stores logical even elements in the high nibble and odd
    // elements in the low nibble: ((q[::2] << 4) | q[1::2]).
    const int code = (logical_index & 1) == 0 ? (byte >> 4) : (byte & 0x0f);
    return kNf4Codebook[code] * absmax[logical_index / blocksize];
}

}  // namespace kudacuda
