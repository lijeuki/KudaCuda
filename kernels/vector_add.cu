#include <cuda_runtime.h>

#include <cstdint>

namespace kudacuda {

namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void vector_add_kernel(
    const float* a,
    const float* b,
    float* output,
    int64_t count
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = a[index] + b[index];
    }
}

}  // namespace

cudaError_t launch_vector_add(
    const float* a,
    const float* b,
    float* output,
    int64_t count,
    cudaStream_t stream
) {
    const auto blocks = static_cast<unsigned int>(
        (count + kThreadsPerBlock - 1) / kThreadsPerBlock
    );
    vector_add_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(a, b, output, count);
    return cudaGetLastError();
}

}  // namespace kudacuda
