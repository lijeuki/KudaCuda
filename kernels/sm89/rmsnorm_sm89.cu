#include "../common/rmsnorm.cuh"

namespace kudacuda {

extern "C" cudaError_t kudacuda_launch_rmsnorm(
    const float* input,
    const float* weight,
    float* output,
    int64_t rows,
    int64_t hidden,
    float epsilon,
    cudaStream_t stream
) {
    return kudacuda::launch_rmsnorm(input, weight, output, rows, hidden, epsilon, stream);
}

}  // namespace kudacuda
