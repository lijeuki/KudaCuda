#include "../common/nf4_swiglu.cuh"

extern "C" cudaError_t kudacuda_launch_nf4_swiglu(
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
    return kudacuda::launch_nf4_swiglu(
        input, norm_weight, gate_weight, gate_absmax, up_weight, up_absmax,
        output, rows, hidden, intermediate, blocksize, epsilon, stream
    );
}
