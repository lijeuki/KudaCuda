#include "../common/nf4_tiled_gemv.cuh"

extern "C" cudaError_t kudacuda_launch_nf4_tiled_rmsnorm_qkv(
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
    return kudacuda::launch_nf4_tiled_rmsnorm_qkv(
        input, norm_weight, q_weight, q_absmax, q_bias, k_weight, k_absmax, k_bias,
        v_weight, v_absmax, v_bias, normalized, q_output, k_output, v_output,
        rows, hidden, q_out, kv_out, blocksize, epsilon, stream
    );
}
