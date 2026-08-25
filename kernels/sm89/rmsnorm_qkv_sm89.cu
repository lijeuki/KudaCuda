#include "../common/rmsnorm_qkv_reference.cuh"

extern "C" cudaError_t kudacuda_launch_fused_rmsnorm_qkv(
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
    return kudacuda::launch_fused_rmsnorm_qkv(
        input,
        norm_weight,
        qkv_weight,
        bias,
        output,
        rows,
        hidden,
        epsilon,
        stream
    );
}
