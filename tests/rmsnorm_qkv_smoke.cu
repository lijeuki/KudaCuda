#include "kudacuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool check_cuda(cudaError_t status, const char* expression) {
    if (status == cudaSuccess) {
        return true;
    }
    std::cerr << expression << " failed: " << cudaGetErrorString(status) << '\n';
    return false;
}

float to_float(const __half& value) {
    return __half2float(value);
}

}  // namespace

int main(int argc, char** argv) {
    const int device_id = argc > 1 ? std::stoi(argv[1]) : 0;
    constexpr int rows = 2;
    constexpr int hidden = 16;
    constexpr int qkv_hidden = hidden * 3;
    constexpr float epsilon = 1.0e-5F;
    constexpr float tolerance = 5.0e-3F;

    if (!check_cuda(cudaSetDevice(device_id), "cudaSetDevice")) {
        return 1;
    }

    std::vector<__half> host_input(rows * hidden);
    std::vector<__half> host_norm_weight(hidden);
    std::vector<__half> host_qkv_weight(qkv_hidden * hidden);
    std::vector<__half> host_bias(qkv_hidden);
    std::vector<__half> host_output(rows * qkv_hidden);
    std::vector<__half> host_expected(rows * qkv_hidden);

    for (int index = 0; index < rows * hidden; ++index) {
        host_input[index] = __float2half(std::sin(static_cast<float>(index) * 0.13F));
    }
    for (int index = 0; index < hidden; ++index) {
        host_norm_weight[index] = __float2half(0.75F + static_cast<float>(index) * 0.01F);
    }
    for (int index = 0; index < qkv_hidden * hidden; ++index) {
        host_qkv_weight[index] = __float2half(std::cos(static_cast<float>(index) * 0.017F) * 0.2F);
    }
    for (int index = 0; index < qkv_hidden; ++index) {
        host_bias[index] = __float2half(std::sin(static_cast<float>(index) * 0.07F) * 0.05F);
    }

    for (int row = 0; row < rows; ++row) {
        float sum_squares = 0.0F;
        for (int dimension = 0; dimension < hidden; ++dimension) {
            const float value = to_float(host_input[row * hidden + dimension]);
            sum_squares += value * value;
        }
        const float inverse_rms = 1.0F / std::sqrt(sum_squares / hidden + epsilon);
        std::vector<float> normalized(hidden);
        for (int dimension = 0; dimension < hidden; ++dimension) {
            normalized[dimension] = to_float(host_input[row * hidden + dimension]) *
                inverse_rms * to_float(host_norm_weight[dimension]);
        }
        for (int output_index = 0; output_index < qkv_hidden; ++output_index) {
            float value = to_float(host_bias[output_index]);
            for (int dimension = 0; dimension < hidden; ++dimension) {
                value += normalized[dimension] *
                    to_float(host_qkv_weight[output_index * hidden + dimension]);
            }
            host_expected[row * qkv_hidden + output_index] = __float2half(value);
        }
    }

    __half* device_input = nullptr;
    __half* device_norm_weight = nullptr;
    __half* device_qkv_weight = nullptr;
    __half* device_bias = nullptr;
    __half* device_output = nullptr;
    cudaStream_t stream = nullptr;
    void* handle = nullptr;
    int exit_code = 1;

    if (!check_cuda(cudaMalloc(&device_input, host_input.size() * sizeof(__half)), "cudaMalloc(input)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_norm_weight, host_norm_weight.size() * sizeof(__half)), "cudaMalloc(norm_weight)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_qkv_weight, host_qkv_weight.size() * sizeof(__half)), "cudaMalloc(qkv_weight)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_bias, host_bias.size() * sizeof(__half)), "cudaMalloc(bias)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_output, host_output.size() * sizeof(__half)), "cudaMalloc(output)")) goto cleanup;
    if (!check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(device_input, host_input.data(), host_input.size() * sizeof(__half), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(input)")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(device_norm_weight, host_norm_weight.data(), host_norm_weight.size() * sizeof(__half), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(norm_weight)")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(device_qkv_weight, host_qkv_weight.data(), host_qkv_weight.size() * sizeof(__half), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(qkv_weight)")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(device_bias, host_bias.data(), host_bias.size() * sizeof(__half), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(bias)")) goto cleanup;

    handle = kudacuda_create(device_id);
    if (handle == nullptr || kudacuda_freeze(handle) != 0) {
        std::cerr << "KudaCuda initialization failed\n";
        goto cleanup;
    }
    if (kudacuda_fused_rmsnorm_qkv(
            handle,
            reinterpret_cast<uint64_t>(device_input),
            reinterpret_cast<uint64_t>(device_norm_weight),
            reinterpret_cast<uint64_t>(device_qkv_weight),
            reinterpret_cast<uint64_t>(device_bias),
            reinterpret_cast<uint64_t>(device_output),
            rows,
            hidden,
            epsilon,
            reinterpret_cast<uint64_t>(stream)) != 0) {
        std::cerr << "kudacuda_fused_rmsnorm_qkv failed: " << kudacuda_last_error(handle) << '\n';
        goto cleanup;
    }
    if (!check_cuda(cudaMemcpyAsync(host_output.data(), device_output, host_output.size() * sizeof(__half), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(output)")) goto cleanup;
    if (!check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) goto cleanup;

    for (std::size_t index = 0; index < host_output.size(); ++index) {
        if (std::fabs(to_float(host_output[index]) - to_float(host_expected[index])) > tolerance) {
            std::cerr << "Fused RMSNorm-QKV mismatch at " << index << '\n';
            goto cleanup;
        }
    }

    std::cout << "KudaCuda fused RMSNorm-QKV output matched the independent FP32 reference\n";
    exit_code = 0;

cleanup:
    if (handle != nullptr) kudacuda_destroy(handle);
    if (stream != nullptr) cudaStreamDestroy(stream);
    cudaFree(device_output);
    cudaFree(device_bias);
    cudaFree(device_qkv_weight);
    cudaFree(device_norm_weight);
    cudaFree(device_input);
    return exit_code;
}
