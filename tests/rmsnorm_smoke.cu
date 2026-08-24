#include "kudacuda.h"

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
    std::cerr << expression << " failed (code " << static_cast<int>(status)
              << ", " << cudaGetErrorName(status) << "): "
              << cudaGetErrorString(status) << '\n';
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    const int device_id = argc > 1 ? std::stoi(argv[1]) : 0;
    constexpr int rows = 7;
    constexpr int hidden = 513;
    constexpr float epsilon = 1.0e-5F;
    constexpr float tolerance = 5.0e-5F;

    if (!check_cuda(cudaSetDevice(device_id), "cudaSetDevice")) {
        return 1;
    }

    std::vector<float> host_input(rows * hidden);
    std::vector<float> host_weight(hidden);
    std::vector<float> host_output(rows * hidden, 0.0F);
    std::vector<float> host_expected(rows * hidden, 0.0F);
    for (int index = 0; index < rows * hidden; ++index) {
        host_input[index] = std::sin(static_cast<float>(index) * 0.013F) * 0.75F;
    }
    for (int index = 0; index < hidden; ++index) {
        host_weight[index] = 0.5F + static_cast<float>(index % 17) * 0.03125F;
    }
    for (int row = 0; row < rows; ++row) {
        double sum_squares = 0.0;
        for (int column = 0; column < hidden; ++column) {
            const double value = host_input[row * hidden + column];
            sum_squares += value * value;
        }
        const float inverse_rms = static_cast<float>(
            1.0 / std::sqrt(sum_squares / hidden + epsilon)
        );
        for (int column = 0; column < hidden; ++column) {
            host_expected[row * hidden + column] =
                host_input[row * hidden + column] * inverse_rms * host_weight[column];
        }
    }

    float* device_input = nullptr;
    float* device_weight = nullptr;
    float* device_output = nullptr;
    cudaStream_t stream = nullptr;
    void* handle = nullptr;
    int status = 0;
    int exit_code = 1;
    const std::size_t input_bytes = host_input.size() * sizeof(float);
    const std::size_t weight_bytes = host_weight.size() * sizeof(float);

    if (!check_cuda(cudaMalloc(&device_input, input_bytes), "cudaMalloc(input)")) {
        goto cleanup;
    }
    if (!check_cuda(cudaMalloc(&device_weight, weight_bytes), "cudaMalloc(weight)")) {
        goto cleanup;
    }
    if (!check_cuda(cudaMalloc(&device_output, input_bytes), "cudaMalloc(output)")) {
        goto cleanup;
    }
    if (!check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags")) {
        goto cleanup;
    }
    if (!check_cuda(cudaMemcpyAsync(device_input, host_input.data(), input_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(input)")) {
        goto cleanup;
    }
    if (!check_cuda(cudaMemcpyAsync(device_weight, host_weight.data(), weight_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(weight)")) {
        goto cleanup;
    }

    handle = kudacuda_create(device_id);
    if (handle == nullptr || kudacuda_freeze(handle) != 0) {
        std::cerr << "KudaCuda initialization failed\n";
        goto cleanup;
    }
    status = kudacuda_rmsnorm(
        handle,
        reinterpret_cast<uint64_t>(device_input),
        reinterpret_cast<uint64_t>(device_weight),
        reinterpret_cast<uint64_t>(device_output),
        rows,
        hidden,
        epsilon,
        reinterpret_cast<uint64_t>(stream)
    );
    if (status != 0) {
        std::cerr << "kudacuda_rmsnorm failed: " << kudacuda_last_error(handle) << '\n';
        goto cleanup;
    }
    if (!check_cuda(cudaMemcpyAsync(host_output.data(), device_output, input_bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(output)")) {
        goto cleanup;
    }
    if (!check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) {
        goto cleanup;
    }

    for (std::size_t index = 0; index < host_output.size(); ++index) {
        if (std::fabs(host_output[index] - host_expected[index]) > tolerance) {
            std::cerr << "RMSNorm mismatch at " << index << ": expected "
                      << host_expected[index] << ", got " << host_output[index] << '\n';
            goto cleanup;
        }
    }

    std::cout << "KudaCuda RMSNorm output matched the independent FP64 reference\n";
    exit_code = 0;

cleanup:
    if (handle != nullptr) {
        kudacuda_destroy(handle);
    }
    if (stream != nullptr) {
        cudaStreamDestroy(stream);
    }
    cudaFree(device_output);
    cudaFree(device_weight);
    cudaFree(device_input);
    return exit_code;
}
