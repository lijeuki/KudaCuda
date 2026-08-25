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
    std::cerr << expression << " failed (" << cudaGetErrorString(status) << ")\n";
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    const int device_id = argc > 1 ? std::stoi(argv[1]) : 0;
    constexpr int hidden = 513;
    constexpr float epsilon = 1.0e-5F;
    constexpr float tolerance = 5.0e-5F;
    const std::size_t bytes = hidden * sizeof(float);

    if (!check_cuda(cudaSetDevice(device_id), "cudaSetDevice")) {
        return 1;
    }

    std::vector<float> host_a(hidden);
    std::vector<float> host_b(hidden);
    std::vector<float> host_weight(hidden);
    std::vector<float> host_output(hidden, 0.0F);
    std::vector<float> host_expected(hidden, 0.0F);
    for (int index = 0; index < hidden; ++index) {
        host_a[index] = std::sin(static_cast<float>(index) * 0.017F);
        host_b[index] = std::cos(static_cast<float>(index) * 0.011F) * 0.25F;
        host_weight[index] = 0.75F + static_cast<float>(index % 13) * 0.03125F;
    }
    double sum_squares = 0.0;
    for (int index = 0; index < hidden; ++index) {
        const float value = host_a[index] + host_b[index];
        sum_squares += static_cast<double>(value) * static_cast<double>(value);
    }
    const float inverse_rms = static_cast<float>(1.0 / std::sqrt(sum_squares / hidden + epsilon));
    for (int index = 0; index < hidden; ++index) {
        host_expected[index] = (host_a[index] + host_b[index]) * inverse_rms * host_weight[index];
    }

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_weight = nullptr;
    float* device_intermediate = nullptr;
    float* device_output = nullptr;
    cudaStream_t stream = nullptr;
    void* handle = nullptr;
    int exit_code = 1;

    if (!check_cuda(cudaMalloc(&device_a, bytes), "cudaMalloc(a)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_b, bytes), "cudaMalloc(b)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_weight, bytes), "cudaMalloc(weight)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_intermediate, bytes), "cudaMalloc(intermediate)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_output, bytes), "cudaMalloc(output)")) goto cleanup;
    if (!check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice, stream), "copy(a)")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice, stream), "copy(b)")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(device_weight, host_weight.data(), bytes, cudaMemcpyHostToDevice, stream), "copy(weight)")) goto cleanup;

    handle = kudacuda_create(device_id);
    if (handle == nullptr || kudacuda_freeze(handle) != 0) {
        std::cerr << "KudaCuda initialization failed\n";
        goto cleanup;
    }
    if (kudacuda_begin_capture(handle, reinterpret_cast<uint64_t>(stream)) != 0) goto cleanup;
    if (kudacuda_vector_add(
            handle,
            reinterpret_cast<uint64_t>(device_a),
            reinterpret_cast<uint64_t>(device_b),
            reinterpret_cast<uint64_t>(device_intermediate),
            hidden,
            reinterpret_cast<uint64_t>(stream)) != 0) goto cleanup;
    if (kudacuda_rmsnorm(
            handle,
            reinterpret_cast<uint64_t>(device_intermediate),
            reinterpret_cast<uint64_t>(device_weight),
            reinterpret_cast<uint64_t>(device_output),
            1,
            hidden,
            epsilon,
            reinterpret_cast<uint64_t>(stream)) != 0) goto cleanup;
    if (kudacuda_end_capture(handle, reinterpret_cast<uint64_t>(stream)) != 0) goto cleanup;
    if (kudacuda_replay_graph(handle, reinterpret_cast<uint64_t>(stream)) != 0) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(host_output.data(), device_output, bytes, cudaMemcpyDeviceToHost, stream), "copy(output)")) goto cleanup;
    if (!check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) goto cleanup;

    for (int index = 0; index < hidden; ++index) {
        if (std::fabs(host_output[index] - host_expected[index]) > tolerance) {
            std::cerr << "Graph mismatch at " << index << "\n";
            goto cleanup;
        }
    }
    if (kudacuda_reset_graph(handle) != 0) goto cleanup;

    std::cout << "KudaCuda multi-launch CUDA graph replay matched the independent reference\n";
    exit_code = 0;

cleanup:
    if (handle != nullptr) kudacuda_destroy(handle);
    if (stream != nullptr) cudaStreamDestroy(stream);
    cudaFree(device_output);
    cudaFree(device_intermediate);
    cudaFree(device_weight);
    cudaFree(device_b);
    cudaFree(device_a);
    return exit_code;
}
