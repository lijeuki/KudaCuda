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
    constexpr int count = 4096;

    if (!check_cuda(cudaSetDevice(device_id), "cudaSetDevice")) {
        return 1;
    }

    cudaDeviceProp properties{};
    if (!check_cuda(cudaGetDeviceProperties(&properties, device_id), "cudaGetDeviceProperties")) {
        return 1;
    }

    std::vector<float> host_a(count);
    std::vector<float> host_b(count);
    std::vector<float> host_output(count, 0.0F);
    for (int index = 0; index < count; ++index) {
        host_a[index] = static_cast<float>(index) * 0.25F;
        host_b[index] = static_cast<float>(count - index) * 0.5F;
    }

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_output = nullptr;
    cudaStream_t stream = nullptr;
    void* handle = nullptr;
    int status = 0;
    int exit_code = 1;

    if (!check_cuda(cudaMalloc(&device_a, count * sizeof(float)), "cudaMalloc(device_a)")) {
        goto cleanup;
    }
    if (!check_cuda(cudaMalloc(&device_b, count * sizeof(float)), "cudaMalloc(device_b)")) {
        goto cleanup;
    }
    if (!check_cuda(cudaMalloc(&device_output, count * sizeof(float)), "cudaMalloc(device_output)")) {
        goto cleanup;
    }
    if (!check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags")) {
        goto cleanup;
    }
    if (!check_cuda(cudaMemcpyAsync(device_a, host_a.data(), count * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(a)")) {
        goto cleanup;
    }
    if (!check_cuda(cudaMemcpyAsync(device_b, host_b.data(), count * sizeof(float), cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(b)")) {
        goto cleanup;
    }

    handle = kudacuda_create(device_id);
    if (handle == nullptr) {
        std::cerr << "kudacuda_create failed\n";
        goto cleanup;
    }
    if (kudacuda_freeze(handle) != 0) {
        std::cerr << "kudacuda_freeze failed: " << kudacuda_last_error(handle) << '\n';
        goto cleanup;
    }

    status = kudacuda_vector_add(
        handle,
        reinterpret_cast<uint64_t>(device_a),
        reinterpret_cast<uint64_t>(device_b),
        reinterpret_cast<uint64_t>(device_output),
        count,
        reinterpret_cast<uint64_t>(stream)
    );
    if (status != 0) {
        std::cerr << "kudacuda_vector_add failed: " << kudacuda_last_error(handle) << '\n';
        goto cleanup;
    }
    if (!check_cuda(cudaMemcpyAsync(host_output.data(), device_output, count * sizeof(float), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(output)")) {
        goto cleanup;
    }
    if (!check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) {
        goto cleanup;
    }

    for (int index = 0; index < count; ++index) {
        const float expected = host_a[index] + host_b[index];
        if (std::fabs(host_output[index] - expected) > 1.0e-6F) {
            std::cerr << "Mismatch at " << index << ": expected " << expected
                      << ", got " << host_output[index] << '\n';
            goto cleanup;
        }
    }

    if (!check_cuda(cudaMemsetAsync(device_output, 0, count * sizeof(float), stream), "cudaMemsetAsync(graph output)")) {
        goto cleanup;
    }
    if (kudacuda_begin_capture(handle, reinterpret_cast<uint64_t>(stream)) != 0) {
        std::cerr << "kudacuda_begin_capture failed: " << kudacuda_last_error(handle) << '\n';
        goto cleanup;
    }
    status = kudacuda_vector_add(
        handle,
        reinterpret_cast<uint64_t>(device_a),
        reinterpret_cast<uint64_t>(device_b),
        reinterpret_cast<uint64_t>(device_output),
        count,
        reinterpret_cast<uint64_t>(stream)
    );
    if (status != 0) {
        std::cerr << "captured kudacuda_vector_add failed: " << kudacuda_last_error(handle) << '\n';
        goto cleanup;
    }
    if (kudacuda_end_capture(handle, reinterpret_cast<uint64_t>(stream)) != 0) {
        std::cerr << "kudacuda_end_capture failed: " << kudacuda_last_error(handle) << '\n';
        goto cleanup;
    }
    if (kudacuda_replay_graph(handle, reinterpret_cast<uint64_t>(stream)) != 0) {
        std::cerr << "kudacuda_replay_graph failed: " << kudacuda_last_error(handle) << '\n';
        goto cleanup;
    }
    if (!check_cuda(cudaMemcpyAsync(host_output.data(), device_output, count * sizeof(float), cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(graph output)")) {
        goto cleanup;
    }
    if (!check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize(graph)")) {
        goto cleanup;
    }
    for (int index = 0; index < count; ++index) {
        const float expected = host_a[index] + host_b[index];
        if (std::fabs(host_output[index] - expected) > 1.0e-6F) {
            std::cerr << "Graph mismatch at " << index << ": expected " << expected
                      << ", got " << host_output[index] << '\n';
            goto cleanup;
        }
    }

    std::cout << "KudaCuda vector_add passed on " << properties.name
              << " using a caller-owned non-default CUDA stream and graph replay\n";
    exit_code = 0;

cleanup:
    if (handle != nullptr) {
        kudacuda_destroy(handle);
    }
    if (stream != nullptr) {
        cudaStreamDestroy(stream);
    }
    cudaFree(device_output);
    cudaFree(device_b);
    cudaFree(device_a);
    return exit_code;
}
