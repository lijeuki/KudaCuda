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
    std::cerr << expression << " failed (code " << static_cast<int>(status)
              << ", " << cudaGetErrorName(status) << "): "
              << cudaGetErrorString(status) << '\n';
    return false;
}

float to_float(const __half& value) {
    return __half2float(value);
}

}  // namespace

int main(int argc, char** argv) {
    const int device_id = argc > 1 ? std::stoi(argv[1]) : 0;
    constexpr int batch_size = 1;
    constexpr int query_len = 1;
    constexpr int kv_len = 37;
    constexpr int query_heads = 16;
    constexpr int kv_heads = 2;
    constexpr int head_dim = 128;
    constexpr float scale = 1.0F / 11.313708498984761F;
    constexpr float tolerance = 3.5e-3F;

    if (!check_cuda(cudaSetDevice(device_id), "cudaSetDevice")) {
        return 1;
    }

    const std::size_t query_count = batch_size * query_heads * query_len * head_dim;
    const std::size_t cache_count = batch_size * kv_heads * kv_len * head_dim;
    std::vector<__half> host_query(query_count);
    std::vector<__half> host_key(cache_count);
    std::vector<__half> host_value(cache_count);
    std::vector<__half> host_output(query_count);
    std::vector<__half> host_expected(query_count);

    for (std::size_t index = 0; index < query_count; ++index) {
        host_query[index] = __float2half(std::sin(static_cast<float>(index) * 0.017F) * 0.75F);
    }
    for (std::size_t index = 0; index < cache_count; ++index) {
        host_key[index] = __float2half(std::cos(static_cast<float>(index) * 0.011F) * 0.55F);
        host_value[index] = __float2half(std::sin(static_cast<float>(index) * 0.023F) * 0.65F);
    }

    for (int query_head = 0; query_head < query_heads; ++query_head) {
        const int kv_head = query_head / (query_heads / kv_heads);
        std::vector<float> scores(kv_len);
        float maximum = -INFINITY;
        for (int token = 0; token < kv_len; ++token) {
            float dot = 0.0F;
            for (int dimension = 0; dimension < head_dim; ++dimension) {
                const std::size_t query_index = query_head * head_dim + dimension;
                const std::size_t cache_index =
                    (kv_head * kv_len + token) * head_dim + dimension;
                dot += to_float(host_query[query_index]) * to_float(host_key[cache_index]);
            }
            scores[token] = dot * scale;
            maximum = std::fmax(maximum, scores[token]);
        }
        float denominator = 0.0F;
        for (float& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        for (int dimension = 0; dimension < head_dim; ++dimension) {
            float result = 0.0F;
            for (int token = 0; token < kv_len; ++token) {
                const std::size_t cache_index =
                    (kv_head * kv_len + token) * head_dim + dimension;
                result += (scores[token] / denominator) * to_float(host_value[cache_index]);
            }
            host_expected[query_head * head_dim + dimension] = __float2half(result);
        }
    }

    __half* device_query = nullptr;
    __half* device_key = nullptr;
    __half* device_value = nullptr;
    __half* device_output = nullptr;
    cudaStream_t stream = nullptr;
    void* handle = nullptr;
    int exit_code = 1;
    const std::size_t query_bytes = query_count * sizeof(__half);
    const std::size_t cache_bytes = cache_count * sizeof(__half);

    if (!check_cuda(cudaMalloc(&device_query, query_bytes), "cudaMalloc(query)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_key, cache_bytes), "cudaMalloc(key)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_value, cache_bytes), "cudaMalloc(value)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_output, query_bytes), "cudaMalloc(output)")) goto cleanup;
    if (!check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(device_query, host_query.data(), query_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(query)")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(device_key, host_key.data(), cache_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(key)")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(device_value, host_value.data(), cache_bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync(value)")) goto cleanup;

    handle = kudacuda_create(device_id);
    if (handle == nullptr || kudacuda_freeze(handle) != 0) {
        std::cerr << "KudaCuda initialization failed\n";
        goto cleanup;
    }
    if (kudacuda_can_dispatch(handle, "qwen2_gqa_decode_sm86_fp16", 1, 1) != 1) {
        std::cerr << "Qwen2 GQA tactic was not dispatchable\n";
        goto cleanup;
    }
    if (kudacuda_qwen2_gqa_decode(
            handle,
            reinterpret_cast<uint64_t>(device_query),
            reinterpret_cast<uint64_t>(device_key),
            reinterpret_cast<uint64_t>(device_value),
            reinterpret_cast<uint64_t>(device_output),
            batch_size,
            query_len,
            kv_len,
            query_heads,
            kv_heads,
            head_dim,
            scale,
            reinterpret_cast<uint64_t>(stream)) != 0) {
        std::cerr << "kudacuda_qwen2_gqa_decode failed: " << kudacuda_last_error(handle) << '\n';
        goto cleanup;
    }
    if (!check_cuda(cudaMemcpyAsync(host_output.data(), device_output, query_bytes, cudaMemcpyDeviceToHost, stream), "cudaMemcpyAsync(output)")) goto cleanup;
    if (!check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) goto cleanup;

    for (std::size_t index = 0; index < host_output.size(); ++index) {
        if (std::fabs(to_float(host_output[index]) - to_float(host_expected[index])) > tolerance) {
            std::cerr << "Qwen2 GQA mismatch at " << index << ": expected "
                      << to_float(host_expected[index]) << ", got " << to_float(host_output[index]) << '\n';
            goto cleanup;
        }
    }

    std::cout << "KudaCuda Qwen2 GQA decode output matched the independent FP32 reference\n";
    exit_code = 0;

cleanup:
    if (handle != nullptr) kudacuda_destroy(handle);
    if (stream != nullptr) cudaStreamDestroy(stream);
    cudaFree(device_output);
    cudaFree(device_value);
    cudaFree(device_key);
    cudaFree(device_query);
    return exit_code;
}
