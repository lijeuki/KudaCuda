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

}  // namespace

int main(int argc, char** argv) {
    const int device_id = argc > 1 ? std::stoi(argv[1]) : 0;
    constexpr int query_heads = 16;
    constexpr int kv_heads = 2;
    constexpr int head_dim = 128;
    constexpr int kv_capacity = 512;
    constexpr float scale = 1.0F / 11.313708498984761F;
    constexpr int workspace_floats = query_heads * 8 * (2 + head_dim);
    const int kv_lens[] = {1, 7, 64, 128, 512};

    if (!check_cuda(cudaSetDevice(device_id), "cudaSetDevice")) {
        return 1;
    }

    std::vector<__half> host_query(query_heads * head_dim);
    std::vector<__half> host_key(kv_heads * kv_capacity * head_dim);
    std::vector<__half> host_value(kv_heads * kv_capacity * head_dim);
    std::vector<__half> host_output(query_heads * head_dim);
    std::vector<float> expected(query_heads * head_dim);
    for (std::size_t index = 0; index < host_query.size(); ++index) {
        host_query[index] = __float2half(std::sin(static_cast<float>(index) * 0.013F));
    }
    for (std::size_t index = 0; index < host_key.size(); ++index) {
        host_key[index] = __float2half(std::cos(static_cast<float>(index) * 0.009F));
        host_value[index] = __float2half(std::sin(static_cast<float>(index) * 0.007F));
    }

    __half* device_query = nullptr;
    __half* device_key = nullptr;
    __half* device_value = nullptr;
    __half* device_output = nullptr;
    float* device_workspace = nullptr;
    int32_t* device_kv_len = nullptr;
    cudaStream_t stream = nullptr;
    void* handle = nullptr;
    int exit_code = 1;

    if (!check_cuda(cudaMalloc(&device_query, host_query.size() * sizeof(__half)), "cudaMalloc(query)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_key, host_key.size() * sizeof(__half)), "cudaMalloc(key)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_value, host_value.size() * sizeof(__half)), "cudaMalloc(value)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_output, host_output.size() * sizeof(__half)), "cudaMalloc(output)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_workspace, workspace_floats * sizeof(float)), "cudaMalloc(workspace)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_kv_len, sizeof(int32_t)), "cudaMalloc(kv_len)")) goto cleanup;
    if (!check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(device_query, host_query.data(), host_query.size() * sizeof(__half), cudaMemcpyHostToDevice, stream), "copy query")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(device_key, host_key.data(), host_key.size() * sizeof(__half), cudaMemcpyHostToDevice, stream), "copy key")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(device_value, host_value.data(), host_value.size() * sizeof(__half), cudaMemcpyHostToDevice, stream), "copy value")) goto cleanup;

    handle = kudacuda_create(device_id);
    if (handle == nullptr || kudacuda_freeze(handle) != 0) {
        std::cerr << "KudaCuda initialization failed\n";
        goto cleanup;
    }

    for (int kv_len : kv_lens) {
        if (!check_cuda(cudaMemcpyAsync(device_kv_len, &kv_len, sizeof(int32_t), cudaMemcpyHostToDevice, stream), "copy kv_len")) goto cleanup;
        if (kudacuda_qwen2_gqa_decode_workspace_v3(
                handle,
                reinterpret_cast<uint64_t>(device_query),
                reinterpret_cast<uint64_t>(device_key),
                reinterpret_cast<uint64_t>(device_value),
                reinterpret_cast<uint64_t>(device_output),
                reinterpret_cast<uint64_t>(device_workspace),
                reinterpret_cast<uint64_t>(device_kv_len),
                1, 1, kv_capacity, query_heads, kv_heads, head_dim, scale,
                reinterpret_cast<uint64_t>(stream)) != 0) {
            std::cerr << "workspace GQA launch failed for kv_len=" << kv_len << ": " << kudacuda_last_error(handle) << '\n';
            goto cleanup;
        }
        if (!check_cuda(cudaMemcpyAsync(host_output.data(), device_output, host_output.size() * sizeof(__half), cudaMemcpyDeviceToHost, stream), "copy output")) goto cleanup;
        if (!check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) goto cleanup;

        for (int query_head = 0; query_head < query_heads; ++query_head) {
            const int kv_head = query_head / (query_heads / kv_heads);
            float max_score = -INFINITY;
            for (int position = 0; position < kv_len; ++position) {
                float score = 0.0F;
                for (int dimension = 0; dimension < head_dim; ++dimension) {
                    score += __half2float(host_query[query_head * head_dim + dimension]) *
                        __half2float(host_key[(kv_head * kv_capacity + position) * head_dim + dimension]);
                }
                max_score = fmaxf(max_score, score * scale);
            }
            float denominator = 0.0F;
            for (int position = 0; position < kv_len; ++position) {
                float score = 0.0F;
                for (int dimension = 0; dimension < head_dim; ++dimension) {
                    score += __half2float(host_query[query_head * head_dim + dimension]) *
                        __half2float(host_key[(kv_head * kv_capacity + position) * head_dim + dimension]);
                }
                denominator += expf(score * scale - max_score);
            }
            for (int dimension = 0; dimension < head_dim; ++dimension) {
                float value = 0.0F;
                for (int position = 0; position < kv_len; ++position) {
                    float score = 0.0F;
                    for (int score_dimension = 0; score_dimension < head_dim; ++score_dimension) {
                        score += __half2float(host_query[query_head * head_dim + score_dimension]) *
                            __half2float(host_key[(kv_head * kv_capacity + position) * head_dim + score_dimension]);
                    }
                    value += expf(score * scale - max_score) *
                        __half2float(host_value[(kv_head * kv_capacity + position) * head_dim + dimension]);
                }
                expected[query_head * head_dim + dimension] = value / denominator;
                if (std::fabs(__half2float(host_output[query_head * head_dim + dimension]) - expected[query_head * head_dim + dimension]) > 0.006F) {
                    std::cerr << "workspace GQA mismatch at kv_len=" << kv_len << ", query_head=" << query_head << ", dimension=" << dimension << '\n';
                    goto cleanup;
                }
            }
        }
    }

    std::cout << "KudaCuda workspace GQA output matched the independent FP32 reference for kv_len=1,7,64,128,512\n";
    exit_code = 0;

cleanup:
    if (handle != nullptr) kudacuda_destroy(handle);
    if (stream != nullptr) cudaStreamDestroy(stream);
    cudaFree(device_kv_len);
    cudaFree(device_workspace);
    cudaFree(device_output);
    cudaFree(device_value);
    cudaFree(device_key);
    cudaFree(device_query);
    return exit_code;
}
