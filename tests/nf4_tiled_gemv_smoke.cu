#include "kudacuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr float kCodebook[16] = {
    -1.0F, -0.6961928009986877F, -0.5250730514526367F, -0.39491748809814453F,
    -0.28444138169288635F, -0.18477343022823334F, -0.09105003625154495F, 0.0F,
    0.07958029955625534F, 0.16093020141124725F, 0.24611230194568634F,
    0.33791524171829224F, 0.44070982933044434F, 0.5626170039176941F,
    0.7229568362236023F, 1.0F,
};

bool check(cudaError_t status, const char* name) {
    if (status == cudaSuccess) return true;
    std::cerr << name << " failed: " << cudaGetErrorString(status) << '\n';
    return false;
}

void pack(std::vector<uint8_t>& output, int logical_count, int salt) {
    for (int index = 0; index < logical_count; index += 2) {
        const int even = (index * 5 + salt) & 0x0f;
        const int odd = ((index + 1) * 5 + salt) & 0x0f;
        output[static_cast<std::size_t>(index / 2)] =
            static_cast<uint8_t>((even << 4) | odd);
    }
}

float decode(const std::vector<uint8_t>& weight, const std::vector<float>& scales, int index) {
    const uint8_t byte = weight[static_cast<std::size_t>(index / 2)];
    const int code = (index & 1) == 0 ? byte >> 4 : byte & 0x0f;
    return kCodebook[code] * scales[static_cast<std::size_t>(index / 64)];
}

}  // namespace

int main(int argc, char** argv) {
    constexpr int hidden = 2048;
    constexpr int q_out = 2048;
    constexpr int kv_out = 256;
    constexpr float epsilon = 1.0e-5F;
    constexpr float tolerance = 0.035F;
    const int device_id = argc > 1 ? std::stoi(argv[1]) : 0;
    if (!check(cudaSetDevice(device_id), "cudaSetDevice")) return 1;

    std::vector<__half> input(hidden), norm_weight(hidden), q_bias(q_out), k_bias(kv_out), v_bias(kv_out);
    std::vector<uint8_t> q_weight(q_out * hidden / 2), k_weight(kv_out * hidden / 2), v_weight(kv_out * hidden / 2);
    std::vector<float> q_scales(q_out * hidden / 64), k_scales(kv_out * hidden / 64), v_scales(kv_out * hidden / 64);
    std::vector<__half> q_output(q_out), k_output(kv_out), v_output(kv_out);
    std::vector<float> normalized(hidden);
    std::vector<__half> q_expected(q_out), k_expected(kv_out), v_expected(kv_out);

    for (int i = 0; i < hidden; ++i) {
        input[i] = __float2half(std::sin(i * 0.013F));
        norm_weight[i] = __float2half(0.75F + (i % 23) * 0.004F);
    }
    pack(q_weight, q_out * hidden, 1);
    pack(k_weight, kv_out * hidden, 3);
    pack(v_weight, kv_out * hidden, 7);
    for (std::size_t i = 0; i < q_scales.size(); ++i) q_scales[i] = 0.2F + (i % 19) * 0.009F;
    for (std::size_t i = 0; i < k_scales.size(); ++i) k_scales[i] = 0.25F + (i % 17) * 0.011F;
    for (std::size_t i = 0; i < v_scales.size(); ++i) v_scales[i] = 0.18F + (i % 13) * 0.013F;
    for (int i = 0; i < q_out; ++i) q_bias[i] = __float2half((i % 11 - 5) * 0.002F);
    for (int i = 0; i < kv_out; ++i) {
        k_bias[i] = __float2half((i % 7 - 3) * 0.003F);
        v_bias[i] = __float2half((i % 5 - 2) * 0.004F);
    }

    float sum_squares = 0.0F;
    for (int i = 0; i < hidden; ++i) {
        const float value = __half2float(input[i]);
        sum_squares += value * value;
    }
    const float inverse_rms = 1.0F / std::sqrt(sum_squares / hidden + epsilon);
    for (int i = 0; i < hidden; ++i) {
        normalized[i] = __half2float(input[i]) * inverse_rms * __half2float(norm_weight[i]);
    }
    for (int output_index = 0; output_index < q_out; ++output_index) {
        float sum = 0.0F;
        for (int dimension = 0; dimension < hidden; ++dimension) {
            sum += normalized[dimension] * decode(q_weight, q_scales, output_index * hidden + dimension);
        }
        q_expected[output_index] = __float2half(sum + __half2float(q_bias[output_index]));
    }
    for (int output_index = 0; output_index < kv_out; ++output_index) {
        float key_sum = 0.0F;
        float value_sum = 0.0F;
        for (int dimension = 0; dimension < hidden; ++dimension) {
            const int index = output_index * hidden + dimension;
            key_sum += normalized[dimension] * decode(k_weight, k_scales, index);
            value_sum += normalized[dimension] * decode(v_weight, v_scales, index);
        }
        k_expected[output_index] = __float2half(key_sum + __half2float(k_bias[output_index]));
        v_expected[output_index] = __float2half(value_sum + __half2float(v_bias[output_index]));
    }

    __half *d_input = nullptr, *d_norm_weight = nullptr, *d_q_bias = nullptr, *d_k_bias = nullptr, *d_v_bias = nullptr;
    __half *d_q_output = nullptr, *d_k_output = nullptr, *d_v_output = nullptr;
    uint8_t *d_q_weight = nullptr, *d_k_weight = nullptr, *d_v_weight = nullptr;
    float *d_q_scales = nullptr, *d_k_scales = nullptr, *d_v_scales = nullptr, *d_normalized = nullptr;
    cudaStream_t stream = nullptr;
    void* handle = nullptr;
    int exit_code = 1;

    if (!check(cudaMalloc(&d_input, hidden * sizeof(__half)), "cudaMalloc(input)")) goto cleanup;
    if (!check(cudaMalloc(&d_norm_weight, hidden * sizeof(__half)), "cudaMalloc(norm)")) goto cleanup;
    if (!check(cudaMalloc(&d_q_weight, q_weight.size()), "cudaMalloc(q_weight)")) goto cleanup;
    if (!check(cudaMalloc(&d_k_weight, k_weight.size()), "cudaMalloc(k_weight)")) goto cleanup;
    if (!check(cudaMalloc(&d_v_weight, v_weight.size()), "cudaMalloc(v_weight)")) goto cleanup;
    if (!check(cudaMalloc(&d_q_scales, q_scales.size() * sizeof(float)), "cudaMalloc(q_scales)")) goto cleanup;
    if (!check(cudaMalloc(&d_k_scales, k_scales.size() * sizeof(float)), "cudaMalloc(k_scales)")) goto cleanup;
    if (!check(cudaMalloc(&d_v_scales, v_scales.size() * sizeof(float)), "cudaMalloc(v_scales)")) goto cleanup;
    if (!check(cudaMalloc(&d_q_bias, q_bias.size() * sizeof(__half)), "cudaMalloc(q_bias)")) goto cleanup;
    if (!check(cudaMalloc(&d_k_bias, k_bias.size() * sizeof(__half)), "cudaMalloc(k_bias)")) goto cleanup;
    if (!check(cudaMalloc(&d_v_bias, v_bias.size() * sizeof(__half)), "cudaMalloc(v_bias)")) goto cleanup;
    if (!check(cudaMalloc(&d_normalized, hidden * sizeof(float)), "cudaMalloc(normalized)")) goto cleanup;
    if (!check(cudaMalloc(&d_q_output, q_output.size() * sizeof(__half)), "cudaMalloc(q_output)")) goto cleanup;
    if (!check(cudaMalloc(&d_k_output, k_output.size() * sizeof(__half)), "cudaMalloc(k_output)")) goto cleanup;
    if (!check(cudaMalloc(&d_v_output, v_output.size() * sizeof(__half)), "cudaMalloc(v_output)")) goto cleanup;
    if (!check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate")) goto cleanup;

    if (!check(cudaMemcpyAsync(d_input, input.data(), input.size() * sizeof(__half), cudaMemcpyHostToDevice, stream), "copy input")) goto cleanup;
    if (!check(cudaMemcpyAsync(d_norm_weight, norm_weight.data(), norm_weight.size() * sizeof(__half), cudaMemcpyHostToDevice, stream), "copy norm")) goto cleanup;
    if (!check(cudaMemcpyAsync(d_q_weight, q_weight.data(), q_weight.size(), cudaMemcpyHostToDevice, stream), "copy q weight")) goto cleanup;
    if (!check(cudaMemcpyAsync(d_k_weight, k_weight.data(), k_weight.size(), cudaMemcpyHostToDevice, stream), "copy k weight")) goto cleanup;
    if (!check(cudaMemcpyAsync(d_v_weight, v_weight.data(), v_weight.size(), cudaMemcpyHostToDevice, stream), "copy v weight")) goto cleanup;
    if (!check(cudaMemcpyAsync(d_q_scales, q_scales.data(), q_scales.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "copy q scales")) goto cleanup;
    if (!check(cudaMemcpyAsync(d_k_scales, k_scales.data(), k_scales.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "copy k scales")) goto cleanup;
    if (!check(cudaMemcpyAsync(d_v_scales, v_scales.data(), v_scales.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "copy v scales")) goto cleanup;
    if (!check(cudaMemcpyAsync(d_q_bias, q_bias.data(), q_bias.size() * sizeof(__half), cudaMemcpyHostToDevice, stream), "copy q bias")) goto cleanup;
    if (!check(cudaMemcpyAsync(d_k_bias, k_bias.data(), k_bias.size() * sizeof(__half), cudaMemcpyHostToDevice, stream), "copy k bias")) goto cleanup;
    if (!check(cudaMemcpyAsync(d_v_bias, v_bias.data(), v_bias.size() * sizeof(__half), cudaMemcpyHostToDevice, stream), "copy v bias")) goto cleanup;

    handle = kudacuda_create(device_id);
    if (handle == nullptr || kudacuda_freeze(handle) != 0) goto cleanup;
    if (kudacuda_qwen2_fused_rmsnorm_qkv_nf4_v2(
            handle,
            reinterpret_cast<uint64_t>(d_input), reinterpret_cast<uint64_t>(d_norm_weight),
            reinterpret_cast<uint64_t>(d_q_weight), reinterpret_cast<uint64_t>(d_q_scales), reinterpret_cast<uint64_t>(d_q_bias),
            reinterpret_cast<uint64_t>(d_k_weight), reinterpret_cast<uint64_t>(d_k_scales), reinterpret_cast<uint64_t>(d_k_bias),
            reinterpret_cast<uint64_t>(d_v_weight), reinterpret_cast<uint64_t>(d_v_scales), reinterpret_cast<uint64_t>(d_v_bias),
            reinterpret_cast<uint64_t>(d_normalized), reinterpret_cast<uint64_t>(d_q_output),
            reinterpret_cast<uint64_t>(d_k_output), reinterpret_cast<uint64_t>(d_v_output),
            1, hidden, q_out, kv_out, 64, epsilon, reinterpret_cast<uint64_t>(stream)) != 0) {
        std::cerr << kudacuda_last_error(handle) << '\n';
        goto cleanup;
    }
    if (!check(cudaMemcpyAsync(q_output.data(), d_q_output, q_output.size() * sizeof(__half), cudaMemcpyDeviceToHost, stream), "copy q output")) goto cleanup;
    if (!check(cudaMemcpyAsync(k_output.data(), d_k_output, k_output.size() * sizeof(__half), cudaMemcpyDeviceToHost, stream), "copy k output")) goto cleanup;
    if (!check(cudaMemcpyAsync(v_output.data(), d_v_output, v_output.size() * sizeof(__half), cudaMemcpyDeviceToHost, stream), "copy v output")) goto cleanup;
    if (!check(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) goto cleanup;
    for (std::size_t i = 0; i < q_output.size(); ++i) if (std::fabs(__half2float(q_output[i]) - __half2float(q_expected[i])) > tolerance) goto cleanup;
    for (std::size_t i = 0; i < k_output.size(); ++i) if (std::fabs(__half2float(k_output[i]) - __half2float(k_expected[i])) > tolerance || std::fabs(__half2float(v_output[i]) - __half2float(v_expected[i])) > tolerance) goto cleanup;

    std::cout << "KudaCuda tiled NF4 GEMV output matched the independent FP32 reference\n";
    exit_code = 0;

cleanup:
    if (handle != nullptr) kudacuda_destroy(handle);
    if (stream != nullptr) cudaStreamDestroy(stream);
    cudaFree(d_v_output); cudaFree(d_k_output); cudaFree(d_q_output); cudaFree(d_normalized);
    cudaFree(d_v_bias); cudaFree(d_k_bias); cudaFree(d_q_bias);
    cudaFree(d_v_scales); cudaFree(d_k_scales); cudaFree(d_q_scales);
    cudaFree(d_v_weight); cudaFree(d_k_weight); cudaFree(d_q_weight);
    cudaFree(d_norm_weight); cudaFree(d_input);
    return exit_code;
}
