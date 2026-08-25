#include "kudacuda.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr float kNf4Codebook[16] = {
    -1.0F, -0.6961928009986877F, -0.5250730514526367F, -0.39491748809814453F,
    -0.28444138169288635F, -0.18477343022823334F, -0.09105003625154495F, 0.0F,
    0.07958029955625534F, 0.16093020141124725F, 0.24611230194568634F,
    0.33791524171829224F, 0.44070982933044434F, 0.5626170039176941F,
    0.7229568362236023F, 1.0F,
};

bool check_cuda(cudaError_t status, const char* expression) {
    if (status == cudaSuccess) {
        return true;
    }
    std::cerr << expression << " failed: " << cudaGetErrorString(status) << '\n';
    return false;
}

uint8_t pack_nf4_code(int first, int second) {
    return static_cast<uint8_t>(((first & 0x0f) << 4) | (second & 0x0f));
}

float decode_nf4(const std::vector<uint8_t>& packed, const std::vector<float>& absmax, int index) {
    const uint8_t byte = packed[static_cast<std::size_t>(index / 2)];
    const int code = (index & 1) == 0 ? (byte >> 4) : (byte & 0x0f);
    return kNf4Codebook[code] * absmax[static_cast<std::size_t>(index / 64)];
}

template <typename T>
void fill_packed(std::vector<uint8_t>& packed, int logical_count, int salt) {
    for (int index = 0; index < logical_count; index += 2) {
        const int first = (index * 7 + salt) & 0x0f;
        const int second = ((index + 1) * 7 + salt) & 0x0f;
        packed[static_cast<std::size_t>(index / 2)] = pack_nf4_code(first, second);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const int device_id = argc > 1 ? std::stoi(argv[1]) : 0;
    constexpr int rows = 1;
    constexpr int hidden = 2048;
    constexpr int q_out = 2048;
    constexpr int kv_out = 256;
    constexpr int blocksize = 64;
    constexpr float epsilon = 1.0e-5F;
    constexpr float tolerance = 0.035F;

    if (!check_cuda(cudaSetDevice(device_id), "cudaSetDevice")) {
        return 1;
    }

    const int q_weight_count = q_out * hidden;
    const int kv_weight_count = kv_out * hidden;
    std::vector<__half> host_input(hidden);
    std::vector<__half> host_norm_weight(hidden);
    std::vector<uint8_t> host_q_weight(q_weight_count / 2);
    std::vector<uint8_t> host_k_weight(kv_weight_count / 2);
    std::vector<uint8_t> host_v_weight(kv_weight_count / 2);
    std::vector<float> host_q_absmax(q_weight_count / blocksize);
    std::vector<float> host_k_absmax(kv_weight_count / blocksize);
    std::vector<float> host_v_absmax(kv_weight_count / blocksize);
    std::vector<__half> host_q_bias(q_out);
    std::vector<__half> host_k_bias(kv_out);
    std::vector<__half> host_v_bias(kv_out);
    std::vector<__half> host_q_output(q_out);
    std::vector<__half> host_k_output(kv_out);
    std::vector<__half> host_v_output(kv_out);
    std::vector<__half> host_q_expected(q_out);
    std::vector<__half> host_k_expected(kv_out);
    std::vector<__half> host_v_expected(kv_out);

    for (int index = 0; index < hidden; ++index) {
        host_input[index] = __float2half(std::sin(static_cast<float>(index) * 0.017F));
        host_norm_weight[index] = __float2half(0.8F + static_cast<float>(index % 31) * 0.003F);
    }
    fill_packed<int>(host_q_weight, q_weight_count, 3);
    fill_packed<int>(host_k_weight, kv_weight_count, 5);
    fill_packed<int>(host_v_weight, kv_weight_count, 9);
    for (std::size_t index = 0; index < host_q_absmax.size(); ++index) {
        host_q_absmax[index] = 0.25F + static_cast<float>(index % 17) * 0.01F;
    }
    for (std::size_t index = 0; index < host_k_absmax.size(); ++index) {
        host_k_absmax[index] = 0.3F + static_cast<float>(index % 13) * 0.012F;
        host_v_absmax[index] = 0.2F + static_cast<float>(index % 11) * 0.014F;
    }
    for (int index = 0; index < q_out; ++index) {
        host_q_bias[index] = __float2half(static_cast<float>(index % 17 - 8) * 0.003F);
    }
    for (int index = 0; index < kv_out; ++index) {
        host_k_bias[index] = __float2half(static_cast<float>(index % 13 - 6) * 0.004F);
        host_v_bias[index] = __float2half(static_cast<float>(index % 11 - 5) * 0.005F);
    }

    float sum_squares = 0.0F;
    for (int index = 0; index < hidden; ++index) {
        const float value = __half2float(host_input[index]);
        sum_squares += value * value;
    }
    const float inverse_rms = 1.0F / std::sqrt(sum_squares / hidden + epsilon);
    std::vector<float> normalized(hidden);
    for (int index = 0; index < hidden; ++index) {
        normalized[index] = __half2float(host_input[index]) * inverse_rms *
            __half2float(host_norm_weight[index]);
    }
    for (int output_index = 0; output_index < q_out; ++output_index) {
        float value = 0.0F;
        for (int dimension = 0; dimension < hidden; ++dimension) {
            value += normalized[dimension] * decode_nf4(
                host_q_weight,
                host_q_absmax,
                output_index * hidden + dimension
            );
        }
        host_q_expected[output_index] = __float2half(value + __half2float(host_q_bias[output_index]));
    }
    for (int output_index = 0; output_index < kv_out; ++output_index) {
        float key_value = 0.0F;
        float value_value = 0.0F;
        for (int dimension = 0; dimension < hidden; ++dimension) {
            const int weight_index = output_index * hidden + dimension;
            key_value += normalized[dimension] * decode_nf4(host_k_weight, host_k_absmax, weight_index);
            value_value += normalized[dimension] * decode_nf4(host_v_weight, host_v_absmax, weight_index);
        }
        host_k_expected[output_index] = __float2half(key_value + __half2float(host_k_bias[output_index]));
        host_v_expected[output_index] = __float2half(value_value + __half2float(host_v_bias[output_index]));
    }

    __half* device_input = nullptr;
    __half* device_norm_weight = nullptr;
    uint8_t* device_q_weight = nullptr;
    float* device_q_absmax = nullptr;
    uint8_t* device_k_weight = nullptr;
    float* device_k_absmax = nullptr;
    uint8_t* device_v_weight = nullptr;
    float* device_v_absmax = nullptr;
    __half* device_q_bias = nullptr;
    __half* device_k_bias = nullptr;
    __half* device_v_bias = nullptr;
    __half* device_q_output = nullptr;
    __half* device_k_output = nullptr;
    __half* device_v_output = nullptr;
    cudaStream_t stream = nullptr;
    void* handle = nullptr;
    int exit_code = 1;

    if (!check_cuda(cudaMalloc(&device_input, host_input.size() * sizeof(__half)), "cudaMalloc(input)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_norm_weight, host_norm_weight.size() * sizeof(__half)), "cudaMalloc(norm_weight)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_q_weight, host_q_weight.size()), "cudaMalloc(q_weight)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_q_absmax, host_q_absmax.size() * sizeof(float)), "cudaMalloc(q_absmax)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_k_weight, host_k_weight.size()), "cudaMalloc(k_weight)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_k_absmax, host_k_absmax.size() * sizeof(float)), "cudaMalloc(k_absmax)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_v_weight, host_v_weight.size()), "cudaMalloc(v_weight)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_v_absmax, host_v_absmax.size() * sizeof(float)), "cudaMalloc(v_absmax)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_q_bias, host_q_bias.size() * sizeof(__half)), "cudaMalloc(q_bias)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_k_bias, host_k_bias.size() * sizeof(__half)), "cudaMalloc(k_bias)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_v_bias, host_v_bias.size() * sizeof(__half)), "cudaMalloc(v_bias)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_q_output, host_q_output.size() * sizeof(__half)), "cudaMalloc(q_output)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_k_output, host_k_output.size() * sizeof(__half)), "cudaMalloc(k_output)")) goto cleanup;
    if (!check_cuda(cudaMalloc(&device_v_output, host_v_output.size() * sizeof(__half)), "cudaMalloc(v_output)")) goto cleanup;
    if (!check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags")) goto cleanup;

    if (!check_cuda(cudaMemcpyAsync(device_input, host_input.data(), host_input.size() * sizeof(__half), cudaMemcpyHostToDevice, stream), "copy input")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(device_norm_weight, host_norm_weight.data(), host_norm_weight.size() * sizeof(__half), cudaMemcpyHostToDevice, stream), "copy norm")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(device_q_weight, host_q_weight.data(), host_q_weight.size(), cudaMemcpyHostToDevice, stream), "copy q weight")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(device_q_absmax, host_q_absmax.data(), host_q_absmax.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "copy q absmax")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(device_k_weight, host_k_weight.data(), host_k_weight.size(), cudaMemcpyHostToDevice, stream), "copy k weight")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(device_k_absmax, host_k_absmax.data(), host_k_absmax.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "copy k absmax")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(device_v_weight, host_v_weight.data(), host_v_weight.size(), cudaMemcpyHostToDevice, stream), "copy v weight")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(device_v_absmax, host_v_absmax.data(), host_v_absmax.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "copy v absmax")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(device_q_bias, host_q_bias.data(), host_q_bias.size() * sizeof(__half), cudaMemcpyHostToDevice, stream), "copy q bias")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(device_k_bias, host_k_bias.data(), host_k_bias.size() * sizeof(__half), cudaMemcpyHostToDevice, stream), "copy k bias")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(device_v_bias, host_v_bias.data(), host_v_bias.size() * sizeof(__half), cudaMemcpyHostToDevice, stream), "copy v bias")) goto cleanup;

    handle = kudacuda_create(device_id);
    if (handle == nullptr || kudacuda_freeze(handle) != 0) {
        std::cerr << "KudaCuda initialization failed\n";
        goto cleanup;
    }
    if (kudacuda_qwen2_fused_rmsnorm_qkv_nf4(
            handle,
            reinterpret_cast<uint64_t>(device_input),
            reinterpret_cast<uint64_t>(device_norm_weight),
            reinterpret_cast<uint64_t>(device_q_weight),
            reinterpret_cast<uint64_t>(device_q_absmax),
            reinterpret_cast<uint64_t>(device_q_bias),
            reinterpret_cast<uint64_t>(device_k_weight),
            reinterpret_cast<uint64_t>(device_k_absmax),
            reinterpret_cast<uint64_t>(device_k_bias),
            reinterpret_cast<uint64_t>(device_v_weight),
            reinterpret_cast<uint64_t>(device_v_absmax),
            reinterpret_cast<uint64_t>(device_v_bias),
            reinterpret_cast<uint64_t>(device_q_output),
            reinterpret_cast<uint64_t>(device_k_output),
            reinterpret_cast<uint64_t>(device_v_output),
            rows,
            hidden,
            q_out,
            kv_out,
            blocksize,
            epsilon,
            reinterpret_cast<uint64_t>(stream)) != 0) {
        std::cerr << "NF4 fused QKV failed: " << kudacuda_last_error(handle) << '\n';
        goto cleanup;
    }
    if (!check_cuda(cudaMemcpyAsync(host_q_output.data(), device_q_output, host_q_output.size() * sizeof(__half), cudaMemcpyDeviceToHost, stream), "copy q output")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(host_k_output.data(), device_k_output, host_k_output.size() * sizeof(__half), cudaMemcpyDeviceToHost, stream), "copy k output")) goto cleanup;
    if (!check_cuda(cudaMemcpyAsync(host_v_output.data(), device_v_output, host_v_output.size() * sizeof(__half), cudaMemcpyDeviceToHost, stream), "copy v output")) goto cleanup;
    if (!check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) goto cleanup;

    for (std::size_t index = 0; index < host_q_output.size(); ++index) {
        if (std::fabs(__half2float(host_q_output[index]) - __half2float(host_q_expected[index])) > tolerance) {
            std::cerr << "NF4 Q mismatch at " << index << '\n';
            goto cleanup;
        }
    }
    for (std::size_t index = 0; index < host_k_output.size(); ++index) {
        if (std::fabs(__half2float(host_k_output[index]) - __half2float(host_k_expected[index])) > tolerance ||
            std::fabs(__half2float(host_v_output[index]) - __half2float(host_v_expected[index])) > tolerance) {
            std::cerr << "NF4 K/V mismatch at " << index << '\n';
            goto cleanup;
        }
    }

    std::cout << "KudaCuda NF4 fused RMSNorm-QKV output matched the independent FP32 reference\n";
    exit_code = 0;

cleanup:
    if (handle != nullptr) kudacuda_destroy(handle);
    if (stream != nullptr) cudaStreamDestroy(stream);
    cudaFree(device_v_output);
    cudaFree(device_k_output);
    cudaFree(device_q_output);
    cudaFree(device_v_absmax);
    cudaFree(device_v_weight);
    cudaFree(device_k_absmax);
    cudaFree(device_k_weight);
    cudaFree(device_q_absmax);
    cudaFree(device_q_weight);
    cudaFree(device_v_bias);
    cudaFree(device_k_bias);
    cudaFree(device_q_bias);
    cudaFree(device_norm_weight);
    cudaFree(device_input);
    return exit_code;
}
