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
        output[static_cast<std::size_t>(index / 2)] = static_cast<uint8_t>((even << 4) | odd);
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
    constexpr int intermediate = 11008;
    constexpr float epsilon = 1.0e-5F;
    constexpr float tolerance = 0.045F;
    const int device_id = argc > 1 ? std::stoi(argv[1]) : 0;
    if (!check(cudaSetDevice(device_id), "cudaSetDevice")) return 1;

    std::vector<__half> input(hidden), norm_weight(hidden), output(intermediate), expected(intermediate);
    std::vector<uint8_t> gate_weight(intermediate * hidden / 2), up_weight(intermediate * hidden / 2);
    std::vector<float> gate_scales(intermediate * hidden / 64), up_scales(intermediate * hidden / 64);
    for (int i = 0; i < hidden; ++i) {
        input[i] = __float2half(std::sin(i * 0.013F));
        norm_weight[i] = __float2half(0.75F + (i % 23) * 0.004F);
    }
    pack(gate_weight, intermediate * hidden, 1);
    pack(up_weight, intermediate * hidden, 7);
    for (std::size_t i = 0; i < gate_scales.size(); ++i) gate_scales[i] = 0.12F + (i % 19) * 0.003F;
    for (std::size_t i = 0; i < up_scales.size(); ++i) up_scales[i] = 0.15F + (i % 17) * 0.004F;

    float sum_squares = 0.0F;
    for (int i = 0; i < hidden; ++i) {
        const float value = __half2float(input[i]);
        sum_squares += value * value;
    }
    const float inverse_rms = 1.0F / std::sqrt(sum_squares / hidden + epsilon);
    std::vector<float> normalized(hidden);
    for (int i = 0; i < hidden; ++i) {
        normalized[i] = __half2float(input[i]) * inverse_rms * __half2float(norm_weight[i]);
    }
    for (int output_index = 0; output_index < intermediate; ++output_index) {
        float gate_sum = 0.0F;
        float up_sum = 0.0F;
        for (int dimension = 0; dimension < hidden; ++dimension) {
            const int index = output_index * hidden + dimension;
            gate_sum += normalized[dimension] * decode(gate_weight, gate_scales, index);
            up_sum += normalized[dimension] * decode(up_weight, up_scales, index);
        }
        const float silu = gate_sum / (1.0F + std::exp(-gate_sum));
        expected[output_index] = __float2half(silu * up_sum);
    }

    __half *d_input = nullptr, *d_norm_weight = nullptr, *d_output = nullptr;
    uint8_t *d_gate_weight = nullptr, *d_up_weight = nullptr;
    float *d_gate_scales = nullptr, *d_up_scales = nullptr;
    cudaStream_t stream = nullptr;
    void* handle = nullptr;
    int exit_code = 1;

    if (!check(cudaMalloc(&d_input, hidden * sizeof(__half)), "cudaMalloc(input)")) goto cleanup;
    if (!check(cudaMalloc(&d_norm_weight, hidden * sizeof(__half)), "cudaMalloc(norm)")) goto cleanup;
    if (!check(cudaMalloc(&d_gate_weight, gate_weight.size()), "cudaMalloc(gate)")) goto cleanup;
    if (!check(cudaMalloc(&d_up_weight, up_weight.size()), "cudaMalloc(up)")) goto cleanup;
    if (!check(cudaMalloc(&d_gate_scales, gate_scales.size() * sizeof(float)), "cudaMalloc(gate scales)")) goto cleanup;
    if (!check(cudaMalloc(&d_up_scales, up_scales.size() * sizeof(float)), "cudaMalloc(up scales)")) goto cleanup;
    if (!check(cudaMalloc(&d_output, output.size() * sizeof(__half)), "cudaMalloc(output)")) goto cleanup;
    if (!check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate")) goto cleanup;
    if (!check(cudaMemcpyAsync(d_input, input.data(), input.size() * sizeof(__half), cudaMemcpyHostToDevice, stream), "copy input")) goto cleanup;
    if (!check(cudaMemcpyAsync(d_norm_weight, norm_weight.data(), norm_weight.size() * sizeof(__half), cudaMemcpyHostToDevice, stream), "copy norm")) goto cleanup;
    if (!check(cudaMemcpyAsync(d_gate_weight, gate_weight.data(), gate_weight.size(), cudaMemcpyHostToDevice, stream), "copy gate")) goto cleanup;
    if (!check(cudaMemcpyAsync(d_up_weight, up_weight.data(), up_weight.size(), cudaMemcpyHostToDevice, stream), "copy up")) goto cleanup;
    if (!check(cudaMemcpyAsync(d_gate_scales, gate_scales.data(), gate_scales.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "copy gate scales")) goto cleanup;
    if (!check(cudaMemcpyAsync(d_up_scales, up_scales.data(), up_scales.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "copy up scales")) goto cleanup;

    handle = kudacuda_create(device_id);
    if (handle == nullptr || kudacuda_freeze(handle) != 0) goto cleanup;
    if (kudacuda_qwen2_swiglu_nf4(
            handle,
            reinterpret_cast<uint64_t>(d_input), reinterpret_cast<uint64_t>(d_norm_weight),
            reinterpret_cast<uint64_t>(d_gate_weight), reinterpret_cast<uint64_t>(d_gate_scales),
            reinterpret_cast<uint64_t>(d_up_weight), reinterpret_cast<uint64_t>(d_up_scales),
            reinterpret_cast<uint64_t>(d_output), 1, hidden, intermediate, 64, epsilon,
            reinterpret_cast<uint64_t>(stream)) != 0) {
        std::cerr << kudacuda_last_error(handle) << '\n';
        goto cleanup;
    }
    if (!check(cudaMemcpyAsync(output.data(), d_output, output.size() * sizeof(__half), cudaMemcpyDeviceToHost, stream), "copy output")) goto cleanup;
    if (!check(cudaStreamSynchronize(stream), "cudaStreamSynchronize")) goto cleanup;
    for (std::size_t i = 0; i < output.size(); ++i) {
        if (std::fabs(__half2float(output[i]) - __half2float(expected[i])) > tolerance) goto cleanup;
    }

    std::cout << "KudaCuda NF4 SwiGLU output matched the independent FP32 reference\n";
    exit_code = 0;

cleanup:
    if (handle != nullptr) kudacuda_destroy(handle);
    if (stream != nullptr) cudaStreamDestroy(stream);
    cudaFree(d_output); cudaFree(d_up_scales); cudaFree(d_gate_scales);
    cudaFree(d_up_weight); cudaFree(d_gate_weight); cudaFree(d_norm_weight); cudaFree(d_input);
    return exit_code;
}
