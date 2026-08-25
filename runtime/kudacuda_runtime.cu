#include "kudacuda.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstring>
#include <new>
#include <string>

namespace kudacuda {

cudaError_t launch_vector_add(
    const float* a,
    const float* b,
    float* output,
    int64_t count,
    cudaStream_t stream
);

extern "C" cudaError_t kudacuda_launch_rmsnorm(
    const float* input,
    const float* weight,
    float* output,
    int64_t rows,
    int64_t hidden,
    float epsilon,
    cudaStream_t stream
);

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
);

extern "C" cudaError_t kudacuda_launch_nf4_fused_rmsnorm_qkv(
    const __half* input,
    const __half* norm_weight,
    const uint8_t* q_weight,
    const float* q_absmax,
    const __half* q_bias,
    const uint8_t* k_weight,
    const float* k_absmax,
    const __half* k_bias,
    const uint8_t* v_weight,
    const float* v_absmax,
    const __half* v_bias,
    __half* q_output,
    __half* k_output,
    __half* v_output,
    int64_t rows,
    int64_t hidden,
    int64_t q_out,
    int64_t kv_out,
    int64_t blocksize,
    float epsilon,
    cudaStream_t stream
);

extern "C" cudaError_t kudacuda_launch_nf4_tiled_rmsnorm_qkv(
    const __half* input,
    const __half* norm_weight,
    const uint8_t* q_weight,
    const float* q_absmax,
    const __half* q_bias,
    const uint8_t* k_weight,
    const float* k_absmax,
    const __half* k_bias,
    const uint8_t* v_weight,
    const float* v_absmax,
    const __half* v_bias,
    float* normalized,
    __half* q_output,
    __half* k_output,
    __half* v_output,
    int64_t rows,
    int64_t hidden,
    int64_t q_out,
    int64_t kv_out,
    int64_t blocksize,
    float epsilon,
    cudaStream_t stream
);

extern "C" cudaError_t kudacuda_launch_nf4_tiled_rmsnorm_qkv_v3(
    const __half* input,
    const __half* norm_weight,
    const uint8_t* q_weight,
    const float* q_absmax,
    const __half* q_bias,
    const uint8_t* k_weight,
    const float* k_absmax,
    const __half* k_bias,
    const uint8_t* v_weight,
    const float* v_absmax,
    const __half* v_bias,
    float* normalized,
    __half* q_output,
    __half* k_output,
    __half* v_output,
    int64_t rows,
    int64_t hidden,
    int64_t q_out,
    int64_t kv_out,
    int64_t blocksize,
    float epsilon,
    cudaStream_t stream
);

extern "C" cudaError_t kudacuda_launch_nf4_swiglu(
    const __half* input,
    const __half* norm_weight,
    const uint8_t* gate_weight,
    const float* gate_absmax,
    const uint8_t* up_weight,
    const float* up_absmax,
    __half* output,
    int64_t rows,
    int64_t hidden,
    int64_t intermediate,
    int64_t blocksize,
    float epsilon,
    cudaStream_t stream
);

extern "C" cudaError_t kudacuda_launch_qwen2_gqa_decode_workspace_v3(
    const __half* query,
    const __half* key_cache,
    const __half* value_cache,
    __half* output,
    float* workspace,
    const int32_t* kv_len_device,
    int64_t batch,
    int64_t query_len,
    int64_t kv_capacity,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    cudaStream_t stream
);

extern "C" cudaError_t kudacuda_launch_qwen2_gqa_decode(
    const __half* query,
    const __half* key_cache,
    const __half* value_cache,
    __half* output,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_len,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    cudaStream_t stream
);

extern "C" cudaError_t kudacuda_launch_qwen2_gqa_decode_graph(
    const __half* query,
    const __half* key_cache,
    const __half* value_cache,
    __half* output,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_capacity,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    const int32_t* kv_len_device,
    cudaStream_t stream
);

extern "C" cudaError_t kudacuda_launch_qwen2_gqa_decode_group_tiled(
    const __half* query,
    const __half* key_cache,
    const __half* value_cache,
    __half* output,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_len,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    cudaStream_t stream
);

extern "C" cudaError_t kudacuda_launch_qwen2_gqa_decode_vectorized(
    const __half* query,
    const __half* key_cache,
    const __half* value_cache,
    __half* output,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_len,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    cudaStream_t stream
);

struct Handle {
    explicit Handle(int device_id_value)
        : device_id(device_id_value) {}

    ~Handle() {
        if (graph_exec != nullptr) {
            cudaGraphExecDestroy(graph_exec);
        }
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
        }
    }

    void set_error(const std::string& message) {
        last_error = message;
    }

    void set_cuda_error(const char* operation, cudaError_t status) {
        last_error = std::string(operation) + ": " + cudaGetErrorString(status);
    }

    int device_id;
    bool device_ready = false;
    bool frozen = false;
    bool capturing = false;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    std::string last_error;
};

bool is_valid_handle(const Handle* handle) {
    return handle != nullptr;
}

}  // namespace kudacuda

extern "C" {

void* kudacuda_create(int device_id) {
    auto* handle = new (std::nothrow) kudacuda::Handle(device_id);
    if (handle == nullptr) {
        return nullptr;
    }

    const cudaError_t status = cudaSetDevice(device_id);
    if (status != cudaSuccess) {
        handle->set_cuda_error("cudaSetDevice", status);
        return handle;
    }

    handle->device_ready = true;
    return handle;
}

int kudacuda_freeze(void* raw_handle) {
    auto* handle = reinterpret_cast<kudacuda::Handle*>(raw_handle);
    if (!kudacuda::is_valid_handle(handle)) {
        return 1;
    }
    if (!handle->device_ready) {
        if (handle->last_error.empty()) {
            handle->set_error("CUDA device is not ready");
        }
        return 1;
    }
    handle->frozen = true;
    handle->last_error.clear();
    return 0;
}

int kudacuda_can_dispatch(
    void* raw_handle,
    const char* tactic_name,
    int64_t batch_size,
    int64_t seq_len
) {
    const auto* handle = reinterpret_cast<const kudacuda::Handle*>(raw_handle);
    if (!kudacuda::is_valid_handle(handle) || !handle->frozen || tactic_name == nullptr) {
        return 0;
    }
    if (batch_size <= 0 || seq_len <= 0) {
        return 0;
    }
    const bool qwen_tactic =
        std::strcmp(tactic_name, "qwen2_gqa_decode_sm86_fp16") == 0 ||
        std::strcmp(tactic_name, "qwen2_gqa_decode_sm89_fp16") == 0 ||
        std::strcmp(tactic_name, "qwen2_gqa_decode_sm86_fp16_group_tiled_v2") == 0 ||
        std::strcmp(tactic_name, "qwen2_gqa_decode_sm89_fp16_group_tiled_v2") == 0 ||
        std::strcmp(tactic_name, "qwen2_gqa_decode_sm86_fp16_vectorized_v2") == 0 ||
        std::strcmp(tactic_name, "qwen2_gqa_decode_sm89_fp16_vectorized_v2") == 0 ||
        std::strcmp(tactic_name, "qwen2_gqa_decode_sm86_fp16_graph_v1") == 0 ||
        std::strcmp(tactic_name, "qwen2_gqa_decode_sm89_fp16_graph_v1") == 0;
    if (qwen_tactic) {
        return batch_size == 1 && seq_len == 1 ? 1 : 0;
    }
    return std::strcmp(tactic_name, "vector_add") == 0 ||
                   std::strcmp(tactic_name, "rmsnorm") == 0
               ? 1
               : 0;
}

int kudacuda_vector_add(
    void* raw_handle,
    uint64_t a,
    uint64_t b,
    uint64_t output,
    int64_t count,
    uint64_t stream
) {
    auto* handle = reinterpret_cast<kudacuda::Handle*>(raw_handle);
    if (!kudacuda::is_valid_handle(handle)) {
        return 1;
    }
    if (!handle->frozen) {
        handle->set_error("kudacuda_vector_add called before kudacuda_freeze");
        return 1;
    }
    if (a == 0 || b == 0 || output == 0) {
        handle->set_error("vector_add requires non-null device pointers");
        return 1;
    }
    if (count <= 0) {
        handle->set_error("vector_add count must be positive");
        return 1;
    }

    const auto status = kudacuda::launch_vector_add(
        reinterpret_cast<const float*>(a),
        reinterpret_cast<const float*>(b),
        reinterpret_cast<float*>(output),
        count,
        reinterpret_cast<cudaStream_t>(stream)
    );
    if (status != cudaSuccess) {
        handle->set_cuda_error("vector_add_kernel", status);
        return 1;
    }
    handle->last_error.clear();
    return 0;
}

int kudacuda_rmsnorm(
    void* raw_handle,
    uint64_t input,
    uint64_t weight,
    uint64_t output,
    int64_t rows,
    int64_t hidden,
    float epsilon,
    uint64_t stream
) {
    auto* handle = reinterpret_cast<kudacuda::Handle*>(raw_handle);
    if (!kudacuda::is_valid_handle(handle)) {
        return 1;
    }
    if (!handle->frozen) {
        handle->set_error("kudacuda_rmsnorm called before kudacuda_freeze");
        return 1;
    }
    if (input == 0 || weight == 0 || output == 0) {
        handle->set_error("rmsnorm requires non-null device pointers");
        return 1;
    }
    if (rows <= 0 || hidden <= 0) {
        handle->set_error("rmsnorm rows and hidden must be positive");
        return 1;
    }
    if (!(epsilon > 0.0F)) {
        handle->set_error("rmsnorm epsilon must be positive");
        return 1;
    }

    const auto status = kudacuda::kudacuda_launch_rmsnorm(
        reinterpret_cast<const float*>(input),
        reinterpret_cast<const float*>(weight),
        reinterpret_cast<float*>(output),
        rows,
        hidden,
        epsilon,
        reinterpret_cast<cudaStream_t>(stream)
    );
    if (status != cudaSuccess) {
        handle->set_cuda_error("rmsnorm_kernel", status);
        return 1;
    }
    handle->last_error.clear();
    return 0;
}

int kudacuda_fused_rmsnorm_qkv(
    void* raw_handle,
    uint64_t input,
    uint64_t norm_weight,
    uint64_t qkv_weight,
    uint64_t bias,
    uint64_t output,
    int64_t rows,
    int64_t hidden,
    float epsilon,
    uint64_t stream
) {
    auto* handle = reinterpret_cast<kudacuda::Handle*>(raw_handle);
    if (!kudacuda::is_valid_handle(handle)) {
        return 1;
    }
    if (!handle->frozen) {
        handle->set_error("kudacuda_fused_rmsnorm_qkv called before kudacuda_freeze");
        return 1;
    }
    if (input == 0 || norm_weight == 0 || qkv_weight == 0 || output == 0) {
        handle->set_error("fused_rmsnorm_qkv requires non-null input, weights, and output pointers");
        return 1;
    }
    if (rows <= 0 || hidden <= 0 || !(epsilon > 0.0F)) {
        handle->set_error("fused_rmsnorm_qkv requires positive rows, hidden, and epsilon");
        return 1;
    }

    const auto status = kudacuda::kudacuda_launch_fused_rmsnorm_qkv(
        reinterpret_cast<const __half*>(input),
        reinterpret_cast<const __half*>(norm_weight),
        reinterpret_cast<const __half*>(qkv_weight),
        reinterpret_cast<const __half*>(bias),
        reinterpret_cast<__half*>(output),
        rows,
        hidden,
        epsilon,
        reinterpret_cast<cudaStream_t>(stream)
    );
    if (status != cudaSuccess) {
        handle->set_cuda_error("fused_rmsnorm_qkv_kernel", status);
        return 1;
    }
    handle->last_error.clear();
    return 0;
}

int kudacuda_qwen2_fused_rmsnorm_qkv_nf4(
    void* raw_handle,
    uint64_t input,
    uint64_t norm_weight,
    uint64_t q_weight_packed,
    uint64_t q_absmax,
    uint64_t q_bias,
    uint64_t k_weight_packed,
    uint64_t k_absmax,
    uint64_t k_bias,
    uint64_t v_weight_packed,
    uint64_t v_absmax,
    uint64_t v_bias,
    uint64_t q_output,
    uint64_t k_output,
    uint64_t v_output,
    int64_t rows,
    int64_t hidden,
    int64_t q_out,
    int64_t kv_out,
    int64_t blocksize,
    float epsilon,
    uint64_t stream
) {
    auto* handle = reinterpret_cast<kudacuda::Handle*>(raw_handle);
    if (!kudacuda::is_valid_handle(handle)) {
        return 1;
    }
    if (!handle->frozen) {
        handle->set_error("kudacuda_qwen2_fused_rmsnorm_qkv_nf4 called before kudacuda_freeze");
        return 1;
    }
    if (input == 0 || norm_weight == 0 || q_weight_packed == 0 || q_absmax == 0 || q_bias == 0 ||
        k_weight_packed == 0 || k_absmax == 0 || k_bias == 0 ||
        v_weight_packed == 0 || v_absmax == 0 || v_bias == 0 ||
        q_output == 0 || k_output == 0 || v_output == 0) {
        handle->set_error("qwen2_fused_rmsnorm_qkv_nf4 requires non-null device pointers");
        return 1;
    }
    if (rows != 1 || hidden != 2048 || q_out != 2048 || kv_out != 256 || blocksize != 64 ||
        !(epsilon > 0.0F)) {
        handle->set_error("qwen2_fused_rmsnorm_qkv_nf4 requires rows=1, hidden=2048, q_out=2048, kv_out=256, blocksize=64, and positive epsilon");
        return 1;
    }

    const auto status = kudacuda::kudacuda_launch_nf4_fused_rmsnorm_qkv(
        reinterpret_cast<const __half*>(input),
        reinterpret_cast<const __half*>(norm_weight),
        reinterpret_cast<const uint8_t*>(q_weight_packed),
        reinterpret_cast<const float*>(q_absmax),
        reinterpret_cast<const __half*>(q_bias),
        reinterpret_cast<const uint8_t*>(k_weight_packed),
        reinterpret_cast<const float*>(k_absmax),
        reinterpret_cast<const __half*>(k_bias),
        reinterpret_cast<const uint8_t*>(v_weight_packed),
        reinterpret_cast<const float*>(v_absmax),
        reinterpret_cast<const __half*>(v_bias),
        reinterpret_cast<__half*>(q_output),
        reinterpret_cast<__half*>(k_output),
        reinterpret_cast<__half*>(v_output),
        rows,
        hidden,
        q_out,
        kv_out,
        blocksize,
        epsilon,
        reinterpret_cast<cudaStream_t>(stream)
    );
    if (status != cudaSuccess) {
        handle->set_cuda_error("qwen2_fused_rmsnorm_qkv_nf4_kernel", status);
        return 1;
    }
    handle->last_error.clear();
    return 0;
}

int kudacuda_qwen2_fused_rmsnorm_qkv_nf4_v2(
    void* raw_handle,
    uint64_t input,
    uint64_t norm_weight,
    uint64_t q_weight_packed,
    uint64_t q_absmax,
    uint64_t q_bias,
    uint64_t k_weight_packed,
    uint64_t k_absmax,
    uint64_t k_bias,
    uint64_t v_weight_packed,
    uint64_t v_absmax,
    uint64_t v_bias,
    uint64_t normalized_workspace,
    uint64_t q_output,
    uint64_t k_output,
    uint64_t v_output,
    int64_t rows,
    int64_t hidden,
    int64_t q_out,
    int64_t kv_out,
    int64_t blocksize,
    float epsilon,
    uint64_t stream
) {
    auto* handle = reinterpret_cast<kudacuda::Handle*>(raw_handle);
    if (!kudacuda::is_valid_handle(handle)) {
        return 1;
    }
    if (!handle->frozen) {
        handle->set_error("kudacuda_qwen2_fused_rmsnorm_qkv_nf4_v2 called before kudacuda_freeze");
        return 1;
    }
    if (input == 0 || norm_weight == 0 || q_weight_packed == 0 || q_absmax == 0 || q_bias == 0 ||
        k_weight_packed == 0 || k_absmax == 0 || k_bias == 0 || v_weight_packed == 0 ||
        v_absmax == 0 || v_bias == 0 || normalized_workspace == 0 || q_output == 0 ||
        k_output == 0 || v_output == 0) {
        handle->set_error("qwen2_fused_rmsnorm_qkv_nf4_v2 requires non-null device pointers");
        return 1;
    }
    if (rows != 1 || hidden != 2048 || q_out != 2048 || kv_out != 256 || blocksize != 64 ||
        !(epsilon > 0.0F)) {
        handle->set_error("qwen2_fused_rmsnorm_qkv_nf4_v2 requires rows=1, hidden=2048, q_out=2048, kv_out=256, blocksize=64, and positive epsilon");
        return 1;
    }

    const auto status = kudacuda::kudacuda_launch_nf4_tiled_rmsnorm_qkv(
        reinterpret_cast<const __half*>(input),
        reinterpret_cast<const __half*>(norm_weight),
        reinterpret_cast<const uint8_t*>(q_weight_packed),
        reinterpret_cast<const float*>(q_absmax),
        reinterpret_cast<const __half*>(q_bias),
        reinterpret_cast<const uint8_t*>(k_weight_packed),
        reinterpret_cast<const float*>(k_absmax),
        reinterpret_cast<const __half*>(k_bias),
        reinterpret_cast<const uint8_t*>(v_weight_packed),
        reinterpret_cast<const float*>(v_absmax),
        reinterpret_cast<const __half*>(v_bias),
        reinterpret_cast<float*>(normalized_workspace),
        reinterpret_cast<__half*>(q_output),
        reinterpret_cast<__half*>(k_output),
        reinterpret_cast<__half*>(v_output),
        rows,
        hidden,
        q_out,
        kv_out,
        blocksize,
        epsilon,
        reinterpret_cast<cudaStream_t>(stream)
    );
    if (status != cudaSuccess) {
        handle->set_cuda_error("nf4_tiled_rmsnorm_qkv", status);
        return 1;
    }
    handle->last_error.clear();
    return 0;
}

int kudacuda_qwen2_fused_rmsnorm_qkv_nf4_v3(
    void* raw_handle,
    uint64_t input,
    uint64_t norm_weight,
    uint64_t q_weight_packed,
    uint64_t q_absmax,
    uint64_t q_bias,
    uint64_t k_weight_packed,
    uint64_t k_absmax,
    uint64_t k_bias,
    uint64_t v_weight_packed,
    uint64_t v_absmax,
    uint64_t v_bias,
    uint64_t normalized_workspace,
    uint64_t q_output,
    uint64_t k_output,
    uint64_t v_output,
    int64_t rows,
    int64_t hidden,
    int64_t q_out,
    int64_t kv_out,
    int64_t blocksize,
    float epsilon,
    uint64_t stream
) {
    auto* handle = reinterpret_cast<kudacuda::Handle*>(raw_handle);
    if (!kudacuda::is_valid_handle(handle)) {
        return 1;
    }
    if (!handle->frozen) {
        handle->set_error("kudacuda_qwen2_fused_rmsnorm_qkv_nf4_v3 called before kudacuda_freeze");
        return 1;
    }
    if (input == 0 || norm_weight == 0 || q_weight_packed == 0 || q_absmax == 0 || q_bias == 0 ||
        k_weight_packed == 0 || k_absmax == 0 || k_bias == 0 || v_weight_packed == 0 ||
        v_absmax == 0 || v_bias == 0 || normalized_workspace == 0 || q_output == 0 ||
        k_output == 0 || v_output == 0) {
        handle->set_error("qwen2_fused_rmsnorm_qkv_nf4_v3 requires non-null device pointers");
        return 1;
    }
    if (rows != 1 || hidden != 2048 || q_out != 2048 || kv_out != 256 || blocksize != 64 ||
        !(epsilon > 0.0F)) {
        handle->set_error("qwen2_fused_rmsnorm_qkv_nf4_v3 requires rows=1, hidden=2048, q_out=2048, kv_out=256, blocksize=64, and positive epsilon");
        return 1;
    }

    const auto status = kudacuda::kudacuda_launch_nf4_tiled_rmsnorm_qkv_v3(
        reinterpret_cast<const __half*>(input),
        reinterpret_cast<const __half*>(norm_weight),
        reinterpret_cast<const uint8_t*>(q_weight_packed),
        reinterpret_cast<const float*>(q_absmax),
        reinterpret_cast<const __half*>(q_bias),
        reinterpret_cast<const uint8_t*>(k_weight_packed),
        reinterpret_cast<const float*>(k_absmax),
        reinterpret_cast<const __half*>(k_bias),
        reinterpret_cast<const uint8_t*>(v_weight_packed),
        reinterpret_cast<const float*>(v_absmax),
        reinterpret_cast<const __half*>(v_bias),
        reinterpret_cast<float*>(normalized_workspace),
        reinterpret_cast<__half*>(q_output),
        reinterpret_cast<__half*>(k_output),
        reinterpret_cast<__half*>(v_output),
        rows,
        hidden,
        q_out,
        kv_out,
        blocksize,
        epsilon,
        reinterpret_cast<cudaStream_t>(stream)
    );
    if (status != cudaSuccess) {
        handle->set_cuda_error("nf4_tiled_rmsnorm_qkv_v3", status);
        return 1;
    }
    handle->last_error.clear();
    return 0;
}

int kudacuda_qwen2_swiglu_nf4(
    void* raw_handle,
    uint64_t input,
    uint64_t norm_weight,
    uint64_t gate_weight_packed,
    uint64_t gate_absmax,
    uint64_t up_weight_packed,
    uint64_t up_absmax,
    uint64_t output,
    int64_t rows,
    int64_t hidden,
    int64_t intermediate,
    int64_t blocksize,
    float epsilon,
    uint64_t stream
) {
    auto* handle = reinterpret_cast<kudacuda::Handle*>(raw_handle);
    if (!kudacuda::is_valid_handle(handle)) {
        return 1;
    }
    if (!handle->frozen) {
        handle->set_error("kudacuda_qwen2_swiglu_nf4 called before kudacuda_freeze");
        return 1;
    }
    if (input == 0 || norm_weight == 0 || gate_weight_packed == 0 || gate_absmax == 0 ||
        up_weight_packed == 0 || up_absmax == 0 || output == 0) {
        handle->set_error("qwen2_swiglu_nf4 requires non-null device pointers");
        return 1;
    }
    if (rows != 1 || hidden != 2048 || intermediate != 11008 || blocksize != 64 || !(epsilon > 0.0F)) {
        handle->set_error("qwen2_swiglu_nf4 requires rows=1, hidden=2048, intermediate=11008, blocksize=64, and positive epsilon");
        return 1;
    }

    const auto status = kudacuda::kudacuda_launch_nf4_swiglu(
        reinterpret_cast<const __half*>(input),
        reinterpret_cast<const __half*>(norm_weight),
        reinterpret_cast<const uint8_t*>(gate_weight_packed),
        reinterpret_cast<const float*>(gate_absmax),
        reinterpret_cast<const uint8_t*>(up_weight_packed),
        reinterpret_cast<const float*>(up_absmax),
        reinterpret_cast<__half*>(output),
        rows,
        hidden,
        intermediate,
        blocksize,
        epsilon,
        reinterpret_cast<cudaStream_t>(stream)
    );
    if (status != cudaSuccess) {
        handle->set_cuda_error("nf4_swiglu", status);
        return 1;
    }
    return 0;
}

int kudacuda_qwen2_gqa_decode_workspace_v3(
    void* raw_handle,
    uint64_t query,
    uint64_t key_cache,
    uint64_t value_cache,
    uint64_t output,
    uint64_t workspace,
    uint64_t kv_len_device,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_capacity,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    uint64_t stream
) {
    auto* handle = reinterpret_cast<kudacuda::Handle*>(raw_handle);
    if (!kudacuda::is_valid_handle(handle)) {
        return 1;
    }
    if (!handle->frozen) {
        handle->set_error("kudacuda_qwen2_gqa_decode_workspace_v3 called before kudacuda_freeze");
        return 1;
    }
    if (query == 0 || key_cache == 0 || value_cache == 0 || output == 0 || workspace == 0 || kv_len_device == 0) {
        handle->set_error("qwen2_gqa_decode_workspace_v3 requires non-null device pointers");
        return 1;
    }
    if (batch_size != 1 || query_len != 1 || kv_capacity != 512 || query_heads != 16 || kv_heads != 2 ||
        head_dim != 128 || !(scale > 0.0F)) {
        handle->set_error("qwen2_gqa_decode_workspace_v3 requires 1x1, capacity=512, heads=16/2, head_dim=128, and positive scale");
        return 1;
    }
    const auto status = kudacuda::kudacuda_launch_qwen2_gqa_decode_workspace_v3(
        reinterpret_cast<const __half*>(query),
        reinterpret_cast<const __half*>(key_cache),
        reinterpret_cast<const __half*>(value_cache),
        reinterpret_cast<__half*>(output),
        reinterpret_cast<float*>(workspace),
        reinterpret_cast<const int32_t*>(kv_len_device),
        batch_size,
        query_len,
        kv_capacity,
        query_heads,
        kv_heads,
        head_dim,
        scale,
        reinterpret_cast<cudaStream_t>(stream)
    );
    if (status != cudaSuccess) {
        handle->set_cuda_error("qwen2_gqa_decode_workspace_v3_kernel", status);
        return 1;
    }
    handle->last_error.clear();
    return 0;
}

int kudacuda_qwen2_gqa_decode(
    void* raw_handle,
    uint64_t query,
    uint64_t key_cache,
    uint64_t value_cache,
    uint64_t output,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_len,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    uint64_t stream
) {
    auto* handle = reinterpret_cast<kudacuda::Handle*>(raw_handle);
    if (!kudacuda::is_valid_handle(handle)) {
        return 1;
    }
    if (!handle->frozen) {
        handle->set_error("kudacuda_qwen2_gqa_decode called before kudacuda_freeze");
        return 1;
    }
    if (query == 0 || key_cache == 0 || value_cache == 0 || output == 0) {
        handle->set_error("qwen2_gqa_decode requires non-null device pointers");
        return 1;
    }
    if (batch_size != 1 || query_len != 1 || kv_len < 1 || kv_len > 512 ||
        query_heads != 16 || kv_heads != 2 || head_dim != 128) {
        handle->set_error("qwen2_gqa_decode requires batch=1, query_len=1, kv_len=1..512, q_heads=16, kv_heads=2, head_dim=128");
        return 1;
    }
    if (!(scale > 0.0F)) {
        handle->set_error("qwen2_gqa_decode scale must be positive");
        return 1;
    }

    const auto status = kudacuda::kudacuda_launch_qwen2_gqa_decode(
        reinterpret_cast<const __half*>(query),
        reinterpret_cast<const __half*>(key_cache),
        reinterpret_cast<const __half*>(value_cache),
        reinterpret_cast<__half*>(output),
        batch_size,
        query_len,
        kv_len,
        query_heads,
        kv_heads,
        head_dim,
        scale,
        reinterpret_cast<cudaStream_t>(stream)
    );
    if (status != cudaSuccess) {
        handle->set_cuda_error("qwen2_gqa_decode_kernel", status);
        return 1;
    }
    handle->last_error.clear();
    return 0;
}

int kudacuda_qwen2_gqa_decode_graph(
    void* raw_handle,
    uint64_t query,
    uint64_t key_cache,
    uint64_t value_cache,
    uint64_t output,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_capacity,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    uint64_t kv_len_device,
    uint64_t stream
) {
    auto* handle = reinterpret_cast<kudacuda::Handle*>(raw_handle);
    if (!kudacuda::is_valid_handle(handle)) {
        return 1;
    }
    if (!handle->frozen) {
        handle->set_error("kudacuda_qwen2_gqa_decode_graph called before kudacuda_freeze");
        return 1;
    }
    if (query == 0 || key_cache == 0 || value_cache == 0 || output == 0 || kv_len_device == 0) {
        handle->set_error("qwen2_gqa_decode_graph requires non-null device pointers");
        return 1;
    }
    if (batch_size != 1 || query_len != 1 || kv_capacity < 1 || kv_capacity > 512 ||
        query_heads != 16 || kv_heads != 2 || head_dim != 128) {
        handle->set_error("qwen2_gqa_decode_graph requires batch=1, query_len=1, kv_capacity=1..512, q_heads=16, kv_heads=2, head_dim=128");
        return 1;
    }
    if (!(scale > 0.0F)) {
        handle->set_error("qwen2_gqa_decode_graph scale must be positive");
        return 1;
    }

    const auto status = kudacuda::kudacuda_launch_qwen2_gqa_decode_graph(
        reinterpret_cast<const __half*>(query),
        reinterpret_cast<const __half*>(key_cache),
        reinterpret_cast<const __half*>(value_cache),
        reinterpret_cast<__half*>(output),
        batch_size,
        query_len,
        kv_capacity,
        query_heads,
        kv_heads,
        head_dim,
        scale,
        reinterpret_cast<const int32_t*>(kv_len_device),
        reinterpret_cast<cudaStream_t>(stream)
    );
    if (status != cudaSuccess) {
        handle->set_cuda_error("qwen2_gqa_decode_graph_kernel", status);
        return 1;
    }
    handle->last_error.clear();
    return 0;
}

int kudacuda_qwen2_gqa_decode_group_tiled(
    void* raw_handle,
    uint64_t query,
    uint64_t key_cache,
    uint64_t value_cache,
    uint64_t output,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_len,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    uint64_t stream
) {
    auto* handle = reinterpret_cast<kudacuda::Handle*>(raw_handle);
    if (!kudacuda::is_valid_handle(handle)) {
        return 1;
    }
    if (!handle->frozen) {
        handle->set_error("kudacuda_qwen2_gqa_decode_group_tiled called before kudacuda_freeze");
        return 1;
    }
    if (query == 0 || key_cache == 0 || value_cache == 0 || output == 0) {
        handle->set_error("qwen2_gqa_decode_group_tiled requires non-null device pointers");
        return 1;
    }
    if (batch_size != 1 || query_len != 1 || kv_len < 1 || kv_len > 512 ||
        query_heads != 16 || kv_heads != 2 || head_dim != 128) {
        handle->set_error("qwen2_gqa_decode_group_tiled requires batch=1, query_len=1, kv_len=1..512, q_heads=16, kv_heads=2, head_dim=128");
        return 1;
    }
    if (!(scale > 0.0F)) {
        handle->set_error("qwen2_gqa_decode_group_tiled scale must be positive");
        return 1;
    }

    const auto status = kudacuda::kudacuda_launch_qwen2_gqa_decode_group_tiled(
        reinterpret_cast<const __half*>(query),
        reinterpret_cast<const __half*>(key_cache),
        reinterpret_cast<const __half*>(value_cache),
        reinterpret_cast<__half*>(output),
        batch_size,
        query_len,
        kv_len,
        query_heads,
        kv_heads,
        head_dim,
        scale,
        reinterpret_cast<cudaStream_t>(stream)
    );
    if (status != cudaSuccess) {
        handle->set_cuda_error("qwen2_gqa_decode_group_tiled_kernel", status);
        return 1;
    }
    handle->last_error.clear();
    return 0;
}

int kudacuda_qwen2_gqa_decode_vectorized(
    void* raw_handle,
    uint64_t query,
    uint64_t key_cache,
    uint64_t value_cache,
    uint64_t output,
    int64_t batch_size,
    int64_t query_len,
    int64_t kv_len,
    int64_t query_heads,
    int64_t kv_heads,
    int64_t head_dim,
    float scale,
    uint64_t stream
) {
    auto* handle = reinterpret_cast<kudacuda::Handle*>(raw_handle);
    if (!kudacuda::is_valid_handle(handle)) {
        return 1;
    }
    if (!handle->frozen) {
        handle->set_error("kudacuda_qwen2_gqa_decode_vectorized called before kudacuda_freeze");
        return 1;
    }
    if (query == 0 || key_cache == 0 || value_cache == 0 || output == 0) {
        handle->set_error("qwen2_gqa_decode_vectorized requires non-null device pointers");
        return 1;
    }
    if (batch_size != 1 || query_len != 1 || kv_len < 1 || kv_len > 512 ||
        query_heads != 16 || kv_heads != 2 || head_dim != 128) {
        handle->set_error("qwen2_gqa_decode_vectorized requires batch=1, query_len=1, kv_len=1..512, q_heads=16, kv_heads=2, head_dim=128");
        return 1;
    }
    if (!(scale > 0.0F)) {
        handle->set_error("qwen2_gqa_decode_vectorized scale must be positive");
        return 1;
    }

    const auto status = kudacuda::kudacuda_launch_qwen2_gqa_decode_vectorized(
        reinterpret_cast<const __half*>(query),
        reinterpret_cast<const __half*>(key_cache),
        reinterpret_cast<const __half*>(value_cache),
        reinterpret_cast<__half*>(output),
        batch_size,
        query_len,
        kv_len,
        query_heads,
        kv_heads,
        head_dim,
        scale,
        reinterpret_cast<cudaStream_t>(stream)
    );
    if (status != cudaSuccess) {
        handle->set_cuda_error("qwen2_gqa_decode_vectorized_kernel", status);
        return 1;
    }
    handle->last_error.clear();
    return 0;
}

int kudacuda_begin_capture(void* raw_handle, uint64_t stream) {
    auto* handle = reinterpret_cast<kudacuda::Handle*>(raw_handle);
    if (!kudacuda::is_valid_handle(handle) || !handle->frozen) {
        if (handle != nullptr) {
            handle->set_error("kudacuda_begin_capture requires a frozen handle");
        }
        return 1;
    }
    if (handle->capturing) {
        handle->set_error("CUDA graph capture is already active");
        return 1;
    }
    const auto status = cudaStreamBeginCapture(
        reinterpret_cast<cudaStream_t>(stream),
        cudaStreamCaptureModeGlobal
    );
    if (status != cudaSuccess) {
        handle->set_cuda_error("cudaStreamBeginCapture", status);
        return 1;
    }
    handle->capturing = true;
    handle->last_error.clear();
    return 0;
}

int kudacuda_end_capture(void* raw_handle, uint64_t stream) {
    auto* handle = reinterpret_cast<kudacuda::Handle*>(raw_handle);
    if (!kudacuda::is_valid_handle(handle) || !handle->capturing) {
        if (handle != nullptr) {
            handle->set_error("kudacuda_end_capture called without active capture");
        }
        return 1;
    }
    const auto end_status = cudaStreamEndCapture(
        reinterpret_cast<cudaStream_t>(stream),
        &handle->graph
    );
    handle->capturing = false;
    if (end_status != cudaSuccess) {
        handle->set_cuda_error("cudaStreamEndCapture", end_status);
        return 1;
    }
    const auto instantiate_status = cudaGraphInstantiate(
        &handle->graph_exec,
        handle->graph,
        nullptr,
        nullptr,
        0
    );
    if (instantiate_status != cudaSuccess) {
        handle->set_cuda_error("cudaGraphInstantiate", instantiate_status);
        cudaGraphDestroy(handle->graph);
        handle->graph = nullptr;
        return 1;
    }
    cudaGraphDestroy(handle->graph);
    handle->graph = nullptr;
    handle->last_error.clear();
    return 0;
}

int kudacuda_replay_graph(void* raw_handle, uint64_t stream) {
    auto* handle = reinterpret_cast<kudacuda::Handle*>(raw_handle);
    if (!kudacuda::is_valid_handle(handle) || handle->graph_exec == nullptr) {
        if (handle != nullptr) {
            handle->set_error("kudacuda_replay_graph called before graph instantiation");
        }
        return 1;
    }
    const auto status = cudaGraphLaunch(
        handle->graph_exec,
        reinterpret_cast<cudaStream_t>(stream)
    );
    if (status != cudaSuccess) {
        handle->set_cuda_error("cudaGraphLaunch", status);
        return 1;
    }
    handle->last_error.clear();
    return 0;
}

int kudacuda_reset_graph(void* raw_handle) {
    auto* handle = reinterpret_cast<kudacuda::Handle*>(raw_handle);
    if (!kudacuda::is_valid_handle(handle)) {
        return 1;
    }
    if (handle->capturing) {
        handle->set_error("kudacuda_reset_graph called during active capture");
        return 1;
    }
    if (handle->graph_exec != nullptr) {
        const auto status = cudaGraphExecDestroy(handle->graph_exec);
        if (status != cudaSuccess) {
            handle->set_cuda_error("cudaGraphExecDestroy", status);
            return 1;
        }
        handle->graph_exec = nullptr;
    }
    if (handle->graph != nullptr) {
        const auto status = cudaGraphDestroy(handle->graph);
        if (status != cudaSuccess) {
            handle->set_cuda_error("cudaGraphDestroy", status);
            return 1;
        }
        handle->graph = nullptr;
    }
    handle->last_error.clear();
    return 0;
}

const char* kudacuda_last_error(void* raw_handle) {
    const auto* handle = reinterpret_cast<const kudacuda::Handle*>(raw_handle);
    if (!kudacuda::is_valid_handle(handle)) {
        return "invalid KudaCuda handle";
    }
    return handle->last_error.c_str();
}

void kudacuda_destroy(void* raw_handle) {
    delete reinterpret_cast<kudacuda::Handle*>(raw_handle);
}

}  // extern "C"
