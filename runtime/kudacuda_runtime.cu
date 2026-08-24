#include "kudacuda.h"

#include <cuda_runtime.h>

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
    const bool supported =
        std::strcmp(tactic_name, "vector_add") == 0 ||
        std::strcmp(tactic_name, "rmsnorm") == 0;
    return supported ? 1 : 0;
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
