#include "kudacuda_registry.h"

#include <sstream>

namespace kudacuda {

namespace {

int set_error(std::string* error, const char* operation, CUresult result) {
    if (error == nullptr) {
        return 1;
    }
    const char* name = nullptr;
    const char* message = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &message);
    std::ostringstream stream;
    stream << operation << " failed: " << (name != nullptr ? name : "CUDA_ERROR_UNKNOWN")
           << " (" << (message != nullptr ? message : "unknown") << ")";
    *error = stream.str();
    return 1;
}

int set_text_error(std::string* error, const std::string& message) {
    if (error != nullptr) {
        *error = message;
    }
    return 1;
}

}  // namespace

KudaCudaModuleRegistry::KudaCudaModuleRegistry(int device_id)
    : device_id_(device_id) {
    if (cuInit(0) != CUDA_SUCCESS) {
        return;
    }
    CUdevice device = 0;
    if (cuDeviceGet(&device, device_id_) != CUDA_SUCCESS) {
        return;
    }
    cuDevicePrimaryCtxRetain(&context_, device);
    if (context_ != nullptr) {
        cuCtxSetCurrent(context_);
    }
}

KudaCudaModuleRegistry::~KudaCudaModuleRegistry() {
    for (const auto& entry : modules_) {
        if (entry.second != nullptr) {
            cuModuleUnload(entry.second);
        }
    }
    if (context_ != nullptr) {
        CUdevice device = 0;
        if (cuDeviceGet(&device, device_id_) == CUDA_SUCCESS) {
            cuDevicePrimaryCtxRelease(device);
        }
    }
}

int KudaCudaModuleRegistry::load(
    const std::string& name,
    const std::string& path,
    std::string* error
) {
    if (frozen_) {
        return set_text_error(error, "module registry is frozen");
    }
    if (context_ == nullptr) {
        return set_text_error(error, "CUDA primary context is unavailable");
    }
    if (modules_.find(name) != modules_.end()) {
        return set_text_error(error, "module already loaded: " + name);
    }
    CUmodule module = nullptr;
    const CUresult result = cuModuleLoad(&module, path.c_str());
    if (result != CUDA_SUCCESS) {
        return set_error(error, "cuModuleLoad", result);
    }
    modules_.emplace(name, module);
    return 0;
}

int KudaCudaModuleRegistry::resolve(
    const std::string& module_name,
    const std::string& symbol,
    CUfunction* function,
    std::string* error
) {
    if (frozen_) {
        return set_text_error(error, "module registry is frozen");
    }
    if (function == nullptr) {
        return set_text_error(error, "function output pointer is null");
    }
    const auto module = modules_.find(module_name);
    if (module == modules_.end()) {
        return set_text_error(error, "module is not loaded: " + module_name);
    }
    const CUresult result = cuModuleGetFunction(function, module->second, symbol.c_str());
    if (result != CUDA_SUCCESS) {
        return set_error(error, "cuModuleGetFunction", result);
    }
    functions_[module_name + "::" + symbol] = *function;
    return 0;
}

int KudaCudaModuleRegistry::freeze(std::string* error) {
    if (context_ == nullptr) {
        return set_text_error(error, "CUDA primary context is unavailable");
    }
    frozen_ = true;
    return 0;
}

}  // namespace kudacuda
