#pragma once

#include <cuda.h>

#include <string>
#include <unordered_map>

namespace kudacuda {

struct KudaCudaKernelDescriptor {
    const char* name;
    const char* symbol;
    unsigned int grid_x;
    unsigned int grid_y;
    unsigned int grid_z;
    unsigned int block_x;
    unsigned int block_y;
    unsigned int block_z;
    unsigned int shared_memory_bytes;
};

class KudaCudaModuleRegistry {
public:
    explicit KudaCudaModuleRegistry(int device_id);
    ~KudaCudaModuleRegistry();

    KudaCudaModuleRegistry(const KudaCudaModuleRegistry&) = delete;
    KudaCudaModuleRegistry& operator=(const KudaCudaModuleRegistry&) = delete;

    int load(const std::string& name, const std::string& path, std::string* error);
    int resolve(const std::string& module_name, const std::string& symbol, CUfunction* function, std::string* error);
    int freeze(std::string* error);
    bool frozen() const { return frozen_; }

private:
    CUcontext context_ = nullptr;
    int device_id_ = 0;
    bool frozen_ = false;
    std::unordered_map<std::string, CUmodule> modules_;
    std::unordered_map<std::string, CUfunction> functions_;
};

}  // namespace kudacuda
