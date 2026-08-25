cmake_minimum_required(VERSION 3.24)

project(KudaCuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architecture for local RTX 3050 builds")
endif()

if("${CMAKE_CUDA_ARCHITECTURES}" MATCHES "89")
    set(KUDACUDA_RMSNORM_SOURCE kernels/sm89/rmsnorm_sm89.cu)
    set(KUDACUDA_RMSNORM_QKV_SOURCE kernels/sm89/rmsnorm_qkv_sm89.cu)
    set(KUDACUDA_NF4_QKV_SOURCE kernels/sm89/nf4_fused_rmsnorm_qkv_sm89.cu)
    set(KUDACUDA_NF4_TILED_QKV_SOURCE kernels/sm89/nf4_tiled_gemv_sm89.cu)
    set(KUDACUDA_NF4_SWIGLU_SOURCE kernels/sm89/nf4_swiglu_sm89.cu)
    set(KUDACUDA_QWEN_SOURCE kernels/sm89/qwen2_gqa_decode_sm89.cu)
    set(KUDACUDA_QWEN_GRAPH_SOURCE kernels/sm89/qwen2_gqa_graph_sm89.cu)
    set(KUDACUDA_QWEN_GROUP_SOURCE kernels/sm89/qwen2_gqa_group_tiled_sm89.cu)
    set(KUDACUDA_QWEN_VECTORIZED_SOURCE kernels/sm89/qwen2_gqa_vectorized_sm89.cu)
    set(KUDACUDA_QWEN_WORKSPACE_SOURCE kernels/sm89/qwen2_gqa_workspace_sm89.cu)
else()
    set(KUDACUDA_RMSNORM_SOURCE kernels/sm86/rmsnorm_sm86.cu)
    set(KUDACUDA_RMSNORM_QKV_SOURCE kernels/sm86/rmsnorm_qkv_sm86.cu)
    set(KUDACUDA_NF4_QKV_SOURCE kernels/sm86/nf4_fused_rmsnorm_qkv_sm86.cu)
    set(KUDACUDA_NF4_TILED_QKV_SOURCE kernels/sm86/nf4_tiled_gemv_sm86.cu)
    set(KUDACUDA_NF4_SWIGLU_SOURCE kernels/sm86/nf4_swiglu_sm86.cu)
    set(KUDACUDA_QWEN_SOURCE kernels/sm86/qwen2_gqa_decode_sm86.cu)
    set(KUDACUDA_QWEN_GRAPH_SOURCE kernels/sm86/qwen2_gqa_graph_sm86.cu)
    set(KUDACUDA_QWEN_GROUP_SOURCE kernels/sm86/qwen2_gqa_group_tiled_sm86.cu)
    set(KUDACUDA_QWEN_VECTORIZED_SOURCE kernels/sm86/qwen2_gqa_vectorized_sm86.cu)
    set(KUDACUDA_QWEN_WORKSPACE_SOURCE kernels/sm86/qwen2_gqa_workspace_sm86.cu)
endif()

add_library(kudacuda SHARED
    kernels/vector_add.cu
    ${KUDACUDA_RMSNORM_SOURCE}
    ${KUDACUDA_RMSNORM_QKV_SOURCE}
    ${KUDACUDA_NF4_QKV_SOURCE}
    ${KUDACUDA_NF4_TILED_QKV_SOURCE}
    ${KUDACUDA_NF4_SWIGLU_SOURCE}
    ${KUDACUDA_QWEN_SOURCE}
    ${KUDACUDA_QWEN_GRAPH_SOURCE}
    ${KUDACUDA_QWEN_GROUP_SOURCE}
    ${KUDACUDA_QWEN_VECTORIZED_SOURCE}
    ${KUDACUDA_QWEN_WORKSPACE_SOURCE}
    runtime/kudacuda_runtime.cu
    runtime/kudacuda_registry.cu
)

target_include_directories(kudacuda
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(kudacuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    OUTPUT_NAME kudacuda
)

target_compile_options(kudacuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

if(WIN32)
    set_target_properties(kudacuda PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

    # The NVIDIA display driver installs nvcuda.dll, but CUDA Toolkit 13.3
    # does not ship nvcuda.lib on this Windows setup. Generate the small
    # import library from the driver ABI definition at build time so the
    # native DLL links against the installed driver without copying it.
    set(KUDACUDA_NVCUDA_IMPORT_LIB "${CMAKE_CURRENT_BINARY_DIR}/nvcuda.lib")
    add_custom_command(
        OUTPUT "${KUDACUDA_NVCUDA_IMPORT_LIB}"
        COMMAND lib.exe
            /def:${CMAKE_CURRENT_SOURCE_DIR}/runtime/nvcuda.def
            /machine:x64
            /out:${KUDACUDA_NVCUDA_IMPORT_LIB}
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/runtime/nvcuda.def"
        VERBATIM
    )
    add_custom_target(kudacuda_nvcuda_import_lib
        DEPENDS "${KUDACUDA_NVCUDA_IMPORT_LIB}"
    )
    add_dependencies(kudacuda kudacuda_nvcuda_import_lib)
    target_link_libraries(kudacuda PRIVATE "${KUDACUDA_NVCUDA_IMPORT_LIB}")
else()
    target_link_libraries(kudacuda PRIVATE cuda)
endif()

add_executable(kudacuda_vector_add_smoke
    tests/vector_add_smoke.cu
)

target_include_directories(kudacuda_vector_add_smoke
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(kudacuda_vector_add_smoke PRIVATE kudacuda)

set_target_properties(kudacuda_vector_add_smoke PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(kudacuda_rmsnorm_smoke
    tests/rmsnorm_smoke.cu
)

target_include_directories(kudacuda_rmsnorm_smoke
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(kudacuda_rmsnorm_smoke PRIVATE kudacuda)

set_target_properties(kudacuda_rmsnorm_smoke PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(kudacuda_qwen2_gqa_smoke
    tests/qwen2_gqa_smoke.cu
)

target_include_directories(kudacuda_qwen2_gqa_smoke
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(kudacuda_qwen2_gqa_smoke PRIVATE kudacuda)

set_target_properties(kudacuda_qwen2_gqa_smoke PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(kudacuda_rmsnorm_qkv_smoke
    tests/rmsnorm_qkv_smoke.cu
)

target_include_directories(kudacuda_rmsnorm_qkv_smoke
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(kudacuda_rmsnorm_qkv_smoke PRIVATE kudacuda)

set_target_properties(kudacuda_rmsnorm_qkv_smoke PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(kudacuda_nf4_fused_qkv_smoke
    tests/nf4_fused_qkv_smoke.cu
)

target_include_directories(kudacuda_nf4_fused_qkv_smoke
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(kudacuda_nf4_fused_qkv_smoke PRIVATE kudacuda)

set_target_properties(kudacuda_nf4_fused_qkv_smoke PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(kudacuda_nf4_tiled_gemv_smoke
    tests/nf4_tiled_gemv_smoke.cu
)

target_include_directories(kudacuda_nf4_tiled_gemv_smoke
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(kudacuda_nf4_tiled_gemv_smoke PRIVATE kudacuda)

set_target_properties(kudacuda_nf4_tiled_gemv_smoke PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(kudacuda_nf4_swiglu_smoke
    tests/nf4_swiglu_smoke.cu
)

target_include_directories(kudacuda_nf4_swiglu_smoke
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(kudacuda_nf4_swiglu_smoke PRIVATE kudacuda)

set_target_properties(kudacuda_nf4_swiglu_smoke PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(kudacuda_qwen2_gqa_workspace_smoke
    tests/qwen2_gqa_workspace_smoke.cu
)

target_include_directories(kudacuda_qwen2_gqa_workspace_smoke
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(kudacuda_qwen2_gqa_workspace_smoke PRIVATE kudacuda)

set_target_properties(kudacuda_qwen2_gqa_workspace_smoke PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

add_executable(kudacuda_cuda_graph_smoke
    tests/cuda_graph_smoke.cu
)

target_include_directories(kudacuda_cuda_graph_smoke
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(kudacuda_cuda_graph_smoke PRIVATE kudacuda)

set_target_properties(kudacuda_cuda_graph_smoke PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
