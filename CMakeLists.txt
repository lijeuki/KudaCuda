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
else()
    set(KUDACUDA_RMSNORM_SOURCE kernels/sm86/rmsnorm_sm86.cu)
endif()

add_library(kudacuda SHARED
    kernels/vector_add.cu
    ${KUDACUDA_RMSNORM_SOURCE}
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
