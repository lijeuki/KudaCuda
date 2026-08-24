#pragma once

#include <stdint.h>

#if defined(_WIN32)
#  if defined(kudacuda_EXPORTS)
#    define KUDACUDA_API __declspec(dllexport)
#  else
#    define KUDACUDA_API __declspec(dllimport)
#  endif
#else
#  define KUDACUDA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

KUDACUDA_API void* kudacuda_create(int device_id);

KUDACUDA_API int kudacuda_freeze(void* handle);

KUDACUDA_API int kudacuda_can_dispatch(
    void* handle,
    const char* tactic_name,
    int64_t batch_size,
    int64_t seq_len
);

KUDACUDA_API int kudacuda_vector_add(
    void* handle,
    uint64_t a,
    uint64_t b,
    uint64_t output,
    int64_t count,
    uint64_t stream
);

KUDACUDA_API int kudacuda_rmsnorm(
    void* handle,
    uint64_t input,
    uint64_t weight,
    uint64_t output,
    int64_t rows,
    int64_t hidden,
    float epsilon,
    uint64_t stream
);

KUDACUDA_API int kudacuda_begin_capture(void* handle, uint64_t stream);

KUDACUDA_API int kudacuda_end_capture(void* handle, uint64_t stream);

KUDACUDA_API int kudacuda_replay_graph(void* handle, uint64_t stream);

KUDACUDA_API const char* kudacuda_last_error(void* handle);

KUDACUDA_API void kudacuda_destroy(void* handle);

#ifdef __cplusplus
}
#endif
