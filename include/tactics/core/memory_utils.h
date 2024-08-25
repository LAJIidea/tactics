#ifndef TACTICS_CORE_MEMORY_UTILS_H
#define TACTICS_CORE_MEMORY_UTILS_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MEMORY_ALIGN_DEFAULT 64

// alloc memory with given size & alignment.
void* memory_alloc_align(size_t size, size_t align);

// alloc memory with given size & alignment, and fill memory space with 0.
void* memory_calloc_align(size_t size, size_t align);

// @brief free aligned memory pointer.
void memory_free_align(void* mem);

#ifdef __cplusplus
}
#endif

#endif // TACTICS_CORE_MEMORY_UTILS_H