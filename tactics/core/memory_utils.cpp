//===------------------------tactics/core/memory_utils.h------------------------===//
//
// Copyright © 2018, Alibaba Group Holding Limited
// Copy from MNN project
//
//===---------------------------------------------------------------------------===//
//
/// This file defines the memory utils implement
///
//===--------------------------------------------------------------------------===//
#include "tactics/core/memory_utils.h"
#include <cassert>
#include <stdint.h>
#include <stdlib.h>

static inline void **alignPointer(void **ptr, size_t alignment) {
    return (void **)((intptr_t)((unsigned char *)ptr + alignment - 1) & -alignment);
}

extern "C" void *memory_alloc_align(size_t size, size_t alignment) {
    assert(size > 0);

#ifdef MNN_DEBUG_MEMORY
    return malloc(size);
#else
    void **origin = (void **)malloc(size + sizeof(void *) + alignment);
    assert(origin != NULL);
    if (!origin) {
        return NULL;
    }

    void **aligned = alignPointer(origin + 1, alignment);
    aligned[-1]    = origin;
    return aligned;
#endif
}

extern "C" void *memory_calloc_align(size_t size, size_t alignment) {
    assert(size > 0);

#ifdef MNN_DEBUG_MEMORY
    return calloc(size, 1);
#else
    void **origin = (void **)calloc(size + sizeof(void *) + alignment, 1);
    assert(origin != NULL);
    if (!origin) {
        return NULL;
    }
    void **aligned = alignPointer(origin + 1, alignment);
    aligned[-1]    = origin;
    return aligned;
#endif
}

extern "C" void memory_free_align(void *aligned) {
#ifdef MNN_DEBUG_MEMORY
    free(aligned);
#else
    if (aligned) {
        void *origin = ((void **)aligned)[-1];
        free(origin);
    }
#endif
}
