//===------------------------tactics/core/auto_storage.h------------------------===//
//
// Copyright © 2018, Alibaba Group Holding Limited
// Copy from MNN project
//
//===--------------------------------------------------------------------------===//
//
/// This file defines the storage utils
///
//===--------------------------------------------------------------------------===//
#ifndef TACTICS_CORE_AUTO_STORAGE_H
#define TACTICS_CORE_AUTO_STORAGE_H

#include "tactics/core/memory_utils.h"
#include <cassert>
#include <stdint.h>
#include <string.h>

namespace tactics {

template <typename T>

// self-managed memory storage
class AutoStorage {
public:
  AutoStorage() {
    mSize = 0;
    mData = NULL;
  }

  AutoStorage(int size) {
    mData = (T *)memory_alloc_align(sizeof(T) * size, MEMORY_ALIGN_DEFAULT);
    mSize = size;
  }

  ~AutoStorage() {
    if (NULL != mData) {
      MNNMemoryFreeAlign(mData);
    }
  }

  inline int size() const { return mSize; }

  void set(T *data, int size) {
    if (NULL != mData && mData != data) {
      MNNMemoryFreeAlign(mData);
    }
    mData = data;
    mSize = size;
  }

  void reset(int size) {
    if (NULL != mData) {
      MNNMemoryFreeAlign(mData);
    }
    mData = (T *)memory_alloc_align(sizeof(T) * size, MEMORY_ALIGN_DEFAULT);
    mSize = size;
  }

  void release() {
    if (NULL != mData) {
      MNNMemoryFreeAlign(mData);
      mData = NULL;
      mSize = 0;
    }
  }

  void clear() { ::memset(mData, 0, mSize * sizeof(T)); }

  T *get() const { return mData; }

private:
  T *mData = NULL;
  int mSize = 0;
};

// Auto Release Class
template <typename T> class AutoRelease {
public:
  AutoRelease(T *d = nullptr) { mData = d; }
  ~AutoRelease() {
    if (NULL != mData) {
      delete mData;
    }
  }
  AutoRelease(const AutoRelease &) = delete;
  T *operator->() { return mData; }
  void reset(T *d) {
    if (nullptr != mData) {
      delete mData;
    }
    mData = d;
  }
  T *get() { return mData; }
  const T *get() const { return mData; }

private:
  T *mData = NULL;
};

class RefCount {
public:
  void addRef() const { mNum++; }
  void decRef() const {
    --mNum;
    assert(mNum >= 0);
    if (0 >= mNum) {
      delete this;
    }
  }
  inline int count() const { return mNum; }

protected:
  RefCount() : mNum(1) {}
  RefCount(const RefCount &f) : mNum(f.mNum) {}
  void operator=(const RefCount &f) {
    if (this != &f) {
      mNum = f.mNum;
    }
  }
  virtual ~RefCount() {}

private:
  mutable int mNum;
};

#define SAFE_UNREF(x)                                                          \
  if (NULL != (x)) {                                                           \
    (x)->decRef();                                                             \
  }
#define SAFE_REF(x)                                                            \
  if (NULL != (x))                                                             \
    (x)->addRef();

#define SAFE_ASSIGN(dst, src)                                                  \
  {                                                                            \
    if (src != NULL) {                                                         \
      src->addRef();                                                           \
    }                                                                          \
    if (dst != NULL) {                                                         \
      dst->decRef();                                                           \
    }                                                                          \
    dst = src;                                                                 \
  }
template <typename T> class SharedPtr {
public:
  SharedPtr() : mT(NULL) {}
  SharedPtr(T *obj) : mT(obj) {}
  SharedPtr(const SharedPtr &o) : mT(o.mT) { SAFE_REF(mT); }
  ~SharedPtr() { SAFE_UNREF(mT); }

  SharedPtr &operator=(const SharedPtr &rp) {
    SAFE_ASSIGN(mT, rp.mT);
    return *this;
  }
  SharedPtr &operator=(T *obj) {
    SAFE_UNREF(mT);
    mT = obj;
    return *this;
  }

  T *get() const { return mT; }
  T &operator*() const { return *mT; }
  T *operator->() const { return mT; }

private:
  T *mT;
};

struct BufferStorage {
  size_t size() const { return allocated_size - offset; }

  const uint8_t *buffer() const { return storage + offset; }
  ~BufferStorage() {
    if (nullptr != storage) {
      delete[] storage;
    }
  }
  size_t allocated_size;
  size_t offset;
  uint8_t *storage = nullptr;
};

} // namespace tactics

#endif // TACTICS_CORE_AUTO_STORAGE_H