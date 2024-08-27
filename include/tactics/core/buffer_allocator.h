//===------------------------tactics/core/buffer_allocator.h------------------------===//
//
// Copyright © 2018, Alibaba Group Holding Limited
// Copy from MNN project
//
//===-------------------------------------------------------------------------------===//
//
/// This file defines the buffer_allocator
///
//===-------------------------------------------------------------------------------===//
#ifndef TACTICS_CORE_BUFFER_ALLOCATOR_H
#define TACTICS_CORE_BUFFER_ALLOCATOR_H

#include <map>
#include <memory>
#include <set>
#include <vector>

#include "tactics/core/auto_storage.h"
#include "tactics/core/memory_utils.h"
#include "tactics/core/tensor.h"

namespace tactics {

// memory utils wrapper. provides memory reusing with alignment ability.
class EagerBufferAllocator;
class DeferBufferAllocator;
class DefaultAllocator;

struct MemNode {
public:
  MemNode(size_t s) : size(s) {}
  ~MemNode() {}
  size_t size = 0, offset = 0;
  void *base = nullptr;
  bool usage = true;
  MemNode *left = nullptr, *right = nullptr;
  std::vector<MemNode *> children;
  std::vector<Tensor *> tensors;
};

struct ChunkBySize {
public:
  ChunkBySize(MemNode *ch) : chunk(ch) {}
  MemNode *chunk;
  bool operator<(const ChunkBySize &rhs) const {
    return chunk->size < rhs.chunk->size;
  }
};

struct MemChunk {
public:
  MemChunk() = default;
  MemChunk(void *base, size_t offset = 0) : first(base), second(offset) {}
  MemChunk(std::pair<void *, size_t> pointer)
      : first(pointer.first), second(pointer.second) {}
  MemChunk(MemNode *node) : mNode(node) {}
  ~MemChunk() = default;
  MemChunk operator+(size_t offset);
  void *base() const;
  size_t offset() const;
  bool invalid() const;
  void attach(Tensor *tensor);
  uint8_t *ptr() const {
    if (mNode) {
      return static_cast<uint8_t *>(mNode->base) + mNode->offset + second;
    }
    return static_cast<uint8_t *>(first) + second;
  }

public:
  void *first = nullptr;
  size_t second = 0;

private:
  MemNode *mNode = nullptr;
  friend class DeferBufferAllocator;
  friend class EagerBufferAllocator;
  friend class DefaultAllocator;
};

class BufferAllocator {
public:
  class Allocator {
  public:
    Allocator() = default;
    virtual ~Allocator() = default;
    virtual MemChunk on_alloc(size_t size, size_t align) = 0;
    virtual void on_release(MemChunk chunk) = 0;
    static std::shared_ptr<Allocator> create_default();
    static std::shared_ptr<Allocator> create_recurse(BufferAllocator *parent);
  };
  BufferAllocator() = default;
  virtual ~BufferAllocator() = default;
  virtual MemChunk alloc(size_t size, bool separate = false,
                         size_t align = 0) = 0;
  virtual bool free(MemChunk chunk) = 0;
  virtual void release(bool allRelease = true) = 0;
  virtual size_t total_size() const = 0;
  virtual void barrier_begin() {}
  virtual void barrier_end() {}
  virtual void begin_group() {}
  virtual void end_group() {}
  virtual void reset() {}
  virtual bool compute();
};

class EagerBufferAllocator : public BufferAllocator {
public:
  EagerBufferAllocator(std::shared_ptr<Allocator> parent,
                       size_t align = MEMORY_ALIGN_DEFAULT)
      : mAllocator(parent), mAlign(align) {}

  ~EagerBufferAllocator() { release(); }

  MemChunk alloc(size_t size, bool separate = false, size_t align = 0) override;

  bool free(MemChunk chunk) override;
  void release(bool allRelease = true) override;

  size_t total_size() const override { return mTotalSize; }

  void barrier_begin() override;
  void barrier_end() override;
  void begin_group() override;
  void end_group() override;

private:
  class Node : public RefCount {
  public:
    ~Node();
    std::pair<void *, size_t> pointer;
    SharedPtr<Node> parent = nullptr;
    size_t size;
    size_t useCount = 0;
    Allocator *outside = nullptr;
  };

  typedef std::multimap<size_t, SharedPtr<Node>> FREELIST;

  static void returnMemory(FREELIST *list, SharedPtr<Node> node,
                           bool permitMerge = true);
  std::pair<void *, size_t> getFromFreeList(FREELIST *list, size_t size,
                                            bool permiteSplit, size_t align);

  std::map<std::pair<void *, size_t>, SharedPtr<Node>> mUsedList;
  FREELIST mFreeList;
  size_t mTotalSize = 0;

  FREELIST *mCurrentFreeList = nullptr;
  std::vector<std::shared_ptr<FREELIST>> mGroups;
  std::shared_ptr<Allocator> mAllocator;
  size_t mAlign;
};
typedef void (*MemChunkApplyToTensor)(uint8_t *ptr, size_t offset,
                                      Tensor *tensor);

class DeferBufferAllocator : public BufferAllocator {
public:
  DeferBufferAllocator(std::shared_ptr<Allocator> parent,
                       size_t align = MEMORY_ALIGN_DEFAULT,
                       MemChunkApplyToTensor func = nullptr);
  ~DeferBufferAllocator() { reset(); }

public:
  MemChunk alloc(size_t size, bool separate = false, size_t align = 0) override;
  bool free(MemChunk chunk) override;
  void release(bool allRelease = true) override;
  size_t total_size() const override;
  void barrier_begin() override;
  void barrier_end() override;
  void begin_group() override;
  void end_group() override;
  void reset() override;
  bool compute() override;

private:
  std::vector<std::unique_ptr<MemNode>> mChunks;
  MemNode *mHead = nullptr, *mTail = nullptr;
  std::multiset<ChunkBySize> mFreeList;
  // std::unique_ptr<uint8_t[]> mPtr;
  MemChunk mPtr;
  size_t mTotalSize = 0;
  std::shared_ptr<Allocator> mAllocator;
  size_t mAlign;
  // barrier
  bool mBarrrier = false;
  std::vector<MemChunk> mBarrrierFreeChunks;

  MemNode *createMemNode(size_t size);
  MemNode *fuse_to_left(MemNode *left, MemNode *right);
  void erase_node(MemNode *chunk);
  void insert_after(MemNode *chunk, MemNode *pos = nullptr);
  void insertFree(MemNode *chunk);
  void eraseFree(MemNode *chunk);
  void visiChildren(MemNode *chunk);
  MemChunkApplyToTensor mApplyFunction;
};

} // namespace tactics

#endif // TACTICS_CORE_BUFFER_ALLOCATOR_H