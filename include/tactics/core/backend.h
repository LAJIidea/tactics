#ifndef TACTICS_CORE_BACKEND_H
#define TACTICS_CORE_BACKEND_H

#include "tactics/core/buffer_allocator.h"
#include <atomic>
#include <future>
#include <map>

namespace tactics {

typedef enum {
  FORWARD_CPU = 0,

  // Firtly find the first available backends not equal to CPU
  // If no other backends, use cpu
  FORWARD_AUTO = 4,

  // Hand write metal
  FORWARD_METAL = 1,

  // NVIDIA GPU API
  FORWARD_CUDA = 2,

  // Android / Common Device GPU API
  FORWARD_OPENCL = 3,
  FORWARD_OPENGL = 6,
  FORWARD_VULKAN = 7,

  // Android 8.1's NNAPI or CoreML for ios*/
  FORWARD_NN = 5,

  // User can use API from Backend.hpp to add or search Backend*/
  FORWARD_USER_0 = 8,
  FORWARD_USER_1 = 9,
  FORWARD_USER_2 = 10,
  FORWARD_USER_3 = 11,

  FORWARD_ALL,

  // Apply arm extension instruction set to accelerate some Ops, this forward
  // type is only used in MNN internal, and will be active automatically when
  // user set forward type to be MNN_FORWARD_CPU and extension instruction set
  // is valid on hardware.
  FORWARD_CPU_EXTENSION

} ForwardType;

struct BackendConfig {
  enum MemoryMode { Memory_Normal = 0, Memory_High, Memory_Low };

  MemoryMode memory = Memory_Normal;

  enum PowerMode { Power_Normal = 0, Power_High, Power_Low };

  PowerMode power = Power_Normal;

  enum PrecisionMode {
    Precision_Normal = 0,
    Precision_High,
    Precision_Low,
    Precision_Low_BF16
  };

  PrecisionMode precision = Precision_Normal;

  // user defined context
  union {
    void *sharedContext = nullptr;
    size_t flags; // Valid for CPU Backend
  };
};

class Backend {
public:
  // info used to create backend
  struct Info {
    // forward type.
    ForwardType type = FORWARD_CPU;
    // numThread for CPU . number of threads.  gpuMode for GPU only.
    // tuning/memory Mode setting
    union {
      int numThread = 4;
      int gpuMode;
    };
    // user data.
    BackendConfig *user = NULL;
    enum Mode {
      // The Op will be run in execution->onExecute
      DIRECT = 0,

      // The Op will be recorded. Run in onExecuteBegin and Wait in onExecuteEnd
      INDIRECT = 1
    };
    Mode mode = DIRECT;
  };

  // backend buffer storage type
  enum StorageType {

    // use NOT reusable memory.
    // - allocates memory when `onAcquireBuffer` is called.
    // - releases memory when `onReleaseBuffer` is called or when the backend is
    // deleted.
    // - do NOTHING when `onClearBuffer` is called.
    STATIC,
    // use reusable memory.
    // - allocates or reuses memory when `onAcquireBuffer` is called. prefers
    // reusing.
    // - collects memory for reuse when `onReleaseBuffer` is called.
    // - releases memory when `onClearBuffer` is called or when the backend is
    // deleted.
    DYNAMIC,
    // use NOT reusable memory.
    // - allocates memory when `onAcquireBuffer` is called.
    // - do NOTHING when `onReleaseBuffer` is called.
    // - releases memory when `onClearBuffer` is called or when the backend is
    // deleted.
    DYNAMIC_SEPERATE
  };

public:
  Backend(ForwardType type) : mType(type) {
    // nothing to do
  }

  virtual ~Backend() = default;

public:
  // create execution for op with input and output tensors.
  // virtual Execution* on_create(const std::vector<Tensor*>& inputs, const
  // std::vector<Tensor*>& outputs,
  //                             const Op* op) = 0;

  // @brief callback before resize ops.
  virtual void on_resize_begin() {
    // nothing to do
  }
  // callback after resize ops.
  virtual bool on_resize_end() = 0;

  // callback before executing ops.
  virtual void on_execute_begin() const = 0;
  // callback after executing ops.
  virtual void on_execute_end() const = 0;
  // virtual const Runtime* get_runtime() {
  //     return nullptr;
  // }

  // allocate buffer of tensor for given storage type.
  bool on_acquire_buffer(const Tensor *tensor, StorageType storageType);

  // release buffer of tensor for given storage type.
  bool on_release_buffer(const Tensor *tensor, StorageType storageType);

  class MemObj : public RefCount {
  public:
    MemObj() {}
    virtual ~MemObj() {}
    virtual MemChunk chunk() { return MemChunk(); }
  };

  // allocate buffer of tensor for given storage type.
  virtual MemObj *on_acquire(const Tensor *tensor, StorageType storageType) = 0;

  virtual bool on_select_dynamic_allocator(int index, int maxIndex) {
    return false;
  }
  // get buffer from tensor directly
  virtual bool on_get_tensor_info(const Tensor *tensor, void *dstInfo) {
    return false;
  }

  // clear all dynamic buffers.
  virtual bool on_clear_buffer() = 0;

  // copy buffer from tensor to tensor.
  virtual void on_copy_buffer(const Tensor *srcTensor,
                              const Tensor *dstTensor) const = 0;

public:
  // get forward type.
  inline ForwardType type() const { return mType; }

public:
  // get Gpu Tensor map host ptr/ unmap
  virtual void *on_map_tensor(Tensor::MapType mtype, const Tensor *srcTensor) {
    return nullptr;
  }

  virtual bool on_unmap_tensor(Tensor::MapType mtype, const Tensor *dstTensor,
                               void *mapPtr) {
    return false;
  }

  virtual int on_sync(Tensor::MapType mtype, bool toCpu,
                      const Tensor *dstTensor) {
    return 0;
  }

private:
  const ForwardType mType;
};

} // namespace tactics

#endif // TACTICS_CORE_BACKEND_H