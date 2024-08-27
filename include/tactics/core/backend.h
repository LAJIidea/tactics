//===------------------------tactics/core/backend.h------------------------===//
//
// Copyright © 2018, Alibaba Group Holding Limited
// Copy from MNN project
//
//===--------------------------------------------------------------------===//
//
/// This file defines the backend base class 
///
//===--------------------------------------------------------------------===//
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

// acquire runtime status by Runtime::get_current_status with following keys
enum RuntimeStatus {
  // get status whether this runtime support 16-bits float point arithmetic
  STATUS_SUPPORT_FP16,

  // get status whether this runtime support dot-product arithmetic
  STATUS_SUPPORT_DOT_PRODUCT,

  // get status whether this runtime support power-low (means low priority for
  // opencl)
  STATUS_SUPPORT_POWER_LOW,

  // emum total number
  STATUS_COUNT
};

struct RuntimeHint {
  // 0: Defer, 1: Eager
  int memoryAllocatorType = 0;
  int winogradMemoryUsed = 3;

  // 0-100, 50 means litter core has 50% capacity of large core
  int cpuDecreaseRate = 50;
  int dynamicQuantOption = 0;

  // 0: Do not quantize kvcache, just store float
  // 1: Only quantize key cache, use int8 asymmetric quantization
  // 2: Only quantize value cache, use fp8 quantization
  // 3: quantize both key and value cache as described above
  int kvcacheQuantOption = 0;

  // the kvcache size limit of each layer
  // if the size of kvcache in memory exceeds the limit
  // it will be moved to disk to save memory
  // -1 for no limit
  int kvcacheSizeLimit = -1;

  // path of the kvcache directory
  std::string kvcacheDirPath = "/tmp";
};

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
  virtual bool on_resize_end() { return false; };

  // callback before executing ops.
  virtual void on_execute_begin() const {};
  // callback after executing ops.
  virtual void on_execute_end() const {};

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
  virtual MemObj *on_acquire(const Tensor *tensor, StorageType storageType) {
    return nullptr;
  };

  virtual bool on_select_dynamic_allocator(int index, int maxIndex) {
    return false;
  }
  // get buffer from tensor directly
  virtual bool on_get_tensor_info(const Tensor *tensor, void *dstInfo) {
    return false;
  }

  // clear all dynamic buffers.
  virtual bool on_clear_buffer() { return false; };

  // copy buffer from tensor to tensor.
  virtual void on_copy_buffer(const Tensor *srcTensor,
                              const Tensor *dstTensor) const {};

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

// Each backend belong to a runtime
class Runtime {
public:
  // Origin Op -> (Compiler) -> New Op -> Backend
  // Default use Compiler_Geometry, Origin Op -> Compiler_Geometry -> Little Op
  // For serveral Backend, we can't use Geometry to decompose origin op, then it
  // set Compiler_Origin
  enum CompilerType {
    Compiler_Geometry = 0,
    Compiler_Origin = 1,
    Compiler_Loop = 2,
  };

  enum AllocatorType {
    Allocator_Defer = 0,
    Allocator_Eager = 1,
  };
  void set_rumtime_hint(const RuntimeHint &hint) { mHint = hint; }
  const RuntimeHint &hint() const { return mHint; }

  virtual CompilerType on_get_compiler_type() const { return Compiler_Loop; }

  virtual ~Runtime() = default;

  // create backend
  virtual Backend *on_create(const BackendConfig *config = nullptr) const {return nullptr;}

  // reset runtime
  virtual void on_reset(int numberThread, const BackendConfig *config,
                       bool full) {
    // Do nothing
  }

  // clear unuseful resource
  virtual void on_gabage_collect(int level) {}

  // Measure the memory it used in MB
  virtual float on_get_memory_inmb() { return 0.0f; }

  // If buffer is not nullptr, try copy cache, else delete cache
  virtual bool on_set_cache(const void *buffer, size_t size) {
    // default cache valid, avoid being reset
    return true;
  }

  virtual std::pair<const void *, size_t> on_get_cache() {
    return std::make_pair(nullptr, 0);
  }
  virtual int on_get_runtime_status(RuntimeStatus statusEnum) const { return 0; }
  // If the info user set can't be match by runtime, return false and set real
  // info
  virtual bool on_check_info(Backend::Info &info) const { return true; }
  struct OpInfo {
    bool initCostLong;
    float exeutionCost; // In ms
    float initCost;     // In ms
  };

  // measure the cost for op with input and output tensors.
  // virtual bool onMeasure(const std::vector<Tensor*>& inputs, const
  // std::vector<Tensor*>& outputs,
  //                                          const Op* op, OpInfo& dstInfo)
  //                                          const {
  //     return true;
  // }

  // // FIXME: Temply use to mask cache valid, in future will delete
  // virtual void onMaskOpReady(const std::vector<Tensor*>& inputs, const
  // std::vector<Tensor*>& outputs,
  //                            const Op* op) {
  //     // Do nothing
  // }
  // FIXME: Temply used, in future will refract
  std::atomic_bool mCancelled = ATOMIC_VAR_INIT(false);
  bool has_async_work() const;
  void set_async_work(std::future<int> &&future);
  void wait_async_work();

private:
  std::future<int> mFuture;
  RuntimeHint mHint;
};

// abstract Runtime register
class RuntimeCreator {
public:
  // initializer.
  virtual ~RuntimeCreator() = default;

  virtual Runtime *on_create(const Backend::Info &info) const {return nullptr;}

  // Turn info to supported.
  virtual bool on_valid(Backend::Info &info) const {
    info.mode = Backend::Info::DIRECT;
    return true;
  }

protected:
  // deinitializer.
  RuntimeCreator() = default;
};

// get registered backend creator for given forward type.

const RuntimeCreator *get_extra_runtime_creator(ForwardType type);

// register backend creator for given forward type.
bool insert_extra_runtime_creator(ForwardType type,
                                  const RuntimeCreator *creator,
                                  bool needCheck = false);

bool cpu_copy_buffer(const Tensor *srcTensor, const Tensor *dstTensor);

} // namespace tactics

#endif // TACTICS_CORE_BACKEND_H