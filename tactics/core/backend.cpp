//===------------------------tactics/core/backend.cpp------------------------===//
//
// Copyright © 2018, Alibaba Group Holding Limited
// Copy from MNN project
//
//===-----------------------------------------------------------------------===//
//
/// This file defines the backend base class and runtime base class implement
///
//===----------------------------------------------------------------------===//
#include "tactics/core/backend.h"
#include "tactics/core/tensor_utils.h"
#include <cassert>
#include <map>
#include <mutex>
#include <utility>

namespace tactics {

static std::map<ForwardType, std::pair<const RuntimeCreator *, bool>> &
get_extra_creator() {
  static std::once_flag g_init_flag;
  static std::map<ForwardType, std::pair<const RuntimeCreator *, bool>>
      *g_extra_creator;
  std::call_once(g_init_flag, [&]() {
    g_extra_creator =
        new std::map<ForwardType, std::pair<const RuntimeCreator *, bool>>;
  });
  return *g_extra_creator;
}

extern void register_cpu_runtime_creator();

#if OPENCL_ENABLED
extern void register_opencl_runtime_creator();
#endif
#if OPENMP_ENABLED
extern void register_openmp_runtime_creator();
#endif
#if CUDA_ENABLED
extern void register_cuda_runtime_creator();
#endif
#if AMDGPU_ENABLED
extern void register_amdgpu_runtime_creator();
#endif
#if MUSA_ENABLED
extern void register_musa_runtime_creator();
#endif
#if APPLE_ENABLED
extern void register_apple_runtime_creator();
#endif
#if HEXAGON_ENABLED
extern void register_hexagon_runtime_creator();
#endif
#if TPU_ENABLE
extern void register_tpu_runtime_creator();
#endif
#if HABANA_ENABLE
extern void register_gaudi_runtime_creator();
#endif
#if DATAFLOW_ENABLE
extern void register_dataflow_runtime_creator();
#endif

static std::once_flag s_flag;
void register_backend() {}

const RuntimeCreator *get_extra_runtime_creator(ForwardType type) {
  register_backend();

  auto &gExtraCreator = get_extra_creator();
  auto iter = gExtraCreator.find(type);
  if (iter == gExtraCreator.end()) {
    return nullptr;
  }
  if (!iter->second.second) {
    return iter->second.first;
  }
  Backend::Info info;
  info.type = type;
  std::shared_ptr<Runtime> bn(iter->second.first->on_create(info));
  if (nullptr != bn.get()) {
    return iter->second.first;
  }
  return nullptr;
}

bool insert_extra_runtime_creator(ForwardType type,
                                  const RuntimeCreator *creator,
                                  bool needCheck) {
  auto &gExtraCreator = get_extra_creator();
  if (gExtraCreator.find(type) != gExtraCreator.end()) {
    assert(false && "duplicate type");
    return false;
  }
  gExtraCreator.insert(
      std::make_pair(type, std::make_pair(creator, needCheck)));
  return true;
}

bool cpu_copy_buffer(const Tensor *srcTensor, const Tensor *dstTensor) {
  auto &srcBuffer = srcTensor->buffer();
  auto &dstBuffer = dstTensor->buffer();

  assert(srcBuffer.dimensions == dstBuffer.dimensions);
  assert(srcBuffer.type == dstBuffer.type);
  if (nullptr == srcBuffer.host || nullptr == dstBuffer.host) {
    return false;
  }
  // auto code = CPUTensorConverter::convert(srcTensor, dstTensor);
  // if (NO_ERROR != code) {
  //     MNN_ERROR("Error in CPUBackend::onCopyBuffer\n");
  // }
  return true;
}

bool Backend::on_acquire_buffer(const Tensor *tensor, StorageType storageType) {
  auto mem = this->on_acquire(tensor, storageType);
  if (nullptr == mem) {
    return false;
  }
  if (mem == TensorUtils::get_describe_origin(tensor)->mem.get()) {
    return true;
  }
  TensorUtils::get_describe_origin(tensor)->mem = mem;
  return true;
}
bool Backend::on_release_buffer(const Tensor *tensor, StorageType storageType) {
  TensorUtils::get_describe_origin(tensor)->mem = nullptr;
  return true;
}

bool Runtime::has_async_work() const { return mFuture.valid(); }
void Runtime::set_async_work(std::future<int> &&future) {
  mFuture = std::move(future);
}
void Runtime::wait_async_work() {
  if (mFuture.valid()) {
    mFuture.wait();
  }
}

} // namespace tactics