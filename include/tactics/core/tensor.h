//===------------------------tactics/tensor/tensor.h------------------------===//
//
// Copyright (c) RISC-X Organizations, see https://risc-x.org
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
//===----------------------------------------------------------------------===//
//
/// This file defines the tensor
///
//===----------------------------------------------------------------------===//

#ifndef TACTICS_TENSOR_TENSOR_H
#define TACTICS_TENSOR_TENSOR_H

#include <vector>

#include "HalideRuntime.h"

#define ALIMIN(x, y) ((x) < (y) ? (x) : (y))
#define ALIMAX(x, y) ((x) > (y) ? (x) : (y))

#define UP_DIV(x, y) (((x) + (y) - (1)) / (y))
#define ROUND_UP(x, y) (((x) + (y) - (1)) / (y) * (y))
#define ALIGN_UP4(x) ROUND_UP((x), 4)
#define ALIGN_UP8(x) ROUND_UP((x), 8)

namespace tactics {

class Tensor {
public:
  struct InsideDescribe;

  // dimension type used to create tensor
  // enum DimensionType {
  //   // for tensorflow net type. uses NHWC as data format.
  //   TENNSORFLOW,
  //   // for caffe net type. use NCHW as data format.

  // };

  // handle type
  enum HandleDataType {
    // default handle type
    HANDLE_NONE = 0,
    // string handle type
    HANDLE_STRING = 1
  };

  // Tensor map type: Read or Write
  enum MapType {
    // map Tensor for writing data
    MAP_TENSOR_WRITE = 0,
    MAP_TENSOR_READ = 1
  };

  Tensor(int dimSize = 4);

  Tensor(const Tensor *tensor, bool allocMemory = true);

  ~Tensor();

  static Tensor *create_device(const std::vector<int> &shape,
                               halide_type_t type);

  template <typename T>
  static Tensor *create_device(const std::vector<int> &shape) {
    return create_device(shape, halide_type_of<T>());
  }

  static Tensor *create(const std::vector<int> &shape, halide_type_t type,
                        void *data = nullptr);

  template <typename T>
  static Tensor *create(const std::vector<int> &shape, void *data = nullptr) {
    return create(shape, halide_type_of<T>(), data);
  }

  static Tensor *clone(const Tensor *src, bool deepCopy = false);

  static void destroy(Tensor *tensor);

  bool copy_from_host_tensor(const Tensor *tensor);

  bool copy_to_host_tensor(Tensor *hostTensor) const;

  static Tensor *create_host_tensor_from_device(const Tensor *deviceTensor,
                                                bool copyData = true);

  const halide_buffer_t &buffer() const { return m_buffer; }

  halide_buffer_t &buffer() { return m_buffer; }

  HandleDataType get_handle_data_type() const;

  // set data type.
  void setType(int type);

  // get data type.
  inline halide_type_t getType() const { return m_buffer.type; }

  // visit host memory, data type is represented by `T`.
  template <typename T> T *host() const { return (T *)m_buffer.host; }

  // visit device memory.
  uint64_t deviceId() const { return m_buffer.device; }

public:
  int dimensions() const { return m_buffer.dimensions; }

  // get all dimensions' extent.
  std::vector<int> shape() const;

  // calculate number of bytes needed to store data taking reordering flag into
  // account.
  int size() const;
  size_t usize() const;

  // calculate number of elements needed to store data taking reordering flag
  // into account.
  inline int elementSize() const { return size() / m_buffer.type.bytes(); }

public:
  inline int width() const { return m_buffer.dim[3].extent; }
  inline int height() const { return m_buffer.dim[2].extent; }
  inline int channel() const { return m_buffer.dim[1].extent; }
  inline int batch() const { return m_buffer.dim[0].extent; }

  // visit dimension's extent & stride
  inline int stride(int index) const { return m_buffer.dim[index].stride; }
  inline int length(int index) const { return m_buffer.dim[index].extent; }
  inline void setStride(int index, int stride) {
    m_buffer.dim[index].stride = stride;
  }
  inline void setLength(int index, int length) {
    m_buffer.dim[index].extent = length;
  }

  // For GPU and Other Device, get memory directly, see MNNSharedContext for
  // detail
  bool getDeviceInfo(void *dst, int forwardType) const;

public:
  // print tensor data. for DEBUG use only.
  void print() const;

  // f print tensor shape
  void printShape() const;

public:
  // map/umap GPU Tensor, to get host ptr
  void *map(MapType mtype);
  void unmap(MapType mtype, void *mapPtr);
  // wait until the tensor is ready to read / write
  int wait(MapType mtype, bool finish);
  // set GPU tensor device ptr, and inform memory type
  bool setDevicePtr(const void *devicePtr, int memoryType);

private:
  Tensor(bool deepCopy, const Tensor *tensor);
  // remove all assignment operator
  Tensor(const Tensor &tensor) = delete;
  Tensor(const Tensor &&tensor) = delete;
  Tensor &operator=(const Tensor &) = delete;
  Tensor &operator=(const Tensor &&) = delete;

  halide_buffer_t m_buffer;
  struct InsideDescribe *m_describe;

  friend class TensorUtils;
};

} // namespace tactics

#endif // TACTICS_TENSOR_TENSOR_H