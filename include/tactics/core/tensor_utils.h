//===------------------------tactics/core/tensor_utils.h------------------------===//
//
// Copyright © 2018, Alibaba Group Holding Limited
// Copy from MNN project
//
//===---------------------------------------------------------------------------===//
//
/// This file defines the tensor utils
///
//===--------------------------------------------------------------------------===//
#ifndef TACTICS_TENSOR_TENSOR_UTILS_H
#define TACTICS_TENSOR_TENSOR_UTILS_H

#include "tactics/core/backend.h"
#include "tactics/core/tensor.h"
#include <cstdint>
#include <memory>
#include <vector>

#define MAX_TENSOR_DIM 8

namespace tactics {

enum DataType {
  DataType_DT_INVALID = 0,
  DataType_DT_FLOAT = 1,
  DataType_DT_DOUBLE = 2,
  DataType_DT_INT32 = 3,
  DataType_DT_UINT8 = 4,
  DataType_DT_INT16 = 5,
  DataType_DT_INT8 = 6,
  DataType_DT_STRING = 7,
  DataType_DT_COMPLEX64 = 8,
  DataType_DT_INT64 = 9,
  DataType_DT_BOOL = 10,
  DataType_DT_QINT8 = 11,
  DataType_DT_QUINT8 = 12,
  DataType_DT_QINT32 = 13,
  DataType_DT_BFLOAT16 = 14,
  DataType_DT_QINT16 = 15,
  DataType_DT_QUINT16 = 16,
  DataType_DT_UINT16 = 17,
  DataType_DT_COMPLEX128 = 18,
  DataType_DT_HALF = 19,
  DataType_DT_RESOURCE = 20,
  DataType_DT_VARIANT = 21,
  DataType_MIN = DataType_DT_INVALID,
  DataType_MAX = DataType_DT_VARIANT
};

enum DATA_FORMAT {
  DATA_FORMAT_NCHW = 0,
  DATA_FORMAT_NHWC = 1,
  DATA_FORMAT_NC4HW4 = 2,
  DATA_FORMAT_NHWC4 = 3,
  DATA_FORMAT_UNKNOWN = 4,
  DATA_FORMAT_MIN = DATA_FORMAT_NCHW,
  MDATA_FORMAT_MAX = DATA_FORMAT_UNKNOWN
};

struct TensorArrayAttr {
  // array size is dynamic or not
  bool is_dynamic = false;
  // element shape is idenetical or not
  bool is_idententical = false;
  // the number of element
  uint32_t array_size = 0;
  // the shape of element
  std::vector<std::vector<int>> element_shape;
};

struct QuantAttr {
  float scale;
  float zero = 0.0f;
  float min = -127.0f;
  float max = 1278.0f;
};

struct Tensor::InsideDescribe {
  struct View {
    int32_t offset = 0;
    int32_t stride[3] = {1, 1, 1};
  };
  struct Region {
    View src;
    View dst;
    int32_t size[3] = {1, 1, 1};
    Tensor *origin;
  };
  struct pad {
    int32_t left = 0;
    int32_t right = 0;
    int32_t bottom = 0;
    int32_t top = 0;
  };
  enum MemoryType {
    // The tenor's memory come from backend
    MEMORY_BACKEND = 0,

    // host memory is owned by tensor or not
    MEMORY_HOST,

    // The tensor don't has memory
    MEMORY_VIRTUAL,

    // host memory is owned by tensor or not
    MEMORY_OUTSIDE,
  };
  enum Usage {
    NORMAL,
    INPUT,
    OUTPUT,
    CONSTANT,
    // Whether the tensor is a trainable parameter. Trainable parameter should
    // be stored in a different area.
    TRAINABLE,
  };
  // For Mask
  enum StageInfo {
    GEMOETRY_STAGE = 1,
    CONVERTED_STAGE = 1 << 1,
    COMPUTE_SHAPE_STAGE = 1 << 2,
    CONTENT_NOT_CHANGE = 1 << 3,
  };
  // extra tensor info container
  struct NativeInsideDescribe {
  public:
    // dimension format
    DATA_FORMAT dimension_format = DATA_FORMAT_NC4HW4;
    union {
      // Serperate memory offset
      int offset;
      // function used to free handle
      void (*handleFreeFunction)(void *);
    } extra;
    MemoryType memoryType = MEMORY_BACKEND;
    // std::weak_ptr<Command> rasterCommand;
    // for DEVICE tensor only.
    int useCount = 0;
    Usage usage = NORMAL;
    std::vector<Region> regions;
    halide_dimension_t dims[MAX_TENSOR_DIM];
    // TensorArray Attribute
    std::shared_ptr<TensorArrayAttr> tensorArrayAttr;
    // Tensor Quant Attribute
    std::shared_ptr<QuantAttr> quantAttr;
    // Only valid when quantAttr is not nullptr
    DataType type = DataType_DT_FLOAT;
    bool isMutable = true;
    int index = -1;
    int group = 0;
    int channel_pack_num = 4;
    bool support_pack16 = true;
    pad mPads;
    // For isMutable = false Tensor , determine whether the content can be
    // convert to main backend
    uint32_t stageMask = 0;
  };
  std::shared_ptr<NativeInsideDescribe> m_content;
  // SharedPtr type for assign
  SharedPtr<Backend::MemObj> mem;
  inline Backend *getBackend() const { return backend; }
  inline void setBackend(Backend *bn) { backend = bn; }

private:
  // for DEVICE tensor only. backend used to manage tensor's device memory
  Backend *backend = nullptr;
};

class TensorUtils {
public:
    // get extra tensor info.
    static Tensor::InsideDescribe::NativeInsideDescribe* get_describe(const Tensor* tensor);

    static Tensor::InsideDescribe* get_describe_origin(const Tensor* tensor);

    // copy shape from source tensor to dest tensor.
    static void copy_shape(const Tensor* source, Tensor* dest, bool copyFormat = false, bool copyRef = false);

    // set shape for dest tensor from a common int vector.
    static void set_shape(Tensor* dest, const std::vector<int>& alldims);

    // auto update tensor's strides according to extents and reorder flags
    static void set_linear_layout(Tensor* tensor);

    // compare tensor to expected with tolerance.
    static bool compare_tensors(const Tensor* compareTensor, const Tensor* toTensor, float tolerance = 0,
                               bool overall = false, bool printsError = true, bool printsTensors = false);

    static void setup_tensor_info(const Tensor* tensor, Tensor* wrapTensor, DATA_FORMAT mMidFormat);
    static Tensor::InsideDescribe::Region make_full_slice(Tensor* input);
    static bool region_is_full(Tensor* input);
    static bool is_copy_region(const Tensor::InsideDescribe::Region& region);
    static bool is_transpose_region(const Tensor::InsideDescribe::Region& region);
    static bool is_tile_region(const Tensor::InsideDescribe::Region& region);
    static bool is_depth_to_space_regions(const Tensor* output);
    static bool reshape_slice(Tensor::InsideDescribe::Region& slice, int outside, int inside, int axis);
    
    class FuseRegionStatus;
    class FuseWrap {
    public:
        FuseWrap();
        ~ FuseWrap();
        bool match(const Tensor::InsideDescribe::Region& srcReg, const Tensor::InsideDescribe::Region& dstReg);
        void apply(const Tensor::InsideDescribe::Region& srcReg, Tensor::InsideDescribe::Region& dstReg);
    private:
        FuseRegionStatus* mStatus;
    };
    static void adjust_tensor_for_compability(Tensor* t);
    static std::vector<float> get_quant_info(const Tensor* t);
    
    static size_t get_raw_size(const Tensor* t);
    // static void setRasterInputs(Command* cmd);
    
    static bool ref_tensor_content(Tensor* dst, const Tensor* src);

    static int get_tensor_channel_pack(const Tensor* tensor);

    static void set_tensor_channel_pack(const Tensor* tensor, int pack);

    static void set_tensor_support_pack(const Tensor* tensor, bool flag);

    static void set_tensor_pad(const Tensor* tensor, int left, int right, int bottom, int top);
};

} // namespace tactics

#endif // TACTICS_TENSOR_TENSOR_UTILS_H