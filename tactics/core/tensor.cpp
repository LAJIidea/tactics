#include "tactics/core/tensor.h"
#include "HalideRuntime.h"
#include "tactics/core/memory_utils.h"
#include "tactics/core/tensor_utils.h"
#include <cassert>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <string>

namespace tactics {

Tensor::Tensor(int dim_size) {
  assert(dim_size <= MAX_TENSOR_DIM);
  m_describe = new InsideDescribe();
  m_describe->m_content.reset(new InsideDescribe::NativeInsideDescribe());
  auto native_desribe = m_describe->m_content.get();
  m_buffer.dimensions = dim_size;
  m_buffer.type = halide_type_of<float>();
  m_buffer.device = 0;
  m_buffer.host = nullptr;
  m_buffer.dim = &native_desribe->dims[0];
  m_buffer.flags = 0;

  native_desribe->dimension_format = DATA_FORMAT_NCHW;
}

Tensor::Tensor(const Tensor* tensor, bool alloc_memory) {
  assert(tensor != nullptr);

  auto buffer = tensor->buffer();
  m_describe = new InsideDescribe();
  m_describe->m_content.reset(new InsideDescribe::NativeInsideDescribe());
  auto native_desribe = m_describe->m_content.get();
  m_buffer.dimensions = buffer.dimensions;
  m_buffer.type = buffer.type;
  m_buffer.device = 0;
  m_buffer.host = nullptr;
  m_buffer.dim = &native_desribe->dims[0];
  m_buffer.flags = 0;
  for (int i = 0; i < buffer.dimensions; ++i) {
    m_buffer.dim[i].extent = buffer.dim[i].extent;
  }

  native_desribe->dimension_format = DATA_FORMAT_NCHW;
  
  // format mapping
  // auto origin_type = tensor

  TensorUtils::set_linear_layout(this);

  for (int i = m_buffer.dimensions; i < 4; i++) {
    m_buffer.dim[i].extent = 1;
  }

  if (alloc_memory) {
    auto memory_size = size();
    if (memory_size > 0) {
      native_desribe->memoryType = Tensor::InsideDescribe::MEMORY_HOST;
      m_buffer.host = (uint8_t*)memory_alloc_align(size(), MEMORY_ALIGN_DEFAULT);
    }
  }
}

Tensor::Tensor(bool deepCopy, const Tensor* tensor) {
    m_describe = new InsideDescribe;
    m_describe->m_content = tensor->m_describe->m_content;
    m_describe->setBackend(tensor->m_describe->getBackend());
    m_describe->mem = tensor->m_describe->mem;
    m_buffer.dim = TensorUtils::get_describe(tensor)->dims;
    m_buffer.type = tensor->getType();
    m_buffer.device = tensor->deviceId();
    m_buffer.host = tensor->buffer().host;
    m_buffer.dimensions = tensor->buffer().dimensions;
    m_buffer.flags = tensor->buffer().flags;
}

Tensor::~Tensor() {
    auto native_desribe = m_describe->m_content.get();
    if (native_desribe->memoryType == InsideDescribe::MEMORY_HOST) {
        if (nullptr != m_buffer.host) {
            memory_free_align(m_buffer.host);
        }
    }
    delete m_describe;
}

Tensor* Tensor::create_device(const std::vector<int>& dims, halide_type_t type) {
    auto shapeTensor = new Tensor((int)dims.size());
    for (int i = 0; i < dims.size(); ++i) {
        shapeTensor->setLength(i, dims[i]);
    }
    shapeTensor->buffer().type = type;
    TensorUtils::set_linear_layout(shapeTensor);
    return shapeTensor;
}

Tensor* Tensor::create(const std::vector<int>& dims, halide_type_t type, void* userData) {
    Tensor shapeTensor((int)dims.size());
    for (int i = 0; i < dims.size(); ++i) {
        shapeTensor.setLength(i, dims[i]);
    }
    shapeTensor.buffer().type = type;

    bool ownData = userData == nullptr;
    auto result  = new Tensor(&shapeTensor, ownData);
    if (nullptr != userData) {
        result->buffer().host = (uint8_t*)userData;
    }
    return result;
}

Tensor* Tensor::clone(const Tensor* src, bool deepCopy) {
    return new Tensor(deepCopy, src);
}


bool Tensor::copy_from_host_tensor(const Tensor* hostTensor) {
    auto bn = m_describe->getBackend();
    if (nullptr == bn) {
        return false;
    }
    bn->on_copy_buffer(hostTensor, this);
    return true;
}

bool Tensor::copy_to_host_tensor(Tensor* hostTensor) const {
    auto bn = m_describe->getBackend();
    if (nullptr == bn) {
        return false;
    }
    bn->on_copy_buffer(this, hostTensor);
    return true;
}

Tensor* Tensor::create_host_tensor_from_device(const Tensor* device, bool copyContent) {
    auto tensor = Tensor::create(device->shape(), device->getType(), nullptr);
    if (copyContent) {
        device->copy_to_host_tensor(tensor);
    }
    return tensor;
}


Tensor::HandleDataType Tensor::get_handle_data_type() const {
    if (halide_type_handle != m_buffer.type.code) {
        return HANDLE_NONE;
    }
    return HANDLE_STRING;
}
void Tensor::setType(int type) {
    auto nativeDescribe = m_describe->m_content.get();
    switch (type) {
        case DataType_DT_DOUBLE:
        case DataType_DT_FLOAT:
            m_buffer.type = halide_type_of<float>();
            break;
        case DataType_DT_BFLOAT16:
            m_buffer.type = halide_type_t(halide_type_bfloat, 16);
            break;
        case DataType_DT_QINT32:
        case DataType_DT_INT32:
        case DataType_DT_BOOL:
        case DataType_DT_INT64:
            m_buffer.type = halide_type_of<int32_t>();
            break;
        case DataType_DT_QINT8:
        case DataType_DT_INT8:
            m_buffer.type = halide_type_of<int8_t>();
            break;
        case DataType_DT_QUINT8:
        case DataType_DT_UINT8:
            m_buffer.type = halide_type_of<uint8_t>();
            break;
        case DataType_DT_QUINT16:
        case DataType_DT_UINT16:
            m_buffer.type = halide_type_of<uint16_t>();
            break;
        case DataType_DT_QINT16:
        case DataType_DT_INT16:
            m_buffer.type = halide_type_of<int16_t>();
            break;
        default:
            printf("Unsupported data type! %d\n", type);
            assert(false);
            break;
    }
}

std::vector<int> Tensor::shape() const {
    std::vector<int> result;
    for (int i = 0; i < m_buffer.dimensions; ++i) {
        result.push_back(m_buffer.dim[i].extent);
    }
    return result;
}
template <typename T>
void printData(const Tensor* tensor, const void* data, const char* fmt) {
    const T* buffer = (const T*)data;
    if (tensor->dimensions() != 4) {
        auto size = tensor->elementSize();
        for (int i = 0; i < size; i++) {
            printf(fmt, buffer[i]);
        }
        printf("\n");
        return;
    }

    // auto tf      = tensor->getDimensionType() == Tensor::TENSORFLOW;
    auto batch   = tensor->batch();
    auto channel = tensor->channel();
    auto height  = tensor->height();
    auto width   = tensor->width();

    auto unit = sizeof(T);
    

        auto bytesPerRow   = width * unit;
        auto bytesPerImage = height * bytesPerRow;
        auto bytesPerBatch = channel * bytesPerImage;

        for (int b = 0; b < batch; b++) {
            auto bytes = buffer + b * bytesPerBatch / unit;
            printf("batch %d:\n", b);

            for (int c = 0; c < channel; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        printf(fmt, bytes[c * width * height + h * width + w]);
                    }
                    printf("\n");
                }
                printf("--------------\n");
            }
        }
    
}
void Tensor::print() const {
    // print dimensions
    printf("====== Tensor %p ======", this);
    printf("\nDimension: ");
    for (int i = 0; i < m_buffer.dimensions; i++) {
        printf("%d, ", m_buffer.dim[i].extent);
    }

    // convert to host if needed
    auto printee = this;
    bool device  = this->buffer().host == NULL && this->buffer().device != 0;
    if (device) {
        printee = this->create_host_tensor_from_device(this, true);
    }
    auto buffer = printee->buffer().host;

    printf("\nData: ");
    if (printee->getType().code == halide_type_int) {
        if (printee->getType().bits == 8) { // int8
            printData<int8_t>(printee, buffer, "%d, ");
        } else if (printee->getType().bits == 16) { // int16
            printData<int16_t>(printee, buffer, "%d, ");
        } else if (printee->getType().bits == 32) { // int32
            printData<int32_t>(printee, buffer, "%d, ");
        } else {
            printf("\nunsupported data type");
        }
    } else if (printee->getType().code == halide_type_uint) {
        if (printee->getType().bits == 8) { // uint8
            printData<uint8_t>(printee, buffer, "%d, ");
        } else {
            printf("\nunsupported data type");
        }
    } else if (printee->getType().code == halide_type_float) {
        if (printee->getType().bits == 32) { // float32
            printData<float>(printee, buffer, "%f, ");
        } else {
            printf("\nunsupported data type\n");
        }
    } else {
        printf("\nunsupported data type");
    }

    // clean up
    if (printee != this) {
        delete printee;
    }
}

void Tensor::printShape() const {
    const int dims = this->dimensions();
    printf("\t**Tensor shape**: ");
    if (dims == 0) {
        printf("\t*Scalar*");
    }
    for (int i = 0; i < dims; ++i) {
        printf("%d, ", this->length(i));
    }
    printf("\n");
}

size_t Tensor::usize() const {
    size_t dataSize = m_buffer.type.bytes();
    assert(dataSize >= 1);
    auto nativeDescribe = m_describe->m_content.get();
    for (int i = 0; i < this->buffer().dimensions; i++) {
        int currentDimSize = m_buffer.dim[i].extent;
        if (nativeDescribe->dimension_format == DATA_FORMAT_NC4HW4 && 1 == i) {
            currentDimSize = ALIGN_UP4(currentDimSize);
        }
        dataSize *= currentDimSize;
    }
    return dataSize;
}

int Tensor::size() const {
    return static_cast<int>(usize());
}

void* Tensor::map(MapType mtype) {
    auto nativeDescribe = m_describe;
    auto bn = nativeDescribe->getBackend();
    if (nullptr == bn) {
        return m_buffer.host;
    }

    auto mapPtr = bn->on_map_tensor(mtype, this);
    if(mapPtr != nullptr) {
        // Get mapPtr in specific backend
        return mapPtr;
    }

    /* Common backend */
    auto needSize = this->size();
    void* hostPtr = malloc(needSize);

    if(mtype == Tensor::MAP_TENSOR_READ) {
        //tmpTensor alloc
        Tensor tmp_tensor(this, false);
        tmp_tensor.buffer().host = (uint8_t *)hostPtr;

        //use onCopyBuffer
        bn->on_copy_buffer(this, &tmp_tensor);
    }
    return hostPtr;
}

void Tensor::unmap(MapType mtype, void *mapPtr) {
    auto nativeDescribe = m_describe;
    auto bn = nativeDescribe->getBackend();
    if (nullptr == bn) {
        return;
    }

    bool ret = bn->on_unmap_tensor(mtype, this, mapPtr);
    if(true == ret) {
        //do unmap already, just return
        return;
    }

    if(mtype == Tensor::MAP_TENSOR_WRITE) {
        //srcTensor alloc
        Tensor src_tensor(this, false);
        src_tensor.buffer().host = (uint8_t *)mapPtr;

        //use onCopyBuffer
        bn->on_copy_buffer(&src_tensor, this);
    }
    if(mapPtr != nullptr) {
        free(mapPtr);
        mapPtr = nullptr;
    }
}
int Tensor::wait(MapType mtype, bool finish) {
    auto nativeDescribe = m_describe;
    auto bn = nativeDescribe->getBackend();
    if (nullptr == bn) {
        return 0;
    }
    return bn->on_sync(mtype, finish, this);
}

bool Tensor::setDevicePtr(const void* devicePtr, int memoryType) {
    m_buffer.flags = memoryType;
    m_buffer.device = (uint64_t)devicePtr;
    // To use memoryType afterwards
    return true;
}

void Tensor::destroy(Tensor* tensor) {
    if (nullptr != tensor) {
        delete tensor;
    }
}
bool Tensor::getDeviceInfo(void* dst, int type) const {
    auto nativeDescribe = m_describe;
    if (nullptr == nativeDescribe->getBackend()) {
        return false;
    }
    if (nativeDescribe->getBackend()->type() != type) {
        return false;
    }
    return nativeDescribe->getBackend()->on_get_tensor_info(this, dst);
}

} // namespace tactics