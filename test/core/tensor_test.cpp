#include <cassert>
#include <tactics/core/tensor.h>

using namespace tactics;

int main() {
  Tensor base(3);
  assert(base.dimensions() == 3 && "dimension error");
  assert(base.getType().bits == 32 && "type width error");
  assert(base.getType().code == halide_type_float && "type error");
  assert(base.host<void>() == nullptr && "host error");
  assert(base.deviceId() == 0 && "device error");

  assert(base.length(0) == 0 && "dim[0] extent error");
  assert(base.length(1) == 0 && "dim[1] extent error");
  assert(base.length(2) == 0 && "dim[2] extent error");
  base.setLength(0, 3);
  base.setLength(1, 5);
  base.setLength(2, 7);
  assert(base.stride(0) == 0 && "dim[0] stride error");
  assert(base.stride(1) == 0 && "dim[1] stride error");
  assert(base.stride(2) == 0 && "dim[2] stride error");

  {
  Tensor *tensor = Tensor::create_device<int16_t>({1, 2, 3, 4});
  assert(tensor->dimensions() == 4);
  assert(tensor->getType().bits == 16);
  assert(tensor->getType().code == halide_type_int);
  assert(tensor->host<void>() == nullptr);
  assert(tensor->deviceId() == 0);
  assert(tensor->length(0) == 1);
  assert(tensor->length(1) == 2);
  assert(tensor->length(2) == 3);
  assert(tensor->length(3) == 4);
  delete tensor;
  }

  uint8_t data[] = {
    0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02,
    0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02,
  };
  Tensor *tensor = Tensor::create<uint8_t>({1, 2, 3, 4}, data);
  assert(tensor->dimensions() == 4);
  assert(tensor->getType().bits == 8);
  assert(tensor->getType().code == halide_type_uint);
  assert(tensor->host<void>() != nullptr);
  assert(tensor->deviceId() == 0);
  assert(tensor->length(0) == 1);
  assert(tensor->length(1) == 2);
  assert(tensor->length(2) == 3);
  assert(tensor->length(3) == 4);
  assert(tensor->elementSize() == 1 * 2 * 3 * 4);
  for (int i = 0; i < tensor->elementSize(); i++) {
      assert(tensor->host<uint8_t>()[i] == data[i]);
  }
  delete tensor;
}