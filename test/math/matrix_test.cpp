#include <tactics/math/matrix.h>
#include <tactics/core/tensor.h>

using namespace tactics;

int main() {
  uint8_t data1[] = {
    0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02,
    0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02,
  };
  Tensor *tensor1 = Tensor::create<uint8_t>({1, 2, 3, 4}, data1);

  uint8_t data2[] = {
    0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02,
    0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02, 0x01, 0x02,
  };
  Tensor *tensor2 = Tensor::create<uint8_t>({1, 2, 3, 4}, data2);

  
}