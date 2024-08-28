//===------------------------tactics/pytthon/binding.cpp------------------------===//
//
// Copyright (c) RISC-X Organizations, see https://risc-x.org
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
//===--------------------------------------------------------------------------===//
//
/// This file defines the python binding
///
//===--------------------------------------------------------------------------===//
#include <iostream>
#include <ostream>
#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "HalideRuntime.h"
#include "tactics/core/tensor.h"
#include "tactics/math/matrix.h"

namespace py = pybind11;
using namespace tactics;
using namespace fmt;

template<typename T>
Tensor* create_tensor(const std::vector<int> &shape, py::array_t<T> data) {
  py::buffer_info buf = data.request();

  void *ptr = static_cast<void*>(buf.ptr);
  halide_type_t ty = halide_type_of<T>();
  // if (buf.format == py::format_descriptor<int>::format()) {
  //   ptr = static_cast<void*>(buf.ptr);
  //   ty = halide_type_of<int>(); 
  // } else if (buf.format == py::format_descriptor<float>::format()) {
  //   ptr = static_cast<void*>(buf.ptr);
  //   ty = halide_type_of<float>(); 
  // } else {
  //   throw std::runtime_error("Unsupported data type");
  // }

  return Tensor::create(shape, ty, ptr);
}

std::string tensor_to_string(const Tensor &tensor) {
  std::ostringstream oss;
  auto data = tensor.host<float>();
  auto shape = tensor.shape();
  int total_elements = 1;
  for (int dim : shape) {
    total_elements *= dim;
  }
  int dimension = shape.size();
  std::vector<int> counters(dimension, 0); // trace dimension index

  // handler outer [
  for (int i = 0; i < shape.size() - 1; ++i) {
    if (counters[i] == 0) {
      oss << "[";
    }
  }

  for (int i = 0; i < total_elements; ++i) {

    // print element
    oss << data[i];

    // update index counter
    for (int j = shape.size() - 1; j >= 0; --j) {
      counters[j]++;
      if (counters[j] < shape[j])
        break;
      counters[j] = 0;
      oss << "]";
      if (j > 0) {
        oss << ",\n";
        for (int k = 0; k < j; ++k)
          oss << " ";
      }
    }

    // add ', '
    if (i < total_elements - 1) {
      oss << ", ";
    }
  }
  oss << "]";

  return oss.str();
}

void print_tensor_recursive(const Tensor &tensor, int depth = 0, int offset = 0) {
  if (depth == tensor.shape().size() - 1) {
    // print inner data
    for (int i = 0; i < tensor.shape()[depth]; ++i) {
      std::cout << tensor.host<float>()[offset + i] << " ";
    }
    std::cout << std::endl;
  } else {
    // recursive print high dimension element
    int stride = 1;
    for (int i = depth + 1; i < tensor.shape().size(); ++i) {
      stride *= tensor.shape()[i];
    }
    for (int i = 0; i < tensor.shape()[depth]; ++i) {
      print_tensor_recursive(tensor, depth + 1, offset + i * stride);
    }
    std::cout << std::endl; // end line
  }
}

void print_tensor(const Tensor &tensor) {
  int total_elements = 1;
  for (int dim : tensor.shape()) {
    total_elements *= dim;
  }

  int dimension = tensor.shape().size();
  std::vector<int> counters(dimension, 0); // trace dimension index

  for (int i = 0; i < total_elements; ++i) {
    // print elements
    std::cout << tensor.host<float>()[i] << " ";

    // update index counter
    for (int j = dimension - 1; j >= 0; --j) {
      counters[j]++;
      if (counters[j] < tensor.shape()[j]) {
        break;
      }
      // reset current dimension index, end line
      counters[j] = 0;
      if (j > 0) {
        std::cout << std::endl;
      }
    }
  }
  std::cout << std::endl;
}

PYBIND11_MODULE(tactics_bind, m) {
  py::class_<Tensor>(m, "Tensor")
    .def(py::init(&create_tensor<int>), py::arg("shape"), py::arg("data"))
    .def(py::init(&create_tensor<float>), py::arg("shape"), py::arg("data"))
    .def("matrix_dump", [&](const Tensor& self) {
      Matrix::print(&self);
    })
    .def("dump", [&](const Tensor& self) {
      print_tensor_recursive(self);
    })
    .def("__str__", &tensor_to_string);
}