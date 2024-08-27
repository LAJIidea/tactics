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
#include <pybind11/buffer_info.h>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <vector>
#include "HalideRuntime.h"
#include "tactics/core/tensor.h"
#include "tactics/math/matrix.h"

namespace py = pybind11;
using namespace tactics;

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

PYBIND11_MODULE(tactics_bind, m) {
  py::class_<Tensor>(m, "Tensor")
    .def(py::init(&create_tensor<int>), py::arg("shape"), py::arg("data"))
    .def(py::init(&create_tensor<float>), py::arg("shape"), py::arg("data"))
    .def("dump", [&](const Tensor& self) {
      Matrix::print(&self);
    });
}