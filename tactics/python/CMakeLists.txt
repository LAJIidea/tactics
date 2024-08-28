include(${PROJECT_SOURCE_DIR}/cmake/PythonNumpyPybind11.cmake)

file(GLOB pybind_src
  *.cpp
)

pybind11_add_module(tactics_bind ${pybind_src})

target_include_directories(tactics_bind PRIVATE ${PROJECT_SOURCE_DIR}/tpl/fmt/include)
target_link_libraries(tactics_bind PRIVATE tactics fmt)