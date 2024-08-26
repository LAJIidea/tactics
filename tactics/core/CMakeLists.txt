file(GLOB CORE_SRC 
          tensor.cpp
          tensor_utils.cpp
          memory_utils.cpp
          buffer_alloc.cpp
          backend.cpp)

add_library(tactics_tensor ${CORE_SRC})