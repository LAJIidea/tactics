add_subdirectory(core)

file(GLOB TACTICS_SRC 
            tactics.cpp
            math/math.cpp
            math/common.cpp)

add_library(tactics ${TACTICS_SRC})

target_link_libraries(tactics tactics_core)