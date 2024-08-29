cmake_minimum_required(VERSION 3.22)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# option()

project(tactics)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17)

option(USE_CUDA "Use CUDA" OFF)
option(USE_SSE "Use SSE optimization for x86 if possible" ON)
option(USE_RVV "Use Vector Extension for RISC-V if possible" OFF)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(X86_64)|(x64)|(X64)|(amd64)|(AMD64)|(i686)" AND USE_SSE)
  message(STATUS "${CMAKE_SYSTEM_PROCESSOR}: Open SSE")
  add_definitions(-DUSE_SSE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(riscv)" AND USE_RVV)
  message(STATUS "{CMAKE_SYSTEM_PROCESSOR}: Open Vector Extension")
  add_definitions(-DUSE_RVV)
endif()


include_directories(${CMAKE_SOURCE_DIR}/include)

add_subdirectory(tactics)
add_subdirectory(tpl)
add_subdirectory(test)