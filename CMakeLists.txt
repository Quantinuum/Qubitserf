cmake_minimum_required(VERSION 3.20)
project(qminweight LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

option(QMINWEIGHT_METAL "Build the Metal GPU backend (Apple only)" ON)
option(QMINWEIGHT_CUDA  "Build the CUDA GPU backend if a CUDA toolkit is found" ON)

set(QMINWEIGHT_CORE_SOURCES
  src/gf2.cpp
  src/css.cpp
  src/stab.cpp
  src/bz.cpp
  src/mitm.cpp
  src/cc.cpp
  src/op_weight.cpp
  src/backend_cpu.cpp
  src/capi.cpp)

set(QMINWEIGHT_GPU_SOURCES "")
set(QMINWEIGHT_GPU_LIBS "")
set(QMINWEIGHT_DEFS "")

# ---- Metal backend (Apple) -------------------------------------------------------
if(APPLE AND QMINWEIGHT_METAL)
  enable_language(OBJCXX)
  list(APPEND QMINWEIGHT_GPU_SOURCES src/metal/backend_metal.mm)
  list(APPEND QMINWEIGHT_DEFS QMINWEIGHT_METAL=1)
  set_source_files_properties(src/metal/backend_metal.mm PROPERTIES
    COMPILE_OPTIONS "-fobjc-arc")
  list(APPEND QMINWEIGHT_GPU_LIBS "-framework Metal" "-framework Foundation")
  message(STATUS "qminweight: Metal backend enabled")
endif()

# ---- CUDA backend (optional) -----------------------------------------------------
if(QMINWEIGHT_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    list(APPEND QMINWEIGHT_GPU_SOURCES src/cuda/backend_cuda.cu)
    list(APPEND QMINWEIGHT_DEFS QMINWEIGHT_CUDA=1)
    set(CMAKE_CUDA_STANDARD 17)
    message(STATUS "qminweight: CUDA backend enabled (${CMAKE_CUDA_COMPILER})")
  else()
    message(STATUS "qminweight: CUDA toolkit not found, skipping CUDA backend")
  endif()
endif()

add_library(qminweight SHARED ${QMINWEIGHT_CORE_SOURCES} ${QMINWEIGHT_GPU_SOURCES})
target_include_directories(qminweight PUBLIC include)
target_compile_definitions(qminweight PRIVATE ${QMINWEIGHT_DEFS})
target_link_libraries(qminweight PRIVATE ${QMINWEIGHT_GPU_LIBS})
find_package(Threads REQUIRED)
target_link_libraries(qminweight PRIVATE Threads::Threads)

# For a plain/dev build (build.sh) drop the dylib straight into the Python package so it is
# importable without installing. For a wheel build (scikit-build-core sets SKBUILD) we
# instead install it into the wheel via the install() rule below.
if(NOT SKBUILD)
  set_target_properties(qminweight PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/qminweight/_lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/qminweight/_lib)
endif()

# Install rule used when building a wheel: the shared library lands inside the package at
# qminweight/_lib so the ctypes loader (python/qminweight/_native.py) finds it.
install(TARGETS qminweight
  LIBRARY DESTINATION qminweight/_lib
  RUNTIME DESTINATION qminweight/_lib)

# ---- tests -----------------------------------------------------------------------
option(QMINWEIGHT_TESTS "Build C++ test executables" ON)
if(QMINWEIGHT_TESTS)
  add_executable(qminweight_smoke src/tests/smoke.cpp)
  target_link_libraries(qminweight_smoke PRIVATE qminweight)
  add_executable(qminweight_compare src/tests/backend_compare.cpp)
  target_link_libraries(qminweight_compare PRIVATE qminweight)
  add_executable(qminweight_op_smoke src/tests/op_smoke.cpp)
  target_link_libraries(qminweight_op_smoke PRIVATE qminweight)
endif()

# ---- C++ command-line tool (Qubitserf-style) -------------------------------------
# Not built for the Python wheel (scikit-build sets QMINWEIGHT_CLI=OFF); there the
# `qminweight` console script (qminweight.cli:main) is the command-line entry point.
option(QMINWEIGHT_CLI "Build the native qminweight command-line tool" ON)
if(QMINWEIGHT_CLI)
  add_executable(qminweight_cli src/cli/qminweight_cli.cpp)
  target_link_libraries(qminweight_cli PRIVATE qminweight)
  set_target_properties(qminweight_cli PROPERTIES OUTPUT_NAME qminweight)
endif()
