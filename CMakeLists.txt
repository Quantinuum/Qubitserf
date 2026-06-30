cmake_minimum_required(VERSION 3.20)
project(qubitserf LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

option(QUBITSERF_METAL "Build the Metal GPU backend (Apple only)" ON)
option(QUBITSERF_CUDA  "Build the CUDA GPU backend if a CUDA toolkit is found" ON)

set(QUBITSERF_CORE_SOURCES
  src/gf2.cpp
  src/css.cpp
  src/stab.cpp
  src/bz.cpp
  src/mitm.cpp
  src/cc.cpp
  src/op_weight.cpp
  src/backend_cpu.cpp
  src/capi.cpp)

set(QUBITSERF_GPU_SOURCES "")
set(QUBITSERF_GPU_LIBS "")
set(QUBITSERF_DEFS "")

# ---- Metal backend (Apple) -------------------------------------------------------
if(APPLE AND QUBITSERF_METAL)
  enable_language(OBJCXX)
  list(APPEND QUBITSERF_GPU_SOURCES src/metal/backend_metal.mm)
  list(APPEND QUBITSERF_DEFS QUBITSERF_METAL=1)
  set_source_files_properties(src/metal/backend_metal.mm PROPERTIES
    COMPILE_OPTIONS "-fobjc-arc")
  list(APPEND QUBITSERF_GPU_LIBS "-framework Metal" "-framework Foundation")
  message(STATUS "qubitserf: Metal backend enabled")
endif()

# ---- CUDA backend (optional) -----------------------------------------------------
if(QUBITSERF_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    list(APPEND QUBITSERF_GPU_SOURCES src/cuda/backend_cuda.cu)
    list(APPEND QUBITSERF_DEFS QUBITSERF_CUDA=1)
    set(CMAKE_CUDA_STANDARD 17)
    message(STATUS "qubitserf: CUDA backend enabled (${CMAKE_CUDA_COMPILER})")
  else()
    message(STATUS "qubitserf: CUDA toolkit not found, skipping CUDA backend")
  endif()
endif()

add_library(qubitserf SHARED ${QUBITSERF_CORE_SOURCES} ${QUBITSERF_GPU_SOURCES})
target_include_directories(qubitserf PUBLIC include)
target_compile_definitions(qubitserf PRIVATE ${QUBITSERF_DEFS})
target_link_libraries(qubitserf PRIVATE ${QUBITSERF_GPU_LIBS})
find_package(Threads REQUIRED)
target_link_libraries(qubitserf PRIVATE Threads::Threads)

# For a plain/dev build (build.sh) drop the dylib straight into the Python package so it is
# importable without installing. For a wheel build (scikit-build-core sets SKBUILD) we
# instead install it into the wheel via the install() rule below.
if(NOT SKBUILD)
  set_target_properties(qubitserf PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/qubitserf/_lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/qubitserf/_lib)
endif()

# Install rule used when building a wheel: the shared library lands inside the package at
# qubitserf/_lib so the ctypes loader (python/qubitserf/_native.py) finds it.
install(TARGETS qubitserf
  LIBRARY DESTINATION qubitserf/_lib
  RUNTIME DESTINATION qubitserf/_lib)

# ---- tests -----------------------------------------------------------------------
option(QUBITSERF_TESTS "Build C++ test executables" ON)
if(QUBITSERF_TESTS)
  add_executable(qubitserf_smoke src/tests/smoke.cpp)
  target_link_libraries(qubitserf_smoke PRIVATE qubitserf)
  add_executable(qubitserf_compare src/tests/backend_compare.cpp)
  target_link_libraries(qubitserf_compare PRIVATE qubitserf)
  add_executable(qubitserf_op_smoke src/tests/op_smoke.cpp)
  target_link_libraries(qubitserf_op_smoke PRIVATE qubitserf)
endif()

# ---- C++ command-line tool (Qubitserf-style) -------------------------------------
# Not built for the Python wheel (scikit-build sets QUBITSERF_CLI=OFF); there the
# `qubitserf` console script (qubitserf.cli:main) is the command-line entry point.
option(QUBITSERF_CLI "Build the native qubitserf command-line tool" ON)
if(QUBITSERF_CLI)
  add_executable(qubitserf_cli src/cli/qubitserf_cli.cpp)
  target_link_libraries(qubitserf_cli PRIVATE qubitserf)
  set_target_properties(qubitserf_cli PROPERTIES OUTPUT_NAME qubitserf)
endif()
