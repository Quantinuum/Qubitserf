cmake_minimum_required(VERSION 3.20)
project(qubitserf LANGUAGES CXX)

# ---------------------------------------------------------------------------------------
# qubitserf builds TWO independent native shared libraries, each exposing a flat extern "C"
# ABI loaded from its Python subpackage via ctypes (no pybind11 / Cython):
#
#   * libdistfind  (src/distfind, include/distfind)  -> qubitserf/distfind/_lib
#       exact code distance: Brouwer-Zimmermann, connected-cluster, meet-in-the-middle.
#   * libcodeaut   (src/codeaut,  include/codeaut)   -> qubitserf/codeaut/_lib
#       automorphism groups: Leon + Brouwer-Zimmermann low-weight enumeration.
#
# Each library optionally carries CUDA / Metal GPU backends.  For a dev build (build.sh) the
# libraries drop straight into their Python subpackage's _lib/ so they import without an
# install; for a wheel build (scikit-build-core sets SKBUILD) they are installed there instead.
# ---------------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

find_package(Threads REQUIRED)

option(DISTFIND_METAL "Build the Metal GPU backend for distfind (Apple only)"   ON)
option(DISTFIND_CUDA  "Build the CUDA GPU backend for distfind if found"        ON)
option(DISTFIND_TESTS "Build distfind C++ test executables"                     ON)
option(DISTFIND_CLI   "Build the native distfind command-line tool"             ON)
option(CODEAUT_METAL  "Build the Metal BZ backend for codeaut (Apple only)"     ON)
option(CODEAUT_CUDA   "Build the CUDA BZ backend for codeaut if found"          ON)

# ---- Detect optional languages once, shared by both libraries --------------------------
set(_HAVE_CUDA OFF)
if(DISTFIND_CUDA OR CODEAUT_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(_HAVE_CUDA ON)
    message(STATUS "qubitserf: CUDA toolkit found (${CMAKE_CUDA_COMPILER})")
  else()
    message(STATUS "qubitserf: no CUDA toolkit; CUDA backends disabled")
  endif()
endif()

set(_HAVE_OBJCXX OFF)
if(APPLE AND (DISTFIND_METAL OR CODEAUT_METAL))
  enable_language(OBJCXX)
  set(_HAVE_OBJCXX ON)
endif()

# =======================================================================================
# distfind
# =======================================================================================
set(DISTFIND_CORE_SOURCES
  src/distfind/gf2.cpp
  src/distfind/css.cpp
  src/distfind/stab.cpp
  src/distfind/bz.cpp
  src/distfind/mitm.cpp
  src/distfind/cc.cpp
  src/distfind/op_weight.cpp
  src/distfind/backend_cpu.cpp
  src/distfind/capi.cpp)
set(DISTFIND_GPU_SOURCES "")
set(DISTFIND_DEFS "")
set(DISTFIND_GPU_LIBS "")

if(_HAVE_OBJCXX AND DISTFIND_METAL)
  list(APPEND DISTFIND_GPU_SOURCES src/distfind/metal/backend_metal.mm)
  list(APPEND DISTFIND_DEFS DISTFIND_METAL=1)
  set_source_files_properties(src/distfind/metal/backend_metal.mm PROPERTIES
    COMPILE_OPTIONS "-fobjc-arc")
  list(APPEND DISTFIND_GPU_LIBS "-framework Metal" "-framework Foundation")
  message(STATUS "distfind: Metal backend enabled")
endif()
if(_HAVE_CUDA AND DISTFIND_CUDA)
  list(APPEND DISTFIND_GPU_SOURCES src/distfind/cuda/backend_cuda.cu)
  list(APPEND DISTFIND_DEFS DISTFIND_CUDA=1)
  message(STATUS "distfind: CUDA backend enabled")
endif()

add_library(distfind SHARED ${DISTFIND_CORE_SOURCES} ${DISTFIND_GPU_SOURCES})
target_include_directories(distfind PUBLIC include)
target_compile_definitions(distfind PRIVATE ${DISTFIND_DEFS})
target_link_libraries(distfind PRIVATE ${DISTFIND_GPU_LIBS} Threads::Threads)

if(NOT SKBUILD)
  set_target_properties(distfind PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/qubitserf/distfind/_lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/qubitserf/distfind/_lib)
endif()
install(TARGETS distfind
  LIBRARY DESTINATION qubitserf/distfind/_lib
  RUNTIME DESTINATION qubitserf/distfind/_lib)

if(DISTFIND_TESTS)
  add_executable(distfind_smoke        src/distfind/tests/smoke.cpp)
  add_executable(distfind_compare      src/distfind/tests/backend_compare.cpp)
  add_executable(distfind_op_smoke     src/distfind/tests/op_smoke.cpp)
  add_executable(distfind_cpu_enum_ref src/distfind/tests/cpu_enum_ref.cpp)
  foreach(t distfind_smoke distfind_compare distfind_op_smoke distfind_cpu_enum_ref)
    target_link_libraries(${t} PRIVATE distfind)
  endforeach()
endif()

# Native distfind CLI (not built for the wheel; there the `distfind` console script is the CLI).
if(DISTFIND_CLI)
  add_executable(distfind_cli src/distfind/cli/distfind_cli.cpp)
  target_link_libraries(distfind_cli PRIVATE distfind)
  set_target_properties(distfind_cli PROPERTIES OUTPUT_NAME distfind)
endif()

# =======================================================================================
# codeaut
# =======================================================================================
file(GLOB CODEAUT_CPU_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/codeaut/*.cpp")
set(CODEAUT_GPU_SOURCES "")
set(CODEAUT_DEFS "")
set(CODEAUT_GPU_LIBS "")

if(_HAVE_CUDA AND CODEAUT_CUDA)
  list(APPEND CODEAUT_GPU_SOURCES "${CMAKE_SOURCE_DIR}/src/codeaut/cuda/backend_cuda.cu")
  list(APPEND CODEAUT_DEFS CODEAUT_CUDA=1)
  message(STATUS "codeaut: CUDA backend enabled")
endif()
if(_HAVE_OBJCXX AND CODEAUT_METAL)
  list(APPEND CODEAUT_GPU_SOURCES "${CMAKE_SOURCE_DIR}/src/codeaut/metal/backend_metal.mm")
  list(APPEND CODEAUT_DEFS CODEAUT_METAL=1)
  set_source_files_properties("${CMAKE_SOURCE_DIR}/src/codeaut/metal/backend_metal.mm"
    PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
  list(APPEND CODEAUT_GPU_LIBS "-framework Metal" "-framework Foundation")
  message(STATUS "codeaut: Metal backend enabled")
endif()

add_library(codeaut SHARED ${CODEAUT_CPU_SOURCES} ${CODEAUT_GPU_SOURCES})
target_include_directories(codeaut PRIVATE include "include/codeaut")
target_compile_definitions(codeaut PRIVATE ${CODEAUT_DEFS})
target_link_libraries(codeaut PRIVATE ${CODEAUT_GPU_LIBS} Threads::Threads)
set_target_properties(codeaut PROPERTIES PREFIX "lib" OUTPUT_NAME "codeaut")

if(NOT SKBUILD)
  set_target_properties(codeaut PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/qubitserf/codeaut/_lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/qubitserf/codeaut/_lib)
endif()
install(TARGETS codeaut
  LIBRARY DESTINATION qubitserf/codeaut/_lib
  RUNTIME DESTINATION qubitserf/codeaut/_lib)
