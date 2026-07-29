cmake_minimum_required(VERSION 3.20)
project(qubitserf LANGUAGES CXX)

# ---------------------------------------------------------------------------------------
# qubitserf builds ONE native shared library, libqubitserf, exposing both flat extern "C"
# ABIs loaded from Python via ctypes (no pybind11 / Cython):
#
#   * distfind_*              (src/distfind, include/distfind)
#       exact code distance: Brouwer-Zimmermann, connected-cluster, meet-in-the-middle.
#   * codeaut_bz_* / qaut_*   (src/codeaut, include/codeaut)
#       automorphism groups: Leon + Brouwer-Zimmermann low-weight enumeration.
#
# Both engines run on the SHARED core in src/core + include/qsf (bit-packed GF(2),
# combinatorics, the two-level enumeration kernel, and the CPU/CUDA/Metal backends --
# one Backend serving the min-weight mode for distfind and the collect mode for codeaut).
#
# For a dev build (build.sh) the library drops into python/qubitserf/_lib so it imports
# without an install; for a wheel build (scikit-build-core sets SKBUILD) it is installed
# there instead.
# ---------------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

find_package(Threads REQUIRED)

option(QUBITSERF_METAL "Build the Metal GPU backend (Apple only)"        ON)
option(QUBITSERF_CUDA  "Build the CUDA GPU backend if a toolkit is found" ON)
option(QUBITSERF_TESTS "Build the C++ test executables"                  ON)
option(QUBITSERF_CLI   "Build the native distfind command-line tool"     ON)

# ---- Detect optional languages --------------------------------------------------------
set(_HAVE_CUDA OFF)
if(QUBITSERF_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(_HAVE_CUDA ON)
    message(STATUS "qubitserf: CUDA toolkit found (${CMAKE_CUDA_COMPILER})")
  else()
    message(STATUS "qubitserf: no CUDA toolkit; CUDA backend disabled")
  endif()
endif()

set(_HAVE_OBJCXX OFF)
if(APPLE AND QUBITSERF_METAL)
  enable_language(OBJCXX)
  set(_HAVE_OBJCXX ON)
endif()

# ---- the one library ------------------------------------------------------------------
set(QUBITSERF_SOURCES
  # shared core
  src/core/gf2.cpp
  src/core/backend_cpu.cpp
  # distfind engine (min-weight mode)
  src/distfind/css.cpp
  src/distfind/stab.cpp
  src/distfind/bz.cpp
  src/distfind/mitm.cpp
  src/distfind/cc.cpp
  src/distfind/op_weight.cpp
  src/distfind/capi.cpp
  # codeaut engine (collect mode + Leon)
  src/codeaut/leon.cpp
  src/codeaut/bz.cpp
  src/codeaut/capi.cpp)
set(QUBITSERF_DEFS "")
set(QUBITSERF_GPU_LIBS "")

if(_HAVE_OBJCXX)
  list(APPEND QUBITSERF_SOURCES src/core/metal/backend_metal.mm)
  list(APPEND QUBITSERF_DEFS QSF_METAL=1)
  set_source_files_properties(src/core/metal/backend_metal.mm PROPERTIES
    COMPILE_OPTIONS "-fobjc-arc")
  list(APPEND QUBITSERF_GPU_LIBS "-framework Metal" "-framework Foundation")
  message(STATUS "qubitserf: Metal backend enabled")
endif()
if(_HAVE_CUDA)
  list(APPEND QUBITSERF_SOURCES src/core/cuda/backend_cuda.cu)
  list(APPEND QUBITSERF_DEFS QSF_CUDA=1)
  message(STATUS "qubitserf: CUDA backend enabled")
endif()

add_library(qubitserf SHARED ${QUBITSERF_SOURCES})
target_include_directories(qubitserf PUBLIC include)
# codeaut sources include their own headers unqualified ("capi.h").
target_include_directories(qubitserf PRIVATE include/codeaut)
target_compile_definitions(qubitserf PRIVATE ${QUBITSERF_DEFS})
target_link_libraries(qubitserf PRIVATE ${QUBITSERF_GPU_LIBS} Threads::Threads)

if(NOT SKBUILD)
  set_target_properties(qubitserf PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/qubitserf/_lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/qubitserf/_lib)
endif()
install(TARGETS qubitserf
  LIBRARY DESTINATION qubitserf/_lib
  RUNTIME DESTINATION qubitserf/_lib)

if(QUBITSERF_TESTS)
  add_executable(distfind_smoke        src/distfind/tests/smoke.cpp)
  add_executable(distfind_compare      src/distfind/tests/backend_compare.cpp)
  add_executable(distfind_op_smoke     src/distfind/tests/op_smoke.cpp)
  add_executable(distfind_cpu_enum_ref src/distfind/tests/cpu_enum_ref.cpp)
  foreach(t distfind_smoke distfind_compare distfind_op_smoke distfind_cpu_enum_ref)
    target_link_libraries(${t} PRIVATE qubitserf)
  endforeach()
endif()

# Native distfind CLI (not built for the wheel; there the `distfind` console script is the CLI).
if(QUBITSERF_CLI)
  add_executable(distfind_cli src/distfind/cli/distfind_cli.cpp)
  target_link_libraries(distfind_cli PRIVATE qubitserf)
  set_target_properties(distfind_cli PROPERTIES OUTPUT_NAME distfind)
endif()
