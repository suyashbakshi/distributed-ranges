# SPDX-FileCopyrightText: Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.20)
project(
  distributed_ranges
  VERSION 0.1
  DESCRIPTION "Distributed ranges")

include(FetchContent)

option(ENABLE_ISHMEM OFF)

# Project wide defaults, not needed when another project uses the library
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

  include(CheckLanguage)
  include(CTest)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    # Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24
    cmake_policy(SET CMP0135 NEW)
  endif()
  include(ExternalProject)

  option(DISABLED_TESTS "Run disabled tests" OFF)
  option(ENABLE_SYCL "Build sycl shp examples" OFF)
  option(ENABLE_CUDA "Build for cuda" OFF)
  option(ENABLE_FORMAT "Build with format library" ON)
  option(GCC_TOOLCHAIN, "GCC toolchain to be used by clang-based compilers" OFF)

  #
  # C++: generic configuration
  #
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_CXX_STANDARD 20)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-fconcepts-diagnostics-depth=10)
  endif()

  if(GCC_TOOLCHAIN)
    add_compile_options(--gcc-toolchain=${GCC_TOOLCHAIN})
    add_link_options(--gcc-toolchain=${GCC_TOOLCHAIN})
  endif()

  if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    set(ENABLE_SYCL on)
  endif()

  if(ENABLE_SYCL)
    add_compile_options(-fsycl)
    add_link_options(-fsycl)
    if(ENABLE_CUDA)
      add_compile_options(-fsycl-targets=nvptx64-nvidia-cuda
                          -Wno-error=unknown-cuda-version)
      add_link_options(-fsycl-targets=nvptx64-nvidia-cuda
                       -Wno-error=unknown-cuda-version)
    endif()
  endif()

  set(buildTypes Release Debug)

  if(NOT CMAKE_BUILD_TYPE)
    message(
      STATUS "No build type selected (CMAKE_BUILD_TYPE), defaulting to Release")
    set(CMAKE_BUILD_TYPE
        "Release"
        CACHE STRING "Choose the type of build, options are: Release Debug ..."
              FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${buildTypes})
  else()
    message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
    if(NOT CMAKE_BUILD_TYPE IN_LIST buildTypes)
      message(
        WARNING
          "Unusual build type was set, please make sure it's a proper one. "
          "Only following are supported by default: ${buildTypes}.")
    endif()
  endif()

  set(CMAKE_C_FLAGS_DEBUG "-O0 -g -ggdb")
  set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -ggdb")
  # When we define NDEBUG, ranges-v3 uses printf, which is not ok in sycl
  # kernels. When we do not define NDEBUG, google bench warns about timing
  # error. Need to investigate if there is another way to stop ranges-v3 from
  # using printf
  #
  # set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG") set(CMAKE_CXX_FLAGS_RELEASE "-O3
  # -DNDEBUG")
  set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

  #
  # Common dependencies for examples
  #
  find_package(MKL REQUIRED)
  find_package(MPI REQUIRED)
  find_package(OpenMP)
  message("MPI exec set to ${MPIEXEC_EXECUTABLE}")

  #
  # Google test
  #
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1)
  # For Windows: Prevent overriding the parent project's compiler/linker
  # settings
  set(gtest_force_shared_crt
      ON
      CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  #
  # Command-line options parser
  #
  FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.0.0)
  FetchContent_MakeAvailable(cxxopts)

endif()

FetchContent_Declare(
  range-v3
  GIT_REPOSITORY https://github.com/BenBrock/range-v3.git
  GIT_TAG 5300fe3)
FetchContent_MakeAvailable(range-v3)

FetchContent_Declare(
  cpp-format
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 0b0f7cf)
FetchContent_MakeAvailable(cpp-format)

FetchContent_Declare(
  mdspan
  GIT_REPOSITORY https://github.com/kokkos/mdspan.git
  GIT_TAG mdspan-0.6.0)
FetchContent_MakeAvailable(mdspan)

if(ENABLE_ISHMEM)

  include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
  link_directories(${CMAKE_CURRENT_BINARY_DIR}/lib)

  # use it if there are no L0 headers installed
  if(ENABLE_L0)
    ExternalProject_Add(
      level-zero
      GIT_REPOSITORY https://github.com/oneapi-src/level-zero.git
      GIT_TAG v1.14.0
      INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}
      CMAKE_CACHE_ARGS "-DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}"
    )

    set(SOS_L0_PARAM "-DL0_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}")
    set(SOS_DEPS "level-zero")
  else()
    set(SOS_L0_PARAM "")
    set(SOS_DEPS "")
  endif()

  # ExternalProject_Add( ofi GIT_REPOSITORY git@github.com:ofiwg/libfabric.git
  # GIT_TAG v1.19.0 SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/ofisrc DEPENDS
  # level-zero CONFIGURE_COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/ofisrc &&
  # ./autogen.sh && ./configure --prefix=${CMAKE_CURRENT_BINARY_DIR}
  # --enable-debug --enable-psm3=auto --enable-cxi=auto --enable-rxm=auto
  # --enable-tcp=auto BUILD_IN_SOURCE ON BUILD_COMMAND make INSTALL_COMMAND make
  # install)

  if(NOT DEFINED OFI_DIR)
    set(OFI_DIR /usr)
  endif()

  ExternalProject_Add(
    sos
    GIT_REPOSITORY git@github.com:Sandia-OpenSHMEM/SOS.git
    GIT_TAG main
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/sossrc
    DEPENDS ${SOS_DEPS}
    CONFIGURE_COMMAND
      cd ${CMAKE_CURRENT_BINARY_DIR}/sossrc && ./autogen.sh && ./configure
      --prefix=${CMAKE_CURRENT_BINARY_DIR} --enable-pmi-simple
      --with-ofi=${OFI_DIR} --disable-fortran --enable-mr-endpoint
      --enable-ofi-mr=basic --enable-ofi-manual-progress --disable-ofi-inject
      --enable-ofi-hmem
    BUILD_IN_SOURCE ON
    BUILD_COMMAND make
    INSTALL_COMMAND make install)

  ExternalProject_Add(
    ishmem
    GIT_REPOSITORY
      git@github.com:intel-innersource/libraries.runtimes.hpc.shmem.ishmem.git
    GIT_TAG main
    INSTALL_DIR ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS sos
    CMAKE_CACHE_ARGS
      "-DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}"
      "-DSHMEM_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}" ${SOS_L0_PARAM})

  add_custom_target(shmem DEPENDS ishmem)

  function(add_mhp_ctest test_name name processes)
    add_test(
      NAME ${test_name}
      COMMAND
        ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${processes}
        ${MPIEXEC_PREFLAGS} -outfile-pattern "${test_name}.out.%r.log"
        -errfile-pattern "${test_name}.err.%r.log"
        ${CMAKE_BINARY_DIR}/bin/ishmrun gdb --ex r -ex bt --ex q --batch --args
        ./${name} --log --logprefix=${test_name} ${ARGN} COMMAND_EXPAND_LISTS)
  endfunction()

  add_compile_definitions(DRISHMEM)

else()

  function(add_mhp_ctest test_name name processes)
    add_test(NAME ${test_name}
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${processes}
                     ${MPIEXEC_PREFLAGS} ./${name} ${ARGN} COMMAND_EXPAND_LISTS)
    set_property(TEST ${test_name} PROPERTY LABELS TESTLABEL MHP)
  endfunction()

endif()

function(add_mhp_offload_ctest test_name name processes)
  # intel mpi needs I_MPI_OFFLOAD=1, mpich does not need anything but it is
  # harmless so pass it anyway
  add_test(
    NAME ${test_name}
    COMMAND
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${processes}
      ${MPIEXEC_PREFLAGS} -genv I_MPI_OFFLOAD=1 ./${name} ${ARGN}
      COMMAND_EXPAND_LISTS)
  set_property(TEST ${test_name} PROPERTY LABELS TESTLABEL MHP)
endfunction()

function(add_shp_ctest test_name name)
  add_test(NAME ${test_name} COMMAND ./${name} ${ARGN})
  set_property(TEST ${test_name} PROPERTY LABELS TESTLABEL SHP)
endfunction()

install(DIRECTORY include DESTINATION ${CMAKE_INSTALL_PREFIX})

add_subdirectory(include)

# Examples are not needed when another project uses the library
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

  include_directories(examples/include)
  include_directories(test/gtest/include)

  include(GoogleTest)

  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Werror>)

  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wno-error=stringop-overflow=)
  endif()

  # This must appear after -Wall
  if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    # Most code get warnings with -ffast-math
    add_compile_options(-Wno-tautological-constant-compare)
    # DPL turns on -fopenmp-simd, triggering optimization warnings in parallel
    # stl
    add_compile_options(-Wno-error=pass-failed)
    add_compile_options(-Wno-error=unknown-pragmas)

    if(CMAKE_BUILD_TYPE STREQUAL "Release")
      add_compile_options(-qopenmp -qopt-streaming-stores=always
                          -qopt-zmm-usage=high)
      add_link_options(-qopenmp)
    endif()
  endif()

  add_subdirectory(test/gtest/mhp)

  if(ENABLE_SYCL)
    # disables rng::detail::box_compress::coalesce which causes rng::box to use
    # global non-const variable, which can not be used in SYCL kernels
    add_compile_definitions(RANGES_WORKAROUND_MSVC_249830)

    add_subdirectory(examples/shp)
    add_subdirectory(test/gtest/shp)
  endif()

  add_subdirectory(examples/serial)
  add_subdirectory(test/gtest/serial)
  add_subdirectory(examples/mhp)
  add_subdirectory(benchmarks/gbench)

  # Requires clang, icpx/llvm nightly do not support the tools
  if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND NOT ENABLE_SYCL)
    add_subdirectory(test/fuzz/cpu)
  endif()

endif()
