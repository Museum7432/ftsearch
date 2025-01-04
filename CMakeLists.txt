cmake_minimum_required(VERSION 3.25)


project(ftsearch)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

include(ExternalProject)
include(FetchContent)

# find_package(Python COMPONENTS Interpreter Development)
# find_package(pybind11 CONFIG)

add_subdirectory(pybind11)

# ###################################################################
# # OpenBlas
# ###################################################################

# set(CUSTOM_OPENBLAS_DIR "${CMAKE_BINARY_DIR}/openblas_0tbuild")
# set(CUSTOM_OPENBLAS_incl "${CUSTOM_OPENBLAS_DIR}/include/openblas")
# # static linking
# set(CUSTOM_OPENBLAS_lib "${CUSTOM_OPENBLAS_DIR}/lib/libopenblas.a")

# # Custom OpenBLAS
# ExternalProject_Add(
#     OpenBLAS_single_thread
#     GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS.git
#     GIT_TAG 24c5152fbd8027a39759acf261e1595326ab0114
#     CMAKE_ARGS -DUSE_OPENMP=1 -DCMAKE_INSTALL_PREFIX=${CUSTOM_OPENBLAS_DIR}
#     # BUILD_COMMAND make USE_THREAD=0
#     # INSTALL_COMMAND make install
#     BUILD_ALWAYS OFF
#     UPDATE_COMMAND ""  # Skip checking for updates
# )

###################################################################
# LibTorch
###################################################################
# FetchContent_Declare(
#         Torch
#         URL https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip
#         DOWNLOAD_EXTRACT_TIMESTAMP TRUE
# )
# FetchContent_MakeAvailable(Torch)
# list(APPEND CMAKE_PREFIX_PATH "${torch_SOURCE_DIR}")
# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

###################################################################
# exe
###################################################################
include_directories(
    # ${CUSTOM_OPENBLAS_incl}
    ftsearch/include
    )

# OpenBLAS::OpenBLAS
find_package(OpenBLAS REQUIRED)
find_package(OpenMP REQUIRED)

add_library(utils ftsearch/src/utils.cpp)

add_library(ftsearch ftsearch/src/ftsearch.cpp)

# target_link_libraries(utils ${CUSTOM_OPENBLAS_lib} OpenMP::OpenMP_CXX)
target_link_libraries(utils OpenBLAS::OpenBLAS OpenMP::OpenMP_CXX)

target_link_libraries(ftsearch utils)


pybind11_add_module(ftsearch_module ftsearch/src/bindings.cpp)
target_link_libraries(ftsearch_module PUBLIC ftsearch)

install(TARGETS ftsearch_module LIBRARY DESTINATION ftsearch)


# add_executable(ftsearch_test misc/test.cpp)
# target_link_libraries(ftsearch_test ftsearch)

