cmake_minimum_required(VERSION 3.15)


project(seqretrival LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -ffast-math")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")


# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto")
# set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -flto")
# set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -flto")

include(FetchContent)
# fetch fp16 support
FetchContent_Declare(
  half_lib
  URL https://sourceforge.net/projects/half/files/half/2.2.1/half-2.2.1.zip/download
  URL_HASH "MD5=5f4f849c94b73abf9a9e1f8719fc559e"
  DOWNLOAD_EXTRACT_TIMESTAMP true
)
FetchContent_MakeAvailable(half_lib)
set(HALF_INCLUDE_DIR "${half_lib_SOURCE_DIR}/include")

message(${HALF_INCLUDE_DIR})

# Fetch GoogleTest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
  BUILD_ALWAYS OFF
  UPDATE_COMMAND ""  # Skip checking for updates
  DOWNLOAD_EXTRACT_TIMESTAMP true
)
FetchContent_MakeAvailable(googletest)
enable_testing()
include(GoogleTest)

# set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}" ${CMAKE_PREFIX_PATH})
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(OpenBLAS REQUIRED)
find_package(OpenMP REQUIRED)


include_directories(
    ${OpenBLAS_INCLUDE_DIR}
)


add_subdirectory(src)
add_subdirectory(tests)