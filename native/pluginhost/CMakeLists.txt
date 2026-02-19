cmake_minimum_required(VERSION 3.21)
project(soundgen_pluginhost LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# CLAP is header-only for our purposes. Fetch the official SDK headers.
FetchContent_Declare(
  clap
  GIT_REPOSITORY https://github.com/free-audio/clap.git
  GIT_TAG main
)
FetchContent_MakeAvailable(clap)

add_executable(soundgen_pluginhost
  src/main.cpp
)

target_include_directories(soundgen_pluginhost PRIVATE
  ${clap_SOURCE_DIR}/include
)

if (WIN32)
  target_compile_definitions(soundgen_pluginhost PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
