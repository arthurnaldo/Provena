cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)
find_package(Threads REQUIRED)

nanobind_add_module(_kernel NB_STATIC src/provena_audit/kernel.cpp)
target_compile_features(_kernel PRIVATE cxx_std_20)
target_link_libraries(_kernel PRIVATE Threads::Threads)
# Hardware popcount on x86 (x86-64-v2 baseline); ARM has it already.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64" AND NOT MSVC)
  target_compile_options(_kernel PRIVATE -mpopcnt)
endif()

install(TARGETS _kernel LIBRARY DESTINATION provena_audit)
