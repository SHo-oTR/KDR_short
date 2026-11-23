cmake_minimum_required(VERSION 3.16)
project(kdr_short LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

# Header-only library target
add_library(kdr_sort INTERFACE)
add_library(kdr::sort ALIAS kdr_sort)
add_library(kdr_short ALIAS kdr_sort)
target_include_directories(kdr_sort INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(kdr_sort INTERFACE Threads::Threads)

option(KDR_BUILD_TESTS "Build KDR unit tests" ON)
option(KDR_BUILD_BENCHMARKS "Build KDR benchmark suite" ON)
option(KDR_ENABLE_NATIVE "Enable -march=native (AVX2/SIMD)" ON)

if(KDR_ENABLE_NATIVE AND (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"))
    set(KDR_OPT_FLAGS -O3 -march=native -Wall -Wextra)
elseif(MSVC)
    set(KDR_OPT_FLAGS /O2 /arch:AVX2 /W4)
else()
    set(KDR_OPT_FLAGS -O3)
endif()

if(KDR_BUILD_TESTS)
    enable_testing()
    add_executable(test_kdr tests/test_kdr.cpp)
    target_link_libraries(test_kdr PRIVATE kdr_sort)
    target_compile_options(test_kdr PRIVATE ${KDR_OPT_FLAGS})
    add_test(NAME kdr_unit_tests COMMAND test_kdr)

    add_executable(test_new_fixes tests/test_new_fixes.cpp)
    target_link_libraries(test_new_fixes PRIVATE kdr_sort)
    target_compile_options(test_new_fixes PRIVATE ${KDR_OPT_FLAGS})
    add_test(NAME kdr_new_fixes_tests COMMAND test_new_fixes)

    add_executable(extended_tests tests/extended_tests.cpp)
    target_link_libraries(extended_tests PRIVATE kdr_sort)
    target_compile_options(extended_tests PRIVATE ${KDR_OPT_FLAGS})
    add_test(NAME kdr_extended_tests COMMAND extended_tests)
endif()

if(KDR_BUILD_BENCHMARKS)
    add_executable(bench_kdr benchmarks/bench_kdr.cpp)
    target_link_libraries(bench_kdr PRIVATE kdr_sort)
    target_compile_options(bench_kdr PRIVATE ${KDR_OPT_FLAGS})
endif()
