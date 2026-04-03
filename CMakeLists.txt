cmake_minimum_required(VERSION 3.20)
project(myml CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()


# ──────────────────────────────────────────────────────────────
# Sources
# ──────────────────────────────────────────────────────────────

file(GLOB_RECURSE LIB_SOURCES      CONFIGURE_DEPENDS "src/*.cpp")
file(GLOB_RECURSE UNIT_TEST_SOURCES    CONFIGURE_DEPENDS "tests/unit/*.cpp")
file(GLOB_RECURSE SMOKE_TEST_SOURCES   CONFIGURE_DEPENDS "tests/smoke/*.cpp")
file(GLOB_RECURSE DATASET_TEST_SOURCES CONFIGURE_DEPENDS "tests/datasets/*.cpp")


# ──────────────────────────────────────────────────────────────
# Dependencies
# ──────────────────────────────────────────────────────────────

find_package(GTest REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS)          # optional — enables BLAS reference in bench_ops


# ──────────────────────────────────────────────────────────────
# Core library (two variants: normal and coverage-instrumented)
# ──────────────────────────────────────────────────────────────

add_library(myml_lib ${LIB_SOURCES})
target_include_directories(myml_lib PUBLIC src)
target_link_libraries(myml_lib PUBLIC OpenMP::OpenMP_CXX)

add_library(myml_lib_cov ${LIB_SOURCES})
target_include_directories(myml_lib_cov PUBLIC src)
target_compile_options(myml_lib_cov PRIVATE --coverage -O0 -fno-inline)
target_link_options(myml_lib_cov    PRIVATE --coverage)
target_link_libraries(myml_lib_cov  PUBLIC OpenMP::OpenMP_CXX)


# ──────────────────────────────────────────────────────────────
# Test binaries
# ──────────────────────────────────────────────────────────────

add_executable(unit_tests    ${UNIT_TEST_SOURCES})
add_executable(smoke_tests   ${SMOKE_TEST_SOURCES})
add_executable(dataset_tests ${DATASET_TEST_SOURCES})

target_link_libraries(unit_tests    PRIVATE myml_lib GTest::gtest_main)
target_link_libraries(smoke_tests   PRIVATE myml_lib GTest::gtest_main)
target_link_libraries(dataset_tests PRIVATE myml_lib GTest::gtest_main)

# Coverage variant (unit tests only — smoke/dataset are too slow to instrument)
add_executable(unit_tests_cov ${UNIT_TEST_SOURCES})
target_compile_options(unit_tests_cov PRIVATE --coverage -O0 -fno-inline)
target_link_options(unit_tests_cov    PRIVATE --coverage)
target_link_libraries(unit_tests_cov  PRIVATE myml_lib_cov GTest::gtest_main)


# ──────────────────────────────────────────────────────────────
# Test discovery (CTest)
# ──────────────────────────────────────────────────────────────

enable_testing()
include(GoogleTest)

gtest_discover_tests(unit_tests    PROPERTIES LABELS "unit")
gtest_discover_tests(smoke_tests   PROPERTIES LABELS "smoke")
gtest_discover_tests(dataset_tests PROPERTIES LABELS "datasets")


# ──────────────────────────────────────────────────────────────
# Main training binary
#
#   make run_main [mode=debug|opt] [epochs=10]
# ──────────────────────────────────────────────────────────────

add_executable(debug_main main.cpp)
target_link_libraries(debug_main PRIVATE myml_lib)


# ──────────────────────────────────────────────────────────────
# Profiling binary (gprof)
#
#   make profile_gprof     [profile_epochs=3]
#   make profile_gprof_top [profile_epochs=3]
# ──────────────────────────────────────────────────────────────

add_executable(profile_main_gprof main.cpp)
target_compile_options(profile_main_gprof PRIVATE -O3 -march=native -pg)
target_link_options(profile_main_gprof    PRIVATE -pg)
target_link_libraries(profile_main_gprof  PRIVATE myml_lib)


# ──────────────────────────────────────────────────────────────
# Benchmark binaries
#
#   make bench op=matmul mode=forward [size=512] [iters=100] [warmup=10]
#   make bench_mnist [batches=200] [batch_size=64]
# ──────────────────────────────────────────────────────────────

set(BENCH_SIZE   "512" CACHE STRING "Default --size for bench_ops")
set(BENCH_ITERS  "100" CACHE STRING "Default --iters for bench_ops")
set(BENCH_WARMUP "10"  CACHE STRING "Default --warmup for bench_ops")

set(BENCH_MNIST_BATCHES   "200" CACHE STRING "Default N_BATCHES for bench_mnist")
set(BENCH_MNIST_BATCHSIZE "64"  CACHE STRING "Default BATCH_SIZE for bench_mnist")

add_executable(bench_mnist tests/profiling/bench_mnist.cpp)
target_compile_options(bench_mnist PRIVATE -O3 -march=native -fno-omit-frame-pointer)
target_link_libraries(bench_mnist  PRIVATE myml_lib)

add_executable(bench_ops tests/profiling/bench_ops.cpp)
target_compile_options(bench_ops PRIVATE -O3 -march=native -fno-omit-frame-pointer)
target_link_libraries(bench_ops  PRIVATE myml_lib)
if(BLAS_FOUND)
    target_compile_definitions(bench_ops PRIVATE HAVE_BLAS)
    target_link_libraries(bench_ops PRIVATE ${BLAS_LIBRARIES})
endif()

add_custom_target(run_bench_mnist
    COMMAND ${CMAKE_COMMAND} -E env
            N_BATCHES=${BENCH_MNIST_BATCHES}
            BATCH_SIZE=${BENCH_MNIST_BATCHSIZE}
            $<TARGET_FILE:bench_mnist>
    DEPENDS bench_mnist
    USES_TERMINAL
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

foreach(OP matmul mul add relu sigmoid cross_entropy linear)
    foreach(MODE forward backward)
        add_custom_target(bench_${OP}_${MODE}
            COMMAND $<TARGET_FILE:bench_ops>
                    --op ${OP} --mode ${MODE}
                    --size ${BENCH_SIZE} --iters ${BENCH_ITERS} --warmup ${BENCH_WARMUP}
            DEPENDS bench_ops
            USES_TERMINAL
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        )
    endforeach()
endforeach()
