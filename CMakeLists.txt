cmake_minimum_required(VERSION 3.20)
project(myml CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to Debug when no build type is explicitly provided.
if (NOT CMAKE_BUILD_TYPE)
     set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()


# ──────────────────────────────────────────────────────────────
# sources
# ──────────────────────────────────────────────────────────────

file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS "src/*.cpp")

file(GLOB_RECURSE UNIT_TEST_SOURCES
     CONFIGURE_DEPENDS
     "tests/unit/*.cpp")

file(GLOB_RECURSE SMOKE_TEST_SOURCES
     CONFIGURE_DEPENDS
     "tests/smoke/*.cpp")

file(GLOB_RECURSE DATASET_TEST_SOURCES
     CONFIGURE_DEPENDS
     "tests/datasets/*.cpp")

# ──────────────────────────────────────────────────────────────
# libraries
# ──────────────────────────────────────────────────────────────

add_library(myml_lib ${LIB_SOURCES})
target_include_directories(myml_lib PUBLIC src)

add_library(myml_lib_cov ${LIB_SOURCES})
target_include_directories(myml_lib_cov PUBLIC src)
target_compile_options(myml_lib_cov PRIVATE --coverage -O3 -fno-inline)
target_link_options(myml_lib_cov    PRIVATE --coverage)

find_package(GTest REQUIRED)
find_package(OpenMP REQUIRED)

target_link_libraries(myml_lib      PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(myml_lib_cov  PUBLIC OpenMP::OpenMP_CXX)

# ──────────────────────────────────────────────────────────────
# unit test binaries
# ──────────────────────────────────────────────────────────────

add_executable(unit_tests ${UNIT_TEST_SOURCES})
target_link_libraries(unit_tests PRIVATE myml_lib GTest::gtest_main)

add_executable(unit_tests_cov ${UNIT_TEST_SOURCES})
target_compile_options(unit_tests_cov PRIVATE --coverage -O3 -fno-inline)
target_link_options(unit_tests_cov    PRIVATE --coverage)
target_link_libraries(unit_tests_cov  PRIVATE myml_lib_cov GTest::gtest_main)

# ──────────────────────────────────────────────────────────────
# smoke test binaries
# ──────────────────────────────────────────────────────────────

add_executable(smoke_tests ${SMOKE_TEST_SOURCES})
target_link_libraries(smoke_tests PRIVATE myml_lib GTest::gtest_main)

add_executable(smoke_tests_cov ${SMOKE_TEST_SOURCES})
target_compile_options(smoke_tests_cov PRIVATE --coverage -O3 -fno-inline)
target_link_options(smoke_tests_cov    PRIVATE --coverage)
target_link_libraries(smoke_tests_cov  PRIVATE myml_lib_cov GTest::gtest_main)

# ──────────────────────────────────────────────────────────────
# dataset test binaries
# ──────────────────────────────────────────────────────────────

add_executable(dataset_tests ${DATASET_TEST_SOURCES})
target_link_libraries(dataset_tests PRIVATE myml_lib GTest::gtest_main)

# ──────────────────────────────────────────────────────────────
# test discovery
# ──────────────────────────────────────────────────────────────

enable_testing()
include(GoogleTest)

gtest_discover_tests(unit_tests  PROPERTIES LABELS "unit")
gtest_discover_tests(smoke_tests PROPERTIES LABELS "smoke")
gtest_discover_tests(dataset_tests PROPERTIES LABELS "datasets")


# ──────────────────────────────────────────────────────────────
# Profiling build of main.cpp (gprof)
#
#   make profile_gprof     [epochs=3]
#   make profile_gprof_top [epochs=3]
# ──────────────────────────────────────────────────────────────

add_executable(profile_main_gprof main.cpp)
target_compile_options(profile_main_gprof PRIVATE -O3 -march=native -pg)
target_link_options(profile_main_gprof    PRIVATE -pg)
target_link_libraries(profile_main_gprof  PRIVATE myml_lib)


# ──────────────────────────────────────────────────────────────
# benchmark binaries
#
# bench_mnist : full MNIST training-loop throughput benchmark
# bench_ops   : per-op microbenchmark (forward / backward)
#
# Both are compiled with -O3 -march=native and kept out of CTest.
#
# Run targets (build + execute in one step):
#
#   cmake --build build --target run_bench_mnist
#
#   cmake --build build --target bench_matmul_forward
#   cmake --build build --target bench_relu_backward
#   ... (all 7 ops × 2 modes available)
#
# Override size / iterations without touching the source:
#
#   cmake -DBENCH_SIZE=1024 -DBENCH_ITERS=20 -DBENCH_WARMUP=5 build
#   cmake --build build --target bench_matmul_forward
# ──────────────────────────────────────────────────────────────

set(BENCH_SIZE   "512" CACHE STRING "Primary dimension for bench_ops targets (--size)")
set(BENCH_ITERS  "100" CACHE STRING "Timed iterations for bench_ops targets (--iters)")
set(BENCH_WARMUP "10"  CACHE STRING "Warmup iterations for bench_ops targets (--warmup)")

set(BENCH_MNIST_BATCHES   "200" CACHE STRING "Number of batches for run_bench_mnist (N_BATCHES)")
set(BENCH_MNIST_BATCHSIZE "64"  CACHE STRING "Batch size for run_bench_mnist (BATCH_SIZE)")

add_executable(bench_mnist tests/profiling/bench_mnist.cpp)
target_compile_options(bench_mnist PRIVATE -O3 -march=native -fno-omit-frame-pointer)
target_link_libraries(bench_mnist  PRIVATE myml_lib)

add_executable(bench_ops tests/profiling/bench_ops.cpp)
target_compile_options(bench_ops PRIVATE -O3 -march=native -fno-omit-frame-pointer)
target_link_libraries(bench_ops  PRIVATE myml_lib)

# Run the full MNIST training-loop benchmark.
add_custom_target(run_bench_mnist
    COMMAND ${CMAKE_COMMAND} -E env
            N_BATCHES=${BENCH_MNIST_BATCHES}
            BATCH_SIZE=${BENCH_MNIST_BATCHSIZE}
            $<TARGET_FILE:bench_mnist>
    DEPENDS bench_mnist
    USES_TERMINAL
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Per-op run targets: bench_<op>_<forward|backward>
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
