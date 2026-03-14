cmake_minimum_required(VERSION 3.20)
project(myml CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)


# ──────────────────────────────────────────────────────────────
# sources
# ──────────────────────────────────────────────────────────────

file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS "src/*.cpp")

file(GLOB_RECURSE UNIT_TEST_SOURCES
     CONFIGURE_DEPENDS
     "tests/unit/test_*.cpp")

# file(GLOB_RECURSE SMOKE_TEST_SOURCES
#      CONFIGURE_DEPENDS
#      "tests/smoke/test_*.cpp")

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

# add_executable(smoke_tests ${SMOKE_TEST_SOURCES})
# target_link_libraries(smoke_tests PRIVATE myml_lib GTest::gtest_main)

# add_executable(smoke_tests_cov ${SMOKE_TEST_SOURCES})
# target_compile_options(smoke_tests_cov PRIVATE --coverage -O3 -fno-inline)
# target_link_options(smoke_tests_cov    PRIVATE --coverage)
# target_link_libraries(smoke_tests_cov  PRIVATE myml_lib_cov GTest::gtest_main)

# ──────────────────────────────────────────────────────────────
# test discovery
# ──────────────────────────────────────────────────────────────

enable_testing()
include(GoogleTest)

gtest_discover_tests(unit_tests  PROPERTIES LABELS "unit")
# gtest_discover_tests(smoke_tests PROPERTIES LABELS "smoke")


# ──────────────────────────────────────────────────────────────
# gprof profiling binary  (add this block to your CMakeLists.txt)
#
# -pg instruments every function call; gprof reads gmon.out after.
# Keep -O2 so the profile reflects realistic code, but -pg
# disables inlining automatically where it needs to for accuracy.
# ──────────────────────────────────────────────────────────────
# file(GLOB_RECURSE PROFILE_SOURCES
#      CONFIGURE_DEPENDS
#      "tests/profiling/*.cpp")

# add_executable(profile_mnist ${PROFILE_SOURCES})
# target_compile_options(profile_mnist PRIVATE -O3 -march=native -g -fno-omit-frame-pointer)
# target_link_libraries(profile_mnist PRIVATE myml_lib)

# add_executable(profile_mnist_gprof ${PROFILE_SOURCES})
# target_compile_options(profile_mnist_gprof PRIVATE -O3 -march=native -pg)
# target_link_options(profile_mnist_gprof    PRIVATE -pg)
# target_link_libraries(profile_mnist_gprof  PRIVATE myml_lib)


# ──────────────────────────────────────────────────────────────
# debug / scratch binary
# ──────────────────────────────────────────────────────────────

add_executable(debug_main main.cpp)
target_link_libraries(debug_main PRIVATE myml_lib)