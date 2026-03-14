CXX      = g++
CXXFLAGS = -std=c++20 -Wall -Wextra

CXXFLAGS += -fopenmp
LDFLAGS  += -fopenmp

BUILD_DEBUG = build
BUILD_OPT   = build_opt

HEADERS  = $(shell find src -name "*.h")
SOURCES  = $(shell find src -name "*.cpp")

UNIT_TESTS  = $(shell find tests/unit  -name "*.cpp")
SMOKE_TESTS = $(shell find tests/smoke -name "*.cpp")
DATASET_TESTS = $(shell find tests/datasets -name "*.cpp")

# ──────────────────────────────────────────────────────────────
# cmake configure
# ──────────────────────────────────────────────────────────────

$(BUILD_DEBUG)/build.ninja: CMakeLists.txt
	cmake -B $(BUILD_DEBUG) -G Ninja -DCMAKE_BUILD_TYPE=Debug

$(BUILD_OPT)/build.ninja: CMakeLists.txt
	cmake -B $(BUILD_OPT) -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3"

# ──────────────────────────────────────────────────────────────
# unit tests
# ──────────────────────────────────────────────────────────────

$(BUILD_DEBUG)/unit_tests: $(BUILD_DEBUG)/build.ninja $(SOURCES) $(HEADERS) $(UNIT_TESTS)
	cmake --build $(BUILD_DEBUG) --target unit_tests

.PHONY: unit_tests
unit_tests: $(BUILD_DEBUG)/unit_tests
ifdef filter
	./$(BUILD_DEBUG)/unit_tests --gtest_filter="$(filter)" --gtest_color=yes
else
	./$(BUILD_DEBUG)/unit_tests --gtest_color=yes
endif

# ──────────────────────────────────────────────────────────────
# smoke tests
# ──────────────────────────────────────────────────────────────

$(BUILD_DEBUG)/smoke_tests: $(BUILD_DEBUG)/build.ninja $(SOURCES) $(HEADERS) $(SMOKE_TESTS)
	cmake --build $(BUILD_DEBUG) --target smoke_tests

.PHONY: smoke_tests
smoke_tests: $(BUILD_DEBUG)/smoke_tests
ifdef filter
	./$(BUILD_DEBUG)/smoke_tests --gtest_filter="$(filter)" --gtest_color=yes
else
	./$(BUILD_DEBUG)/smoke_tests --gtest_color=yes
endif

# ──────────────────────────────────────────────────────────────
# dataset tests
# ──────────────────────────────────────────────────────────────

$(BUILD_DEBUG)/dataset_tests: $(BUILD_DEBUG)/build.ninja $(SOURCES) $(HEADERS) $(DATASET_TESTS)
	cmake --build $(BUILD_DEBUG) --target dataset_tests

.PHONY: test_datasets
test_datasets: $(BUILD_DEBUG)/dataset_tests
ifdef filter
	./$(BUILD_DEBUG)/dataset_tests --gtest_filter="$(filter)" --gtest_color=yes
else
	./$(BUILD_DEBUG)/dataset_tests --gtest_color=yes
endif

# ──────────────────────────────────────────────────────────────
# coverage (unit tests only)
# ──────────────────────────────────────────────────────────────

$(BUILD_DEBUG)/unit_tests_cov: $(BUILD_DEBUG)/build.ninja $(SOURCES) $(HEADERS) $(UNIT_TESTS)
	cmake --build $(BUILD_DEBUG) --target unit_tests_cov

.PHONY: coverage
coverage: $(BUILD_DEBUG)/unit_tests_cov
	find $(BUILD_DEBUG) -name "*.gcda" -delete
	./$(BUILD_DEBUG)/unit_tests_cov --gtest_color=yes || true

	mkdir -p $(BUILD_DEBUG)/coverage_html

	gcovr \
	    --root /workspace \
	    --filter /workspace/src/ \
	    --object-directory $(BUILD_DEBUG)/CMakeFiles/myml_lib_cov.dir \
	    --object-directory $(BUILD_DEBUG)/CMakeFiles/unit_tests_cov.dir \
	    --html-details $(BUILD_DEBUG)/coverage_html/index.html \
	    --print-summary \
	    --exclude-unreachable-branches \
	    --sort-percentage

	@echo ""
	@echo "HTML report → $(BUILD_DEBUG)/coverage_html/index.html"
	@echo "Serve with:"
	@echo "python3 -m http.server 8080 --directory $(BUILD_DEBUG)/coverage_html"

# ──────────────────────────────────────────────────────────────
# main training run (debug or optimized)
# Usage:
#   make run_main mode=debug epochs=10
#   make run_main mode=opt epochs=10
# ──────────────────────────────────────────────────────────────

$(BUILD_DEBUG)/debug_main: $(BUILD_DEBUG)/build.ninja $(SOURCES) $(HEADERS) main.cpp
	cmake --build $(BUILD_DEBUG) --target debug_main

$(BUILD_OPT)/debug_main: $(BUILD_OPT)/build.ninja $(SOURCES) $(HEADERS) main.cpp
	cmake --build $(BUILD_OPT) --target debug_main

.PHONY: run_main run_main_debug run_main_opt
mode ?= debug
epochs ?= 10

run_main:
ifeq ($(mode),debug)
	$(MAKE) run_main_debug epochs=$(epochs)
else ifeq ($(mode),opt)
	$(MAKE) run_main_opt epochs=$(epochs)
else
	$(error Unsupported mode '$(mode)'. Use mode=debug or mode=opt)
endif

run_main_debug: $(BUILD_DEBUG)/debug_main
	./$(BUILD_DEBUG)/debug_main $(epochs)

run_main_opt: $(BUILD_OPT)/debug_main
	./$(BUILD_OPT)/debug_main $(epochs)

# ──────────────────────────────────────────────────────────────
# convenience targets
# ──────────────────────────────────────────────────────────────

.PHONY: check
check: test unit_tests

.PHONY: all_tests
all_tests: unit_tests smoke_tests test_datasets

# ──────────────────────────────────────────────────────────────
# clean
# ──────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	rm -f tests_bin
	rm -rf $(BUILD_DEBUG) $(BUILD_OPT)

# ──────────────────────────────────────────────────────────────
# profiling targets for main.cpp (gprof)
#
#   make profile_gprof     [profile_epochs=3]   # flat profile + call-graph
#   make profile_gprof_top [profile_epochs=3]   # top-20 functions only
#
# Default: profile_epochs=3
# Uses the optimised build (O3 + OpenMP) so the profile is representative.
# ──────────────────────────────────────────────────────────────

profile_epochs ?= 3

$(BUILD_OPT)/profile_main_gprof: $(BUILD_OPT)/build.ninja $(SOURCES) $(HEADERS) main.cpp
	cmake --build $(BUILD_OPT) --target profile_main_gprof

# ── run + full flat profile + call-graph ──────────────────────

.PHONY: profile_gprof
profile_gprof: $(BUILD_OPT)/profile_main_gprof
	./$(BUILD_OPT)/profile_main_gprof $(profile_epochs)
	gprof -b $(BUILD_OPT)/profile_main_gprof gmon.out > $(BUILD_OPT)/gprof_report.txt
	@echo ""
	@echo "=== FLAT PROFILE (top 30 by self time) ==="
	gprof -b -p $(BUILD_OPT)/profile_main_gprof gmon.out | head -35
	@echo ""
	@echo "Full report: $(BUILD_OPT)/gprof_report.txt"

# ── top-20 flat profile only (quick glance) ───────────────────

.PHONY: profile_gprof_top
profile_gprof_top: $(BUILD_OPT)/profile_main_gprof
	./$(BUILD_OPT)/profile_main_gprof $(profile_epochs)
	@echo ""
	@echo "=== TOP 20 FUNCTIONS BY SELF TIME ==="
	gprof -b -p $(BUILD_OPT)/profile_main_gprof gmon.out | head -25

# ──────────────────────────────────────────────────────────────
# MNIST training throughput benchmark
#
#   make bench_mnist [batches=200] [batch_size=64]
# ──────────────────────────────────────────────────────────────

batches    ?= 200
batch_size ?= 64

$(BUILD_OPT)/bench_mnist: $(BUILD_OPT)/build.ninja $(SOURCES) $(HEADERS) tests/profiling/bench_mnist.cpp
	cmake --build $(BUILD_OPT) --target bench_mnist

.PHONY: bench_mnist
bench_mnist: $(BUILD_OPT)/bench_mnist
	N_BATCHES=$(batches) BATCH_SIZE=$(batch_size) ./$(BUILD_OPT)/bench_mnist

# ──────────────────────────────────────────────────────────────
# op microbenchmarks
#
#   make bench op=matmul mode=forward [size=512] [iters=100]
#   make bench op=relu   mode=backward
# ──────────────────────────────────────────────────────────────

op   ?= matmul
mode ?= forward
size ?= 512
iters ?= 100

$(BUILD_OPT)/bench_ops: $(BUILD_OPT)/build.ninja $(SOURCES) $(HEADERS) tests/profiling/bench_ops.cpp
	cmake --build $(BUILD_OPT) --target bench_ops

.PHONY: bench
bench: $(BUILD_OPT)/bench_ops
	./$(BUILD_OPT)/bench_ops --op $(op) --mode $(mode) --size $(size) --iters $(iters)