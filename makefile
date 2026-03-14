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
# gprof profiling targets  (add this block to your Makefile)
#
# Usage:
#   make profile_gprof              # run 200 batches, print report
#   make profile_gprof n_batches=500
#   make profile_gprof_top          # flat profile only (top 20 fns)
# ──────────────────────────────────────────────────────────────

PROFILE_SOURCES = $(shell find tests/profiling -name "*.cpp")

# ── build ──────────────────────────────────────────────────────

$(BUILD_DEBUG)/profile_mnist_gprof: $(BUILD_DEBUG)/build.ninja $(SOURCES) $(HEADERS) $(PROFILE_SOURCES)
	cmake --build $(BUILD_DEBUG) --target profile_mnist_gprof

# ── run + full report ─────────────────────────────────────────
# Runs the harness (gmon.out written automatically by the runtime),
# then converts it to a human-readable report saved to
# build/gprof_report.txt and printed to stdout.

.PHONY: profile_gprof
profile_gprof: $(BUILD_DEBUG)/profile_mnist_gprof
	@echo "--- running harness (gmon.out will be written to cwd) ---"
	  ./$(BUILD_DEBUG)/profile_mnist_gprof $(or $(n_batches),1000)
	@echo ""
	@echo "--- generating gprof report → $(BUILD_DEBUG)/gprof_report.txt ---"
	gprof ./$(BUILD_DEBUG)/profile_mnist_gprof gmon.out > $(BUILD_DEBUG)/gprof_report.txt
	@echo ""
	@echo "=== FLAT PROFILE (top 30 functions by self time) ==="
	gprof -b ./$(BUILD_DEBUG)/profile_mnist_gprof gmon.out | head -50
	@echo ""
	@echo "Full report: $(BUILD_DEBUG)/gprof_report.txt"
	@echo "View call graph with: gprof -b ./$(BUILD_DEBUG)/profile_mnist_gprof gmon.out | less"

# ── flat profile only (quick glance) ─────────────────────────

.PHONY: profile_gprof_top
profile_gprof_top: $(BUILD_DEBUG)/profile_mnist_gprof
	MNIST_PATH=$(or $(MNIST_PATH),data/MNIST) \
	  ./$(BUILD_DEBUG)/profile_mnist_gprof $(or $(n_batches),200)
	@echo ""
	@echo "=== TOP 20 FUNCTIONS BY SELF TIME ==="
	gprof -b -p ./$(BUILD_DEBUG)/profile_mnist_gprof gmon.out | head -30