BUILD_DEBUG = build
BUILD_OPT   = build_opt

SOURCES       = $(shell find src          -name "*.cpp")
HEADERS       = $(shell find src          -name "*.h")
UNIT_TESTS    = $(shell find tests/unit   -name "*.cpp")
SMOKE_TESTS   = $(shell find tests/smoke  -name "*.cpp")
DATASET_TESTS = $(shell find tests/datasets -name "*.cpp")

# ──────────────────────────────────────────────────────────────
# help  (default target)
# ──────────────────────────────────────────────────────────────

.PHONY: help
help:
	@echo ""
	@echo "  Tests"
	@echo "    make unit_tests                      [filter=TestSuite.Name]"
	@echo "    make smoke_tests                     [filter=TestSuite.Name]"
	@echo "    make test_datasets                   [filter=TestSuite.Name]"
	@echo "    make all_tests"
	@echo ""
	@echo "  Coverage  (unit tests, HTML report)"
	@echo "    make coverage"
	@echo ""
	@echo "  Training"
	@echo "    make run_main   [mode=debug|opt]  [epochs=10]"
	@echo ""
	@echo "  Benchmarks"
	@echo "    make bench      op=<op> mode=<forward|backward>"
	@echo "                    [size=512] [iters=100] [warmup=10]"
	@echo "    make bench_mnist [batches=200] [batch_size=64] [bench_model=mlp|cnn]"
	@echo ""
	@echo "  Profiling (gprof)"
	@echo "    make profile_gprof     [profile_epochs=3]"
	@echo "    make profile_gprof_top [profile_epochs=3]"
	@echo ""
	@echo "    make clean"
	@echo ""


# ──────────────────────────────────────────────────────────────
# cmake configure (auto-runs when CMakeLists.txt changes)
# ──────────────────────────────────────────────────────────────

$(BUILD_DEBUG)/build.ninja: CMakeLists.txt
	cmake -B $(BUILD_DEBUG) -G Ninja -DCMAKE_BUILD_TYPE=Debug

$(BUILD_OPT)/build.ninja: CMakeLists.txt
	cmake -B $(BUILD_OPT) -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3"


# ──────────────────────────────────────────────────────────────
# tests
#
#   make unit_tests
#   make smoke_tests
#   make test_datasets
#   make all_tests
#
#   Optional filter:  make unit_tests filter=MatMulOpForwardTest.*
# ──────────────────────────────────────────────────────────────

$(BUILD_DEBUG)/unit_tests: $(BUILD_DEBUG)/build.ninja $(SOURCES) $(HEADERS) $(UNIT_TESTS)
	cmake --build $(BUILD_DEBUG) --target unit_tests

$(BUILD_DEBUG)/smoke_tests: $(BUILD_DEBUG)/build.ninja $(SOURCES) $(HEADERS) $(SMOKE_TESTS)
	cmake --build $(BUILD_DEBUG) --target smoke_tests

$(BUILD_DEBUG)/dataset_tests: $(BUILD_DEBUG)/build.ninja $(SOURCES) $(HEADERS) $(DATASET_TESTS)
	cmake --build $(BUILD_DEBUG) --target dataset_tests

.PHONY: unit_tests smoke_tests test_datasets all_tests

unit_tests: $(BUILD_DEBUG)/unit_tests
ifdef filter
	./$(BUILD_DEBUG)/unit_tests --gtest_filter="$(filter)" --gtest_color=yes
else
	./$(BUILD_DEBUG)/unit_tests --gtest_color=yes
endif

smoke_tests: $(BUILD_DEBUG)/smoke_tests
ifdef filter
	./$(BUILD_DEBUG)/smoke_tests --gtest_filter="$(filter)" --gtest_color=yes
else
	./$(BUILD_DEBUG)/smoke_tests --gtest_color=yes
endif

test_datasets: $(BUILD_DEBUG)/dataset_tests
ifdef filter
	./$(BUILD_DEBUG)/dataset_tests --gtest_filter="$(filter)" --gtest_color=yes
else
	./$(BUILD_DEBUG)/dataset_tests --gtest_color=yes
endif

all_tests: unit_tests smoke_tests test_datasets


# ──────────────────────────────────────────────────────────────
# coverage  (unit tests, HTML report)
#
#   make coverage
#   open build/coverage_html/index.html
# ──────────────────────────────────────────────────────────────

$(BUILD_DEBUG)/unit_tests_cov: $(BUILD_DEBUG)/build.ninja $(SOURCES) $(HEADERS) $(UNIT_TESTS)
	cmake --build $(BUILD_DEBUG) --target unit_tests_cov

.PHONY: coverage
coverage: $(BUILD_DEBUG)/unit_tests_cov
	find $(BUILD_DEBUG) -name "*.gcda" -delete
	./$(BUILD_DEBUG)/unit_tests_cov --gtest_color=yes || true
	mkdir -p $(BUILD_DEBUG)/coverage_html
	gcovr \
	    --root $(CURDIR) \
	    --filter $(CURDIR)/src/ \
	    --object-directory $(BUILD_DEBUG)/CMakeFiles/myml_lib_cov.dir \
	    --object-directory $(BUILD_DEBUG)/CMakeFiles/unit_tests_cov.dir \
	    --html-details $(BUILD_DEBUG)/coverage_html/index.html \
	    --print-summary \
	    --exclude-unreachable-branches \
	    --sort-percentage
	@echo ""
	@echo "HTML report → $(BUILD_DEBUG)/coverage_html/index.html"
	@echo "Serve with:   python3 -m http.server 8080 --directory $(BUILD_DEBUG)/coverage_html"


# ──────────────────────────────────────────────────────────────
# training
#
#   make run_main  [mode=debug|opt]  [epochs=10]
# ──────────────────────────────────────────────────────────────

mode   ?= debug
epochs ?= 10

$(BUILD_DEBUG)/debug_main: $(BUILD_DEBUG)/build.ninja $(SOURCES) $(HEADERS) main.cpp
	cmake --build $(BUILD_DEBUG) --target debug_main

$(BUILD_OPT)/debug_main: $(BUILD_OPT)/build.ninja $(SOURCES) $(HEADERS) main.cpp
	cmake --build $(BUILD_OPT) --target debug_main

.PHONY: run_main
run_main:
ifeq ($(mode),debug)
	$(MAKE) $(BUILD_DEBUG)/debug_main && ./$(BUILD_DEBUG)/debug_main $(epochs)
else ifeq ($(mode),opt)
	$(MAKE) $(BUILD_OPT)/debug_main   && ./$(BUILD_OPT)/debug_main   $(epochs)
else
	$(error mode must be 'debug' or 'opt')
endif


# ──────────────────────────────────────────────────────────────
# profiling (gprof)
#
#   make profile_gprof     [profile_epochs=3]
#   make profile_gprof_top [profile_epochs=3]
# ──────────────────────────────────────────────────────────────

profile_epochs ?= 3

$(BUILD_OPT)/profile_main_gprof: $(BUILD_OPT)/build.ninja $(SOURCES) $(HEADERS) main.cpp
	cmake --build $(BUILD_OPT) --target profile_main_gprof

.PHONY: profile_gprof profile_gprof_top
profile_gprof: $(BUILD_OPT)/profile_main_gprof
	./$(BUILD_OPT)/profile_main_gprof $(profile_epochs)
	gprof -b $(BUILD_OPT)/profile_main_gprof gmon.out > $(BUILD_OPT)/gprof_report.txt
	@echo ""
	@echo "=== FLAT PROFILE (top 30 by self time) ==="
	gprof -b -p $(BUILD_OPT)/profile_main_gprof gmon.out | head -35
	@echo "Full report: $(BUILD_OPT)/gprof_report.txt"

profile_gprof_top: $(BUILD_OPT)/profile_main_gprof
	./$(BUILD_OPT)/profile_main_gprof $(profile_epochs)
	@echo ""
	@echo "=== TOP 20 FUNCTIONS BY SELF TIME ==="
	gprof -b -p $(BUILD_OPT)/profile_main_gprof gmon.out | head -25


# ──────────────────────────────────────────────────────────────
# benchmarks
#
#   make bench op=matmul mode=forward [size=512] [iters=100] [warmup=10]
#   make bench_mnist [batches=200] [batch_size=64] [bench_model=mlp|cnn]
# ──────────────────────────────────────────────────────────────

op     ?= matmul
size   ?= 512
iters  ?= 100
warmup ?= 10

batches    ?= 200
batch_size ?= 64
bench_model ?= mlp

$(BUILD_OPT)/bench_ops: $(BUILD_OPT)/build.ninja $(SOURCES) $(HEADERS) tests/profiling/bench_ops.cpp
	cmake --build $(BUILD_OPT) --target bench_ops

$(BUILD_OPT)/bench_mnist: $(BUILD_OPT)/build.ninja $(SOURCES) $(HEADERS) tests/profiling/bench_mnist.cpp
	cmake --build $(BUILD_OPT) --target bench_mnist

.PHONY: bench bench_mnist
bench: $(BUILD_OPT)/bench_ops
	./$(BUILD_OPT)/bench_ops --op $(op) --mode $(mode) --size $(size) --iters $(iters) --warmup $(warmup)

bench_mnist: $(BUILD_OPT)/bench_mnist
	N_BATCHES=$(batches) BATCH_SIZE=$(batch_size) MODEL=$(bench_model) ./$(BUILD_OPT)/bench_mnist


# ──────────────────────────────────────────────────────────────
# clean
# ──────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	rm -rf $(BUILD_DEBUG) $(BUILD_OPT)
