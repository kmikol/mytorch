CXX      = g++
CXXFLAGS = -std=c++17 -Wall -Wextra

CXXFLAGS += -fopenmp
LDFLAGS  += -fopenmp

HEADERS  = $(shell find src -name "*.h")
SOURCES  = $(shell find src -name "*.cpp")

UNIT_TESTS  = $(shell find tests/unit  -name "*.cpp")
SMOKE_TESTS = $(shell find tests/smoke -name "*.cpp")

# ──────────────────────────────────────────────────────────────
# cmake configure
# ──────────────────────────────────────────────────────────────

build/build.ninja: CMakeLists.txt
	cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug

# ──────────────────────────────────────────────────────────────
# unit tests
# ──────────────────────────────────────────────────────────────

build/unit_tests: build/build.ninja $(SOURCES) $(HEADERS) $(UNIT_TESTS)
	cmake --build build --target unit_tests

.PHONY: unit_tests
unit_tests: build/unit_tests
ifdef filter
	./build/unit_tests --gtest_filter="$(filter)" --gtest_color=yes
else
	./build/unit_tests --gtest_color=yes
endif

# ──────────────────────────────────────────────────────────────
# smoke tests
# ──────────────────────────────────────────────────────────────

build/smoke_tests: build/build.ninja $(SOURCES) $(HEADERS) $(SMOKE_TESTS)
	cmake --build build --target smoke_tests

.PHONY: smoke_tests
smoke_tests: build/smoke_tests
ifdef filter
	./build/smoke_tests --gtest_filter="$(filter)" --gtest_color=yes
else
	./build/smoke_tests --gtest_color=yes
endif

# ──────────────────────────────────────────────────────────────
# coverage (unit tests only)
# ──────────────────────────────────────────────────────────────

build/unit_tests_cov: build/build.ninja $(SOURCES) $(HEADERS) $(UNIT_TESTS)
	cmake --build build --target unit_tests_cov

.PHONY: coverage
coverage: build/unit_tests_cov
	find build -name "*.gcda" -delete
	./build/unit_tests_cov --gtest_color=yes || true

	mkdir -p build/coverage_html

	gcovr \
	    --root /workspace \
	    --filter /workspace/src/ \
	    --object-directory build/CMakeFiles/myml_lib_cov.dir \
	    --object-directory build/CMakeFiles/unit_tests_cov.dir \
	    --html-details build/coverage_html/index.html \
	    --print-summary \
	    --exclude-unreachable-branches \
	    --sort-percentage

	@echo ""
	@echo "HTML report → build/coverage_html/index.html"
	@echo "Serve with:"
	@echo "python3 -m http.server 8080 --directory build/coverage_html"

# ──────────────────────────────────────────────────────────────
# convenience targets
# ──────────────────────────────────────────────────────────────

.PHONY: check
check: test unit_tests

.PHONY: all_tests
all_tests: unit_tests smoke_tests

# ──────────────────────────────────────────────────────────────
# clean
# ──────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	rm -f tests_bin
	rm -rf build

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

build/profile_mnist_gprof: build/build.ninja $(SOURCES) $(HEADERS) $(PROFILE_SOURCES)
	cmake --build build --target profile_mnist_gprof

# ── run + full report ─────────────────────────────────────────
# Runs the harness (gmon.out written automatically by the runtime),
# then converts it to a human-readable report saved to
# build/gprof_report.txt and printed to stdout.

.PHONY: profile_gprof
profile_gprof: build/profile_mnist_gprof
	@echo "--- running harness (gmon.out will be written to cwd) ---"
	  ./build/profile_mnist_gprof $(or $(n_batches),1000)
	@echo ""
	@echo "--- generating gprof report → build/gprof_report.txt ---"
	gprof ./build/profile_mnist_gprof gmon.out > build/gprof_report.txt
	@echo ""
	@echo "=== FLAT PROFILE (top 30 functions by self time) ==="
	gprof -b ./build/profile_mnist_gprof gmon.out | head -50
	@echo ""
	@echo "Full report: build/gprof_report.txt"
	@echo "View call graph with: gprof -b ./build/profile_mnist_gprof gmon.out | less"

# ── flat profile only (quick glance) ─────────────────────────

.PHONY: profile_gprof_top
profile_gprof_top: build/profile_mnist_gprof
	MNIST_PATH=$(or $(MNIST_PATH),data/MNIST) \
	  ./build/profile_mnist_gprof $(or $(n_batches),200)
	@echo ""
	@echo "=== TOP 20 FUNCTIONS BY SELF TIME ==="
	gprof -b -p ./build/profile_mnist_gprof gmon.out | head -30