CXX      = g++
CXXFLAGS = -std=c++17 -Wall -Wextra

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