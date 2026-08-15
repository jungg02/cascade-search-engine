# Plain make rather than CMake: three targets, no external dependencies beyond
# pybind11's headers, and one less thing to install on a fresh machine.

CXX      ?= c++
CXXFLAGS := -std=c++20 -O3 -Wall -Wextra -Wno-unused-parameter -Iindex -Iquery
PYTHON   ?= ../.venv/bin/python

# On macOS, clang searches CommandLineTools/usr/include/c++/v1 before the SDK's
# copy. That directory can be partially populated — on this machine it has 11
# files and no <string> — which fails the build with a confusing "file not
# found". Point at the SDK's libc++ explicitly.
ifeq ($(shell uname -s),Darwin)
SDK_PATH := $(shell xcrun --show-sdk-path)
CXXFLAGS += -nostdinc++ -isystem $(SDK_PATH)/usr/include/c++/v1
endif

BUILD    := build
COMMON   := $(BUILD)/analyzer.o $(BUILD)/porter.o
QUERY    := $(BUILD)/searcher.o

PY_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
PY_SUFFIX   := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

.PHONY: all clean test
# bench_query (the Phase 1 exit-artifact driver) is not written yet.
all: $(BUILD)/build_index $(BUILD)/test_index ../py/cascade_index$(PY_SUFFIX)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: index/%.cc | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/%.o: query/%.cc | $(BUILD)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD)/build_index: index/builder.cc $(COMMON) | $(BUILD)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/bench_query: query/bench_query.cc $(COMMON) $(QUERY) | $(BUILD)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BUILD)/test_index: tests/test_index.cc $(COMMON) $(QUERY) | $(BUILD)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Built directly into py/ so `import cascade_index` works with pythonpath=py.
../py/cascade_index$(PY_SUFFIX): bindings/module.cc $(COMMON) $(QUERY) | $(BUILD)
	$(CXX) $(CXXFLAGS) -shared -fPIC -undefined dynamic_lookup $(PY_INCLUDES) $^ -o $@

test: $(BUILD)/test_index
	$(BUILD)/test_index

clean:
	rm -rf $(BUILD) ../py/cascade_index*.so
