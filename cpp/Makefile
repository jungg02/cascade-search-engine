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

# grpc/protobuf are provided by the Anaconda Python distribution. pkg-config
# is required — direct -lgrpc++ -lprotobuf linking fails on missing abseil
# symbols that protobuf-generated code references directly (verified while
# writing this plan). Override PKG_CONFIG_PATH if your grpc/protobuf live
# elsewhere.
PKG_CONFIG_PATH ?= /opt/anaconda3/lib/pkgconfig
export PKG_CONFIG_PATH

# macOS ships GNU Make 3.81, where an in-Makefile `export` isn't visible to
# $(shell ...) calls evaluated while the Makefile is parsed (only to recipe
# sub-shells) — so PKG_CONFIG_PATH must be passed explicitly here too, or a
# PKG_CONFIG_PATH-less environment silently falls back to the system
# pkg-config search path and fails to find protobuf/grpc++.
GRPC_CFLAGS := $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --cflags protobuf grpc++)
GRPC_LIBS   := $(shell PKG_CONFIG_PATH=$(PKG_CONFIG_PATH) pkg-config --libs protobuf grpc++)

PROTO_DIR := ../proto
GEN_DIR   := $(BUILD)/generated
PROTOC ?= protoc
GRPC_CPP_PLUGIN ?= grpc_cpp_plugin

.PHONY: all clean test
# bench_query (the Phase 1 exit-artifact driver) is not written yet.
all: $(BUILD)/build_index $(BUILD)/test_index ../py/cascade_index$(PY_SUFFIX)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/server:
	mkdir -p $(BUILD)/server

$(GEN_DIR)/search.pb.cc $(GEN_DIR)/search.pb.h $(GEN_DIR)/search.grpc.pb.cc $(GEN_DIR)/search.grpc.pb.h: $(PROTO_DIR)/search.proto | $(BUILD)
	mkdir -p $(GEN_DIR)
	$(PROTOC) -I $(PROTO_DIR) --cpp_out=$(GEN_DIR) --grpc_out=$(GEN_DIR) \
	  --plugin=protoc-gen-grpc=$$(which $(GRPC_CPP_PLUGIN)) $(PROTO_DIR)/search.proto

$(GEN_DIR)/%.o: $(GEN_DIR)/%.cc
	$(CXX) $(CXXFLAGS) $(GRPC_CFLAGS) -I$(GEN_DIR) -c $< -o $@

$(BUILD)/server/%.o: server/%.cc | $(BUILD)/server
	$(CXX) $(CXXFLAGS) $(GRPC_CFLAGS) -Iserver -I$(GEN_DIR) -c $< -o $@

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
