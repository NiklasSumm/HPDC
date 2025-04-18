# Compiler and flags
CXX = mpic++
CXXSTD = -std=c++20
DEBUG_FLAGS = -g -Wall -Wextra -DDEBUG
RELEASE_FLAGS = -O3

# Paths
SRC = template.cpp
DEBUG_OUT = template_dbg
RELEASE_OUT = template

# Check if we're in the right directory
$(if $(wildcard $(SRC)),,$(error Source file $(SRC) not found. Please run make from the directory containing $(SRC)))

# Default target
all: release

# Debug build
debug: $(DEBUG_OUT)

$(DEBUG_OUT): $(SRC)
	$(CXX) $(SRC) -o $(DEBUG_OUT) $(CXXSTD) $(DEBUG_FLAGS)

# Release build
release: $(RELEASE_OUT)

$(RELEASE_OUT): $(SRC)
	$(CXX) $(SRC) -o $(RELEASE_OUT) $(CXXSTD) $(RELEASE_FLAGS)

# Clean up
clean:
	rm -f $(DEBUG_OUT) $(RELEASE_OUT)

# Phony targets
.PHONY: all debug release clean