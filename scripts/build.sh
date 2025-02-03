#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Build configuration
BUILD_TYPE=${1:-Release}
BUILD_DIR="build"
INSTALL_DIR="install"
NUM_CORES=$(nproc)

# Print header
echo -e "${GREEN}=== Glooms Toolkit Build Script ===${NC}"
echo "Build type: $BUILD_TYPE"
echo "Using $NUM_CORES cores for compilation"

# Function to check last command status
check_status() {
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: $1${NC}"
        exit 1
    fi
}

# Check for required tools
echo -e "\n${YELLOW}Checking required tools...${NC}"

# Check CMake
cmake --version > /dev/null 2>&1
check_status "CMake not found. Please install CMake."

# Check OpenCV
pkg-config --modversion opencv4 > /dev/null 2>&1
check_status "OpenCV not found. Please install OpenCV."

# Check CUDA if available
if command -v nvcc &> /dev/null; then
    echo "CUDA found: $(nvcc --version | grep release | awk '{print $5,$6}')"
    CUDA_FLAGS="-DUSE_CUDA=ON"
else
    echo "CUDA not found. Building without GPU support."
    CUDA_FLAGS="-DUSE_CUDA=OFF"
fi

# Create build directory
echo -e "\n${YELLOW}Creating build directory...${NC}"
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
check_status "Failed to create build directory"

# Configure project
echo -e "\n${YELLOW}Configuring project...${NC}"
cd $BUILD_DIR
cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_INSTALL_PREFIX=../$INSTALL_DIR \
    $CUDA_FLAGS \
    -DBUILD_TESTS=ON \
    -DBUILD_EXAMPLES=ON
check_status "CMake configuration failed"

# Build project
echo -e "\n${YELLOW}Building project...${NC}"
cmake --build . -- -j$NUM_CORES
check_status "Build failed"

# Run tests if requested
if [ "$2" == "--test" ]; then
    echo -e "\n${YELLOW}Running tests...${NC}"
    ctest --output-on-failure
    check_status "Tests failed"
fi

# Install
echo -e "\n${YELLOW}Installing...${NC}"
cmake --install .
check_status "Installation failed"

# Build documentation if Doxygen is available
if command -v doxygen &> /dev/null; then
    echo -e "\n${YELLOW}Building documentation...${NC}"
    cmake --build . --target docs
    check_status "Documentation generation failed"
fi

# Create package
echo -e "\n${YELLOW}Creating package...${NC}"
cpack -G "TGZ;ZIP"
check_status "Package creation failed"

# Print success message
echo -e "\n${GREEN}Build completed successfully!${NC}"
echo "Installation directory: $(pwd)/$INSTALL_DIR"
echo "Documentation: $(pwd)/docs/html/index.html"
echo "Packages: $(pwd)/glooms-*.{tar.gz,zip}"

# Print build information
echo -e "\n${YELLOW}Build Information:${NC}"
echo "Build Type: $BUILD_TYPE"
echo "OpenCV Version: $(pkg-config --modversion opencv4)"
if [ ! -z "$CUDA_FLAGS" ]; then
    echo "CUDA Version: $(nvcc --version | grep release | awk '{print $5,$6}')"
fi
echo "CMake Version: $(cmake --version | head -n1)"
echo "Compiler: $(cc --version | head -n1)"

# Usage instructions
echo -e "\n${YELLOW}Usage:${NC}"
echo "  ./glooms_toolkit --help"
echo "  ./examples/vision_example"
echo "  ./tests/unit_tests"

exit 0