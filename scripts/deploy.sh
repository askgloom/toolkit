#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VERSION=$(git describe --tags --always)
DEPLOY_DIR="deploy"
ARTIFACTS_DIR="artifacts"
CONFIG_DIR="config"
MODELS_DIR="models"

# Default values
ENVIRONMENT=${1:-development}
GPU_ENABLED=${2:-true}
CLEAN_DEPLOY=${3:-false}

# Print header
echo -e "${GREEN}=== Glooms Toolkit Deployment Script ===${NC}"
echo "Version: $VERSION"
echo "Environment: $ENVIRONMENT"
echo "GPU Enabled: $GPU_ENABLED"

# Function to check last command status
check_status() {
    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: $1${NC}"
        exit 1
    fi
}

# Function to create directory safely
create_dir() {
    if [ ! -d "$1" ]; then
        mkdir -p "$1"
        check_status "Failed to create directory: $1"
    fi
}

# Clean previous deployment if requested
if [ "$CLEAN_DEPLOY" = true ]; then
    echo -e "\n${YELLOW}Cleaning previous deployment...${NC}"
    rm -rf $DEPLOY_DIR
    check_status "Failed to clean deployment directory"
fi

# Create deployment structure
echo -e "\n${YELLOW}Creating deployment structure...${NC}"
create_dir "$DEPLOY_DIR"
create_dir "$DEPLOY_DIR/bin"
create_dir "$DEPLOY_DIR/lib"
create_dir "$DEPLOY_DIR/config"
create_dir "$DEPLOY_DIR/models"
create_dir "$DEPLOY_DIR/logs"
create_dir "$ARTIFACTS_DIR"

# Copy binaries
echo -e "\n${YELLOW}Copying binaries...${NC}"
cp build/bin/* "$DEPLOY_DIR/bin/"
check_status "Failed to copy binaries"

# Copy libraries
echo -e "\n${YELLOW}Copying libraries...${NC}"
cp build/lib/* "$DEPLOY_DIR/lib/"
check_status "Failed to copy libraries"

# Copy configuration files
echo -e "\n${YELLOW}Copying configuration...${NC}"
cp "$CONFIG_DIR/$ENVIRONMENT/"* "$DEPLOY_DIR/config/"
check_status "Failed to copy configuration"

# Copy models
echo -e "\n${YELLOW}Copying models...${NC}"
cp -r "$MODELS_DIR/"* "$DEPLOY_DIR/models/"
check_status "Failed to copy models"

# Generate version file
echo -e "\n${YELLOW}Generating version information...${NC}"
cat > "$DEPLOY_DIR/version.json" << EOF
{
    "version": "$VERSION",
    "build_date": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "environment": "$ENVIRONMENT",
    "gpu_enabled": $GPU_ENABLED,
    "git_commit": "$(git rev-parse HEAD)",
    "git_branch": "$(git rev-parse --abbrev-ref HEAD)"
}
EOF

# Create deployment package
echo -e "\n${YELLOW}Creating deployment package...${NC}"
PACKAGE_NAME="glooms-toolkit-${VERSION}-${ENVIRONMENT}"
tar -czf "$ARTIFACTS_DIR/${PACKAGE_NAME}.tar.gz" -C "$DEPLOY_DIR" .
check_status "Failed to create deployment package"

# Generate deployment manifest
echo -e "\n${YELLOW}Generating deployment manifest...${NC}"
cat > "$ARTIFACTS_DIR/${PACKAGE_NAME}-manifest.txt" << EOF
Glooms Toolkit Deployment Manifest
=================================
Version: $VERSION
Environment: $ENVIRONMENT
Deployment Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
GPU Enabled: $GPU_ENABLED

Contents:
$(find "$DEPLOY_DIR" -type f | sed 's|'$DEPLOY_DIR'/||' | sort)

Dependencies:
$(ldd "$DEPLOY_DIR/bin/"* | sort -u)

Environment Information:
- OS: $(uname -a)
- GPU: $(if command -v nvidia-smi > /dev/null; then nvidia-smi --query-gpu=gpu_name --format=csv,noheader; else echo "No GPU detected"; fi)
- OpenCV: $(pkg-config --modversion opencv4)
EOF

# Set permissions
echo -e "\n${YELLOW}Setting permissions...${NC}"
chmod -R 755 "$DEPLOY_DIR/bin"
chmod -R 644 "$DEPLOY_DIR/config"
chmod -R 644 "$DEPLOY_DIR/models"

# Verify deployment
echo -e "\n${YELLOW}Verifying deployment...${NC}"
if [ -x "$DEPLOY_DIR/bin/glooms_toolkit" ]; then
    "$DEPLOY_DIR/bin/glooms_toolkit" --version
    check_status "Deployment verification failed"
fi

# Print deployment information
echo -e "\n${BLUE}Deployment Information:${NC}"
echo "Package: $ARTIFACTS_DIR/${PACKAGE_NAME}.tar.gz"
echo "Manifest: $ARTIFACTS_DIR/${PACKAGE_NAME}-manifest.txt"
echo "Version: $VERSION"
echo "Environment: $ENVIRONMENT"
echo "GPU Enabled: $GPU_ENABLED"

echo -e "\n${GREEN}Deployment completed successfully!${NC}"

# Usage instructions
echo -e "\n${YELLOW}Usage:${NC}"
echo "  1. Extract the deployment package:"
echo "     tar -xzf ${PACKAGE_NAME}.tar.gz"
echo "  2. Set environment variables:"
echo "     export GLOOMS_CONFIG_DIR=./config"
echo "     export GLOOMS_MODELS_DIR=./models"
echo "  3. Run the toolkit:"
echo "     ./bin/glooms_toolkit"

exit 0