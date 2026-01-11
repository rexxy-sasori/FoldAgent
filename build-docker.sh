#!/bin/bash

# Docker Buildx Build Script
# Purpose: Simplify the execution of Docker Buildx build command with configurable options
# This script encapsulates the build process with error handling, clear feedback, and configuration options

# Define color variables for better output formatting
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Default values
DEFAULT_REGISTRY="harbor.xa.xshixun.com:7443/hanfeigeng"
DEFAULT_REPO="context-fold-agent"
USE_CACHE=false

# Print help message
print_help() {
    echo -e "${YELLOW}======================================${NC}"
    echo -e "${GREEN}Docker Buildx Build Script${NC}"
    echo -e "${YELLOW}======================================${NC}"
    echo -e "Usage: $0 [OPTIONS] -t <tag>"
    echo -e "\nOptions:"
    echo -e "  -r, --registry <registry>   Docker registry (default: $DEFAULT_REGISTRY)"
    echo -e "  -n, --repo <repo>           Docker repository name (default: $DEFAULT_REPO)"
    echo -e "  -t, --tag <tag>             Docker image tag (required)"
    echo -e "  --use-cache                 Use Docker cache (default: no cache)"
    echo -e "  -h, --help                  Show this help message"
    echo -e "\nExample:"
    echo -e "  $0 -t 0.65-log-llmbackend"
    echo -e "  $0 -r myregistry.com/myuser -n my-app -t 1.0.0 --use-cache"
    echo -e "${YELLOW}======================================${NC}"
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -n|--repo)
            REPO="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        --use-cache)
            USE_CACHE=true
            shift 1
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            print_help
            exit 1
            ;;
    esac
done

# Set default values if not provided
REGISTRY=${REGISTRY:-$DEFAULT_REGISTRY}
REPO=${REPO:-$DEFAULT_REPO}

# Validate required parameters
if [ -z "$TAG" ]; then
    echo -e "${RED}❌ Error: Image tag is required!${NC}"
    print_help
    exit 1
fi

# Print script header and configuration
echo -e "${YELLOW}======================================${NC}"
echo -e "${GREEN}Docker Buildx Build Script${NC}"
echo -e "${YELLOW}======================================${NC}"
echo -e "Building Docker image with configuration:"
echo -e "  Registry: ${YELLOW}$REGISTRY${NC}"
echo -e "  Repository: ${YELLOW}$REPO${NC}"
echo -e "  Tag: ${YELLOW}$TAG${NC}"
echo -e "  Use Cache: ${YELLOW}$USE_CACHE${NC}"
echo -e "${YELLOW}======================================${NC}\n"

# Construct the Docker Buildx command
IMAGE_NAME="$REGISTRY/$REPO:$TAG"
CACHE_OPTION="--no-cache"

if [ "$USE_CACHE" = true ]; then
    CACHE_OPTION=""
fi

BUILD_COMMAND="docker buildx build --platform=linux/amd64 -t $IMAGE_NAME . $CACHE_OPTION"

# Execute the build command and capture exit status
echo -e "${YELLOW}Running command:${NC} $BUILD_COMMAND\n"
$BUILD_COMMAND

# Check if the build succeeded
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}======================================${NC}"
    echo -e "${GREEN}✅ Build successful!${NC}"
    echo -e "Image: ${GREEN}$IMAGE_NAME${NC}"
    echo -e "${GREEN}======================================${NC}"
    exit 0
else
    echo -e "\n${RED}======================================${NC}"
    echo -e "${RED}❌ Build failed!${NC}"
    echo -e "Please check the error messages above."
    echo -e "${RED}======================================${NC}"
    exit 1
fi