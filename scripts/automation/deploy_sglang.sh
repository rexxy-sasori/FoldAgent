#!/bin/bash

# Script to deploy sglang server with different configurations

set -euo pipefail

# Dry run flag
DRY_RUN=false

# Parse arguments
while [[ $# -gt 2 ]]; do
    case $1 in
        --dry-run|-d)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--dry-run|-d] <scheduling_policy> <context_length>"
            echo "Example: $0 lpm 49152"
            echo "Example: $0 fcfs 130172"
            echo "Example: $0 --dry-run lpm 49152"
            exit 1
            ;;
    esac
done

# Default values
SCRIPT_DIR="$(dirname "$0")"
ROOT_DIR="$(dirname "$SCRIPT_DIR")/.."
NAMESPACE="liuyunxin"
DEPLOYMENT_FILE="$ROOT_DIR/deployment/sglang/sglang-deployment.yaml"
TEMP_DEPLOYMENT="/tmp/sglang-deployment-temp.yaml"
CLUSTER_CONNECT_SCRIPT="~/h200-connect.sh"

# Function to execute commands (with dry run support)
exec_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo "DRY RUN: $@"
    else
        echo "EXEC: $@"
        eval "$@"
    fi
}

echo "Starting sglang server deployment script..."

# Connect to the cluster (skip in dry run since it's just sourcing)
if [ "$DRY_RUN" = false ]; then
    echo "Connecting to cluster using $CLUSTER_CONNECT_SCRIPT..."
    source "$CLUSTER_CONNECT_SCRIPT"
    echo "Cluster connection established successfully!"
else
    echo "DRY RUN: Would connect to cluster using $CLUSTER_CONNECT_SCRIPT"
fi
echo

# Parse arguments (after handling options)
if [ $# -ne 2 ]; then
    echo "Usage: $0 [--dry-run|-d] <scheduling_policy> <context_length>"
    echo "Example: $0 lpm 49152"
    echo "Example: $0 fcfs 130172"
    echo "Example: $0 --dry-run lpm 49152"
    exit 1
fi

SCHEDULING_POLICY=$1
CONTEXT_LENGTH=$2

# Validate scheduling policy
if [[ "$SCHEDULING_POLICY" != "lpm" && "$SCHEDULING_POLICY" != "fcfs" ]]; then
    echo "Error: Invalid scheduling policy '$SCHEDULING_POLICY'. Must be 'lpm' or 'fcfs'."
    exit 1
fi

# Validate context length
if [[ "$CONTEXT_LENGTH" != "49152" && "$CONTEXT_LENGTH" != "130172" ]]; then
    echo "Error: Invalid context length '$CONTEXT_LENGTH'. Must be '49152' or '130172'."
    exit 1
fi

echo "Deploying sglang server with:"
echo "  - Scheduling policy: $SCHEDULING_POLICY"
echo "  - Context length: $CONTEXT_LENGTH"
echo

# Create temporary deployment file with modified parameters
echo "Creating temporary deployment file..."

# Only modify files in non-dry run mode
if [ "$DRY_RUN" = false ]; then
    cp "$DEPLOYMENT_FILE" "$TEMP_DEPLOYMENT"
    
    # Update scheduling policy
    sed -i '' "s/--schedule-policy [a-z]\+/--schedule-policy $SCHEDULING_POLICY/" "$TEMP_DEPLOYMENT"
    
    # Update context length
    sed -i '' "s/--context-length [0-9]\+/--context-length $CONTEXT_LENGTH/" "$TEMP_DEPLOYMENT"
    
    echo "Updated deployment parameters in temporary file"
else
    echo "DRY RUN: Would copy '$DEPLOYMENT_FILE' to '$TEMP_DEPLOYMENT'"
    echo "DRY RUN: Would update scheduling-policy to '$SCHEDULING_POLICY'"
    echo "DRY RUN: Would update context-length to '$CONTEXT_LENGTH'"
fi

echo "Applying deployment to Kubernetes..."
exec_cmd "kubectl apply -f '$TEMP_DEPLOYMENT' -n '$NAMESPACE'"

echo "Waiting for deployment to complete..."
exec_cmd "kubectl rollout status deployment/sglang -n '$NAMESPACE' --timeout=10m"

echo

echo "Deployment completed successfully!"
