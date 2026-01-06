#!/bin/bash

# Script to check sglang server health status

set -euo pipefail

# Dry run flag
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run|-d)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--dry-run|-d]"
            exit 1
            ;;
    esac
done

# Default values
HEALTH_CHECK_URL="https://grafana-liuyunxin.xa.xshixun.cn:7443/sglang/health"
MAX_RETRIES=60
RETRY_INTERVAL=10  # seconds
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

echo "Starting sglang server health check..."

# Connect to the cluster if not already connected
if [ "$DRY_RUN" = false ]; then
    echo "Connecting to cluster using $CLUSTER_CONNECT_SCRIPT..."
    source "$CLUSTER_CONNECT_SCRIPT"
    echo "Cluster connection established successfully!"
else
    echo "DRY RUN: Would connect to cluster using $CLUSTER_CONNECT_SCRIPT"
fi
echo

echo "Checking server health at $HEALTH_CHECK_URL..."

success=false
for ((i=1; i<=MAX_RETRIES; i++)); do
    if [ "$DRY_RUN" = true ]; then
        echo "DRY RUN: curl -k -s -o /dev/null -w '%{http_code}' '$HEALTH_CHECK_URL' | grep -q '200'"
        # Simulate success for dry run
        echo "✅ Server is healthy!"
        success=true
        break
    else
        if exec_cmd "curl -k -s -o /dev/null -w '%{http_code}' '$HEALTH_CHECK_URL' | grep -q '200'"; then
            echo "✅ Server is healthy!"
            success=true
            break
        fi
    fi
    
    echo "⏳ Retry $i/$MAX_RETRIES: Server not ready yet..."
    exec_cmd "sleep '$RETRY_INTERVAL'"
done

if [ "$success" = false ]; then
    echo "❌ Server did not become healthy after $MAX_RETRIES retries"
    exit 1
fi

echo

echo "Server is ready for evaluation!"
