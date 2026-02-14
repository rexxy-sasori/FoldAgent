#!/bin/bash
"""
Wrapper script to generate and deploy Kubernetes YAML for downloading Hugging Face models.

Usage: ./scripts/download-model.sh <model-path> [--output <yaml-file>]

Example:
./scripts/download-model.sh ByteDance-Seed/Seed-OSS-36B-Instruct
./scripts/download-model.sh ByteDance-Seed/Seed-OSS-36B-Instruct --output custom-download.yaml
"""

set -e

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not available"
    exit 1
fi

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "Warning: kubectl is not available, will only generate YAML"
    CAN_DEPLOY=false
else
    CAN_DEPLOY=true
fi

# Generate the YAML file using the Python script
echo "Generating download model YAML..."
python3 scripts/generate-download-model-yaml.py "$@"

# Get the output file name from arguments or use the default
OUTPUT_FILE=""
for arg in "$@"; do
    if [ "$arg" == "--output" ] || [ "$arg" == "-o" ]; then
        GET_OUTPUT=true
    elif [ "$GET_OUTPUT" == true ]; then
        OUTPUT_FILE="$arg"
        break
    fi
done

# If no output file specified, use the default
if [ -z "$OUTPUT_FILE" ]; then
    MODEL_PATH="$1"
    MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-' | tr '[:upper:]' '[:lower:]')
    OUTPUT_FILE="download-model-$MODEL_NAME.yaml"
fi

# Deploy the YAML if kubectl is available
if [ "$CAN_DEPLOY" == true ]; then
    echo -e "\n🔄 Deploying the download model pod..."
    kubectl apply -f "$OUTPUT_FILE"
    
    # Show logs command
    POD_NAME=$(grep -A 2 "name:" "$OUTPUT_FILE" | grep -v "name:" | grep -v "metadata:" | tr -d ' "')
    echo -e "\n📊 To check download progress run:"
    echo -e "   kubectl logs -n liuyunxin $POD_NAME -f"
    
    # Show cleanup command
    echo -e "\n🗑️  To cleanup after download completes:"
    echo -e "   kubectl delete -f $OUTPUT_FILE"
fi
