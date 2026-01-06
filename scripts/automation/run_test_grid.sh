#!/bin/bash

# Main orchestration script to run the complete test grid for sglang server
# Tests combinations of: scheduling_policy (lpm/fcfs) × context_length (49152/130172) × workflow (search/search_branch)

# Set script to exit on error
set -e

# Dry run flag
DRY_RUN=false

# Parse command line arguments
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

# Directories and files
SCRIPT_DIR="$(dirname "$0")"
ROOT_DIR="$(dirname "$SCRIPT_DIR")/.."
TEST_CONFIGS_PY="$SCRIPT_DIR/test_configs.py"
DEPLOY_SGLANG_SH="$SCRIPT_DIR/deploy_sglang.sh"
CHECK_HEALTH_SH="$SCRIPT_DIR/check_server_health.sh"
EVAL_DEPLOYMENT_FILE="$ROOT_DIR/deployment/sglang/eval-deployment-sglang.yaml"
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

# Ensure all scripts are executable
exec_cmd "chmod +x '$DEPLOY_SGLANG_SH'"
exec_cmd "chmod +x '$CHECK_HEALTH_SH'"

# Connect to the cluster (skip in dry run since it's just sourcing)
if [ "$DRY_RUN" = false ]; then
    echo "Connecting to cluster using $CLUSTER_CONNECT_SCRIPT..."
    source "$CLUSTER_CONNECT_SCRIPT"
    echo "Cluster connection established successfully!"
else
    echo "DRY RUN: Would connect to cluster using $CLUSTER_CONNECT_SCRIPT"
fi

# Generate test configurations
echo "Generating test configurations..."
TEST_CONFIGS=$(python3 "$TEST_CONFIGS_PY")
echo "Generated test configurations:"
echo "$TEST_CONFIGS"

# Convert Python list to bash array using jq (assuming jq is installed)
# If jq is not available, we'll use a different approach
if command -v jq &> /dev/null; then
    NUM_CONFIGS=$(echo "$TEST_CONFIGS" | jq length)
    echo "Total test configurations: $NUM_CONFIGS"
else
    echo "Warning: jq not found. Using alternative approach."
    # Fallback approach: count lines
    NUM_CONFIGS=$(echo "$TEST_CONFIGS" | grep -c '{')
fi

echo ""
echo "Starting test grid execution..."
echo "="[32m=== Starting Test Grid Execution ===[0m"
echo ""

# Counter for completed tests
COMPLETED_TESTS=0

# Iterate through each configuration
while IFS= read -r config; do
    if [[ -z "$config" || "$config" == "[" || "$config" == "]" || "$config" == " " ]]; then
        continue
    fi
    
    # Increment completed tests counter
    ((COMPLETED_TESTS++))
    
    echo ""
    echo "[33m=== Test $COMPLETED_TESTS/$NUM_CONFIGS ===[0m"
    echo "Configuration: $config"
    echo ""
    
    # Extract parameters using grep/sed (JSON with double quotes)
    SCHEDULING_POLICY=$(echo "$config" | grep -o '"scheduling_policy": "[^"]*"' | sed 's/"scheduling_policy": "\([^"]*\)"/\1/')
    CONTEXT_LENGTH=$(echo "$config" | grep -o '"context_length": [0-9]*' | sed 's/"context_length": \([0-9]*\)/\1/')
    WORKFLOW=$(echo "$config" | grep -o '"workflow": "[^"]*"' | sed 's/"workflow": "\([^"]*\)"/\1/')
    
    echo "Parameters extracted:"
    echo "- Scheduling Policy: $SCHEDULING_POLICY"
    echo "- Context Length: $CONTEXT_LENGTH"
    echo "- Workflow: $WORKFLOW"
    echo ""
    
    # Step 1: Deploy sglang server with current parameters
    echo "[34mStep 1: Deploying sglang server...[0m"
    exec_cmd "'$DEPLOY_SGLANG_SH' '$SCHEDULING_POLICY' '$CONTEXT_LENGTH'"
    echo ""
    
    # Step 2: Check server health
    echo "[34mStep 2: Checking server health...[0m"
    exec_cmd "'$CHECK_HEALTH_SH'"
    echo ""
    
    # Step 3: Deploy eval_bc with current workflow
    echo "[34mStep 3: Deploying eval_bc with workflow '$WORKFLOW'...[0m"
    
    # Create a temporary copy of the eval deployment file
    TEMP_EVAL_DEPLOYMENT="/tmp/eval-deployment-sglang-$WORKFLOW-$SCHEDULING_POLICY-$CONTEXT_LENGTH.yaml"
    
    # Only modify files in non-dry run mode
    if [ "$DRY_RUN" = false ]; then
        cp "$EVAL_DEPLOYMENT_FILE" "$TEMP_EVAL_DEPLOYMENT"
        
        # Update the workflow in the deployment file
        sed -i '' "s/--workflow [a-z_]*/--workflow $WORKFLOW/" "$TEMP_EVAL_DEPLOYMENT"
        
        # Update the output directory to include test parameters
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        OUTPUT_DIR="/root/results/test_${WORKFLOW}_${SCHEDULING_POLICY}_${CONTEXT_LENGTH}_${TIMESTAMP}"
        sed -i '' "s|--output_dir /root/results|--output_dir $OUTPUT_DIR|" "$TEMP_EVAL_DEPLOYMENT"
    else
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        OUTPUT_DIR="/root/results/test_${WORKFLOW}_${SCHEDULING_POLICY}_${CONTEXT_LENGTH}_${TIMESTAMP}"
        echo "DRY RUN: Would copy '$EVAL_DEPLOYMENT_FILE' to '$TEMP_EVAL_DEPLOYMENT'"
        echo "DRY RUN: Would update workflow to '$WORKFLOW'"
        echo "DRY RUN: Would set output directory to '$OUTPUT_DIR'"
    fi
    
    echo "Updated eval deployment with:"
    echo "- Workflow: $WORKFLOW"
    echo "- Output Directory: $OUTPUT_DIR"
    
    # Delete existing eval deployment if it exists
    if [ "$DRY_RUN" = false ]; then
        if kubectl get deployment foldagent-eval-deployment-sglang -n liuyunxin &> /dev/null; then
            echo "Deleting existing eval deployment..."
            exec_cmd "kubectl delete deployment foldagent-eval-deployment-sglang -n liuyunxin"
            # Wait for deployment to be deleted
            sleep 5
        fi
    else
        echo "DRY RUN: Would check for existing eval deployment"
        echo "DRY RUN: Would delete existing eval deployment if it exists"
    fi
    
    # Deploy the updated eval configuration
    echo "Applying updated eval deployment..."
    exec_cmd "kubectl apply -f '$TEMP_EVAL_DEPLOYMENT' -n liuyunxin"
    
    # Wait for eval pod to be ready
    echo "Waiting for eval pod to be ready..."
    exec_cmd "kubectl wait --for=condition=available deployment/foldagent-eval-deployment-sglang -n liuyunxin --timeout=300s"
    
    # Get the eval pod name (mock in dry run)
    if [ "$DRY_RUN" = false ]; then
        EVAL_POD_NAME=$(kubectl get pods -n liuyunxin -l app=foldagent-eval-sglang -o jsonpath='{.items[0].metadata.name}')
    else
        EVAL_POD_NAME="mock-eval-pod-$WORKFLOW-$SCHEDULING_POLICY-$CONTEXT_LENGTH"
    fi
    echo "Eval pod running: $EVAL_POD_NAME"
    
    # Step 4: Monitor the evaluation progress
    echo ""
    echo "[34mStep 4: Monitoring evaluation progress...[0m"
    echo "Viewing logs for eval pod: $EVAL_POD_NAME"
    echo "Press Ctrl+C to stop viewing logs (evaluation will continue in background)"
    echo ""
    
    # Show logs for 2 minutes to monitor progress
    exec_cmd "timeout 120 kubectl logs -f '$EVAL_POD_NAME' -n liuyunxin"
    
    echo ""
    echo "[32mTest $COMPLETED_TESTS/$NUM_CONFIGS completed successfully![0m"
    echo "Results will be available in: $OUTPUT_DIR"
    echo ""
    
    # Clean up temporary file
    exec_cmd "rm '$TEMP_EVAL_DEPLOYMENT'"
done < <(echo "$TEST_CONFIGS" | tr -d '\n' | sed 's/},{/}\n{/g' | tr -d '[]')

echo ""
echo "[32m=== Test Grid Execution Complete ===[0m"
echo "All $NUM_CONFIGS tests have been executed successfully!"
echo "Results are available in the respective output directories on the cluster."
echo "To view results, use: kubectl exec -it <eval-pod-name> -n liuyunxin -- ls -la /root/results/"
