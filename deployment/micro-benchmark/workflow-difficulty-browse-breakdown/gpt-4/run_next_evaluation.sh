#!/bin/bash
set -e

NAMESPACE="liuyunxin"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Paths to dependent deployments
SGLANG_DEPLOYMENT="/Users/rexsasori/FoldAgent/deployment/sglang/sglang-deployment.yaml"
SEARCH_SERVER_DEPLOYMENT="/Users/rexsasori/FoldAgent/deployment/search-server-deployment.yaml"
QWEN_JUDGER_DEPLOYMENT="/Users/rexsasori/FoldAgent/deployment/sglang/sglang-qwen3-32b-judger.yaml"

# Job files
CURRENT_JOB_FILE="${SCRIPT_DIR}/eval-job-search-all-kvcache-logging.yaml"
NEXT_JOB_FILE="${SCRIPT_DIR}/eval-job-search_branch-all-kvcache-logging.yaml"

# Dry run flag
DRY_RUN=false

# Function to get current timestamp
get_timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

# Function to print dry run message
print_dry_run() {
    echo "[$(get_timestamp)] [DRY RUN] $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -n, --dry-run      Dry run (show what would be executed)"
            echo "  -h, --help         Show this help message"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [OPTIONS]"
            echo "Try '$0 --help' for more information."
            exit 1
            ;;
    esac
done

# Function to restart dependent deployments
restart_deployments() {
    local start_time=$(get_timestamp)
    echo ""
    echo "Restarting dependent deployments..."
    echo "-----------------------------------"
    echo "Start Time: ${start_time}"
    
    # Restart sglang deployment
    echo "[$(get_timestamp)] Restarting sglang deployment..."
    if [ -f "${SGLANG_DEPLOYMENT}" ]; then
        if ${DRY_RUN}; then
            print_dry_run "Would execute: kubectl delete -f \"${SGLANG_DEPLOYMENT}\" --ignore-not-found=true -n ${NAMESPACE}"
            print_dry_run "Would execute: kubectl apply -f \"${SGLANG_DEPLOYMENT}\" -n ${NAMESPACE}"
        else
            kubectl delete -f "${SGLANG_DEPLOYMENT}" --ignore-not-found=true -n ${NAMESPACE}
            kubectl apply -f "${SGLANG_DEPLOYMENT}" -n ${NAMESPACE}
        fi
    else
        echo "[$(get_timestamp)] WARNING: sglang deployment file not found: ${SGLANG_DEPLOYMENT}"
        echo "[$(get_timestamp)] Attempting to restart existing deployment..."
        if ${DRY_RUN}; then
            print_dry_run "Would execute: kubectl rollout restart deployment sglang -n ${NAMESPACE}"
        else
            kubectl rollout restart deployment sglang -n ${NAMESPACE}
        fi
    fi
    
    # Restart search-server deployment
    echo "[$(get_timestamp)] Restarting search-server deployment..."
    if [ -f "${SEARCH_SERVER_DEPLOYMENT}" ]; then
        if ${DRY_RUN}; then
            print_dry_run "Would execute: kubectl delete -f \"${SEARCH_SERVER_DEPLOYMENT}\" --ignore-not-found=true -n ${NAMESPACE}"
            print_dry_run "Would execute: kubectl apply -f \"${SEARCH_SERVER_DEPLOYMENT}\" -n ${NAMESPACE}"
        else
            kubectl delete -f "${SEARCH_SERVER_DEPLOYMENT}" --ignore-not-found=true -n ${NAMESPACE}
            kubectl apply -f "${SEARCH_SERVER_DEPLOYMENT}" -n ${NAMESPACE}
        fi
    else
        echo "[$(get_timestamp)] WARNING: search-server deployment file not found: ${SEARCH_SERVER_DEPLOYMENT}"
        echo "[$(get_timestamp)] Attempting to restart existing deployment..."
        if ${DRY_RUN}; then
            print_dry_run "Would execute: kubectl rollout restart deployment search-server -n ${NAMESPACE}"
        else
            kubectl rollout restart deployment search-server -n ${NAMESPACE}
        fi
    fi
    
    echo "[$(get_timestamp)] Waiting for deployments to be ready..."
    echo "-----------------------------------"
    
    # Wait for sglang deployment to be ready
    echo "[$(get_timestamp)] Checking sglang deployment status..."
    if ${DRY_RUN}; then
        print_dry_run "Would execute: kubectl rollout status deployment sglang -n ${NAMESPACE} --timeout=600s"
    else
        kubectl rollout status deployment sglang -n ${NAMESPACE} --timeout=600s
    fi
    
    # Verify sglang pods are ready using kubectl wait
    echo "[$(get_timestamp)] Verifying sglang pods are ready..."
    if ${DRY_RUN}; then
        print_dry_run "Would execute: kubectl wait --for=condition=ready pod -l app=sglang -n ${NAMESPACE} --timeout=300s"
        print_dry_run "sglang deployment is ready!"
    else
        if kubectl wait --for=condition=ready pod -l app=sglang -n ${NAMESPACE} --timeout=300s; then
            echo "[$(get_timestamp)] sglang deployment is ready!"
        else
            echo "[$(get_timestamp)] ERROR: sglang pods not ready within timeout"
            echo "[$(get_timestamp)] Evaluation cannot proceed without sglang deployment"
            return 1
        fi
    fi
    
    # Wait for search-server deployment to be ready
    echo "[$(get_timestamp)] Checking search-server deployment status..."
    if ${DRY_RUN}; then
        print_dry_run "Would execute: kubectl rollout status deployment search-server -n ${NAMESPACE} --timeout=600s"
    else
        kubectl rollout status deployment search-server -n ${NAMESPACE} --timeout=600s
    fi
    
    # Verify search-server pods are ready using kubectl wait
    echo "[$(get_timestamp)] Verifying search-server pods are ready..."
    if ${DRY_RUN}; then
        print_dry_run "Would execute: kubectl wait --for=condition=ready pod -l app=search-server -n ${NAMESPACE} --timeout=300s"
        print_dry_run "search-server deployment is ready!"
    else
        if kubectl wait --for=condition=ready pod -l app=search-server -n ${NAMESPACE} --timeout=300s; then
            echo "[$(get_timestamp)] search-server deployment is ready!"
        else
            echo "[$(get_timestamp)] ERROR: search-server pods not ready within timeout"
            echo "[$(get_timestamp)] Evaluation cannot proceed without search-server deployment"
            return 1
        fi
    fi
    
    # Verify services are reachable
    echo "[$(get_timestamp)] Checking if services are reachable..."
    echo "[$(get_timestamp)] sglang service: http://sglang.liuyunxin:8080"
    echo "[$(get_timestamp)] search-server service: http://search-server.liuyunxin:8000"
    
    local end_time=$(get_timestamp)
    echo ""
    echo "Deployments restarted successfully!"
    echo "-----------------------------------"
    echo "Start Time: ${start_time}"
    echo "End Time: ${end_time}"
}

# Function to deploy and monitor a job
deploy_and_monitor_job() {
    local job_file=$1
    local start_time=$(get_timestamp)
    
    echo ""
    echo "=========================================="
    echo "Deploying and monitoring job:"
    echo "  Job: ${job_file}"
    echo "  Start Time: ${start_time}"
    echo "=========================================="
    
    if [ ! -f "${job_file}" ]; then
        echo "[$(get_timestamp)] ERROR: Job file not found: ${job_file}"
        return 1
    fi
    
    echo "[$(get_timestamp)] Deleting existing job (if any)..."
    if ${DRY_RUN}; then
        print_dry_run "Would execute: kubectl delete -f \"${job_file}\" --ignore-not-found=true -n ${NAMESPACE}"
    else
        kubectl delete -f "${job_file}" --ignore-not-found=true -n ${NAMESPACE}
    fi
    
    echo "[$(get_timestamp)] Creating job..."
    if ${DRY_RUN}; then
        print_dry_run "Would execute: kubectl create -f \"${job_file}\" -n ${NAMESPACE}"
    else
        kubectl create -f "${job_file}" -n ${NAMESPACE}
    fi
    
    # Get the job name from the file
    local job_name=$(grep -A 1 "name:" "${job_file}" | grep -v "metadata:" | grep -v "^--$" | head -1 | awk '{print $2}')
    
    if [ -z "${job_name}" ]; then
        echo "[$(get_timestamp)] ERROR: Could not extract job name from ${job_file}"
        return 1
    fi
    
    echo "[$(get_timestamp)] Found job: ${job_name}"
    echo "[$(get_timestamp)] Monitoring job progress..."
    echo "[$(get_timestamp)] Waiting for job to complete..."

    if ${DRY_RUN}; then
        print_dry_run "Would monitor job: ${job_name}"
        print_dry_run "Would wait for job to complete with 10 hour timeout"
        print_dry_run "Job monitoring would complete"
        return 0
    fi

    # Monitor job logs while waiting
    local start_wait=$(date +%s)
    local timeout=36000  # 10 hours timeout
    local check_interval=60  # Check every 60 seconds
    local elapsed=0

    # Show initial logs
    sleep 5
    echo "[$(get_timestamp)] Showing job logs..."

    # Wait for job to complete
    local job_completed=false
    while [ ${elapsed} -lt ${timeout} ]; do
        # Check if job is complete (with error handling for VPN disconnections)
        if kubectl get job "${job_name}" -n ${NAMESPACE} -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null | grep -q "True"; then
            job_completed=true
            break
        fi
        
        # Check if job failed (with error handling for VPN disconnections)
        if kubectl get job "${job_name}" -n ${NAMESPACE} -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null | grep -q "True"; then
            job_completed=true
            break
        fi
        
        # Show logs periodically (with error handling for VPN disconnections)
        if [ $((elapsed % 60)) -eq 0 ]; then
            echo ""
            echo "[$(get_timestamp)] Job still running..."
            kubectl logs job/"${job_name}" -n ${NAMESPACE} --tail=20 2>/dev/null || echo "[$(get_timestamp)] Unable to fetch logs (may be VPN connection issue)"
            echo ""
        fi
        
        # Sleep for check interval
        sleep ${check_interval}
        elapsed=$((elapsed + check_interval))
    done
    
    # Check if job completed within timeout
    if [ "${job_completed}" = "false" ]; then
        echo ""
        echo "[$(get_timestamp)] WARNING: Job did not complete within ${timeout} seconds"
        local job_status=$(kubectl get job "${job_name}" -n ${NAMESPACE} -o jsonpath='{.status.conditions[0].type}' 2>/dev/null)
        echo "[$(get_timestamp)] Job status: ${job_status}"
    fi
    
    local end_time=$(get_timestamp)
    local job_status=$(kubectl get job "${job_name}" -n ${NAMESPACE} -o jsonpath='{.status.conditions[*].type}' 2>/dev/null)
    
    echo ""
    echo "Job ${job_status}: ${job_file}"
    echo "=========================================="
    echo "Start Time: ${start_time}"
    echo "End Time: ${end_time}"
    echo "Job Status: ${job_status}"
    echo "Results are saved in the container's /root/results directory"
    
    # Get the pod name from the job
    echo "[$(get_timestamp)] Getting pod name for job: ${job_name}"
    local pod_name=$(kubectl get pods -l job-name="${job_name}" -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    
    if [ -z "${pod_name}" ]; then
        echo "[$(get_timestamp)] ERROR: No pod found for job: ${job_name}"
        return 1
    fi
    
    echo "[$(get_timestamp)] Found pod: ${pod_name}"
    
    # Show final logs
    if [[ "${job_status}" != "Complete" ]]; then
        echo ""
        echo "[$(get_timestamp)] Showing full logs for failed job..."
        kubectl logs "${pod_name}" -n ${NAMESPACE} 2>/dev/null || echo "[$(get_timestamp)] Unable to fetch logs (may be VPN connection issue)"
        echo "=========================================="
        return 1
    else
        echo ""
        echo "[$(get_timestamp)] Showing recent logs..."
        kubectl logs "${pod_name}" -n ${NAMESPACE} --tail=50 2>/dev/null || echo "[$(get_timestamp)] Unable to fetch logs (may be VPN connection issue)"
        echo "=========================================="
        return 0
    fi
}

# Main script execution
echo "=========================================="
echo "FoldAgent Evaluation Sequence Runner"
echo "=========================================="
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo "Current job file: ${CURRENT_JOB_FILE}"
echo "Next job file: ${NEXT_JOB_FILE}"

if ${DRY_RUN}; then
    echo ""
    echo "=========================================="
    echo "Dry Run Mode"
    echo "=========================================="
    echo "This is a dry run - no actual changes will be made."
    echo ""
fi

# Step 1: Wait for current job to complete
echo ""
echo "Step 1: Waiting for current job to complete..."
echo "================================================"

# Extract current job name from file
current_job_name=$(grep -A 1 "name:" "${CURRENT_JOB_FILE}" | grep -v "metadata:" | grep -v "^--$" | head -1 | awk '{print $2}')

if [ -z "${current_job_name}" ]; then
    echo "[$(get_timestamp)] ERROR: Could not extract job name from ${CURRENT_JOB_FILE}"
    exit 1
fi

echo "[$(get_timestamp)] Waiting for job to complete: ${current_job_name}"

if ${DRY_RUN}; then
    print_dry_run "Would check job status every 60 seconds for completion"
    print_dry_run "Would timeout after 10 hours"
    print_dry_run "Current job would complete successfully"
else
    # Wait for current job to complete
    start_wait=$(date +%s)
    timeout=36000  # 10 hours timeout
    check_interval=60  # Check every 60 seconds
    elapsed=0

    while [ ${elapsed} -lt ${timeout} ]; do
        # Check if job is complete (with error handling for VPN disconnections)
        if kubectl get job "${current_job_name}" -n ${NAMESPACE} -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null | grep -q "True"; then
            echo "[$(get_timestamp)] Current job completed successfully!"
            break
        fi
        
        # Check if job failed (with error handling for VPN disconnections)
        if kubectl get job "${current_job_name}" -n ${NAMESPACE} -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null | grep -q "True"; then
            echo "[$(get_timestamp)] Current job failed!"
            break
        fi
        
        # Show status periodically
        if [ $((elapsed % 300)) -eq 0 ]; then  # Every 5 minutes
            echo "[$(get_timestamp)] Current job still running..."
            kubectl get job "${current_job_name}" -n ${NAMESPACE}
            echo ""
        fi
        
        # Sleep for check interval
        sleep ${check_interval}
        elapsed=$((elapsed + check_interval))
    done

    # Check if job completed within timeout
    if [ ${elapsed} -ge ${timeout} ]; then
        echo ""
        echo "[$(get_timestamp)] WARNING: Current job did not complete within ${timeout} seconds"
        job_status=$(kubectl get job "${current_job_name}" -n ${NAMESPACE} -o jsonpath='{.status.conditions[0].type}' 2>/dev/null)
        echo "[$(get_timestamp)] Job status: ${job_status}"
    fi
fi

# Step 2: Restart dependent deployments
echo ""
echo "Step 2: Restarting dependent deployments..."
echo "=========================================="

restart_deployments
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to restart deployments"
    exit 1
fi

# Step 3: Deploy and monitor next job
echo ""
echo "Step 3: Deploying and monitoring next job..."
echo "=========================================="

deploy_and_monitor_job "${NEXT_JOB_FILE}"
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to deploy and monitor next job"
    exit 1
fi

echo ""
echo "=========================================="
echo "Evaluation sequence completed!"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
