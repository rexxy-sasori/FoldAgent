#!/bin/bash
set -e

NAMESPACE="liuyunxin"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="${SCRIPT_DIR}"
JUDGE_MODEL_DIR="kimi-2-thinking"
VERSION_SUFFIX=""

# Paths to dependent deployments
SGLANG_DEPLOYMENT="/Users/rexsasori/FoldAgent/deployment/sglang/sglang-deployment.yaml"
SEARCH_SERVER_DEPLOYMENT="/Users/rexsasori/FoldAgent/deployment/search-server-deployment.yaml"

# Paths to the new joint evaluation files
JOINT_EVAL_FILES=(
    "${DEPLOYMENT_DIR}/${JUDGE_MODEL_DIR}/eval-job-search-all.yaml"
    "${DEPLOYMENT_DIR}/${JUDGE_MODEL_DIR}/eval-job-search_branch-all.yaml"
)

echo "=========================================="
echo "FoldAgent Joint Evaluation Runner"
echo "=========================================="
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Function to get current timestamp
get_timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

check_deployments_ready() {
    # Check if sglang deployment is ready
    echo "[$(get_timestamp)] Checking sglang deployment status..."
    if ! kubectl rollout status deployment sglang -n ${NAMESPACE} --timeout=120s > /dev/null 2>&1; then
        echo "[$(get_timestamp)] ERROR: sglang deployment not ready"
        return 1
    fi
    
    # Check if sglang pods are ready
    echo "[$(get_timestamp)] Verifying sglang pods are ready..."
    if ! kubectl wait --for=condition=ready pod -l app=sglang -n ${NAMESPACE} --timeout=60s > /dev/null 2>&1; then
        echo "[$(get_timestamp)] ERROR: sglang pods not ready within timeout"
        return 1
    fi
    
    # Check if search-server deployment is ready
    echo "[$(get_timestamp)] Checking search-server deployment status..."
    if ! kubectl rollout status deployment search-server -n ${NAMESPACE} --timeout=120s > /dev/null 2>&1; then
        echo "[$(get_timestamp)] ERROR: search-server deployment not ready"
        return 1
    fi
    
    # Check if search-server pods are ready
    echo "[$(get_timestamp)] Verifying search-server pods are ready..."
    if ! kubectl wait --for=condition=ready pod -l app=search-server -n ${NAMESPACE} --timeout=60s > /dev/null 2>&1; then
        echo "[$(get_timestamp)] ERROR: search-server pods not ready within timeout"
        return 1
    fi
    
    echo "[$(get_timestamp)] All dependent deployments are ready!"
    return 0
}

restart_deployments() {
    local start_time=$(get_timestamp)
    echo ""
    echo "Restarting dependent deployments..."
    echo "-----------------------------------"
    echo "Start Time: ${start_time}"
    
    # Restart sglang deployment
    echo "[$(get_timestamp)] Restarting sglang deployment..."
    if [ -f "${SGLANG_DEPLOYMENT}" ]; then
        kubectl delete -f "${SGLANG_DEPLOYMENT}" --ignore-not-found=true -n ${NAMESPACE}
        kubectl apply -f "${SGLANG_DEPLOYMENT}" -n ${NAMESPACE}
    else
        echo "[$(get_timestamp)] WARNING: sglang deployment file not found: ${SGLANG_DEPLOYMENT}"
        echo "[$(get_timestamp)] Attempting to restart existing deployment..."
        kubectl rollout restart deployment sglang -n ${NAMESPACE}
    fi
    
    # Restart search-server deployment
    echo "[$(get_timestamp)] Restarting search-server deployment..."
    if [ -f "${SEARCH_SERVER_DEPLOYMENT}" ]; then
        kubectl delete -f "${SEARCH_SERVER_DEPLOYMENT}" --ignore-not-found=true -n ${NAMESPACE}
        kubectl apply -f "${SEARCH_SERVER_DEPLOYMENT}" -n ${NAMESPACE}
    else
        echo "[$(get_timestamp)] WARNING: search-server deployment file not found: ${SEARCH_SERVER_DEPLOYMENT}"
        echo "[$(get_timestamp)] Attempting to restart existing deployment..."
        kubectl rollout restart deployment search-server -n ${NAMESPACE}
    fi
    
    echo "[$(get_timestamp)] Waiting for deployments to be ready..."
    echo "-----------------------------------"
    
    # Wait for sglang deployment to be ready
    echo "[$(get_timestamp)] Checking sglang deployment status..."
    kubectl rollout status deployment sglang -n ${NAMESPACE} --timeout=600s
    
    # Verify sglang pods are ready using kubectl wait
    echo "[$(get_timestamp)] Verifying sglang pods are ready..."
    if kubectl wait --for=condition=ready pod -l app=sglang -n ${NAMESPACE} --timeout=300s; then
        echo "[$(get_timestamp)] sglang deployment is ready!"
    else
        echo "[$(get_timestamp)] ERROR: sglang pods not ready within timeout"
        echo "[$(get_timestamp)] Evaluation cannot proceed without sglang deployment"
        return 1
    fi
    
    # Wait for search-server deployment to be ready
    echo "[$(get_timestamp)] Checking search-server deployment status..."
    kubectl rollout status deployment search-server -n ${NAMESPACE} --timeout=600s
    
    # Verify search-server pods are ready using kubectl wait
    echo "[$(get_timestamp)] Verifying search-server pods are ready..."
    if kubectl wait --for=condition=ready pod -l app=search-server -n ${NAMESPACE} --timeout=300s; then
        echo "[$(get_timestamp)] search-server deployment is ready!"
    else
        echo "[$(get_timestamp)] ERROR: search-server pods not ready within timeout"
        echo "[$(get_timestamp)] Evaluation cannot proceed without search-server deployment"
        return 1
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

run_joint_evaluation() {
    local job_file=$1
    local start_time=$(get_timestamp)
    
    # Extract workflow from filename
    local workflow=""
    if [[ "${job_file}" == *"search-all"* ]]; then
        workflow="search"
    elif [[ "${job_file}" == *"search_branch-all"* ]]; then
        workflow="search_branch"
    fi
    
    echo ""
    echo "=========================================="
    echo "Running joint evaluation:"
    echo "  Workflow: ${workflow}"
    echo "  Job: ${job_file}"
    echo "  Start Time: ${start_time}"
    echo "  Judge Model: ${JUDGE_MODEL_DIR}"
    if [ -n "${VERSION_SUFFIX}" ]; then
        echo "  Version: ${VERSION_SUFFIX}"
    fi
    echo "=========================================="
    
    if [ ! -f "${job_file}" ]; then
        echo "[$(get_timestamp)] ERROR: Job file not found: ${job_file}"
        return 1
    fi
    
    echo "[$(get_timestamp)] Deleting existing job (if any)..."
    kubectl delete -f "${job_file}" --ignore-not-found=true -n ${NAMESPACE}
    
    echo "[$(get_timestamp)] Creating job..."
    kubectl create -f "${job_file}" -n ${NAMESPACE}
    
    # Get the job name
    local label_selector="app=foldagent-eval-${workflow//_/-}-all-kimi-k2-thinking"
    
    # Append version suffix if specified
    if [ -n "${VERSION_SUFFIX}" ]; then
        label_selector="${label_selector}-${VERSION_SUFFIX}"
    fi
    
    local job_name=$(kubectl get jobs -l "${label_selector}" -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}')
    
    echo "[$(get_timestamp)] Monitoring job progress..."
    echo "[$(get_timestamp)] Waiting for job to complete..."

    # Wait for pod to be ready before monitoring
    echo "[$(get_timestamp)] Waiting for pod to be ready..."
    local pod_ready=false
    local pod_wait_timeout=600  # 10 minutes to wait for pod to be ready
    local pod_wait_elapsed=0
    local pod_check_interval=10  # Check every 10 seconds

    while [ ${pod_wait_elapsed} -lt ${pod_wait_timeout} ]; do
        local pod_name=$(kubectl get pods -l job-name="${job_name}" -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
        
        if [ -n "${pod_name}" ]; then
            local pod_phase=$(kubectl get pod "${pod_name}" -n ${NAMESPACE} -o jsonpath='{.status.phase}' 2>/dev/null)
            
            if [ "${pod_phase}" = "Running" ]; then
                pod_ready=true
                echo "[$(get_timestamp)] Pod ${pod_name} is ready"
                break
            elif [ "${pod_phase}" = "Failed" ]; then
                echo "[$(get_timestamp)] ERROR: Pod ${pod_name} failed to start"
                pod_ready=false
                break
            fi
        fi
        
        sleep ${pod_check_interval}
        pod_wait_elapsed=$((pod_wait_elapsed + pod_check_interval))
    done

    if [ "${pod_ready}" = "false" ]; then
        echo "[$(get_timestamp)] WARNING: Pod did not become ready within ${pod_wait_timeout} seconds"
        echo "[$(get_timestamp)] Proceeding with job monitoring anyway..."
    fi

    # Monitor job logs while waiting
    local start_wait=$(date +%s)
    local timeout=36000  # 10 hours timeout
    local check_interval=600  # Check every 10 minutes
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
    echo "Evaluation ${job_status} for ${workflow}"
    echo "=========================================="
    echo "Start Time: ${start_time}"
    echo "End Time: ${end_time}"
    echo "Job Status: ${job_status}"
    echo "Results are saved in the container's /root/results directory"
    
    # Get the pod name from the job
    local pod_name=$(kubectl get pods -l job-name="${job_name}" -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    
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

run_all_joint_evaluations() {
    local succeeded_configs=()
    local failed_configs=()
    
    for job_file in "${JOINT_EVAL_FILES[@]}"; do
        restart_deployments
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to restart deployments for ${job_file}"
            failed_configs+=("${job_file}")
            continue
        fi
        run_joint_evaluation "${job_file}"
        if [ $? -eq 0 ]; then
            succeeded_configs+=($(basename "${job_file}"))
        else
            failed_configs+=($(basename "${job_file}"))
        fi
    done
    
    # Display summary
    echo ""
    echo "=========================================="
    echo "Evaluation Summary"
    echo "=========================================="
    echo "Total evaluations: $((${#succeeded_configs[@]} + ${#failed_configs[@]}))"
    echo "Succeeded: ${#succeeded_configs[@]}"
    echo "Failed: ${#failed_configs[@]}"
    echo ""
    
    if [ ${#succeeded_configs[@]} -gt 0 ]; then
        echo "Succeeded configurations:"
        for config in "${succeeded_configs[@]}"; do
            echo "  ✓ ${config}"
        done
    fi
    
    if [ ${#failed_configs[@]} -gt 0 ]; then
        echo ""
        echo "Failed configurations:"
        for config in "${failed_configs[@]}"; do
            echo "  ✗ ${config}"
        done
    fi
    
    echo "=========================================="
    
    # Return error if any failed
    if [ ${#failed_configs[@]} -gt 0 ]; then
        return 1
    fi
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -v, --version SUFFIX    Version suffix for job names (e.g., v0-74)"
    echo "  -a, --all               Run all joint evaluations (default)"
    echo "  -r, --restart-only      Only restart deployments, don't run evaluation"
    echo "  -n, --dry-run           Dry run (show what would be executed)"
    echo "  -h, --help              Show help message"
    echo ""
    echo "Examples:"
    echo "  $0"
    echo "  $0 --all"
    echo "  $0 --all --version v0-74"
    echo "  $0 --restart-only"
    echo "  $0 --dry-run"
}

main() {
    local run_all=true
    local restart_only=false
    local dry_run=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -a|--all)
                run_all=true
                shift
                ;;
            -r|--restart-only)
                restart_only=true
                shift
                ;;
            -v|--version)
                VERSION_SUFFIX="$2"
                shift 2
                ;;
            -n|--dry-run)
                dry_run=true
                shift
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
    
    if ${dry_run}; then
        echo ""
        echo "=========================================="
        echo "Dry Run Mode"
        echo "=========================================="
        echo "This is a dry run - no actual changes will be made."
        echo ""
        
        if ${restart_only}; then
            echo "Would restart deployments:"
            echo "=========================================="
            echo "Commands that would be executed:"
            echo ""
            echo "# Restart sglang deployment"
            if [ -f "${SGLANG_DEPLOYMENT}" ]; then
                echo "kubectl delete -f \"${SGLANG_DEPLOYMENT}\" --ignore-not-found=true -n ${NAMESPACE}"
                echo "kubectl apply -f \"${SGLANG_DEPLOYMENT}\" -n ${NAMESPACE}"
            else
                echo "kubectl rollout restart deployment sglang -n ${NAMESPACE}"
            fi
            echo ""
            echo "# Restart search-server deployment"
            if [ -f "${SEARCH_SERVER_DEPLOYMENT}" ]; then
                echo "kubectl delete -f \"${SEARCH_SERVER_DEPLOYMENT}\" --ignore-not-found=true -n ${NAMESPACE}"
                echo "kubectl apply -f \"${SEARCH_SERVER_DEPLOYMENT}\" -n ${NAMESPACE}"
            else
                echo "kubectl rollout restart deployment search-server -n ${NAMESPACE}"
            fi
            echo ""
            echo "# Wait for deployments to be ready"
            echo "kubectl rollout status deployment sglang -n ${NAMESPACE} --timeout=600s"
            echo "kubectl rollout status deployment search-server -n ${NAMESPACE} --timeout=600s"
        else
            echo "Would run joint evaluations:"
            echo "=========================================="
            echo "Total: 2 evaluations"
            echo "Judge Model: ${JUDGE_MODEL_DIR}"
            if [ -n "${VERSION_SUFFIX}" ]; then
                echo "Version Suffix: ${VERSION_SUFFIX}"
            fi
            echo ""
            echo "For each evaluation, would execute:"
            echo "1. Restart deployments (as shown above)"
            echo "2. Run evaluation with commands:"
            echo ""
            
            for job_file in "${JOINT_EVAL_FILES[@]}"; do
                echo "# $(basename "${job_file}")"
                echo "kubectl delete -f \"${job_file}\" --ignore-not-found=true -n ${NAMESPACE}"
                echo "kubectl create -f \"${job_file}\" -n ${NAMESPACE}"
                echo "# Monitor job until completion (10 hours timeout)"
                echo "# Show logs while waiting"
                
                local workflow=""
                if [[ "${job_file}" == *"search-all"* ]]; then
                    workflow="search"
                elif [[ "${job_file}" == *"search_branch-all"* ]]; then
                    workflow="search_branch"
                fi
                local label_selector="app=foldagent-eval-${workflow//_/-}-all-kimi-k2-thinking"
                if [ -n "${VERSION_SUFFIX}" ]; then
                    label_selector="${label_selector}-${VERSION_SUFFIX}"
                fi
                echo "kubectl wait --for=condition=complete job \$(kubectl get jobs -l ${label_selector} -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') --timeout=36000s -n ${NAMESPACE}"
                echo "kubectl logs \$(kubectl get pods -l job-name=\$(kubectl get jobs -l ${label_selector} -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE} --tail=50"
                echo ""
            done
        fi
        
        echo ""
        echo "Dry run completed successfully!"
        exit 0
    fi
    
    if ${restart_only}; then
        restart_deployments
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to restart deployments"
            exit 1
        fi
        echo "=========================================="
        echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="
        exit 0
    fi
    
    if ${run_all}; then
        local all_start=$(get_timestamp)
        echo "=========================================="
        echo "Running all joint evaluations"
        echo "Start Time: ${all_start}"
        echo "=========================================="
        run_all_joint_evaluations
        local result=$?
        local all_end=$(get_timestamp)
        echo "=========================================="
        if [ ${result} -eq 0 ]; then
            echo "All joint evaluations completed successfully"
        else
            echo "Some joint evaluations failed (see summary above)"
        fi
        echo "Start Time: ${all_start}"
        echo "End Time: ${all_end}"
        echo "=========================================="
        exit ${result}
    fi
}

main "$@"