#!/bin/bash
set -e

NAMESPACE="liuyunxin"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="${SCRIPT_DIR}"
JUDGE_MODEL_DIR=""
VERSION_SUFFIX=""

# Paths to dependent deployments
SGLANG_DEPLOYMENT="/Users/rexsasori/FoldAgent/deployment/sglang/sglang-deployment.yaml"
SEARCH_SERVER_DEPLOYMENT="/Users/rexsasori/FoldAgent/deployment/search-server-deployment.yaml"
QWEN_JUDGER_DEPLOYMENT="/Users/rexsasori/FoldAgent/deployment/sglang/sglang-qwen3-32b-judger.yaml"

echo "=========================================="
echo "FoldAgent Evaluation Runner"
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
    
    # Check if sglang-qwen-judger deployment is ready (only if using qwen-32B judge model)
    if [ "${JUDGE_MODEL_DIR}" = "qwen-32B" ]; then
        echo "[$(get_timestamp)] Checking sglang-qwen-judger deployment status..."
        if ! kubectl rollout status deployment sglang-qwen-judger -n ${NAMESPACE} --timeout=120s > /dev/null 2>&1; then
            echo "[$(get_timestamp)] ERROR: sglang-qwen-judger deployment not ready"
            return 1
        fi
        
        # Check if sglang-qwen-judger pods are ready
        echo "[$(get_timestamp)] Verifying sglang-qwen-judger pods are ready..."
        if ! kubectl wait --for=condition=ready pod -l app=sglang-qwen-judger -n ${NAMESPACE} --timeout=60s > /dev/null 2>&1; then
            echo "[$(get_timestamp)] ERROR: sglang-qwen-judger pods not ready within timeout"
            return 1
        fi
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
    
    # Restart sglang-qwen-judger deployment (only if using qwen-32B judge model)
    if [ "${JUDGE_MODEL_DIR}" = "qwen-32B" ]; then
        echo "[$(get_timestamp)] Restarting sglang-qwen-judger deployment..."
        if [ -f "${QWEN_JUDGER_DEPLOYMENT}" ]; then
            kubectl delete -f "${QWEN_JUDGER_DEPLOYMENT}" --ignore-not-found=true -n ${NAMESPACE}
            kubectl apply -f "${QWEN_JUDGER_DEPLOYMENT}" -n ${NAMESPACE}
        else
            echo "[$(get_timestamp)] WARNING: sglang-qwen-judger deployment file not found: ${QWEN_JUDGER_DEPLOYMENT}"
            echo "[$(get_timestamp)] Attempting to restart existing deployment..."
            kubectl rollout restart deployment sglang-qwen-judger -n ${NAMESPACE}
        fi
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
    
    # Wait for sglang-qwen-judger deployment to be ready (only if using qwen-32B judge model)
    if [ "${JUDGE_MODEL_DIR}" = "qwen-32B" ]; then
        echo "[$(get_timestamp)] Checking sglang-qwen-judger deployment status..."
        kubectl rollout status deployment sglang-qwen-judger -n ${NAMESPACE} --timeout=600s
        
        # Verify sglang-qwen-judger pods are ready using kubectl wait
        echo "[$(get_timestamp)] Verifying sglang-qwen-judger pods are ready..."
        if kubectl wait --for=condition=ready pod -l app=sglang-qwen-judger -n ${NAMESPACE} --timeout=300s; then
            echo "[$(get_timestamp)] sglang-qwen-judger deployment is ready!"
        else
            echo "[$(get_timestamp)] ERROR: sglang-qwen-judger pods not ready within timeout"
            echo "[$(get_timestamp)] Evaluation cannot proceed without sglang-qwen-judger deployment"
            return 1
        
        fi
    fi
    
    # Verify services are reachable
    echo "[$(get_timestamp)] Checking if services are reachable..."
    echo "[$(get_timestamp)] sglang service: http://sglang.liuyunxin:8080"
    echo "[$(get_timestamp)] search-server service: http://search-server.liuyunxin:8000"
    # Only show sglang-qwen-judger service if using qwen-32B judge model
    if [ "${JUDGE_MODEL_DIR}" = "qwen-32B" ]; then
        echo "[$(get_timestamp)] sglang-qwen-judger service: http://sglang-qwen-judger.liuyunxin:8080"
    fi
    
    local end_time=$(get_timestamp)
    echo ""
    echo "Deployments restarted successfully!"
    echo "-----------------------------------"
    echo "Start Time: ${start_time}"
    echo "End Time: ${end_time}"
} 

run_evaluation() {
    local difficulty=$1
    local workflow=$2
    local start_time=$(get_timestamp)
    
    local job_dir="${DEPLOYMENT_DIR}"
    if [ -n "${JUDGE_MODEL_DIR}" ]; then
        job_dir="${DEPLOYMENT_DIR}/${JUDGE_MODEL_DIR}"
    fi
    
    local job_file="${job_dir}/eval-job-${workflow}-${difficulty}.yaml"
    
    echo ""
    echo "=========================================="
    echo "Running evaluation:"
    echo "  Difficulty: ${difficulty}"
    echo "  Workflow: ${workflow}"
    echo "  Job: ${job_file}"
    echo "  Start Time: ${start_time}"
    if [ -n "${JUDGE_MODEL_DIR}" ]; then
        echo "  Judge Model: ${JUDGE_MODEL_DIR}"
    fi
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
    local label_selector="app=foldagent-eval-${workflow//_/-}-${difficulty}"
    local append_version_suffix=true
    if [ -n "${JUDGE_MODEL_DIR}" ]; then
        # For judge model directories, use a more specific selector
        # Handle different naming conventions based on judge model directory
        if [ "${JUDGE_MODEL_DIR}" = "kimi-2-thinking" ]; then
            label_selector="app=foldagent-eval-${workflow//_/-}-${difficulty}-kimi-k2-thinking"
        elif [ "${JUDGE_MODEL_DIR}" = "gpt-5" ]; then
            label_selector="app=foldagent-eval-${workflow//_/-}-${difficulty}-gpt-5-v0-81"
            append_version_suffix=false
        elif [ "${JUDGE_MODEL_DIR}" = "gpt-4" ]; then
            label_selector="app=foldagent-eval-${workflow//_/-}-${difficulty}-gpt-4-v0-81"
            append_version_suffix=false
        else
            # Default to the basic selector
            label_selector="app=foldagent-eval-${workflow//_/-}-${difficulty}"
        fi
    fi
    
    # Append version suffix if specified and not already included
    if [ -n "${VERSION_SUFFIX}" ] && [ "${append_version_suffix}" = "true" ]; then
        label_selector="${label_selector}-${VERSION_SUFFIX}"
    fi
    
    echo "[$(get_timestamp)] Looking for job with label selector: ${label_selector}"
    local job_name=$(kubectl get jobs -l "${label_selector}" -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}')
    
    if [ -z "${job_name}" ]; then
        echo "[$(get_timestamp)] ERROR: No job found with label selector: ${label_selector}"
        return 1
    fi
    
    echo "[$(get_timestamp)] Found job: ${job_name}"
    echo "[$(get_timestamp)] Monitoring job progress..."
    echo "[$(get_timestamp)] Waiting for job to complete..."

    # Wait for pod to be ready before monitoring
    echo "[$(get_timestamp)] Waiting for pod to be ready..."
    echo "[$(get_timestamp)] Looking for pod with job-name: ${job_name}"
    local pod_ready=false
    local pod_wait_timeout=600  # 10 minutes to wait for pod to be ready
    local pod_wait_elapsed=0
    local pod_check_interval=10  # Check every 10 seconds

    while [ ${pod_wait_elapsed} -lt ${pod_wait_timeout} ]; do
        local pod_name=$(kubectl get pods -l job-name="${job_name}" -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
        
        if [ -n "${pod_name}" ]; then
            echo "[$(get_timestamp)] Found pod: ${pod_name}"
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
        echo "[$(get_timestamp)] Was looking for pod with job-name: ${job_name}"
        echo "[$(get_timestamp)] Proceeding with job monitoring anyway..."
    fi

    # Monitor job logs while waiting
    local start_wait=$(date +%s)
    local timeout=36000  # 10 hours timeout
    local check_interval=60  # Check every 30 seconds
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
    echo "Evaluation ${job_status} for ${difficulty}/${workflow}"
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

run_all_evaluations() {
    local difficulties=("easy" "medium" "hard")
    local workflows=("search" "search_branch")
    local succeeded_configs=()
    local failed_configs=()
    
    for workflow in "${workflows[@]}"; do
        for difficulty in "${difficulties[@]}"; do
            restart_deployments
            if [ $? -ne 0 ]; then
                echo "ERROR: Failed to restart deployments for ${workflow}/${difficulty}"
                failed_configs+=("${workflow}/${difficulty}")
                continue
            fi
            run_evaluation "${difficulty}" "${workflow}"
            if [ $? -eq 0 ]; then
                succeeded_configs+=("${workflow}/${difficulty}")
            else
                failed_configs+=("${workflow}/${difficulty}")
            fi
        done
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
    echo "  -d, --difficulty DIFF   Run specific difficulty (easy|medium|hard)"
    echo "  -w, --workflow WORKFLOW  Run specific workflow (search|search_branch)"
    echo "  -j, --judge-model DIR   Judge model directory (e.g., gpt-5-judge, kimi-2-thinking)"
    echo "  -v, --version SUFFIX    Version suffix for job names (e.g., v0-74)"
    echo "  -a, --all               Run all combinations (3 difficulties × 2 workflows)"
    echo "  -r, --restart-only      Only restart deployments, don't run evaluation"
    echo "  -n, --dry-run           Dry run (show what would be executed)"
    echo "  -h, --help              Show help message"
    echo ""
    echo "Examples:"
    echo "  $0 --difficulty easy --workflow search"
    echo "  $0 --difficulty easy --workflow search --judge-model kimi-2-thinking"
    echo "  $0 --all"
    echo "  $0 --all --judge-model gpt-5-judge"
    echo "  $0 --all --judge-model kimi-2-thinking --version v0-74"
    echo "  $0 --restart-only"
    echo "  $0 --difficulty easy --workflow search --dry-run"
}

main() {
    local difficulty=""
    local workflow=""
    local run_all=false
    local restart_only=false
    local dry_run=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--difficulty)
                difficulty="$2"
                shift 2
                ;;
            -w|--workflow)
                workflow="$2"
                shift 2
                ;;
            -j|--judge-model)
                JUDGE_MODEL_DIR="$2"
                shift 2
                ;;
            -v|--version)
                VERSION_SUFFIX="$2"
                shift 2
                ;;
            -a|--all)
                run_all=true
                shift
                ;;
            -r|--restart-only)
                restart_only=true
                shift
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
            # Only include sglang-qwen-judger deployment commands if using qwen-32B judge model
            if [ "${JUDGE_MODEL_DIR}" = "qwen-32B" ]; then
                echo "# Restart sglang-qwen-judger deployment"
                if [ -f "${QWEN_JUDGER_DEPLOYMENT}" ]; then
                    echo "kubectl delete -f \"${QWEN_JUDGER_DEPLOYMENT}\" --ignore-not-found=true -n ${NAMESPACE}"
                    echo "kubectl apply -f \"${QWEN_JUDGER_DEPLOYMENT}\" -n ${NAMESPACE}"
                else
                    echo "kubectl rollout restart deployment sglang-qwen-judger -n ${NAMESPACE}"
                fi
                echo ""
            fi
            echo "# Wait for deployments to be ready"
            echo "kubectl rollout status deployment sglang -n ${NAMESPACE} --timeout=600s"
            echo "kubectl rollout status deployment search-server -n ${NAMESPACE} --timeout=600s"
            # Only include sglang-qwen-judger rollout status if using qwen-32B judge model
            if [ "${JUDGE_MODEL_DIR}" = "qwen-32B" ]; then
                echo "kubectl rollout status deployment sglang-qwen-judger -n ${NAMESPACE} --timeout=600s"
            fi
        elif ${run_all}; then
            local difficulties=("easy" "medium" "hard")
            local workflows=("search" "search_branch")
            local job_dir="${DEPLOYMENT_DIR}"
            if [ -n "${JUDGE_MODEL_DIR}" ]; then
                job_dir="${DEPLOYMENT_DIR}/${JUDGE_MODEL_DIR}"
            fi
            
            echo "Would run all evaluations:"
            echo "=========================================="
            echo "Difficulties: easy, medium, hard"
            echo "Workflows: search, search_branch"
            echo "Total: 6 evaluations"
            if [ -n "${JUDGE_MODEL_DIR}" ]; then
                echo "Judge Model: ${JUDGE_MODEL_DIR}"
            fi
            if [ -n "${VERSION_SUFFIX}" ]; then
                echo "Version Suffix: ${VERSION_SUFFIX}"
            fi
            echo ""
            echo "For each evaluation, would execute:"
            echo "1. Restart deployments (as shown above)"
            echo "2. Run evaluation with commands:"
            echo ""
            
            for workflow in "${workflows[@]}"; do
                for difficulty in "${difficulties[@]}"; do
                    local job_file="${job_dir}/eval-job-${workflow}-${difficulty}.yaml"
                    echo "# ${workflow} - ${difficulty}"
                    echo "kubectl delete -f \"${job_file}\" --ignore-not-found=true -n ${NAMESPACE}"
                    echo "kubectl create -f \"${job_file}\" -n ${NAMESPACE}"
                    echo "# Monitor job until completion (10 hours timeout)"
                    echo "# Show logs while waiting"
                    local label_selector="app=foldagent-eval-${workflow//_/-}-${difficulty}"
                    if [ -n "${JUDGE_MODEL_DIR}" ] && [ "${JUDGE_MODEL_DIR}" = "kimi-2-thinking" ]; then
                        label_selector="app=foldagent-eval-${workflow//_/-}-${difficulty}-kimi-k2-thinking"
                    fi
                    if [ -n "${VERSION_SUFFIX}" ]; then
                        label_selector="${label_selector}-${VERSION_SUFFIX}"
                    fi
                    echo "kubectl wait --for=condition=complete job \$(kubectl get jobs -l ${label_selector} -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') --timeout=36000s -n ${NAMESPACE}"
                    echo "kubectl get pods -l job-name=\$(kubectl get jobs -l ${label_selector} -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE}"
                    echo "kubectl logs \$(kubectl get pods -l job-name=\$(kubectl get jobs -l ${label_selector} -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE} --tail=50"
                    echo ""
                done
            done
        elif [ -n "${workflow}" ] && [ -z "${difficulty}" ]; then
            local difficulties=("easy" "medium" "hard")
            local job_dir="${DEPLOYMENT_DIR}"
            if [ -n "${JUDGE_MODEL_DIR}" ]; then
                job_dir="${DEPLOYMENT_DIR}/${JUDGE_MODEL_DIR}"
            fi
            
            echo "Would run all difficulties for workflow ${workflow}:"
            echo "=========================================="
            echo "Difficulties: easy, medium, hard"
            echo "Workflow: ${workflow}"
            echo "Total: 3 evaluations"
            if [ -n "${JUDGE_MODEL_DIR}" ]; then
                echo "Judge Model: ${JUDGE_MODEL_DIR}"
            fi
            echo ""
            echo "For each evaluation, would execute:"
            echo "1. Restart deployments (as shown above)"
            echo "2. Run evaluation with commands:"
            echo ""
            
            for difficulty in "${difficulties[@]}"; do
                local job_file="${job_dir}/eval-job-${workflow}-${difficulty}.yaml"
                echo "# ${workflow} - ${difficulty}"
                echo "kubectl delete -f \"${job_file}\" --ignore-not-found=true -n ${NAMESPACE}"
                echo "kubectl create -f \"${job_file}\" -n ${NAMESPACE}"
                echo "# Monitor job until completion (10 hours timeout)"
                echo "# Show logs while waiting"
                if [ -n "${JUDGE_MODEL_DIR}" ] && [ "${JUDGE_MODEL_DIR}" = "kimi-2-thinking" ]; then
                    echo "kubectl wait --for=condition=complete job \$(kubectl get jobs -l app=foldagent-eval-${workflow//_/-}-${difficulty}-kimi-k2-thinking -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') --timeout=36000s -n ${NAMESPACE}"
                    echo "kubectl get pods -l job-name=\$(kubectl get jobs -l app=foldagent-eval-${workflow//_/-}-${difficulty}-kimi-k2-thinking -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE}"
                    echo "kubectl logs \$(kubectl get pods -l job-name=\$(kubectl get jobs -l app=foldagent-eval-${workflow//_/-}-${difficulty}-kimi-k2-thinking -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE} --tail=50"
                elif [ -n "${JUDGE_MODEL_DIR}" ] && [ "${JUDGE_MODEL_DIR}" = "gpt-5" ]; then
                    echo "kubectl wait --for=condition=complete job \$(kubectl get jobs -l app=foldagent-eval-${workflow//_/-}-${difficulty}-gpt-5-v0-81 -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') --timeout=36000s -n ${NAMESPACE}"
                    echo "kubectl get pods -l job-name=\$(kubectl get jobs -l app=foldagent-eval-${workflow//_/-}-${difficulty}-gpt-5-v0-81 -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE}"
                    echo "kubectl logs \$(kubectl get pods -l job-name=\$(kubectl get jobs -l app=foldagent-eval-${workflow//_/-}-${difficulty}-gpt-5-v0-81 -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE} --tail=50"
                elif [ -n "${JUDGE_MODEL_DIR}" ] && [ "${JUDGE_MODEL_DIR}" = "gpt-4" ]; then
                    echo "kubectl wait --for=condition=complete job \$(kubectl get jobs -l app=foldagent-eval-${workflow//_/-}-${difficulty}-gpt-4-v0-81 -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') --timeout=36000s -n ${NAMESPACE}"
                    echo "kubectl get pods -l job-name=\$(kubectl get jobs -l app=foldagent-eval-${workflow//_/-}-${difficulty}-gpt-4-v0-81 -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE}"
                    echo "kubectl logs \$(kubectl get pods -l job-name=\$(kubectl get jobs -l app=foldagent-eval-${workflow//_/-}-${difficulty}-gpt-4-v0-81 -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE} --tail=50"
                else
                    echo "kubectl wait --for=condition=complete job \$(kubectl get jobs -l app=foldagent-eval-${workflow//_/-}-${difficulty} -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') --timeout=36000s -n ${NAMESPACE}"
                    echo "kubectl get pods -l job-name=\$(kubectl get jobs -l app=foldagent-eval-${workflow//_/-}-${difficulty} -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE}"
                    echo "kubectl logs \$(kubectl get pods -l job-name=\$(kubectl get jobs -l app=foldagent-eval-${workflow//_/-}-${difficulty} -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE} --tail=50"
                fi
                echo ""
            done
        elif [ -n "${difficulty}" ] && [ -z "${workflow}" ]; then
            local workflows=("search" "search_branch")
            local job_dir="${DEPLOYMENT_DIR}"
            if [ -n "${JUDGE_MODEL_DIR}" ]; then
                job_dir="${DEPLOYMENT_DIR}/${JUDGE_MODEL_DIR}"
            fi
            
            echo "Would run all workflows for difficulty ${difficulty}:"
            echo "=========================================="
            echo "Difficulty: ${difficulty}"
            echo "Workflows: search, search_branch"
            echo "Total: 2 evaluations"
            if [ -n "${JUDGE_MODEL_DIR}" ]; then
                echo "Judge Model: ${JUDGE_MODEL_DIR}"
            fi
            if [ -n "${VERSION_SUFFIX}" ]; then
                echo "Version Suffix: ${VERSION_SUFFIX}"
            fi
            echo ""
            echo "For each evaluation, would execute:"
            echo "1. Restart deployments (as shown above)"
            echo "2. Run evaluation with commands:"
            echo ""
            
            for workflow in "${workflows[@]}"; do
                local job_file="${job_dir}/eval-job-${workflow}-${difficulty}.yaml"
                echo "# ${workflow} - ${difficulty}"
                echo "kubectl delete -f \"${job_file}\" --ignore-not-found=true -n ${NAMESPACE}"
                echo "kubectl create -f \"${job_file}\" -n ${NAMESPACE}"
                echo "# Monitor job until completion (10 hours timeout)"
                echo "# Show logs while waiting"
                if [ -n "${JUDGE_MODEL_DIR}" ] && [ "${JUDGE_MODEL_DIR}" = "kimi-2-thinking" ]; then
                    echo "kubectl wait --for=condition=complete job \$(kubectl get jobs -l app=foldagent-eval-${workflow//_/-}-${difficulty}-kimi-k2-thinking -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') --timeout=36000s -n ${NAMESPACE}"
                    echo "kubectl get pods -l job-name=\$(kubectl get jobs -l app=foldagent-eval-${workflow//_/-}-${difficulty}-kimi-k2-thinking -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE}"
                    echo "kubectl logs \$(kubectl get pods -l job-name=\$(kubectl get jobs -l app=foldagent-eval-${workflow//_/-}-${difficulty}-kimi-k2-thinking -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE} --tail=50"
                else
                    echo "kubectl wait --for=condition=complete job \$(kubectl get jobs -l app=foldagent-eval-${workflow//_/-}-${difficulty} -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') --timeout=36000s -n ${NAMESPACE}"
                    echo "kubectl get pods -l job-name=\$(kubectl get jobs -l app=foldagent-eval-${workflow//_/-}-${difficulty} -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE}"
                    echo "kubectl logs \$(kubectl get pods -l job-name=\$(kubectl get jobs -l app=foldagent-eval-${workflow//_/-}-${difficulty} -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE} --tail=50"
                fi
                echo ""
            done
        else
            if [ -n "${difficulty}" ] && [ -n "${workflow}" ]; then
                local job_dir="${DEPLOYMENT_DIR}"
                if [ -n "${JUDGE_MODEL_DIR}" ]; then
                    job_dir="${DEPLOYMENT_DIR}/${JUDGE_MODEL_DIR}"
                fi
                
                local job_file="${job_dir}/eval-job-${workflow}-${difficulty}.yaml"
                echo "Would run evaluation:"
                echo "=========================================="
                echo "Difficulty: ${difficulty}"
                echo "Workflow: ${workflow}"
                echo "Job: ${job_file}"
                if [ -n "${JUDGE_MODEL_DIR}" ]; then
                    echo "Judge Model: ${JUDGE_MODEL_DIR}"
                fi
                if [ -n "${VERSION_SUFFIX}" ]; then
                    echo "Version Suffix: ${VERSION_SUFFIX}"
                fi
                echo ""
                echo "Commands that would be executed:"
                echo ""
                echo "# 1. Restart dependent deployments"
                if [ -f "${SGLANG_DEPLOYMENT}" ]; then
                    echo "kubectl delete -f \"${SGLANG_DEPLOYMENT}\" --ignore-not-found=true -n ${NAMESPACE}"
                    echo "kubectl apply -f \"${SGLANG_DEPLOYMENT}\" -n ${NAMESPACE}"
                else
                    echo "kubectl rollout restart deployment sglang -n ${NAMESPACE}"
                fi
                if [ -f "${SEARCH_SERVER_DEPLOYMENT}" ]; then
                    echo "kubectl delete -f \"${SEARCH_SERVER_DEPLOYMENT}\" --ignore-not-found=true -n ${NAMESPACE}"
                    echo "kubectl apply -f \"${SEARCH_SERVER_DEPLOYMENT}\" -n ${NAMESPACE}"
                else
                    echo "kubectl rollout restart deployment search-server -n ${NAMESPACE}"
                fi
                # Only include sglang-qwen-judger deployment commands if using qwen-32B judge model
                if [ "${JUDGE_MODEL_DIR}" = "qwen-32B" ]; then
                    if [ -f "${QWEN_JUDGER_DEPLOYMENT}" ]; then
                        echo "kubectl delete -f \"${QWEN_JUDGER_DEPLOYMENT}\" --ignore-not-found=true -n ${NAMESPACE}"
                        echo "kubectl apply -f \"${QWEN_JUDGER_DEPLOYMENT}\" -n ${NAMESPACE}"
                    else
                        echo "kubectl rollout restart deployment sglang-qwen-judger -n ${NAMESPACE}"
                    fi
                fi
                echo "kubectl rollout status deployment sglang -n ${NAMESPACE} --timeout=600s"
                echo "kubectl rollout status deployment search-server -n ${NAMESPACE} --timeout=600s"
                # Only include sglang-qwen-judger rollout status if using qwen-32B judge model
                if [ "${JUDGE_MODEL_DIR}" = "qwen-32B" ]; then
                    echo "kubectl rollout status deployment sglang-qwen-judger -n ${NAMESPACE} --timeout=600s"
                fi
                echo ""
                echo "# 2. Run evaluation"
                echo "kubectl delete -f \"${job_file}\" --ignore-not-found=true -n ${NAMESPACE}"
                echo "kubectl create -f \"${job_file}\" -n ${NAMESPACE}"
                echo "# Monitor job until completion (10 hours timeout)"
                echo "# Show logs while waiting"
                local label_selector="app=foldagent-eval-${workflow//_/-}-${difficulty}"
                if [ -n "${JUDGE_MODEL_DIR}" ] && [ "${JUDGE_MODEL_DIR}" = "kimi-2-thinking" ]; then
                    label_selector="app=foldagent-eval-${workflow//_/-}-${difficulty}-kimi-k2-thinking"
                fi
                if [ -n "${VERSION_SUFFIX}" ]; then
                    label_selector="${label_selector}-${VERSION_SUFFIX}"
                fi
                echo "kubectl wait --for=condition=complete job \$(kubectl get jobs -l ${label_selector} -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') --timeout=36000s -n ${NAMESPACE}"
                echo "kubectl logs \$(kubectl get pods -l job-name=\$(kubectl get jobs -l ${label_selector} -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE} --tail=50"
            else
                echo "ERROR: Both --difficulty and --workflow must be specified, or use --all"
                print_usage
                exit 1
            fi
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
        echo "Running all evaluations"
        echo "Start Time: ${all_start}"
        echo "=========================================="
        run_all_evaluations
        local result=$?
        local all_end=$(get_timestamp)
        echo "=========================================="
        if [ ${result} -eq 0 ]; then
            echo "All evaluations completed successfully"
        else
            echo "Some evaluations failed (see summary above)"
        fi
        echo "Start Time: ${all_start}"
        echo "End Time: ${all_end}"
        echo "=========================================="
        exit ${result}
    elif [ -n "${workflow}" ] && [ -z "${difficulty}" ]; then
        local all_start=$(get_timestamp)
        local difficulties=("easy" "medium" "hard")
        local succeeded_configs=()
        local failed_configs=()
        
        echo "=========================================="
        echo "Running all difficulties for workflow ${workflow}"
        echo "Start Time: ${all_start}"
        echo "=========================================="
        
        for difficulty in "${difficulties[@]}"; do
            restart_deployments
            if [ $? -ne 0 ]; then
                echo "ERROR: Failed to restart deployments for ${workflow}/${difficulty}"
                failed_configs+=("${workflow}/${difficulty}")
                continue
            fi
            run_evaluation "${difficulty}" "${workflow}"
            if [ $? -eq 0 ]; then
                succeeded_configs+=("${workflow}/${difficulty}")
            else
                failed_configs+=("${workflow}/${difficulty}")
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
            exit 1
        fi
        
        local all_end=$(get_timestamp)
        echo "=========================================="
        echo "All evaluations completed successfully"
        echo "Start Time: ${all_start}"
        echo "End Time: ${all_end}"
        echo "=========================================="
        exit 0
    elif [ -n "${difficulty}" ] && [ -z "${workflow}" ]; then
        local all_start=$(get_timestamp)
        local workflows=("search" "search_branch")
        local succeeded_configs=()
        local failed_configs=()
        
        echo "=========================================="
        echo "Running all workflows for difficulty ${difficulty}"
        echo "Start Time: ${all_start}"
        echo "=========================================="
        
        for workflow in "${workflows[@]}"; do
            restart_deployments
            if [ $? -ne 0 ]; then
                echo "ERROR: Failed to restart deployments for ${workflow}/${difficulty}"
                failed_configs+=("${workflow}/${difficulty}")
                continue
            fi
            run_evaluation "${difficulty}" "${workflow}"
            if [ $? -eq 0 ]; then
                succeeded_configs+=("${workflow}/${difficulty}")
            else
                failed_configs+=("${workflow}/${difficulty}")
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
            exit 1
        fi
        
        local all_end=$(get_timestamp)
        echo "=========================================="
        echo "All evaluations completed successfully"
        echo "Start Time: ${all_start}"
        echo "End Time: ${all_end}"
        echo "=========================================="
        exit 0
    else
        if [ -z "${difficulty}" ] || [ -z "${workflow}" ]; then
            echo "ERROR: Both --difficulty and --workflow must be specified, or use --all"
            print_usage
            exit 1
        fi
        
        restart_deployments
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to restart deployments"
            exit 1
        fi
        run_evaluation "${difficulty}" "${workflow}"
        echo "=========================================="
        echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "=========================================="
    fi
}

main "$@"
