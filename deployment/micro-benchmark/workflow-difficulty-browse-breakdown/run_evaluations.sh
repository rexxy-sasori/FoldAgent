#!/bin/bash
set -e

NAMESPACE="liuyunxin"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_DIR="${SCRIPT_DIR}"

# Paths to dependent deployments
SGLANG_DEPLOYMENT="/Users/rexsasori/FoldAgent/deployment/sglang/sglang-deployment.yaml"
SEARCH_SERVER_DEPLOYMENT="/Users/rexsasori/FoldAgent/deployment/search-server-deployment.yaml"

echo "=========================================="
echo "FoldAgent Evaluation Runner"
echo "=========================================="
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="

# Function to get current timestamp
get_timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
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

run_evaluation() {
    local difficulty=$1
    local workflow=$2
    local start_time=$(get_timestamp)
    
    local job_file="${DEPLOYMENT_DIR}/eval-job-${workflow}-${difficulty}.yaml"
    
    echo ""
    echo "=========================================="
    echo "Running evaluation:"
    echo "  Difficulty: ${difficulty}"
    echo "  Workflow: ${workflow}"
    echo "  Job: ${job_file}"
    echo "  Start Time: ${start_time}"
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
    local job_name=$(kubectl get jobs -l app=foldagent-eval-${workflow}-${difficulty} -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}')
    
    echo "[$(get_timestamp)] Monitoring job progress..."
    echo "[$(get_timestamp)] Waiting for job to complete..."
    
    # Monitor job logs while waiting
    local start_wait=$(date +%s)
    local timeout=1800  # 30 minutes timeout
    
    # Show initial logs
    sleep 5
    echo "[$(get_timestamp)] Showing job logs..."
    
    # Wait for job to complete
    kubectl wait --for=condition=complete job "${job_name}" --timeout=${timeout}s -n ${NAMESPACE} || {
        echo "[$(get_timestamp)] WARNING: Job did not complete within ${timeout} seconds"
        local job_status=$(kubectl get job "${job_name}" -n ${NAMESPACE} -o jsonpath='{.status.conditions[0].type}')
        echo "[$(get_timestamp)] Job status: ${job_status}"
    }
    
    local end_time=$(get_timestamp)
    local job_status=$(kubectl get job "${job_name}" -n ${NAMESPACE} -o jsonpath='{.status.conditions[*].type}')
    
    echo ""
    echo "Evaluation ${job_status} for ${difficulty}/${workflow}"
    echo "=========================================="
    echo "Start Time: ${start_time}"
    echo "End Time: ${end_time}"
    echo "Job Status: ${job_status}"
    echo "Results are saved in the container's /root/results directory"
    
    # Get the pod name from the job
    local pod_name=$(kubectl get pods -l job-name="${job_name}" -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}')
    
    # Show final logs
    if [[ "${job_status}" != "Complete" ]]; then
        echo ""
        echo "[$(get_timestamp)] Showing full logs for failed job..."
        kubectl logs "${pod_name}" -n ${NAMESPACE}
        echo "=========================================="
        return 1
    else
        echo ""
        echo "[$(get_timestamp)] Showing recent logs..."
        kubectl logs "${pod_name}" -n ${NAMESPACE} --tail=50
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
    echo "  -a, --all               Run all combinations (3 difficulties × 2 workflows)"
    echo "  -r, --restart-only      Only restart deployments, don't run evaluation"
    echo "  -n, --dry-run           Dry run (show what would be executed)"
    echo "  -h, --help              Show help message"
    echo ""
    echo "Examples:"
    echo "  $0 --difficulty easy --workflow search"
    echo "  $0 --all"
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
            echo "# Wait for deployments to be ready"
            echo "kubectl rollout status deployment sglang -n ${NAMESPACE} --timeout=600s"
            echo "kubectl rollout status deployment search-server -n ${NAMESPACE} --timeout=600s"
        elif ${run_all}; then
            local difficulties=("easy" "medium" "hard")
            local workflows=("search" "search_branch")
            
            echo "Would run all evaluations:"
            echo "=========================================="
            echo "Difficulties: easy, medium, hard"
            echo "Workflows: search, search_branch"
            echo "Total: 6 evaluations"
            echo ""
            echo "For each evaluation, would execute:"
            echo "1. Restart deployments (as shown above)"
            echo "2. Run evaluation with commands:"
            echo ""
            
            for workflow in "${workflows[@]}"; do
                for difficulty in "${difficulties[@]}"; do
                    local job_file="${DEPLOYMENT_DIR}/eval-job-${workflow}-${difficulty}.yaml"
                    echo "# ${workflow} - ${difficulty}"
                    echo "kubectl delete -f \"${job_file}\" --ignore-not-found=true -n ${NAMESPACE}"
                    echo "kubectl create -f \"${job_file}\" -n ${NAMESPACE}"
                    echo "# Monitor job until completion (30 minute timeout)"
                    echo "# Show logs while waiting"
                    echo "kubectl wait --for=condition=complete job \$(kubectl get jobs -l app=foldagent-eval-${workflow}-${difficulty} -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') --timeout=1800s -n ${NAMESPACE}"
                    echo "kubectl get pods -l job-name=\$(kubectl get jobs -l app=foldagent-eval-${workflow}-${difficulty} -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE}"
                    echo "kubectl logs \$(kubectl get pods -l job-name=\$(kubectl get jobs -l app=foldagent-eval-${workflow}-${difficulty} -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE} --tail=50"
                    echo ""
                done
            done
        else
            if [ -n "${difficulty}" ] && [ -n "${workflow}" ]; then
                local job_file="${DEPLOYMENT_DIR}/eval-job-${workflow}-${difficulty}.yaml"
                echo "Would run evaluation:"
                echo "=========================================="
                echo "Difficulty: ${difficulty}"
                echo "Workflow: ${workflow}"
                echo "Job: ${job_file}"
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
                echo "kubectl rollout status deployment sglang -n ${NAMESPACE} --timeout=600s"
                echo "kubectl rollout status deployment search-server -n ${NAMESPACE} --timeout=600s"
                echo ""
                echo "# 2. Run evaluation"
                echo "kubectl delete -f \"${job_file}\" --ignore-not-found=true -n ${NAMESPACE}"
                echo "kubectl create -f \"${job_file}\" -n ${NAMESPACE}"
                echo "# Monitor job until completion (30 minute timeout)"
                echo "# Show logs while waiting"
                echo "kubectl wait --for=condition=complete job \$(kubectl get jobs -l app=foldagent-eval-${workflow}-${difficulty} -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') --timeout=1800s -n ${NAMESPACE}"
                echo "kubectl logs \$(kubectl get pods -l job-name=\$(kubectl get jobs -l app=foldagent-eval-${workflow}-${difficulty} -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE} -o jsonpath='{.items[0].metadata.name}') -n ${NAMESPACE} --tail=50"
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
    fi
    
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
}

main "$@"
