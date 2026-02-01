# FoldAgent Evaluation Guide

This guide explains how to run the `eval_bc.py` script with different difficulty levels and workflows on Kubernetes.

## Overview

The evaluation system runs `scripts/eval_bc.py` with the following combinations:
- **Difficulties**: `easy`, `medium`, `hard`
- **Workflows**: `search` (ReAct), `search_branch` (Context-Folding)

This creates 6 total evaluation configurations (3 × 2).

## Prerequisites

- Kubernetes cluster access
- `kubectl` configured for the `liuyunxin` namespace
- Required secrets already created:
  - `foldagent-eval-secrets`
  - `hf-token-secret`
  - `postgresql-secrets`

## Quick Start

### Run All Evaluations

To run all 6 combinations sequentially (each with fresh deployments):

```bash
cd /Users/rexsasori/FoldAgent/deployment/micro-benchmark
chmod +x run_evaluations.sh
./run_evaluations.sh --all
```

### Run Specific Evaluation

To run a single evaluation with specific difficulty and workflow:

```bash
./run_evaluations.sh --difficulty easy --workflow search
./run_evaluations.sh --difficulty medium --workflow search_branch
./run_evaluations.sh --difficulty hard --workflow search
```

### Restart Deployments Only

To restart the dependent deployments without running evaluations:

```bash
./run_evaluations.sh --restart-only
```

## Deployment Files

The following deployment files are created in `deployment/micro-benchmark/`:

| File | Difficulty | Workflow |
|------|------------|----------|
| `eval-deployment-search-easy.yaml` | easy | search |
| `eval-deployment-search-medium.yaml` | medium | search |
| `eval-deployment-search-hard.yaml` | hard | search |
| `eval-deployment-search_branch-easy.yaml` | easy | search_branch |
| `eval-deployment-search_branch-medium.yaml` | medium | search_branch |
| `eval-deployment-search_branch-hard.yaml` | hard | search_branch |

## Manual Deployment

If you prefer to deploy manually without the script:

```bash
# Restart dependent services (robust approach)
kubectl delete -f /Users/rexsasori/FoldAgent/deployment/sglang/sglang-deployment.yaml --ignore-not-found=true -n liuyunxin
kubectl apply -f /Users/rexsasori/FoldAgent/deployment/sglang/sglang-deployment.yaml -n liuyunxin

kubectl delete -f /Users/rexsasori/FoldAgent/deployment/search-server-deployment.yaml --ignore-not-found=true -n liuyunxin
kubectl apply -f /Users/rexsasori/FoldAgent/deployment/search-server-deployment.yaml -n liuyunxin

# Wait for services to be ready
kubectl rollout status deployment sglang -n liuyunxin --timeout=600s
kubectl rollout status deployment search-server -n liuyunxin --timeout=600s

# Apply specific evaluation deployment
kubectl apply -f /Users/rexsasori/FoldAgent/deployment/micro-benchmark/eval-deployment-search-easy.yaml -n liuyunxin

# View logs
kubectl logs -f deployment/foldagent-eval-search-easy -n liuyunxin
```

## Robust Deployment Restart

The script now uses a more robust approach to restart deployments:

1. **Delete and Re-apply**: Instead of just rolling out a restart, the script deletes and re-applies the deployment files
2. **File Tracking**: The script keeps track of the exact locations of the deployment files:
   - SGLang: `/Users/rexsasori/FoldAgent/deployment/sglang/sglang-deployment.yaml`
   - Search Server: `/Users/rexsasori/FoldAgent/deployment/search-server-deployment.yaml`
3. **Fallback Mechanism**: If the deployment files are not found, it falls back to the traditional rollout restart approach
4. **Enhanced Readiness Checks**: Before proceeding, the script:
   - Waits for deployment rollout to complete
   - Verifies all pods are ready using `kubectl wait` (more reliable than custom polling)
   - Checks service reachability
   - Provides warnings if deployments aren't ready within reasonable time

This ensures a clean state for each evaluation run and reduces the chance of deployment issues.

## Dry Run Mode

The script now supports a dry run mode to preview what would be executed without making actual changes:

```bash
# Preview all evaluations
./run_evaluations.sh --all --dry-run

# Preview specific evaluation
./run_evaluations.sh --difficulty easy --workflow search --dry-run

# Preview deployment restart
./run_evaluations.sh --restart-only --dry-run
```

Dry run mode shows:
- **Actual commands** that would be executed
- Which deployments would be restarted
- Which evaluation configurations would be run
- Paths to deployment files
- No actual changes to the cluster

### Example Dry Run Output

```bash
==========================================
Dry Run Mode
==========================================
This is a dry run - no actual changes will be made.

Would run evaluation:
==========================================
Difficulty: easy
Workflow: search
Deployment: /Users/rexsasori/FoldAgent/deployment/micro-benchmark/eval-deployment-search-easy.yaml

Commands that would be executed:

# 1. Restart dependent deployments
kubectl delete -f "/Users/rexsasori/FoldAgent/deployment/sglang/sglang-deployment.yaml" --ignore-not-found=true -n liuyunxin
kubectl apply -f "/Users/rexsasori/FoldAgent/deployment/sglang/sglang-deployment.yaml" -n liuyunxin
kubectl delete -f "/Users/rexsasori/FoldAgent/deployment/search-server-deployment.yaml" --ignore-not-found=true -n liuyunxin
kubectl apply -f "/Users/rexsasori/FoldAgent/deployment/search-server-deployment.yaml" -n liuyunxin
kubectl rollout status deployment sglang -n liuyunxin --timeout=600s
kubectl rollout status deployment search-server -n liuyunxin --timeout=600s

# 2. Run evaluation
kubectl delete -f "/Users/rexsasori/FoldAgent/deployment/micro-benchmark/eval-deployment-search-easy.yaml" --ignore-not-found=true -n liuyunxin
kubectl apply -f "/Users/rexsasori/FoldAgent/deployment/micro-benchmark/eval-deployment-search-easy.yaml" -n liuyunxin
sleep 10
kubectl logs -f deployment/foldagent-eval-search-easy -n liuyunxin

Dry run completed successfully!
```

## Timestamp Tracking

The script now includes comprehensive timestamp tracking to help monitor execution times:

### What's Tracked:

1. **Overall Execution**: Start and end times for the entire script run
2. **Deployment Restarts**: Start and end times for the deployment restart process
3. **Individual Evaluations**: Start and end times for each evaluation run
4. **Key Milestones**: Timestamps for important operations (e.g., deployment deletion/apply)

### Example Timestamp Output

```bash
==========================================
FoldAgent Evaluation Runner
==========================================
Start Time: 2026-02-01 10:42:12
==========================================

Restarting dependent deployments...
-----------------------------------
Start Time: 2026-02-01 10:42:12

[2026-02-01 10:42:12] Restarting sglang deployment...
[2026-02-01 10:42:15] Restarting search-server deployment...
[2026-02-01 10:42:18] Waiting for deployments to be ready...
...
Deployments restarted successfully!
-----------------------------------
Start Time: 2026-02-01 10:42:12
End Time: 2026-02-01 10:45:30

==========================================
Running evaluation:
  Difficulty: easy
  Workflow: search
  Deployment: /Users/rexsasori/FoldAgent/deployment/micro-benchmark/eval-deployment-search-easy.yaml
  Start Time: 2026-02-01 10:45:30
==========================================

[2026-02-01 10:45:30] Deleting existing deployment (if any)...
[2026-02-01 10:45:32] Applying deployment...
[2026-02-01 10:45:35] Waiting for deployment to start...
...
Evaluation completed for easy/search
==========================================
Start Time: 2026-02-01 10:45:30
End Time: 2026-02-01 10:55:45
Results are saved in the container's /root/results directory
==========================================

==========================================
End Time: 2026-02-01 10:55:45
==========================================
```

### Benefits of Timestamp Tracking

1. **Performance Monitoring**: Track how long each component takes to execute
2. **Troubleshooting**: Identify bottlenecks in the deployment or evaluation process
3. **Audit Trail**: Maintain a record of when evaluations were run
4. **Scheduling**: Better plan when to run evaluations based on historical timing data

Timestamps use the format `YYYY-MM-DD HH:MM:SS` for consistency and readability.

## Results

Results are saved in the container's `/root/results` directory. Each run creates:
- A timestamped JSON file with evaluation results
- A log file with detailed execution logs

To retrieve results:

```bash
# Copy results from pod
kubectl cp <pod-name>:/root/results ./local_results -n liuyunxin

# Or use kubectl exec to view
kubectl exec -it <pod-name> -n liuyunxin -- ls -la /root/results
```

## Script Options

```bash
./run_evaluations.sh [OPTIONS]

Options:
  -d, --difficulty DIFF   Run specific difficulty (easy|medium|hard)
  -w, --workflow WORKFLOW  Run specific workflow (search|search_branch)
  -a, --all               Run all combinations (3 difficulties × 2 workflows)
  -r, --restart-only      Only restart deployments, don't run evaluation
  -n, --dry-run           Dry run (show what would be executed)
  -h, --help              Show help message
```

## Configuration

Each deployment uses the following configuration:

- **Model**: `ByteDance-Seed/Seed-OSS-36B-Instruct`
- **Workers**: 32 parallel evaluation workers
- **Prompt Length**: 16384 tokens
- **Response Length**: 32768 tokens
- **Max Turns**: 200
- **Max Sessions**: 10

These can be modified in the deployment YAML files if needed.

## Troubleshooting

### Check Deployment Status

```bash
kubectl get deployments -n liuyunxin
kubectl get pods -n liuyunxin
```

### View Logs

```bash
# Evaluation pod logs
kubectl logs deployment/foldagent-eval-search-easy -n liuyunxin

# SGLang server logs
kubectl logs deployment/sglang -n liuyunxin

# Search server logs
kubectl logs deployment/search-server -n liuyunxin
```

### Delete Evaluation Deployment

```bash
kubectl delete -f /Users/rexsasori/FoldAgent/deployment/micro-benchmark/eval-deployment-search-easy.yaml -n liuyunxin
```

## Notes

- The script automatically restarts `sglang` and `search-server` deployments before each evaluation run using a robust delete-and-reapply approach
- Each evaluation run is independent and uses a fresh deployment
- The evaluation container sleeps indefinitely after completion to allow log viewing
- Results are persisted in the container's `/root/results` directory
- All deployment files are now organized in the `deployment/micro-benchmark/` directory
