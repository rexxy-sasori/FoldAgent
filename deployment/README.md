# FoldAgent Kubernetes Deployment Guide

This guide provides instructions for deploying the FoldAgent project on Kubernetes using the local LLM option mentioned in the main README.md.

## Overview

The deployment consists of three main components:
1. **Search Server**: Handles search functionality using embeddings
2. **Local LLM**: Runs the ByteDance-Seed/Seed-OSS-36B-Instruct model using vLLM
3. **Evaluation Service**: Runs the evaluation script using the local LLM

## Prerequisites

- Kubernetes cluster with GPU support (NVIDIA GPUs)
- Docker installed on your local machine
- `kubectl` configured to access your Kubernetes cluster
- NVIDIA GPU Operator installed in the cluster (for GPU support)

## Building Docker Images

First, build the Docker image from the project root (this single image works for both Search Server and Evaluation components):

```bash
# Build image
docker build -t foldagent:latest .

# Push image to your container registry (if required)
docker tag foldagent:latest <your-registry>/foldagent:latest
docker push <your-registry>/foldagent:latest
```

Note: The Local LLM component uses the official `vllm/vllm-openai` image, so no custom build is needed.

## Deploying to Kubernetes

### 1. Create Namespace

```bash
kubectl apply -f foldagent-namespace.yaml
```

### 2. Deploy Search Server

```bash
kubectl apply -f search-server-deployment.yaml
```

### 3. Deploy Local LLM

```bash
kubectl apply -f local-llm-deployment.yaml
```

### 4. Run Evaluation

The evaluation is configured as a Kubernetes Job for one-time execution (recommended):

```bash
# Run evaluation as a Job
kubectl apply -f eval-deployment.yaml
```

This will run the evaluation using the BrowseComp dataset with the Fold Agent workflow (`search_branch`).

## Checking Deployment Status

```bash
# Check all resources in the namespace
kubectl get all -n liuyunxin

# Check pod status
kubectl get pods -n liuyunxin

# Check logs
kubectl logs <pod-name> -n liuyunxin

# Get service URLs
kubectl get services -n liuyunxin
```

## Testing the Deployment

### Check Search Server

```bash
# Forward port to local machine
kubectl port-forward service/search-server 8000:8000 -n liuyunxin

# Test with curl
curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"query": "What is machine learning?", "k": 5}'
```

### Check Local LLM

```bash
# Forward port to local machine
kubectl port-forward service/local-llm 8001:8001 -n liuyunxin

# Test with curl (matches the main README's vLLM setup)
curl -X POST http://localhost:8001/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer dummy" -d '{"model": "ByteDance-Seed/Seed-OSS-36B-Instruct", "messages": [{"role": "user", "content": "Hello"}]}'
```

Note: The Local LLM uses vLLM with the `ByteDance-Seed/Seed-OSS-36B-Instruct` model, configured with a maximum model length of 131072 tokens.

## Accessing Evaluation Results

If using the Job with PersistentVolumeClaim:

```bash
# Find the evaluation pod
kubectl get pods -n liuyunxin -l app=foldagent-eval

# Copy results from pod to local machine
kubectl cp <eval-pod-name>:/app/results /local/path/to/save/results -n liuyunxin
```

## Cleaning Up

```bash
# Delete all resources
kubectl delete -f eval-deployment.yaml
kubectl delete -f local-llm-deployment.yaml
kubectl delete -f search-server-deployment.yaml
kubectl delete -f foldagent-namespace.yaml

# Delete PVC (optional)
kubectl delete pvc foldagent-results-pvc -n liuyunxin
```

## Configuration Options

### Environment Variables

- `NUM_GPUS`: Number of GPUs to use (default: 1)
- `CUDA_VISIBLE_DEVICES`: GPU devices to use (default: 0)
- `OPENAI_API_KEY`: API key for OpenAI (dummy value for local LLM)
- `OPENAI_BASE_URL`: Base URL for OpenAI API (local LLM URL)

### Resource Requirements

You can adjust the resource limits in the deployment files based on your cluster capabilities. Here are the default configurations:

**Search Server:**
```yaml
resources:
  limits:
    nvidia.com/gpu: 1
    memory: "32Gi"
    cpu: "8"
  requests:
    memory: "16Gi"
    cpu: "4"
```

**Local LLM (vLLM):**
```yaml
resources:
  limits:
    nvidia.com/gpu: 1
    memory: "64Gi"  # Higher memory requirement for the LLM
    cpu: "8"
  requests:
    memory: "32Gi"
    cpu: "4"
```

**Evaluation:**
```yaml
resources:
  limits:
    nvidia.com/gpu: 1
    memory: "32Gi"
    cpu: "8"
  requests:
    memory: "16Gi"
    cpu: "4"
```


## Troubleshooting

### GPU Issues
- Ensure NVIDIA GPU Operator is installed: `kubectl get pods -n gpu-operator`
- Check GPU availability: `kubectl describe nodes | grep -A 10 nvidia.com/gpu`

### Model Download Issues
- The first run will download models from Hugging Face. This may take time.
- Ensure your cluster has internet access or use a pre-downloaded model cache.

### Evaluation Failures
- Check logs for API connectivity issues: `kubectl logs <eval-pod-name> -n liuyunxin`
- Verify service URLs are correct: `http://search-server.liuyunxin:8000` and `http://local-llm.liuyunxin:8001/v1`