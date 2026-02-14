# FoldAgent RL Training Deployment

This directory contains Kubernetes deployment files for running FoldAgent RL training on a Kubernetes cluster with H200 GPUs.

## Prerequisites

1. Kubernetes cluster with at least 4 H200 GPUs
2. NVIDIA GPU Operator installed on the cluster
3. Helm installed for package management
4. Docker installed for building the training image

## Files Included

1. **Dockerfile.rl-training**: Dockerfile for building the RL training image
2. **rl-training-pvc.yaml**: PersistentVolumeClaim for storing training data, logs, and checkpoints
3. **rl-training-job.yaml**: Kubernetes Job for running the RL training
4. **wandb-secret.yaml**: Secret for storing WANDB API key
5. **wandb-deployment.yaml**: Deployment for running a local WANDB server with ingress

## Setup Instructions

### 1. Build the Docker Image

```bash
# Navigate to FoldAgent directory
cd /Users/rexsasori/FoldAgent

# Build the Docker image
docker build -f docker/Dockerfile.rl-training -t foldagent-rl-training:latest .

# Push the image to your container registry (if needed)
docker tag foldagent-rl-training:latest <your-registry>/foldagent-rl-training:latest
docker push <your-registry>/foldagent-rl-training:latest
```

### 2. Create Kubernetes Resources

```bash
# Create WANDB secret (update with your actual API key)
kubectl apply -f deployment/rl-training/wandb-secret.yaml

# Deploy WANDB server
kubectl apply -f deployment/rl-training/wandb-deployment.yaml

# Run RL training job
kubectl apply -f deployment/rl-training/rl-training-job.yaml
```

### 3. Configure Training Parameters

Update the following parameters in `scripts/rl_training.sh` to match your H200 GPU setup:

```bash
# Hardware settings (4 GPUs instead of 8)
actor_rollout_ref.rollout.tensor_model_parallel_size=4
actor_rollout_ref.rollout.n=4

# Memory optimization for H200 GPUs
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
```

### 4. Accessing Training Metrics

#### Via WANDB

1. Open a browser and go to: `https://wandb-liuyunxin.xa.xshixun.cn`
2. Log in with your WANDB credentials
3. View training metrics and curves

#### Via TensorBoard

1. Port-forward the tensorboard port from the training pod:

```bash
# Get the training pod name
POD_NAME=$(kubectl get pods -n liuyunxin -l job-name=foldagent-rl-training -o jsonpath="{.items[0].metadata.name}")

# Port-forward
kubectl port-forward -n liuyunxin $POD_NAME 6006:6006
```

2. Open a browser and go to: `http://localhost:6006`

### 5. Monitoring Training Progress

```bash
# Check training job status
kubectl get job -n liuyunxin foldagent-rl-training

# View training logs
kubectl logs -n liuyunxin -f job/foldagent-rl-training
```

### 6. Cleaning Up

```bash
# Delete training job
kubectl delete job -n liuyunxin foldagent-rl-training

# Delete WANDB deployment
kubectl delete -f deployment/rl-training/wandb-deployment.yaml

# Delete secret
kubectl delete secret -n liuyunxin wandb-secret
```

## Resource Requirements

- **CPU**: 32 cores
- **Memory**: 256GiB
- **GPU**: 4 x H200 (141GB VRAM each)
- **Storage**: 500GiB for training data, logs, and checkpoints

## Notes

- The training job is configured to use the `ByteDance-Seed/Seed-OSS-36B-Instruct` model
- Database logging is disabled by default (`LOG_EVENT_TO_DB=false`) to optimize performance
- The training will automatically download the model if not present
- Checkpoints are saved every 10 steps as configured in `rl_training.sh`