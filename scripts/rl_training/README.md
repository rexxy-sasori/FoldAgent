# React Agent Training Guide

Complete guide for training react agents using VERL framework with Ray distributed training.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [Docker Setup](#docker-setup)
7. [Ray Distributed Training](#ray-distributed-training)
8. [H200 GPU Configuration](#h200-gpu-configuration)
9. [Performance Tuning](#performance-tuning)
10. [Troubleshooting](#troubleshooting)

## Overview

### What is React Agent?

React agent implements the ReAct (Reason + Act) paradigm:
- **Think**: Generate reasoning about current state
- **Act**: Execute tool calls based on reasoning
- **Observe**: Process tool results
- **Repeat**: Continue until answer found or limit reached

### Key Differences

| Aspect | React Agent | VERL's Tool Agent |
|---------|-------------|---------------------|
| **Architecture** | Custom agent loop | Built-in VERL agent loop |
| **Tool Integration** | Uses `envs/local_search.py` directly | Uses `verl_tool_adaptor` wrapper |
| **Configuration** | `workflow=search` | `multi_turn.tool_config_path=...` |
| **Complexity** | Simpler, more control | More complex, more automated |
| **Best For** | Custom agent logic | Standard tool use cases |

### Why React Agent?

✅ **Simpler configuration** - Just set `workflow=search`
✅ **More control** - Direct access to environment
✅ **Custom logic** - Implement your own agent behavior
✅ **Less overhead** - No tool wrapper layer

## Architecture

### Data Flow

```
Training Data → ReactAgentLoop → react_agent.process_item() → DataProto 
→ Convert to AgentLoopOutput → VERL Trainer → DataProto
```

### Type Conversion

**Important**: The `process_item()` function returns `DataProto`, but VERL's `AgentLoopBase.run()` expects `AgentLoopOutput`. The `ReactAgentLoop` includes a conversion method `_convert_dataproto_to_agentloopoutput()` to handle this.

### Key Components

#### [`react_training.py`](react_training.py)
- **Purpose**: Training entry point and agent loop implementation
- **Key Class**: `ReactAgentLoop` - Registered as `react_agent`
- **Key Method**: `run()` - Executes react agent workflow
- **Type Conversion**: `_convert_dataproto_to_agentloopoutput()` - Converts DataProto to AgentLoopOutput

#### [`react_training.sh`](react_training.sh)
- **Purpose**: Shell script to configure and run training
- **Configuration**: Sets model paths, data paths, and training hyperparameters

## Prerequisites

### 1. Start Search Server

```bash
# Start the local search server (envs/search_server.py)
# This is the only external service needed for react_agent
python envs/search_server.py
# Server runs on http://localhost:8000
```

**Note**: Unlike VERL's built-in `tool_agent`, `react_agent` does NOT need `verl_tool_adaptor`. It directly uses `envs/local_search.py` which connects to the search server.

### 2. Prepare Data

```bash
# Ensure training data exists
ls /root/react-training/data/bc_train.parquet
```

### 3. Python Package Structure

The `scripts/` directory must be a proper Python package for Ray distributed training:

```
scripts/
├── __init__.py                    # ← Makes scripts/ a package
├── rl_training/
│   ├── __init__.py                # ← Makes scripts.rl_training/ a package
│   ├── react_training.py
│   └── ...
```

**Why this is necessary**: Ray distributed workers need to import `scripts.rl_training.react_training` as a Python package, which requires `__init__.py` files.

## Quick Start

### Single GPU Training

```bash
cd /app/scripts/rl_training
bash react_training.sh
```

### Multi-GPU Training

```bash
cd /app/scripts/rl_training
bash react_training.sh
# Automatically uses all available GPUs via Ray
```

## Configuration

### Environment Variables

```bash
# Docker environment
export PYTHONPATH=/app:$PYTHONPATH

# Local environment (if not using Docker)
# export PYTHONPATH=/Users/rexsasori/FoldAgent:$PYTHONPATH

# Ray distributed training
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
```

### Training Parameters

In [`react_training.sh`](react_training.sh):

```bash
# Model configuration
MODEL_PATH=/app/models/qwen2.5-7b-instruct

# Data configuration
DATA_PATH=/app/data/bc_train.parquet
VAL_DATA_PATH=/app/data/bc_val.parquet

# Training configuration
PROMPT_LENGTH=8192
RESPONSE_LENGTH=32768
MAX_TURNS=20
```

### Agent Configuration

```bash
# Agent loop configuration
actor_rollout_ref.rollout.agent.default_agent_loop=react_agent

# Workflow configuration
actor_rollout_ref.rollout.plugin.workflow=search
actor_rollout_ref.rollout.plugin.max_turn=20
actor_rollout_ref.rollout.plugin.val_max_turn=20

# Log probabilities for PPO
actor_rollout_ref.rollout.calculate_log_probs=True
```

## Docker Setup

### Build Docker Image

```bash
# Build from project root
cd /Users/rexsasori/FoldAgent
docker build -f docker/Dockerfile.rl-training -t foldagent-rl-training:latest .
```

### Run Training in Docker

#### Option 1: Interactive Shell

```bash
docker run -it --gpus all \
    -v /Users/rexsasori/FoldAgent:/app \
    -p 8000:8000 \
    foldagent-rl-training:latest \
    bash

# Inside container
cd /app/scripts/rl_training
bash react_training.sh
```

#### Option 2: Direct Command

```bash
docker run -it --gpus all \
    -v /Users/rexsasori/FoldAgent:/app \
    -p 8000:8000 \
    foldagent-rl-training:latest \
    bash -c "cd /app/scripts/rl_training && bash react_training.sh"
```

### Docker Environment

| Variable | Docker | Local |
|----------|---------|--------|
| **PYTHONPATH** | `/app` | `/Users/rexsasori/FoldAgent` |
| **WORKDIR** | `/app` | `/Users/rexsasori/FoldAgent` |
| **Data Path** | `/app/data/` | `/Users/rexsasori/FoldAgent/data/` |
| **Model Path** | `/app/models/` | `/Users/rexsasori/FoldAgent/models/` |

## Ray Distributed Training

### How It Works

Ray distributed training runs across multiple GPUs, with each GPU having its own Python process:

```
Main Process (GPU 0):
  - Imports scripts.rl_training.react_training
  - @register runs
  - react_agent added to registry ✓
  - Spawns Ray workers

Ray Worker (GPU 1):
  - Starts fresh process
  - Imports scripts.rl_training.react_training
  - @register runs
  - react_agent added to registry ✓

Ray Worker (GPU 2):
  - Same as GPU 1 ✓

Ray Worker (GPU 3):
  - Same as GPU 1 ✓
```

### Key Requirements

1. **Python Package Structure** - `__init__.py` files must exist
2. **PYTHONPATH** - Must include project root
3. **Module Import** - Must import before main execution

### Registration Process

```python
# 1. Module is imported
import scripts.rl_training.react_training

# 2. @register decorator runs
@register("react_agent")
class ReactAgentLoop(AgentLoopBase):
    ...

# 3. Class is added to registry
_agent_loop_registry["react_agent"] = {
    "_target_": "scripts.rl_training.react_training.ReactAgentLoop"
}
```

### Common Issues

#### Issue: "Agent loop react_agent not registered"

**Cause**: Module not imported in Ray worker process

**Solution**: Ensure module is imported before Ray initialization

#### Issue: "ImportError: No module named 'scripts.rl_training'"

**Cause**: Missing `__init__.py` files

**Solution**: Create `__init__.py` in both directories:

```bash
touch scripts/__init__.py
touch scripts/rl_training/__init__.py
```

#### Issue: Works on single GPU but fails on multiple GPUs

**Cause**: Ray workers on additional GPUs don't import custom modules

**Solution**: Ensure `__init__.py` files exist and PYTHONPATH is set correctly

## H200 GPU Configuration

### H200 Specifications

- **HBM3 Memory**: 141GB per GPU
- **Total Memory**: 564GB (4 GPUs)
- **Compute**: High-performance compute
- **Model Support**: Can handle large models (36B+ parameters)

### Key Parameters

```bash
# Model parallelism (1 model across 4 GPUs)
actor_rollout_ref.rollout.tensor_model_parallel_size=4

# Number of GPUs
actor_rollout_ref.rollout.n=4
trainer.n_gpus_per_node=4

# Agent workers (1 per GPU)
actor_rollout_ref.rollout.agent.num_workers=4

# Batch sizes
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
data.train_batch_size=16
```

### Memory Configuration

```bash
# Token lengths (optimized for H200's 141GB memory)
PROMPT_LENGTH=16384        # 16K tokens for prompt
RESPONSE_LENGTH=32768       # 32K tokens for response
MAX_LENGTH=49152           # 49K tokens total

# Maximum token lengths per GPU
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_LENGTH
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_LENGTH
actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=$MAX_LENGTH
```

### Optimization Settings

```bash
# Data type (H200 supports bfloat16 efficiently)
actor_rollout_ref.rollout.dtype=bfloat16

# FSDP configuration (for distributed training)
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
```

### H200-Specific Optimizations

The [`react_training.sh`](react_training.sh) script includes H200-specific environment variables:

```bash
# NCCL optimizations for high-speed interconnect
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5

# vLLM optimizations
export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL=WARN
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

# CUDA optimizations
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=true
```

### Launch Command

```bash
# Docker
docker run -it --gpus all \
    -v /Users/rexsasori/FoldAgent:/app \
    -p 8000:8000 \
    -e NCCL_DEBUG=WARN \
    foldagent-rl-training:latest \
    bash -c "cd /app/scripts/rl_training && bash react_training.sh"
```

### H200-Specific Optimizations

#### 1. Memory Efficiency

H200's 141GB HBM3 memory allows for:
- Larger batch sizes
- Longer sequences
- More aggressive caching

```bash
# Increase batch size if needed
data.train_batch_size=32  # Can be increased from 16

# Increase micro batch size if needed
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2  # Can be increased from 1
```

#### 2. Compute Efficiency

H200's high compute capability allows for:
- Faster training
- More aggressive optimization
- Better throughput

```bash
# Enable mixed precision
actor_rollout_ref.rollout.dtype=bfloat16

# Enable gradient checkpointing (if supported)
# actor_rollout_ref.actor.fsdp_config.use_activation_checkpointing=True
```

#### 3. Network Efficiency

H200's high-speed interconnect allows for:
- Faster communication between GPUs
- Better scaling for distributed training

```bash
# Use NCCL for GPU communication
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
```

## Performance Tuning

### Monitor GPU Memory

```bash
# Watch GPU memory usage
watch -n 1 nvidia-smi

# Check memory usage during training
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Adjust Batch Sizes

If GPU memory is underutilized (<80%):
```bash
# Increase batch size
data.train_batch_size=32
actor_rollout_ref.actor.ppo_mini_batch_size=32
```

If GPU memory is overutilized (>95%):
```bash
# Decrease batch size
data.train_batch_size=8
actor_rollout_ref.actor.ppo_mini_batch_size=8
```

### Adjust Token Lengths

If sequences are too short:
```bash
# Increase token lengths
PROMPT_LENGTH=32768        # 32K tokens
RESPONSE_LENGTH=65536       # 64K tokens
MAX_LENGTH=98304           # 98K tokens
```

If sequences are too long (OOM errors):
```bash
# Decrease token lengths
PROMPT_LENGTH=8192         # 8K tokens
RESPONSE_LENGTH=16384       # 16K tokens
MAX_LENGTH=24576           # 24K tokens
```

### Check NCCL Settings

```bash
# Check network configuration
ibstat  # Check InfiniBand status

# Optimize communication
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
```

## Troubleshooting

### H200-Specific Issues

#### Issue: Out of Memory (OOM)

**Symptoms**: CUDA out of memory errors

**Solutions**:
1. Decrease batch size
2. Decrease token lengths
3. Enable gradient checkpointing
4. Increase model parallelism

```bash
# Decrease batch size
data.train_batch_size=8

# Decrease token lengths
PROMPT_LENGTH=8192
RESPONSE_LENGTH=16384

# Enable gradient checkpointing
# actor_rollout_ref.actor.fsdp_config.use_activation_checkpointing=True
```

#### Issue: Slow Training

**Symptoms**: Training is slower than expected

**Solutions**:
1. Increase batch size
2. Increase micro batch size
3. Check NCCL settings
4. Optimize data loading

```bash
# Increase batch size
data.train_batch_size=32

# Increase micro batch size
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2

# Check NCCL settings
export NCCL_DEBUG=INFO
```

#### Issue: Poor Scaling

**Symptoms**: Adding GPUs doesn't improve performance

**Solutions**:
1. Check network configuration
2. Optimize communication
3. Adjust batch sizes per GPU
4. Check for bottlenecks

```bash
# Check network configuration
ibstat  # Check InfiniBand status

# Optimize communication
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
```

### General Issues

### Verify Package Structure

```bash
# Test that packages are importable
python -c "import scripts; print('✓ scripts package imported')"
python -c "import scripts.rl_training; print('✓ scripts.rl_training package imported')"
python -c "from scripts.rl_training.react_training import ReactAgentLoop; print('✓ ReactAgentLoop imported')"
```

### Verify Registration

```bash
# Test that agent loop is registered
python -c "
from verl.experimental.agent_loop.agent_loop import _agent_loop_registry
print('Registered agent loops:')
for name in sorted(_agent_loop_registry.keys()):
    print(f'  - {name}')
# Should include: react_agent
"
```

### Test Single GPU

```bash
# Should work without issues
python scripts/rl_training/react_training.py \
    actor_rollout_ref.rollout.agent.default_agent_loop=react_agent \
    # ... other config
```

### Test Multi-GPU

```bash
# Should work with proper registration
python scripts/rl_training/react_training.py \
    actor_rollout_ref.rollout.agent.default_agent_loop=react_agent \
    # ... other config
```

## File Structure

```
scripts/rl_training/
├── README.md                    # This guide
├── react_training.py            # Training entry point + agent loop
├── react_training.sh            # Training script with configuration
├── rl_training.py              # Fold agent training (reference)
├── rl_training.sh              # Fold agent training script (reference)
└── __init__.py                # Python package marker
```

## Best Practices

1. **Always set PYTHONPATH** - Ensure Ray workers can find your modules
2. **Create `__init__.py` files** - Make directories proper Python packages
3. **Import before Ray initialization** - Ensure @register runs in all processes
4. **Test on single GPU first** - Verify registration works before scaling
5. **Use same pattern as built-in agents** - Follow VERL's conventions
6. **Verify registration** - Check that agent loop is in registry

### H200-Specific Best Practices

7. **Use bfloat16** - H200 supports it efficiently
8. **Enable FSDP** - For distributed training
9. **Monitor memory** - Adjust batch sizes accordingly
10. **Optimize communication** - Use NCCL settings
11. **Profile performance** - Identify bottlenecks
12. **Use tensor parallelism** - For large models
13. **Balance load** - Distribute work evenly across GPUs

## References

- [VERL Agent Loop](/Users/rexsasori/FoldAgent/external/verl/verl/experimental/agent_loop/agent_loop.py)
- [VERL Built-in Agent Loops](/Users/rexsasori/FoldAgent/external/verl/verl/experimental/agent_loop/__init__.py)
- [React Agent](/Users/rexsasori/FoldAgent/agents/react_agent.py)
- [Local Search Environment](/Users/rexsasori/FoldAgent/envs/local_search.py)
- [Dockerfile.rl-training](/Users/rexsasori/FoldAgent/docker/Dockerfile.rl-training)
