# React Agent Training Guide

This guide explains how to train a React (Reason + Act) agent using VERL framework.

## Overview

The React agent is a simpler alternative to the Fold (Context-Folding) agent. It follows a straightforward ReAct pattern:
- **Reason**: Think about the problem
- **Act**: Take actions using tools
- **Observe**: See the results
- **Repeat**: Continue until answer found or limit reached

## Key Differences: React Agent vs VERL's Tool Agent

| Aspect | React Agent | VERL's Tool Agent |
|---------|-------------|---------------------|
| **Architecture** | Custom agent loop | Built-in VERL agent loop |
| **Tool Integration** | Uses `envs/local_search.py` directly | Uses `verl_tool_adaptor` wrapper |
| **Configuration** | `workflow=search` | `multi_turn.tool_config_path=...` |
| **Complexity** | Simpler, more control | More complex, more automated |
| **Best For** | Custom agent logic | Standard tool use cases |

### Why Two Different Approaches?

**React Agent** (your approach):
- Uses `envs/local_search.py` environment directly
- Calls search server at `http://localhost:8000`
- More control over agent behavior
- Simpler configuration

**VERL's Tool Agent**:
- Uses `verl_tool_adaptor` wrapper
- Requires tool configuration YAML files
- More automated tool management
- Better for standard tool use cases

### What You Need for React Agent Training

✅ **Required**:
- `envs/local_search.py` - Environment implementation
- `envs/search_server.py` - Search server (port 8000)
- Training data with `workflow=search` in extra_info

❌ **Not Required**:
- `envs/verl_tool_adaptor/` - Only for VERL's built-in tool_agent
- Tool configuration YAML files
- `multi_turn.tool_config_path` parameter

| Feature | React Agent | Fold Agent |
|----------|-------------|------------|
| **Complexity** | Simple, single-agent loop | Complex, multi-branch architecture |
| **Parallel Execution** | Sequential tool calls | Parallel branch agents |
| **Context Management** | Linear conversation growth | Context folding with summaries |
| **Best For** | Simple research tasks | Complex, multi-step research |
| **Training Speed** | Faster | Slower (more overhead) |

## Files Created

1. **`scripts/rl-training/react_agent_loop.py`** - Agent loop implementation
2. **`scripts/rl-training/react_training.py`** - Training entry point
3. **`scripts/rl-training/react_training.sh`** - Training shell script

## How It Works

### 1. Agent Loop Registration

The `ReactAgentLoop` class is registered with VERL using the `@register("react_agent")` decorator. This allows VERL to instantiate and use it during training.

### 2. Data Flow

```
Training Data → ReactAgentLoop → react_agent.process_item() → DataProto 
→ Convert to AgentLoopOutput → VERL Trainer → DataProto
```

**Important Type Conversion**: The `process_item()` function returns `DataProto`, but VERL's `AgentLoopBase.run()` method expects `AgentLoopOutput`. The `ReactAgentLoop` includes a conversion method `_convert_dataproto_to_agentloopoutput()` to handle this.

### 3. Key Components

**ReactAgentLoop** (`react_agent_loop.py`):
- Wraps `react_agent.process_item()` for VERL integration
- Creates LLM client and task context
- Returns `DataProto` with training data

**react_agent.process_item()** (`agents/react_agent.py`):
- Initializes environment (LocalSearch)
- Creates system prompt based on workflow
- Runs ReAct loop: think → act → observe
- Computes rewards
- Returns `DataProto` with trajectory data

## Configuration

### Critical Parameters

```bash
# Agent loop type
actor_rollout_ref.rollout.agent.default_agent_loop=react_agent

# Workflow type (determines system prompt and tools)
actor_rollout_ref.rollout.plugin.workflow=search

# Max turns for training/validation
actor_rollout_ref.rollout.plugin.max_turn=20
actor_rollout_ref.rollout.plugin.val_max_turn=20

# Model and rollout settings
actor_rollout_ref.rollout.name=vllm
actor_rollout_ref.rollout.mode=async
actor_rollout_ref.rollout.calculate_log_probs=True
```

### Workflow Options

The `workflow` parameter determines which system prompt and tools are used:

- **`search`** - Standard search workflow with search/open_page/finish tools
- **`search_base`** - Alternative search workflow (BrowseComp-Plus baseline)
- **`search_multi`** - Multi-question search workflow
- **`code`** - Code editing workflow (execute_bash/str_replace_editor/think/finish)

## Running Training

### Prerequisites

1. **Start Search Server**:
```bash
# Start the local search server (envs/search_server.py)
# This is the only external service needed for react_agent
python envs/search_server.py
# Server runs on http://localhost:8000
```

2. **Prepare Data**:
```bash
# Ensure training data exists
ls /root/react-training/data/bc_train.parquet
```

**Note**: Unlike VERL's built-in `tool_agent`, `react_agent` does NOT need `verl_tool_adaptor`. It directly uses `envs/local_search.py` which connects to the search server.

### Training Command

```bash
cd /Users/rexsasori/FoldAgent
bash scripts/rl-training/react_training.sh
```

### Environment Variables

The script sets these important variables:

```bash
WANDB_DIR=/root/react-training/wandb          # Weights & Biases logs
LOG_DIR=/root/react-training/logs              # Training logs
CHECKPOINT_DIR=/root/react-training/checkpoints # Model checkpoints
SEARCH_SERVER_URL=http://localhost:8000       # Search server URL
```

## Training Data Format

Your training data should be a parquet file with:

```python
{
    'ability': 'LocalSearch',           # Environment type
    'extra_info': {
        'instance_id': 'unique_id',
        'workflow': 'search',           # Workflow type
        'problem_statement': 'question...'
    },
    'uid': 'unique_id',
    'reward_model': {
        'ground_truth': 'expected_answer'
    },
    'data_source': 'easy|medium|hard'  # Difficulty level
}
```

## Monitoring Training

### Logs

- **Training logs**: `/root/react-training/logs/`
- **W&B dashboard**: Check your Weights & Biases project

### Key Metrics

- **Reward score**: Average reward per trajectory
- **Success rate**: Percentage of tasks completed successfully
- **Turn count**: Average number of turns per task
- **Context length**: Average conversation length

## Troubleshooting

### Issue: Agent not using search tools

**Solution**: Ensure `workflow=search` is set in config:
```bash
actor_rollout_ref.rollout.plugin.workflow=search
```

### Issue: Training data not found

**Solution**: Check data paths in shell script:
```bash
TRAIN_DATA_PATH=/root/react-training/data/bc_train.parquet
TEST_DATA_PATH=/root/react-training/data/bc_test.parquet
```

### Issue: Search server connection failed

**Solution**: Ensure search server is running and URL is correct:
```bash
export SEARCH_SERVER_URL=http://localhost:8000
# Start search server
```

### Issue: Out of memory during training

**Solution**: Reduce batch sizes or sequence lengths:
```bash
data.train_batch_size=8  # Reduce from 16
actor_rollout_ref.rollout.prompt_length=8192  # Reduce from 16384
actor_rollout_ref.rollout.response_length=16384  # Reduce from 32768
```

## Advanced Configuration

### Custom System Prompt

To use a custom system prompt, modify `agents/prompts.py` or preprocess your data:

```python
# In preprocess_data_with_system_prompt.py
system_prompt = "Your custom system prompt here..."
```

### Adjusting Max Turns

For longer/shorter conversations:

```bash
# Training
actor_rollout_ref.rollout.plugin.max_turn=50

# Validation
actor_rollout_ref.rollout.plugin.val_max_turn=50
```

### Enabling Summary Mode

For very long conversations, enable summary mode:

```bash
actor_rollout_ref.rollout.plugin.enable_summary=True
```

This will create summaries when context approaches limits.

## Next Steps

1. **Run evaluation** to test trained model:
```bash
python scripts/serving/eval_bc.py \
  --workflow search \
  --model_path /root/react-training/checkpoints/step_100 \
  --data_path /root/react-training/data/bc_test.parquet
```

2. **Fine-tune hyperparameters** based on training results

3. **Experiment with different workflows** (search, code, etc.)

## References

- **VERL Documentation**: `/Users/rexsasori/FoldAgent/external/verl/docs/`
- **React Agent Implementation**: `/Users/rexsasori/FoldAgent/agents/react_agent.py`
- **Fold Agent (for comparison)**: `/Users/rexsasori/FoldAgent/agents/fold_agent.py`
- **System Prompts**: `/Users/rexsasori/FoldAgent/agents/prompts.py`
- **Tool Specifications**: `/Users/rexsasori/FoldAgent/agents/tool_spec.py`
