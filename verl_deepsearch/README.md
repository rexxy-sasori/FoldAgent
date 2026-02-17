# DeepSearch RLHF Training Components

This package contains custom components for training ReAct agents with search tool integration and remote judge-based reward computation in the VERL framework.

## Components

### 1. ReActAgentLoop (`react_agent_loop.py`)

Custom agent loop implementation that inherits from `verl.experimental.agent_loop.AgentLoopBase`.

**Features:**
- Multi-turn ReAct-style reasoning with tool use
- Search tool integration via HTTP to LOCAL_SEARCH_URL
- Open-page tool for fetching full document content
- Automatic termination on `<function=finish>` pattern
- Session timeout and max turn limits
- Optional conversation summarization for long contexts
- Comprehensive metrics tracking (generation time, search latency, tool calls)

**Key Methods:**
- `run()`: Main async execution loop with state machine
- `_execute_search()`: HTTP POST to search server
- `_execute_open_page()`: HTTP POST to open page endpoint
- `_detect_tool_calls()`: Regex-based tool pattern detection
- `_check_termination()`: Detect finish pattern

**Configuration:**
- `max_turns`: Maximum turns during training (default: 64)
- `val_max_turns`: Maximum turns during validation (default: 200)
- `session_timeout`: Session timeout in seconds (default: 5400)
- `max_tool_response_length`: Max characters for tool responses (default: 1000)
- `enable_summary`: Enable conversation summarization (default: false)

**Environment Variables:**
- `LOCAL_SEARCH_URL`: Search server endpoint (default: http://localhost:8000)
- `VERL_LOGGING_LEVEL`: Logging level (default: WARN)

### 2. JudgeRewardManager (`judge_reward_manager.py`)

Custom reward manager that inherits from `verl.workers.reward_manager.AbstractRewardManager`.

**Features:**
- Async HTTP calls to remote judge API (OpenAI-compatible)
- Batch processing with concurrent requests
- Structured score parsing with regex
- Comprehensive error handling and retry logic
- Batch-level metrics with gen_uid deduplication

**Key Methods:**
- `__call__()`: Main entry point for reward computation
- `_call_judge()`: Async API call to judge with retry logic
- `_process_single_item()`: Process individual trajectory
- `_compute_batch_metrics()`: Compute batch-level statistics

**Configuration:**
- `JUDGE_OPENAI_API_KEY`: API key for judge service
- `JUDGE_OPENAI_BASE_URL`: Base URL (default: https://lonlie.plus7.plus/v1)
- `JUDGE_OPENAI_MODEL`: Model name (default: gpt-4.1)
- `JUDGE_OPENAI_URL`: Full endpoint URL

**Prompt Template:**
Uses structured grader template with fields:
- `extracted_final_answer`: Extracted answer from response
- `correct`: Binary yes/no judgment
- `reasoning`: Explanation of judgment
- `confidence`: Confidence score (0-100%)

**Score Mapping:**
- `correct="yes"` → score = 1.0
- `correct="no"` → score = 0.0
- Parse error/timeout → score = 0.0

## Configuration Files

### `configs/react_agent.yaml`
Agent loop registration file:
```yaml
- name: react_agent
  _target_: verl_deepsearch.deepsearch_training.react_agent_loop.ReActAgentLoop
```

### `configs/deepsearch_ppo_trainer.yaml`
Main training configuration with:
- Agent loop configuration
- Reward manager configuration
- Multi-turn settings
- PPO hyperparameters
- Environment variable injection

## Launch Script

### `scripts/rl_training/launch_configs/deepsearch_training.sh`

Bash script for launching training with:
- Environment variable setup
- WandB integration
- Data path configuration
- Model path configuration
- All training hyperparameters

**Usage:**
```bash
export LOCAL_SEARCH_URL=http://search-server.liuyunxin:8000
export JUDGE_OPENAI_API_KEY=your_api_key
export WANDB_API_KEY=your_wandb_key

bash scripts/rl_training/launch_configs/deepsearch_training.sh
```

## Integration with VERL

### Registration

The components are automatically registered with VERL's registry system:

```python
# Agent loop
@register("react_agent")
class ReActAgentLoop(AgentLoopBase):
    ...

# Reward manager
@register("judge")
class JudgeRewardManager(AbstractRewardManager):
    ...
```

### Configuration Flow

1. Hydra loads `deepsearch_ppo_trainer.yaml`
2. Agent loop registered from `react_agent.yaml`
3. Reward manager configured with environment variables
4. Training loop uses:
   - `ReActAgentLoop` for rollout
   - `JudgeRewardManager` for reward computation

### Data Flow

```
Dataset Item
    ↓
ReActAgentLoop.run()
    ├── Generate LLM response
    ├── Detect tool calls
    ├── Execute search/open_page tools
    ├── Update chat history
    └── Return AgentLoopOutput
    ↓
JudgeRewardManager.__call__()
    ├── Decode trajectory
    ├── Construct judge prompt
    ├── Call judge API
    ├── Parse score
    └── Return reward tensor
    ↓
PPO Training
```

## Dependencies

- `verl`: VERL framework
- `aiohttp`: Async HTTP client for tool execution
- `openai`: Async OpenAI client for judge API
- `agents.prompts`: System prompts and tool descriptions
- `torch`: PyTorch for tensor operations
- `numpy`: Numerical operations

## Error Handling

### ReActAgentLoop
- Search server timeout: 120 seconds
- Session timeout: Configurable (default: 90 minutes)
- Max turn limits: Prevent infinite loops
- Response length limits: Prevent token overflow
- Graceful degradation on errors

### JudgeRewardManager
- API timeout: 300 seconds
- Retry logic: 3 attempts with exponential backoff
- Parse error handling: Default to 0.0 score
- Batch isolation: One failure doesn't affect others

## Performance Considerations

### ReActAgentLoop
- Async tool execution to avoid blocking
- Efficient regex compilation (class-level)
- Connection pooling for HTTP clients
- Metrics tracking for optimization

### JudgeRewardManager
- Concurrent API calls for batch processing
- Connection pooling in AsyncOpenAI client
- Rate limiting awareness
- Batch-level metrics for monitoring

## Logging

Both components use VERL's logging system:
- `VERL_LOGGING_LEVEL`: Control verbosity (WARN, INFO, DEBUG)
- Structured logging with request IDs
- Performance metrics logging
- Error tracking and debugging

## Example Training Command

```bash
# Set environment variables
export LOCAL_SEARCH_URL=http://search-server.liuyunxin:8000
export JUDGE_OPENAI_API_KEY=sk-...
export JUDGE_OPENAI_MODEL=gpt-4.1
export WANDB_API_KEY=your_wandb_key

# Launch training
python3 scripts/rl_training/main.py \
  --config-name deepsearch_ppo_trainer \
  --config-path verl_deepsearch/deepsearch_training/configs
```

## Architecture Notes

### State Machine (ReActAgentLoop)
```
INITIALIZING → GENERATING → TOOL_DETECTION → TOOL_EXECUTION → GENERATING → ... → TERMINATED
```

### Batch Processing (JudgeRewardManager)
- Concurrent API calls using `asyncio.gather()`
- Error isolation per trajectory
- Metrics aggregation across batch
- Gen_uid deduplication for multi-rollout scenarios

## Troubleshooting

### Common Issues

1. **Search server unavailable**
   - Check `LOCAL_SEARCH_URL` is correct
   - Verify search server is running
   - Check network connectivity

2. **Judge API errors**
   - Verify `JUDGE_OPENAI_API_KEY` is set
   - Check `JUDGE_OPENAI_BASE_URL` is accessible
   - Review judge API logs for specific errors

3. **Training not starting**
   - Verify agent loop is registered
   - Check reward manager is registered
   - Review configuration file syntax
   - Check PYTHONPATH includes verl_deepsearch

## License

Copyright 2025 DeepSearch Contributors
Licensed under the Apache License, Version 2.0