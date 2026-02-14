# Type Conversion: DataProto ↔ AgentLoopOutput

## The Problem

When implementing custom agent loops for VERL training, there's a type mismatch:

- **`process_item()`** (from `react_agent.py` and `fold_agent.py`) returns `DataProto`
- **`AgentLoopBase.run()`** (from VERL) expects `AgentLoopOutput`

## Why This Happens

### DataProto
Used internally by the agent implementations to store training data:
```python
DataProto(
    batch={
        'input_ids': [...],           # Full conversation tokens
        'attention_mask': [...],        # Attention mask
        'rollout_behavior_log_probs': [...],  # Log probabilities
        'model_output_mask': [...],    # Which tokens are model-generated
        # ... other fields
    },
    non_tensor_batch={...},
    meta_info={...}
)
```

### AgentLoopOutput
Used by VERL's agent loop system for trajectory generation:
```python
AgentLoopOutput(
    prompt_ids=[...],              # Prompt tokens only
    response_ids=[...],            # Response tokens only
    response_mask=[...],            # 1 for LLM tokens, 0 for tool responses
    response_logprobs=[...],       # Log probabilities for response
    num_turns=5,                   # Number of conversation turns
    metrics=AgentLoopMetrics(...),   # Performance metrics
    reward_score=0.8,              # Optional reward score
    extra_fields={...}              # Additional metadata
)
```

## The Solution

The `ReactAgentLoop` (and similarly `FoldAgentLoop`) includes a conversion method:

```python
def _convert_dataproto_to_agentloopoutput(
    self, 
    data_proto: DataProto, 
    kwargs: dict
) -> AgentLoopOutput:
    """Convert DataProto from process_item() to AgentLoopOutput format."""
    batch = data_proto.batch
    
    # Split input_ids into prompt and response
    prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
    response_length = self.config.actor_rollout_ref.rollout.response_length
    
    input_ids = batch['input_ids'][0]
    prompt_ids = input_ids[:prompt_length].tolist()
    response_ids = input_ids[prompt_length:prompt_length + response_length].tolist()
    
    # Create response mask (all 1s for react agent)
    response_mask = [1] * len(response_ids)
    
    # Extract log probabilities
    response_logprobs = None
    if 'rollout_behavior_log_probs' in batch:
        log_probs = batch['rollout_behavior_log_probs'][0]
        response_logprobs = log_probs[:len(response_ids)].tolist()
    
    # Extract metadata
    reward_score = None
    num_turns = 1
    if 'extra_data' in data_proto.non_tensor_batch:
        extra_data = data_proto.non_tensor_batch['extra_data'][0]
        if isinstance(extra_data, dict):
            reward_score = extra_data.get('reward', None)
            if 'stats' in extra_data:
                num_turns = extra_data['stats'].get('total_turns', 1)
    
    return AgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        response_mask=response_mask,
        response_logprobs=response_logprobs,
        num_turns=num_turns,
        metrics=AgentLoopMetrics(generate_sequences=0.0, tool_calls=0.0),
        reward_score=reward_score,
        extra_fields={}
    )
```

## Key Differences in Structure

| Aspect | DataProto | AgentLoopOutput |
|---------|-----------|-----------------|
| **Purpose** | Internal training data | VERL agent loop interface |
| **Token Split** | Combined `input_ids` | Separate `prompt_ids` + `response_ids` |
| **Mask** | `attention_mask` + `model_output_mask` | Single `response_mask` |
| **Log Probs** | `rollout_behavior_log_probs` | `response_logprobs` |
| **Metadata** | `non_tensor_batch['extra_data']` | `reward_score`, `num_turns`, `metrics` |

## Flow Through VERL

```
1. ReactAgentLoop.run()
   ↓
2. process_item() → DataProto
   ↓
3. _convert_dataproto_to_agentloopoutput() → AgentLoopOutput
   ↓
4. VERL's _agent_loop_postprocess() → _InternalAgentLoopOutput
   ↓
5. VERL's _postprocess() → DataProto (for training)
```

## Why This Two-Step Conversion?

VERL's agent loop system needs `AgentLoopOutput` because:
- It's designed for multi-turn conversations with tool calls
- It separates prompt from response for efficient processing
- It tracks turn-level metrics
- It handles parallel tool execution

But VERL's training system needs `DataProto` because:
- It's the standard data format for all VERL components
- It supports batch operations
- It integrates with reward managers, actors, and critics

## Implementation Notes

### For React Agent
- All response tokens are LLM-generated → `response_mask = [1, 1, 1, ...]`
- No parallel tool calls → simple sequential flow
- Response mask doesn't need to track tool responses

### For Fold Agent
- Response tokens include tool responses → `response_mask = [1, 1, 0, 0, 1, 1, ...]`
- Parallel branch agents → more complex turn tracking
- Response mask must distinguish LLM from tool tokens

### For Tool Agent (VERL's built-in)
- Already returns `AgentLoopOutput` directly
- No conversion needed
- Handles complex multi-turn scenarios automatically

## Debugging Tips

### Issue: Type mismatch error
```
TypeError: Expected AgentLoopOutput, got DataProto
```
**Solution**: Ensure your agent loop's `run()` method calls the conversion function:
```python
async def run(self, sampling_params, **kwargs):
    data_proto = await process_item(item, context)
    return self._convert_dataproto_to_agentloopoutput(data_proto, kwargs)
```

### Issue: Incorrect prompt/response split
**Symptoms**: Training loss is NaN or very high
**Solution**: Verify prompt_length and response_length match your config:
```python
prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
response_length = self.config.actor_rollout_ref.rollout.response_length
```

### Issue: Missing log probabilities
**Symptoms**: PPO ratio computation fails
**Solution**: Ensure `calculate_log_probs=True` in config:
```bash
actor_rollout_ref.rollout.calculate_log_probs=True
```

## References

- **VERL Agent Loop**: `/Users/rexsasori/FoldAgent/external/verl/verl/experimental/agent_loop/agent_loop.py`
- **React Agent**: `/Users/rexsasori/FoldAgent/agents/react_agent.py`
- **Fold Agent**: `/Users/rexsasori/FoldAgent/agents/fold_agent.py`
- **DataProto**: `/Users/rexsasori/FoldAgent/external/verl/verl/protocol.py`
