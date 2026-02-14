# VERL Tools Configuration for Local Search

This directory contains VERL-compatible tool configurations and handlers for integrating the local search server with VERL's agent loop system.

## Files

- `local_search.yaml` - Tool configuration in VERL format
- `local_search_tool.py` - Tool handler implementation
- `README.md` - This file

## Setup

### 1. Start Search Server

```bash
cd /Users/rexsasori/FoldAgent/envs
python search_server.py --host 0.0.0.0 --port 8000
```

### 2. Configure Tool Agent

Add these parameters to your training script (`rl_training.sh`):

```bash
# Tool agent configuration
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
actor_rollout_ref.rollout.multi_turn.tool_config_path=/app/envs/verl-tools-config/local_search.yaml \
actor_rollout_ref.rollout.multi_turn.format=gpt_oss \
actor_rollout_ref.rollout.multi_turn.max_user_turns=10 \
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10 \
actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right
```

### 3. Add Search Server URL

Add to your Kubernetes job configuration (`rl-training-job.yaml`):

```yaml
env:
  - name: SEARCH_SERVER_URL
    value: "http://search-server:8000"
```

### 4. Update Training Script

Add to `rl_training.sh`:

```bash
# Add search server URL to environment
export SEARCH_SERVER_URL=${SEARCH_SERVER_URL:-"http://localhost:8000"}

# The tool agent will automatically load tools from the config path
```

## Tool Configuration

### Search Tool

- **Name**: `search`
- **Description**: Search for relevant information in corpus
- **Parameters**:
  - `query` (string): The search query
  - `k` (integer): Number of results (default: 20, range: 1-100)

### Open Page Tool

- **Name**: `open_page`
- **Description**: Open a specific document by docid or URL
- **Parameters**:
  - `docid` (string): Document ID (optional)
  - `url` (string): URL to open (optional)
  - **Note**: At least one of `docid` or `url` must be provided

## Usage Example

### In Training Script

```bash
python3 scripts/rl_training.py \
  algorithm.adv_estimator=foldgrpo \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.multi_turn.tool_config_path=/app/envs/verl-tools-config/local_search.yaml \
  actor_rollout_ref.rollout.multi_turn.format=gpt_oss \
  # ... other parameters
```

### Tool Call Format (OpenAI-compatible)

The model will generate tool calls in this format:

```json
{
  "name": "search",
  "arguments": {
    "query": "machine learning algorithms",
    "k": 10
  }
}
```

## Testing

### Test Search Server

```bash
# Health check
curl http://localhost:8000/health

# Test search
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "k": 5}'

# Test open page
curl -X POST http://localhost:8000/open \
  -H "Content-Type: application/json" \
  -d '{"docid": "doc123"}'
```

### Test Tool Integration

1. Deploy the updated job with tool_agent configuration
2. Monitor logs for tool initialization
3. Verify tool calls are being made to search server
4. Check that search results are being used in responses

## Troubleshooting

### Search Server Not Reachable

- Check if search server is running: `curl http://localhost:8000/health`
- Verify network connectivity from training pod
- Check firewall rules and DNS resolution

### Tool Not Found

- Verify tool config path is correct
- Check YAML syntax: `python -c "import yaml; yaml.safe_load(open('/app/envs/verl-tools-config/local_search.yaml'))"`
- Ensure tools are properly registered in VERL

### Tool Call Fails

- Check search server logs for errors
- Verify tool handler implementation
- Check network connectivity and timeouts
- Review VERL agent loop logs for detailed error messages

## Next Steps

1. **Phase 1**: Test tool_agent with local search to verify model improvement
2. **Phase 2**: Convert fold_agent logic to VERL-compatible implementation
3. **Phase 3**: Optimize tool usage and reward calculation

## References

- VERL Agent Loop Documentation: `/Users/rexsasori/FoldAgent/external/verl/verl/experimental/agent_loop/`
- Search Server Implementation: `/Users/rexsasori/FoldAgent/envs/search_server.py`
- Tool Agent Implementation: `/Users/rexsasori/FoldAgent/external/verl/verl/experimental/agent_loop/tool_agent_loop.py`
