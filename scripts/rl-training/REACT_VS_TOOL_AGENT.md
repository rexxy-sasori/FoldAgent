# React Agent vs VERL Tool Agent: What You Need

## The Key Insight

**You're absolutely correct - `react_agent` does NOT need `envs/verl_tool_adaptor/`!**

## Two Different Architectures

### React Agent (Your Approach)

```
Training Data → ReactAgentLoop → react_agent.process_item()
    ↓
envs/local_search.py → envs/search_server.py (port 8000)
    ↓
DataProto → VERL Trainer
```

**What it uses:**
- ✅ `envs/local_search.py` - Environment implementation
- ✅ `envs/search_server.py` - Search server
- ✅ `agents/react_agent.py` - Agent logic
- ✅ `agents/prompts.py` - System prompts
- ✅ `agents/tool_spec.py` - Tool definitions

**What it doesn't use:**
- ❌ `envs/verl_tool_adaptor/` - Only for VERL's built-in tool_agent

### VERL's Tool Agent (Built-in)

```
Training Data → ToolAgentLoop (built-in)
    ↓
verl_tool_adaptor/ → envs/search_server.py (port 8000)
    ↓
AgentLoopOutput → VERL Trainer
```

**What it uses:**
- ✅ `envs/verl_tool_adaptor/` - Tool wrappers
- ✅ Tool configuration YAML files
- ✅ VERL's built-in agent loop
- ✅ `envs/search_server.py` - Search server

## Configuration Differences

### React Agent Configuration

```bash
# Simple - just set workflow
actor_rollout_ref.rollout.agent.default_agent_loop=react_agent
actor_rollout_ref.rollout.plugin.workflow=search
actor_rollout_ref.rollout.plugin.max_turn=20
```

### VERL Tool Agent Configuration

```bash
# Complex - needs tool config path and multi-turn settings
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
actor_rollout_ref.rollout.multi_turn.tool_config_path=/app/envs/verl_tool_adaptor/local_search/local_search.yaml
actor_rollout_ref.rollout.multi_turn.format=gpt_oss
actor_rollout_ref.rollout.multi_turn.max_user_turns=10
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10
actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
```

## Why Two Approaches?

### React Agent Advantages
- **Simpler configuration** - Just set `workflow=search`
- **More control** - Direct access to environment
- **Custom logic** - Implement your own agent behavior
- **Less overhead** - No tool wrapper layer

### VERL Tool Agent Advantages
- **Standardized** - Uses VERL's tool system
- **Multi-tool support** - Easy to add new tools
- **Parallel execution** - Built-in support for parallel tool calls
- **Better for standard cases** - When you just need tool access

## File Comparison

### React Agent Files

```
agents/
├── react_agent.py          # Agent implementation
├── prompts.py              # System prompts
├── tool_spec.py            # Tool definitions
└── utils.py               # Utilities (imports envs.local_search)

envs/
├── local_search.py         # Environment (direct server calls)
└── search_server.py        # Search server (port 8000)

scripts/rl-training/
├── react_agent_loop.py     # VERL integration
├── react_training.py       # Training entry point
└── react_training.sh       # Training script
```

### VERL Tool Agent Files

```
envs/verl_tool_adaptor/
├── local_search/
│   ├── local_search.yaml   # Tool configuration
│   └── local_search_tool.py # Tool wrapper
└── system_prompt_template.txt

# Plus all the above files
```

## Training Script Comparison

### React Agent (Simplified)

```bash
python3 scripts/rl-training/react_training.py \
algorithm.adv_estimator=foldgrpo \
actor_rollout_ref.rollout.agent.default_agent_loop=react_agent \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.mode=async \
actor_rollout_ref.rollout.calculate_log_probs=True \
actor_rollout_ref.rollout.plugin.workflow=search \
actor_rollout_ref.rollout.plugin.max_turn=20 \
actor_rollout_ref.rollout.plugin.val_max_turn=20 \
data.apply_chat_template_kwargs.add_generation_prompt=True \
# ... rest of config
```

### VERL Tool Agent (Complex)

```bash
python3 scripts/rl-training/rl_training.py \
algorithm.adv_estimator=foldgrpo \
actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.mode=async \
actor_rollout_ref.rollout.calculate_log_probs=True \
actor_rollout_ref.rollout.multi_turn.tool_config_path=/app/envs/verl_tool_adaptor/local_search/local_search.yaml \
actor_rollout_ref.rollout.multi_turn.format=gpt_oss \
actor_rollout_ref.rollout.multi_turn.max_user_turns=10 \
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=10 \
actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=right \
data.apply_chat_template_kwargs.add_generation_prompt=True \
# ... rest of config
```

## Summary

| Aspect | React Agent | VERL Tool Agent |
|---------|-------------|------------------|
| **Tool Integration** | Direct via `envs/local_search.py` | Via `verl_tool_adaptor` wrapper |
| **Configuration** | Simple (`workflow=search`) | Complex (tool config path + multi-turn settings) |
| **Files Needed** | `envs/local_search.py` | `envs/verl_tool_adaptor/` + YAML configs |
| **Control** | High (custom agent logic) | Medium (built-in agent loop) |
| **Best For** | Custom agent behavior | Standard tool use cases |

## Recommendation

**Use React Agent when:**
- ✅ You want custom agent logic
- ✅ You need simpler configuration
- ✅ You're building a research agent
- ✅ You want direct control over environment

**Use VERL Tool Agent when:**
- ✅ You need standard tool access
- ✅ You want to use VERL's built-in features
- ✅ You need multi-tool support
- ✅ You prefer standardized configuration

## For Your Use Case

Since you're implementing a **react agent** for research tasks:

✅ **You need:**
- `envs/local_search.py` - Already exists
- `envs/search_server.py` - Already exists
- `agents/react_agent.py` - Already exists
- Training script with `workflow=search` - Created

❌ **You don't need:**
- `envs/verl_tool_adaptor/` - Delete or ignore
- Tool configuration YAML files - Not needed
- `multi_turn.tool_config_path` - Not needed
- `multi_turn.format` - Not needed

**Your setup is correct!** Just run the training script and you're good to go.
