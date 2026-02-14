# VERL Tool Adaptor

This directory contains VERL-compatible tool implementations for integrating external services with VERL's agent loop system.

## Directory Structure

```
verl_tool_adaptor/
├── __init__.py                    # Main package initialization
├── local_search/                  # Local search tool implementation
│   ├── __init__.py               # Local search package initialization
│   ├── local_search.yaml          # Tool configuration
│   ├── local_search_tool.py       # Tool implementation
│   └── README.md                # Tool-specific documentation
├── code_execution/               # (Future) Code execution tool
├── web_browsing/                # (Future) Web browsing tool
└── README.md                    # This file
```

## Adding New Tools

### Step 1: Create Tool Directory

```bash
mkdir -p /Users/rexsasori/FoldAgent/envs/verl_tool_adaptor/your_tool_name
```

### Step 2: Create Tool Implementation

Create `your_tool_name/your_tool.py`:

```python
import aiohttp
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class YourTool:
    def __init__(self, config: Dict[str, Any], tool_schema: Optional[Any] = None):
        self.config = config
        self.tool_schema = tool_schema
        logger.info(f"Initialized YourTool with config: {config}")
    
    async def __call__(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        try:
            if tool_name == "your_tool_action":
                return await self._your_action(**kwargs)
            else:
                return {
                    "tool_call_id": tool_name,
                    "response": {"error": f"Unknown tool: {tool_name}"}
                }
        except Exception as e:
            logger.error(f"Error in {tool_name}: {str(e)}")
            return {
                "tool_call_id": tool_name,
                "response": {"error": f"Tool execution failed: {str(e)}"}
            }
    
    async def _your_action(self, param1: str, param2: int) -> Dict[str, Any]:
        # Your tool implementation
        return {
            "tool_call_id": "your_tool_action",
            "response": {"result": "success"}
        }
```

### Step 3: Create Package Initialization

Create `your_tool_name/__init__.py`:

```python
from .your_tool import YourTool

__all__ = ['YourTool']
```

### Step 4: Create Tool Configuration

Create `your_tool_name/your_tool.yaml`:

```yaml
tools:
  - name: your_tool_action
    class_name: envs.verl_tool_adaptor.your_tool_name.YourTool
    config:
      type: native
      your_param: ${YOUR_ENV_VAR:-"default_value"}
    tool_schema:
      type: function
      function:
        name: your_tool_action
        description: Description of what this tool does
        parameters:
          type: object
          properties:
            param1:
              type: string
              description: First parameter
            param2:
              type: integer
              description: Second parameter
          required:
            - param1
```

### Step 5: Update Main Package

Update `__init__.py` in the root directory:

```python
from .local_search import LocalSearchTool
from .your_tool_name import YourTool

__all__ = ['LocalSearchTool', 'YourTool']
```

### Step 6: Update Training Script

Add to `scripts/rl_training.sh`:

```bash
actor_rollout_ref.rollout.multi_turn.tool_config_path=/app/envs/verl_tool_adaptor/your_tool_name/your_tool.yaml \
```

## Tool Interface Requirements

All tools must implement:

1. **`__init__(self, config: Dict[str, Any], tool_schema: Optional[Any] = None)`**
   - Initialize tool with configuration from YAML
   - Store config and tool_schema for later use

2. **`async def __call__(self, tool_name: str, **kwargs) -> Dict[str, Any]`**
   - Main entry point for tool execution
   - Route to appropriate method based on tool_name
   - Return response in VERL format:
     ```python
     {
         "tool_call_id": "tool_name",
         "response": {"result": "..."}  # or {"error": "..."}
     }
     ```

3. **Error Handling**
   - Catch and log all exceptions
   - Return error responses in standard format
   - Include tool_call_id in all responses

## Configuration Format

### Tool Configuration (YAML)

```yaml
tools:
  - name: tool_name
    class_name: envs.verl_tool_adaptor.tool_directory.ToolClass
    config:
      type: native
      param1: value1
      param2: ${ENV_VAR:-"default"}
    tool_schema:
      type: function
      function:
        name: tool_name
        description: Tool description
        parameters:
          type: object
          properties:
            param1:
              type: string
              description: Parameter description
          required:
            - param1
```

### Environment Variables

Use environment variables in YAML config:

```yaml
config:
  server_url: ${SEARCH_SERVER_URL:-"http://localhost:8000"}
  api_key: ${API_KEY:-"default_key"}
```

## Available Tools

### Local Search

**Directory**: `local_search/`

**Tools**:
- `search` - Search for relevant documents
- `open_page` - Open documents by docid or URL

**Configuration**: `/app/envs/verl_tool_adaptor/local_search/local_search.yaml`

**Backend**: `/Users/rexsasori/FoldAgent/envs/search_server.py`

**Documentation**: [local_search/README.md](local_search/README.md)

## Testing Tools

### Test Tool Configuration

```bash
python -c "import yaml; yaml.safe_load(open('/app/envs/verl_tool_adaptor/local_search/local_search.yaml'))"
```

### Test Tool Import

```bash
python -c "from envs.verl_tool_adaptor.local_search import LocalSearchTool; print(LocalSearchTool)"
```

### Test Tool Execution

```python
import asyncio
from envs.verl_tool_adaptor.local_search import LocalSearchTool

async def test():
    tool = LocalSearchTool(config={"search_server_url": "http://localhost:8000"})
    result = await tool(tool_name="search", query="test", k=5)
    print(result)

asyncio.run(test())
```

## Troubleshooting

### Import Error: Module Not Found

**Problem**: Cannot import tool class

**Solution**: Ensure `__init__.py` files exist in all directories

### Tool Not Found

**Problem**: VERL cannot find tool

**Solution**: Check `class_name` in YAML matches actual import path

### Configuration Error

**Problem**: YAML syntax error

**Solution**: Validate YAML with `python -c "import yaml; yaml.safe_load(open('config.yaml'))"`

### Tool Execution Failed

**Problem**: Tool returns error

**Solution**: Check logs for detailed error messages, verify backend service is running

## Best Practices

1. **Async Operations**: Always use async/await for I/O operations
2. **Error Handling**: Catch all exceptions and return error responses
3. **Logging**: Log important events for debugging
4. **Timeouts**: Set reasonable timeouts for external requests
5. **Environment Variables**: Use environment variables for configuration
6. **Documentation**: Document each tool with README.md
7. **Testing**: Test tools independently before integration

## References

- VERL Agent Loop: `/Users/rexsasori/FoldAgent/external/verl/verl/experimental/agent_loop/`
- VERL Tool Registry: `/Users/rexsasori/FoldAgent/external/verl/verl/tools/utils/tool_registry.py`
- VERL Tool Schemas: `/Users/rexsasori/FoldAgent/external/verl/verl/tools/schemas/`
