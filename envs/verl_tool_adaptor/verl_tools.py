import json
import logging
import os
from typing import Any, Optional
from uuid import uuid4

import httpx

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SearchTool(BaseTool):
    """Search tool for retrieving information using FoldAgent's search_server backend.

    This tool provides search functionality by calling the FoldAgent search_server
    which performs semantic search over a corpus of documents.

    Methods:
        get_openai_tool_schema: Return tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute search tool
        calc_reward: Calculate reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize SearchTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Expected config keys:
            - search_server_url: URL of the search server (default: http://localhost:8000)
            - timeout: Request timeout in seconds (default: 30)
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Search server configuration
        self.search_server_url = config.get("search_server_url", os.getenv("SEARCH_SERVER_URL", "http://localhost:8000"))
        self.timeout = config.get("timeout", 30)

        logger.info(f"Initialized SearchTool with search_server_url: {self.search_server_url}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return OpenAI tool schema."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
            tool_creation_response: The response of the tool when creating instance.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "search_history": [],
            "reward": 0.0,
        }
        return instance_id, ToolResponse()

    async def execute_search(self, query: str, k: int = 10) -> dict:
        """Execute search operation using search_server.

        Args:
            query: Search query string
            k: Number of top results to return

        Returns:
            Dictionary containing search results
        """
        url = f"{self.search_server_url}/search"
        payload = {"query": query, "k": k}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                logger.debug(f"Search result for query '{query}': {len(data.get('results', []))} results")
                return data
        except httpx.HTTPError as e:
            logger.error(f"Search request failed: {e}")
            return {"results": [], "error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            return {"results": [], "error": str(e)}

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute search tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing 'query' and optional 'k'

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response text of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        query = parameters.get("query")
        k = parameters.get("k", 10)

        if not query:
            error_msg = "Error: 'query' parameter is missing in parameters."
            logger.error(f"[SearchTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=json.dumps({"error": error_msg})), 0.0, {}

        # Execute search
        result_data = await self.execute_search(query, k)

        # Format results as text
        if "error" in result_data:
            result_text = json.dumps({"error": result_data["error"]})
            metrics = {"status": "error", "error": result_data["error"]}
        else:
            results = result_data.get("results", [])
            formatted_results = []
            for i, res in enumerate(results, 1):
                formatted_results.append(
                    f"[{i}] docid: {res.get('docid', 'N/A')}\n"
                    f"    url: {res.get('url', 'N/A')}\n"
                    f"    score: {res.get('score', 0.0):.4f}\n"
                    f"    text: {res.get('text', '')[:500]}...\n"
                )
            result_text = "\n".join(formatted_results)
            metrics = {
                "status": "success",
                "query": query,
                "k": k,
                "result_count": len(results),
            }

        # Store search history
        if instance_id in self._instance_dict:
            self._instance_dict[instance_id]["search_history"].append({
                "query": query,
                "k": k,
                "result_count": len(result_data.get("results", [])),
            })

        return ToolResponse(text=result_text), 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate reward of the tool.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The reward of the tool.
        """
        if instance_id in self._instance_dict:
            return self._instance_dict[instance_id]["reward"]
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance.

        Args:
            instance_id: The instance id of the tool.
        """
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class OpenPageTool(BaseTool):
    """Open page tool for retrieving full document content from FoldAgent's search_server.

    This tool provides functionality to open and retrieve the full content of a document
    by its docid or URL using the FoldAgent search_server.

    Methods:
        get_openai_tool_schema: Return tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute open_page tool
        calc_reward: Calculate reward with respect to tool state
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize OpenPageTool with configuration and schema.

        Args:
            config: Configuration dictionary containing tool settings
            tool_schema: OpenAI function tool schema definition

        Expected config keys:
            - search_server_url: URL of the search server (default: http://localhost:8000)
            - timeout: Request timeout in seconds (default: 30)
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Search server configuration
        self.search_server_url = config.get("search_server_url", os.getenv("SEARCH_SERVER_URL", "http://localhost:8000"))
        self.timeout = config.get("timeout", 30)

        logger.info(f"Initialized OpenPageTool with search_server_url: {self.search_server_url}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return OpenAI tool schema."""
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The instance id of the tool.
            tool_creation_response: The response of the tool when creating instance.
        """
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "open_history": [],
            "reward": 0.0,
        }
        return instance_id, ToolResponse()

    async def execute_open_page(self, docid: Optional[str] = None, url: Optional[str] = None) -> dict:
        """Execute open_page operation using search_server.

        Args:
            docid: Document ID from search results
            url: URL from search results

        Returns:
            Dictionary containing document content
        """
        if not docid and not url:
            return {"results": [], "error": "Either 'docid' or 'url' must be provided"}

        endpoint_url = f"{self.search_server_url}/open"
        payload = {}
        if docid:
            payload["docid"] = docid
        if url:
            payload["url"] = url

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(endpoint_url, json=payload)
                response.raise_for_status()
                data = response.json()
                logger.debug(f"Open page result for docid '{docid}' or url '{url}'")
                return data
        except httpx.HTTPError as e:
            logger.error(f"Open page request failed: {e}")
            return {"results": [], "error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error during open_page: {e}")
            return {"results": [], "error": str(e)}

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute open_page tool.

        Args:
            instance_id: The instance ID of the tool
            parameters: Tool parameters containing 'docid' and/or 'url'

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response text of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        docid = parameters.get("docid")
        url = parameters.get("url")

        if not docid and not url:
            error_msg = "Error: Either 'docid' or 'url' parameter must be provided."
            logger.error(f"[OpenPageTool] {error_msg} Received parameters: {parameters}")
            return ToolResponse(text=json.dumps({"error": error_msg})), 0.0, {}

        # Execute open_page
        result_data = await self.execute_open_page(docid, url)

        # Format results as text
        if "error" in result_data:
            result_text = json.dumps({"error": result_data["error"]})
            metrics = {"status": "error", "error": result_data["error"]}
        else:
            results = result_data.get("results", [])
            if results:
                res = results[0]
                result_text = (
                    f"docid: {res.get('docid', 'N/A')}\n"
                    f"url: {res.get('url', 'N/A')}\n"
                    f"text: {res.get('text', '')}\n"
                )
                metrics = {
                    "status": "success",
                    "docid": docid,
                    "url": url,
                    "text_length": len(res.get('text', '')),
                }
            else:
                result_text = "No document content found."
                metrics = {"status": "not_found", "docid": docid, "url": url}

        # Store open history
        if instance_id in self._instance_dict:
            self._instance_dict[instance_id]["open_history"].append({
                "docid": docid,
                "url": url,
                "status": metrics.get("status", "unknown"),
            })

        return ToolResponse(text=result_text), 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate reward of the tool.

        Args:
            instance_id: The instance id of the tool.

        Returns:
            The reward of the tool.
        """
        if instance_id in self._instance_dict:
            return self._instance_dict[instance_id]["reward"]
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance.

        Args:
            instance_id: The instance id of the tool.
        """
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
