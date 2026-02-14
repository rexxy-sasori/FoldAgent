import aiohttp
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LocalSearchTool:
    def __init__(self, config: Dict[str, Any], tool_schema: Optional[Any] = None):
        self.search_server_url = config.get("search_server_url", "http://localhost:8000")
        self.tool_schema = tool_schema
        logger.info(f"Initialized LocalSearchTool with server: {self.search_server_url}")
    
    async def __call__(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        try:
            if tool_name == "search":
                return await self._search(**kwargs)
            elif tool_name == "open_page":
                return await self._open_page(**kwargs)
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
    
    async def _search(self, query: str, k: int = 20, topk: int = None) -> Dict[str, Any]:
        try:
            # Use topk if provided (for backward compatibility)
            search_k = topk if topk is not None else k
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.search_server_url}/search",
                    json={"query": query, "k": search_k},
                    timeout=aiohttp.ClientTimeout(total=30.0)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Search completed for query: {query[:50]}... - Found {len(result['results'])} results")
                        return {
                            "tool_call_id": "search",
                            "response": result
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"Search failed with status {response.status}: {error_text}")
                        return {
                            "tool_call_id": "search",
                            "response": {"error": f"Search failed: {error_text}"}
                        }
        except aiohttp.ClientError as e:
            logger.error(f"Search request failed: {str(e)}")
            return {
                "tool_call_id": "search",
                "response": {"error": f"Connection error: {str(e)}"}
            }
        except Exception as e:
            logger.error(f"Unexpected error in search: {str(e)}")
            return {
                "tool_call_id": "search",
                "response": {"error": f"Unexpected error: {str(e)}"}
            }
    
    async def _open_page(self, docid: Optional[str] = None, url: Optional[str] = None) -> Dict[str, Any]:
        if not docid and not url:
            return {
                "tool_call_id": "open_page",
                "response": {"error": "Either docid or url must be provided"}
            }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.search_server_url}/open",
                    json={"docid": docid, "url": url},
                    timeout=aiohttp.ClientTimeout(total=30.0)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Open page completed for docid: {docid} or url: {url}")
                        return {
                            "tool_call_id": "open_page",
                            "response": result
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"Open page failed with status {response.status}: {error_text}")
                        return {
                            "tool_call_id": "open_page",
                            "response": {"error": f"Open page failed: {error_text}"}
                        }
        except aiohttp.ClientError as e:
            logger.error(f"Open page request failed: {str(e)}")
            return {
                "tool_call_id": "open_page",
                "response": {"error": f"Connection error: {str(e)}"}
            }
        except Exception as e:
            logger.error(f"Unexpected error in open_page: {str(e)}")
            return {
                "tool_call_id": "open_page",
                "response": {"error": f"Unexpected error: {str(e)}"}
            }


def get_tool_handler(config: Dict[str, Any], tool_schema: Optional[Any] = None) -> LocalSearchTool:
    return LocalSearchTool(config=config, tool_schema=tool_schema)
