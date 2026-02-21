import logging
import os
import re
import asyncio
import json
from typing import Any, Optional, List
from uuid import uuid4

import aiohttp
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    AgentLoopMetrics,
    register,
)
from verl.utils.rollout_trace import rollout_trace_op
from verl import DataProto

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("react_agent")
class ReActAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        logger.info("Performing class-level ReActAgentLoop initialization")

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.config = config

        cls.max_turns = config.actor_rollout_ref.rollout.plugin.get("max_turn", 12)
        cls.val_max_turns = config.actor_rollout_ref.rollout.plugin.get("val_max_turn", 12)
        cls.session_timeout = config.actor_rollout_ref.rollout.plugin.get("session_timeout", 90 * 60)
        cls.max_tool_response_length = config.actor_rollout_ref.rollout.plugin.get("max_tool_response_length", 1000)
        cls.tool_response_truncate_side = config.actor_rollout_ref.rollout.plugin.get("tool_response_truncate_side", "right")
        cls.enable_summary = config.actor_rollout_ref.rollout.plugin.get("enable_summary", False)

        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length

        cls.system_prompt = cls._get_system_prompt()
        cls.tool_description = cls._get_tool_description()

        cls.tool_patterns = [
            re.compile(r'<function=search>\s*<parameter=query>(.*?)</parameter>', re.DOTALL),
            re.compile(r'<function=open_page>\s*<parameter=(?:docid|url)>(.*?)</parameter>', re.DOTALL),
            re.compile(r'<function=finish>', re.DOTALL),
        ]

        cls.local_search_url = os.getenv("LOCAL_SEARCH_URL", "http://localhost:8000")
        cls.search_timeout = 2.0
        
        logger.info(f"Initialized ReActAgentLoop with LOCAL_SEARCH_URL={cls.local_search_url}, search_timeout={cls.search_timeout}s")

    @classmethod
    def _get_system_prompt(cls):
        from agents.prompts import SEARCH_SYSTEM_PROMPT
        return SEARCH_SYSTEM_PROMPT

    @classmethod
    def _get_tool_description(cls):
        from agents.prompts import PARALLEL_TOOL_PROMPT, search_tool, convert_tools_to_description
        return PARALLEL_TOOL_PROMPT.format(description=convert_tools_to_description(search_tool()))

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        image_data = kwargs.get("multi_modal_data", {}).get("image", None)
        metrics = {}
        request_id = uuid4().hex

        logger.info(f"[{request_id}] Starting ReActAgentLoop")

        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    tools=None,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages,
                    tools=None,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )

        response_ids = []
        response_mask = []
        response_logprobs = []
        tool_calls_count = 0
        total_generation_time = 0.0
        total_search_time = 0.0

        current_messages = messages.copy()
        max_turn = self.val_max_turns if kwargs.get("validate", False) else self.max_turns
        session_start_time = asyncio.get_event_loop().time()
        init_len = len(prompt_ids)

        logger.info(f"[{request_id}] Max turns: {max_turn}, Session timeout: {self.session_timeout}s")
        logger.info(f"[{request_id}] Initial prompt length: {init_len} tokens")
        logger.info(f"[{request_id}] Response length limit: {self.response_length} tokens")

        iteration = 0
        while iteration < max_turn:
            current_time = asyncio.get_event_loop().time()
            elapsed_time = current_time - session_start_time
            
            if elapsed_time > self.session_timeout:
                logger.info(f"[{request_id}] Session timeout after {elapsed_time:.1f}s")
                break

            if self.enable_summary and len(prompt_ids) - init_len > self.response_length * 0.95:
                logger.info(f"[{request_id}] Context approaching limit, summarizing")
                summary_response = await self._summarize_conversation(current_messages, request_id)
                if summary_response is None:
                    logger.warning(f"[{request_id}] Summary failed, breaking")
                    break

                summary_prompt = (
                    f"For this question, you have already made the following progress in previous session, "
                    f"summarized as follow:\n\n{summary_response}\n\nNow continue work on it."
                )
                current_messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": summary_prompt},
                ]
                if self.processor is not None:
                    raw_prompt = await self.loop.run_in_executor(
                        None,
                        lambda: self.processor.apply_chat_template(
                            current_messages,
                            tools=None,
                            add_generation_prompt=True,
                            tokenize=False,
                            **self.apply_chat_template_kwargs,
                        ),
                    )
                    model_inputs = self.processor(text=[raw_prompt], images=None, return_tensors="pt")
                    prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
                else:
                    prompt_ids = await self.loop.run_in_executor(
                        None,
                        lambda: self.tokenizer.apply_chat_template(
                            current_messages,
                            tools=None,
                            add_generation_prompt=True,
                            tokenize=True,
                            **self.apply_chat_template_kwargs,
                        ),
                    )
                init_len = len(prompt_ids)

            iteration += 1
            logger.debug(f"[{request_id}] Turn {iteration}: Generating response")

            import time
            gen_start = time.time()
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
            )
            gen_time = time.time() - gen_start
            total_generation_time += gen_time

            logger.debug(f"[{request_id}] Turn {iteration}: Generated {len(output.token_ids)} tokens in {gen_time:.2f}s")

            turn_response_ids = output.token_ids
            prompt_ids += turn_response_ids
            response_ids += turn_response_ids
            response_mask += [1] * len(turn_response_ids)

            if output.log_probs:
                response_logprobs += output.log_probs

            response_text = self.tokenizer.decode(turn_response_ids, skip_special_tokens=True)
            logger.debug(f"[{request_id}] Turn {iteration}: Response text (first 200 chars): {response_text[:200]}")

            current_messages.append({"role": "assistant", "content": response_text})

            if self._check_termination(response_text):
                logger.info(f"[{request_id}] Turn {iteration}: Termination detected")
                break

            tool_calls = self._detect_tool_calls(response_text)
            if not tool_calls:
                logger.debug(f"[{request_id}] Turn {iteration}: No tool calls detected")
                if len(response_mask) >= self.response_length:
                    logger.info(f"[{request_id}] Response length limit reached")
                    break
                continue

            tool_calls_count += len(tool_calls)
            logger.info(f"[{request_id}] Turn {iteration}: Detected {len(tool_calls)} tool calls")

            search_tasks = []
            for tool_call in tool_calls:
                if tool_call["function"] == "finish":
                    logger.info(f"[{request_id}] Turn {iteration}: Finish tool detected")
                    break

                logger.debug(f"[{request_id}] Turn {iteration}: Executing tool: {tool_call['function']}")
                logger.debug(f"[{request_id}] Turn {iteration}: Tool arguments: {tool_call['arguments']}")
                search_tasks.append(self._execute_tool(tool_call, request_id))

            if search_tasks:
                search_start = time.time()
                logger.debug(f"[{request_id}] Turn {iteration}: Starting {len(search_tasks)} parallel tool executions")
                observations = await asyncio.gather(*search_tasks)
                search_time = time.time() - search_start
                total_search_time += search_time
                logger.debug(f"[{request_id}] Turn {iteration}: Completed {len(observations)} tool observations in {search_time:.2f}s")

                for i, observation in enumerate(observations):
                    observation_text = observation.get("text", f"Error: {observation.get('error', 'Unknown error')}")
                    error = observation.get("error")
                    logger.debug(f"[{request_id}] Turn {iteration}: Observation {i}: {observation_text[:200]}")
                    if error:
                        logger.debug(f"[{request_id}] Turn {iteration}: Observation {i} error: {error}")
                    
                    current_messages.append({"role": "tool", "content": observation_text})

                    if self.processor is not None:
                        raw_tool_response = await self.loop.run_in_executor(
                            None,
                            lambda: self.processor.apply_chat_template(
                                [{"role": "tool", "content": observation_text}],
                                add_generation_prompt=True,
                                tokenize=False,
                                **self.apply_chat_template_kwargs,
                            ),
                        )
                        model_inputs = self.processor(text=[raw_tool_response], images=None, return_tensors="pt")
                        tool_response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
                    else:
                        tool_response_ids = await self.loop.run_in_executor(
                            None,
                            lambda: self.tokenizer.apply_chat_template(
                                [{"role": "tool", "content": observation_text}],
                                add_generation_prompt=True,
                                tokenize=True,
                            ),
                        )

                    if len(response_mask) + len(tool_response_ids) >= self.response_length:
                        logger.info(f"[{request_id}] Response length limit reached after tool response")
                        break

                    prompt_ids += tool_response_ids
                    response_ids += tool_response_ids
                    response_mask += [0] * len(tool_response_ids)

                    if response_logprobs:
                        response_logprobs += [0.0] * len(tool_response_ids)

            if len(response_mask) >= self.response_length:
                logger.info(f"[{request_id}] Response length limit reached")
                logger.info(f"[{request_id}] Turn {iteration}: Response length limit reached after tool response")
                break

        metrics = AgentLoopMetrics(
            generate_sequences=iteration,
            tool_calls=tool_calls_count,
        )

        final_prompt_ids = prompt_ids[: len(prompt_ids) - len(response_mask)]
        final_response_ids = response_ids[: self.response_length]
        final_response_mask = response_mask[: self.response_length]

        final_response_text = self.tokenizer.decode(final_response_ids, skip_special_tokens=True)
        logger.info(f"[{request_id}] Final response length: {len(final_response_ids)} tokens")
        logger.info(f"[{request_id}] Total turns: {iteration}, Total tool calls: {tool_calls_count}")
        logger.info(f"[{request_id}] Total generation time: {total_generation_time:.2f}s, Total search time: {total_search_time:.2f}s")

        # Determine if this rollout should be masked (True if overlong, False otherwise)
        mask_rollout = len(final_response_ids) >= self.response_length

        output = AgentLoopOutput(
            prompt_ids=final_prompt_ids,
            response_ids=final_response_ids,
            response_mask=final_response_mask,
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data={"image": image_data} if image_data is not None else {},
            num_turns=iteration,
            metrics=metrics,
            extra_fields={
                "generation_time": total_generation_time,
                "search_time": total_search_time,
                "mask_rollout": mask_rollout,
            },
        )

        logger.info(f"[{request_id}] ReActAgentLoop completed: {iteration} turns, {tool_calls_count} tool calls")
        return output

    def _check_termination(self, response_text: str) -> bool:
        return "<function=finish>" in response_text

    def _detect_tool_calls(self, response_text: str) -> list[dict[str, Any]]:
        tool_calls = []

        for pattern in self.tool_patterns:
            matches = pattern.findall(response_text)
            for match in matches:
                if "search" in pattern.pattern:
                    query = match.strip()
                    tool_calls.append({"function": "search", "arguments": {"query": query}})
                elif "open_page" in pattern.pattern:
                    param = match.strip()
                    if "=" in param:
                        key, value = param.split("=", 1)
                        tool_calls.append({"function": "open_page", "arguments": {key.strip(): value.strip()}})
                    else:
                        tool_calls.append({"function": "open_page", "arguments": {"docid": param.strip()}})
                elif "finish" in pattern.pattern:
                    tool_calls.append({"function": "finish", "arguments": {}})

        return tool_calls

    async def _execute_tool(self, tool_call: dict[str, Any], request_id: str) -> dict[str, Any]:
        function_name = tool_call["function"]
        arguments = tool_call["arguments"]

        if function_name == "search":
            return await self._execute_search(arguments.get("query", ""), request_id)
        elif function_name == "open_page":
            docid = arguments.get("docid")
            url = arguments.get("url")
            return await self._execute_open_page(docid=docid, url=url, request_id=request_id)
        elif function_name == "finish":
            return {"text": "Finish", "search_time": 0.0}
        else:
            return {"text": f"Unknown tool: {function_name}", "error": "Unknown tool", "search_time": 0.0}

    async def _execute_search(self, query: str, request_id: str) -> dict[str, Any]:
        import time
        start_time = time.time()

        logger.debug(f"[{request_id}] Executing search: {query}")

        try:
            timeout = aiohttp.ClientTimeout(total=self.search_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                payload = {"query": query, "k": 10}
                logger.debug(f"[{request_id}] Search payload: {payload}")
                async with session.post(f"{self.local_search_url}/search", json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()

                    results = data.get("results", [])
                    logger.debug(f"[{request_id}] Search returned {len(results)} results")
                    formatted_results = []
                    for i, result in enumerate(results[:10]):
                        docid = result.get("docid", f"doc_{i}")
                        text = result.get("text", "")[:500]
                        url = result.get("url", "")
                        score = result.get("score", 0.0)
                        formatted_results.append(f"[{docid}] {text} (URL: {url}, Score: {score:.3f})")

                    search_time = time.time() - start_time
                    logger.info(f"[{request_id}] Search completed in {search_time:.2f}s, returned {len(results)} results")

                    return {
                        "text": "\n".join(formatted_results),
                        "search_time": search_time,
                        "results": results,
                    }
        except asyncio.TimeoutError:
            search_time = time.time() - start_time
            logger.error(f"[{request_id}] Search timeout after {search_time:.2f}s")
            return {"text": f"Search timed out after {self.search_timeout}s", "error": "timeout", "search_time": search_time}
        except Exception as e:
            search_time = time.time() - start_time
            logger.error(f"[{request_id}] Search error: {e}")
            return {"text": f"Search error: {str(e)}", "error": str(e), "search_time": search_time}

    async def _execute_open_page(self, docid: Optional[str], url: Optional[str], request_id: str) -> dict[str, Any]:
        import time
        start_time = time.time()

        logger.debug(f"[{request_id}] Opening page: docid={docid}, url={url}")

        try:
            timeout = aiohttp.ClientTimeout(total=self.search_timeout)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if docid:
                    payload = {"docid": docid}
                    logger.debug(f"[{request_id}] Open page payload: {payload}")
                    async with session.post(f"{self.local_search_url}/open", json=payload) as response:
                        response.raise_for_status()
                        data = await response.json()
                elif url:
                    payload = {"url": url}
                    logger.debug(f"[{request_id}] Open page payload: {payload}")
                    async with session.post(f"{self.local_search_url}/open", json=payload) as response:
                        response.raise_for_status()
                        data = await response.json()
                else:
                    return {"text": "Error: Either docid or url must be provided", "error": "missing_parameter", "search_time": 0.0}

                results = data.get("results", [])
                logger.debug(f"[{request_id}] Open page returned {len(results)} results")
                if results:
                    text = results[0].get("text", "")
                    if len(text) > self.max_tool_response_length:
                        if self.tool_response_truncate_side == "right":
                            text = text[: self.max_tool_response_length] + "\n[Document truncated.]"
                        else:
                            text = text[-self.max_tool_response_length:] + "\n[Document truncated.]"
                    logger.debug(f"[{request_id}] Open page text (first 200 chars): {text[:200]}")
                else:
                    text = "Document not found"
                    logger.debug(f"[{request_id}] Document not found")

                search_time = time.time() - start_time
                logger.info(f"[{request_id}] Open page completed in {search_time:.2f}s")

                return {"text": text, "search_time": search_time}
        except asyncio.TimeoutError:
            search_time = time.time() - start_time
            logger.error(f"[{request_id}] Open page timeout after {search_time:.2f}s")
            return {"text": f"Open page timed out after {self.search_timeout}s", "error": "timeout", "search_time": search_time}
        except Exception as e:
            search_time = time.time() - start_time
            logger.error(f"[{request_id}] Open page error: {e}")
            return {"text": f"Open page error: {str(e)}", "error": str(e), "search_time": search_time}

    async def _summarize_conversation(self, messages: list[dict[str, str]], request_id: str) -> Optional[str]:
        logger.info(f"[{request_id}] Summarizing conversation")

        summary_prompt = "Please summarize the current conversation and progress so far."
        summary_messages = messages + [{"role": "user", "content": summary_prompt}]

        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    summary_messages,
                    tools=None,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=None, return_tensors="pt")
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    summary_messages,
                    tools=None,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )

        sampling_params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 500,
        }

        try:
            output = await self.server_manager.generate(
                request_id=f"{request_id}_summary",
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=None,
            )

            summary_text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
            logger.info(f"[{request_id}] Summary generated: {summary_text[:100]}...")
            return summary_text
        except Exception as e:
            logger.error(f"[{request_id}] Summary generation error: {e}")
            return None
