import logging
import os
import re
import asyncio
import json
import copy
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

# Import LocalSearch environment
from envs.local_search import LocalSearch, extract_fn_call

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

        # Use LocalSearch URL from environment
        cls.local_search_url = os.getenv("LOCAL_SEARCH_URL", "http://localhost:8000")
        
        logger.info(f"Initialized ReActAgentLoop with LOCAL_SEARCH_URL={cls.local_search_url}")

    @classmethod
    def _get_system_prompt(cls):
        from agents.prompts import SEARCH_SYSTEM_PROMPT
        return SEARCH_SYSTEM_PROMPT

    @classmethod
    def _get_tool_description(cls):
        from agents.prompts import PARALLEL_TOOL_PROMPT, search_tool, convert_tools_to_description
        return PARALLEL_TOOL_PROMPT.format(description=convert_tools_to_description(search_tool()))

    def get_max_model_len(self):
        """Helper method to get the maximum model length consistently across the class."""
        max_model_len = getattr(self.config.actor_rollout_ref.rollout, 'max_model_len', None)
        if max_model_len is None:
            max_model_len = self.prompt_length + self.response_length
        return max_model_len

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        tool_description = self.__class__.tool_description
        system_prompt = self.__class__.system_prompt + '\n\n' + tool_description
        messages = list(kwargs["raw_prompt"])
        
        # Check if there's already a system message in the raw_prompt
        # If there is, replace it; otherwise, insert the new system message at the beginning
        system_message_exists = False
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                messages[i] = {"role": "system", "content": system_prompt}
                system_message_exists = True
                break
        
        if not system_message_exists:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        # print("debug messages: - input", messages)

        image_data = kwargs.get("multi_modal_data", {}).get("image", None)
        metrics = {}
        request_id = uuid4().hex

        # Create LocalSearch environment instance
        local_search_env = LocalSearch(
            config=self.config.actor_rollout_ref.rollout,
            tokenizer=self.tokenizer,
            ability="LocalSearch",
            request_id=request_id,
            run_id=request_id
        )

        # Initialize environment with item data
        # Extract item data from kwargs for environment initialization
        item_data = kwargs.get("extra_info", {})
        if item_data:
            local_search_env.question = item_data.get("query")
            local_search_env.label_answer = item_data.get("answer")
            local_search_env.instance_info = copy.deepcopy(item_data)
            if "query" in item_data:
                local_search_env.instance_info['problem_statement'] = item_data['query']
            logger.info(f"[{request_id}] Initialized environment with question: {local_search_env.question[:100] if local_search_env.question else 'None'}...")
        else:
            logger.warning(f"[{request_id}] No extra_info found in kwargs, environment not fully initialized")

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

        iteration = 0
        while iteration < max_turn:
            current_time = asyncio.get_event_loop().time()
            elapsed_time = current_time - session_start_time
            
            if elapsed_time > self.session_timeout:
                print(f"[{request_id}] Session timeout after {elapsed_time:.1f}s")
                break

            iteration += 1

            import time
            gen_start = time.time()
            
            # Calculate max_new_tokens to match CallAPI behavior
            max_model_len = self.get_max_model_len()
            max_new_tokens = max_model_len - len(prompt_ids) - 1
            print(f"debug agent - input_length: {len(prompt_ids)}, max_model_len: {max_model_len}, max_new_tokens: {max_new_tokens}")
            sampling_params_with_max = sampling_params.copy()
            sampling_params_with_max['max_new_tokens'] = max_new_tokens
            
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params_with_max,
                image_data=image_data,
            )
            gen_time = time.time() - gen_start
            total_generation_time += gen_time

            print(f"[{request_id}] Turn {iteration}: input_length: {len(prompt_ids)}, Generated {len(output.token_ids)} tokens in {gen_time:.2f}s")
            
            turn_response_ids = output.token_ids
            # Check if adding model response would exceed total context length
            if len(prompt_ids) + len(turn_response_ids) > max_model_len:
                print(f"[{request_id}] Total context length limit ({max_model_len}) reached after model response @ Turn {iteration}")
                break
                
            prompt_ids += turn_response_ids
            response_ids += turn_response_ids  
            response_mask += [1] * len(turn_response_ids)  

            if output.log_probs:
                response_logprobs += output.log_probs

            response_text = self.tokenizer.decode(turn_response_ids, skip_special_tokens=False)
            
            current_messages.append({"role": "assistant", "content": response_text})

            fn_calls = extract_fn_call(response_text)            
            if fn_calls is None or len(fn_calls) == 0:
                print(f"[{request_id}] Turn {iteration}: No tool calls detected. Terminating to prevent infinite conversational loop.")
                break

            tool_calls_count += 1
            fn_call = fn_calls[0]
            if fn_call.get('function') == 'finish':
                print(f"[{request_id}] Turn {iteration}: Finish action detected. Terminating loop.")
                await local_search_env.run_action(response_text, request_id=request_id)
                break
            else:
                print(f"[{request_id}] Turn {iteration}: Executing tool: {fn_call['function']}")
                observation_result = await local_search_env.run_action(response_text, request_id=request_id)
                observation_text = observation_result.get('observation', str(observation_result))
                
                formatted_observation = f"Tool Observation:\n{observation_text}"
                current_messages.append({"role": "user", "content": formatted_observation})

                if self.processor is not None:
                    raw_tool_response = await self.loop.run_in_executor(
                        None, lambda: self.processor.apply_chat_template(
                            [{"role": "user", "content": formatted_observation}],
                            add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                        )
                    )
                    model_inputs = self.processor(text=[raw_tool_response], images=None, return_tensors="pt")
                    tool_response_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
                else:
                    tool_response_ids = await self.loop.run_in_executor(
                        None, lambda: self.tokenizer.apply_chat_template(
                            [{"role": "user", "content": formatted_observation}],
                            add_generation_prompt=True, tokenize=True
                        )
                    )

                if len(prompt_ids) + len(tool_response_ids) > max_model_len:
                    print(f"[{request_id}] Context limit reached after observation @ Turn {iteration}")
                    break

                prompt_ids += tool_response_ids
                response_ids += tool_response_ids
                response_mask += [0] * len(tool_response_ids)
                if response_logprobs:
                    response_logprobs += [0.0] * len(tool_response_ids)

        metrics = AgentLoopMetrics(
            generate_sequences=iteration,
            tool_calls=tool_calls_count,
        )

        final_prompt_ids = prompt_ids[:init_len]
        final_response_ids = response_ids[: self.response_length]
        final_response_mask = response_mask[: self.response_length]

        final_response_text = self.tokenizer.decode(final_response_ids, skip_special_tokens=True)
        print(f"[{request_id}] Final response length: {len(final_response_ids)} tokens")
        print(f"[{request_id}] Total turns: {iteration}, Total tool calls: {tool_calls_count}")
        print(f"[{request_id}] Total generation time: {total_generation_time:.2f}s, Total search time: {total_search_time:.2f}s")

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

        print(f"[{request_id}] ReActAgentLoop completed: {iteration} turns, {tool_calls_count} tool calls")
        return output

    def _check_termination(self, response_text: str) -> bool:
        # Use the same detection method as LocalSearch
        fn_calls = extract_fn_call(response_text)
        if fn_calls:
            for fn_call in fn_calls:
                if fn_call.get('function') == 'finish':
                    return True
        return False

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
