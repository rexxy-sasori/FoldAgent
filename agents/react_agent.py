import os
import re
import time
import copy
import asyncio
import logging
import uuid
from functools import partial
import random

import numpy as np
import torch

from verl import DataProto
from .utils import CallLLM, Agent, select_env, truncate_text, is_weird, TaskContext, run_action
from .prompts import create_chat

logger = logging.getLogger(__name__)


async def process_item(
        item: DataProto,
        context: TaskContext,
        LLMClass=CallLLM,
) -> DataProto:
    start_time = time.time()
    request_id = str(uuid.uuid4())  # Generate unique request ID
    logger.info(f'[REQUEST {request_id}] Starting process_item')
    tokenizer = context.tokenizer
    config = context.config.actor_rollout_ref.rollout
    is_train = context.is_train

    if not is_train:
        if getattr(config.plugin, "val_response_length", None):
            config.response_length = getattr(config.plugin, "val_response_length", None)
    ability = item.non_tensor_batch['ability'][0]
    # Select env
    EnvClass = select_env(ability, config, )
    logger.debug(f'[REQUEST {request_id}] Environment initialized - is_train: {is_train}, EnvClass: {EnvClass.__name__}')
    env = EnvClass(config, tokenizer, ability, request_id=request_id)

    try:
        await env.init_env(item)
    except Exception as e:
        logger.error(f"[Error] during environment init: {str(e)}")

    user_prompt, agent_config = await env.get_data(item, context)
    workflow = item.non_tensor_batch['extra_info'][0].get('workflow', None) or getattr(config.plugin, "workflow",
                                                                                       "search")
    user_prompt = create_chat(env.instance_info['problem_statement'], workflow, item)
    max_turn = agent_config.get("max_turn", 64)
    host = context.server_host
    port = context.server_port
    # Add request ID to meta_info for LLM client
    meta_info = agent_config.get("meta_info", {})
    meta_info['request_id'] = request_id
    llm_client = LLMClass(host, port, tokenizer, config, meta_info=meta_info, agent_type="react_agent")
    logger.debug(f'[REQUEST {request_id}] LLM client initialized')
    prompt_turn = len(user_prompt)

    agent = Agent(llm_client, user_prompt, tokenizer, config, prompt_turn=prompt_turn)
    iteration = 0
    session_start_time = time.time()
    session_timeout = getattr(config.plugin, "session_timeout", 90 * 60)
    enable_summary = getattr(config.plugin, "enable_summary", False)
    init_len = len(agent.context())
    
    while iteration < max_turn:
        # Check session timeout
        if time.time() - session_start_time > session_timeout:
            logger.info(f'[REQUEST {request_id}] Session Timeout')
            break
            
        # Check context length and summarize if needed
        if enable_summary and len(agent.context()) - init_len > config.response_length * 0.95:
            logger.info(f'[REQUEST {request_id}] Context approaching limit, rolling back last turn')
            agent.rollback(k=2)  # Rollback last turn
            agent.append({'role': 'assistant', 'content': ""})
            summary_prompt = "Please summarize the current conversation and progress so far."
            agent.append({'role': 'user', 'content': summary_prompt})
            summary_response = await agent.step()
            if summary_response is None:
                break
            # Start new session with summary
            next_session_prompt = (
                f"For this question, you have already made the following progress in previous session, "
                f"summarized as follow:\n\n{summary_response}\n\nNow continue work on it."
            )
            agent = Agent(llm_client, user_prompt, tokenizer, config, prompt_turn=prompt_turn)
            agent.append({'role': 'assistant', 'content': ""})
            agent.append({'role': 'user', 'content': next_session_prompt})
        
        iteration += 1
        logger.debug(f'[REQUEST {request_id}] Calling LLM for step {iteration}')
        response = await agent.step()
        logger.debug(f'[REQUEST {request_id}] LLM response received for step {iteration}: {response[:100]}...' if response else f'[REQUEST {request_id}] LLM response was None')
        if response is None:
            break
        observation = await run_action(env, response, request_id=request_id)
        if observation is None:
            break
        agent.append({'role': 'user', 'content': observation})

    logger.info(f'[REQUEST {request_id}] Task Finish, Start Reward')
    try:
        score_msg, reward, reward_dict = await asyncio.wait_for(
            env.get_reward(item, agent.messages(), context), timeout=60 * 10)
        score = (score_msg, reward)
        logger.debug(f'[REQUEST {request_id}] Reward score: {score}')
    except Exception as e:
        logger.error(f"[REQUEST {request_id}] [Error] Getting reward: {e}")
        score, reward_dict = ("", 0), {"ans_reward": 0.0, "format_reward": 0.0, "ref_reward": 0.0}

    out = await agent.dataproto()
    messages = agent.messages()
    out = await env.update_dataproto(out, item, messages, score, reward_dict,
                                         tag='main', metrics=agent.get_metrics())
    
    # Add completion time and context metrics to the output
    end_time = time.time()
    completion_time = end_time - start_time
    session_duration = end_time - session_start_time
    final_context_length = len(agent.context())
    context_growth = final_context_length - init_len
    
    if 'extra_data' not in out.non_tensor_batch:
        out.non_tensor_batch['extra_data'] = np.array([{}], dtype=object)
    if 'stats' not in out.non_tensor_batch['extra_data'][0]:
        out.non_tensor_batch['extra_data'][0]['stats'] = {}
    
    stats = out.non_tensor_batch['extra_data'][0]['stats']
    stats['completion_time'] = completion_time
    stats['request_id'] = request_id
    stats['session_duration'] = session_duration
    stats['final_context_length'] = final_context_length
    stats['context_growth'] = context_growth
    stats['total_turns'] = iteration
    
    res = DataProto.concat([out])
    logger.info(f'[REQUEST {request_id}] process_item completed in {completion_time:.2f} seconds')
    return res


# @register_handler("agent/react_agent")
# class ReActAgent(AsyncAgent):
#     async def __call__(self, item: DataProto, context: TaskContext, **kwargs):
#         return await process_single_batch(item, context)
