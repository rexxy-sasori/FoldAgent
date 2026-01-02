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
    os.environ["no_proxy"] = ""
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
    env = EnvClass(config, tokenizer, ability)

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
    while iteration < max_turn:
        iteration += 1
        logger.debug(f'[REQUEST {request_id}] Calling LLM for step {iteration}')
        response = await agent.step()
        logger.debug(f'[REQUEST {request_id}] LLM response received for step {iteration}: {response[:100]}...' if response else f'[REQUEST {request_id}] LLM response was None')
        if response is None:
            break
        observation = await run_action(env, response)
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
    
    # Add completion time to the output
    end_time = time.time()
    completion_time = end_time - start_time
    if 'extra_data' not in out.non_tensor_batch:
        out.non_tensor_batch['extra_data'] = np.array([{}], dtype=object)
    if 'stats' not in out.non_tensor_batch['extra_data'][0]:
        out.non_tensor_batch['extra_data'][0]['stats'] = {}
    out.non_tensor_batch['extra_data'][0]['stats']['completion_time'] = completion_time
    out.non_tensor_batch['extra_data'][0]['stats']['request_id'] = request_id
    
    res = DataProto.concat([out])
    logger.info(f'[REQUEST {request_id}] process_item completed in {completion_time:.2f} seconds')
    return res


# @register_handler("agent/react_agent")
# class ReActAgent(AsyncAgent):
#     async def __call__(self, item: DataProto, context: TaskContext, **kwargs):
#         return await process_single_batch(item, context)
