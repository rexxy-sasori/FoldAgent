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
from .utils import CallLLM, Agent, select_env, truncate_text, is_weird, TaskContext, CallAPI, run_action
from .prompts import create_chat
from .prompts import BRANCH_MESSAGE_SEARCH, BRANCH_MESSAGE, SUMMARY_PROMPT_CODE, SUMMARY_PROMPT_SEARCH
from .verifier import judge_scope
from .db_client import log_event

logger = logging.getLogger(__name__)


def print_chat(chat):
    chat_str = ""
    for turn in chat:
        if is_weird(str(turn)):
            chat_str += '# ' + turn['role'] + ' **CJK**\n\n' + turn['content'] + "\n\n---\n\n"
        else:
            chat_str += '# ' + turn['role'] + '\n\n' + turn['content'] + "\n\n---\n\n"
    return chat_str

def extract_fn_call(text):
    if text is None:
        return None
    func_matches = re.findall(r'<function=([^>]+)>', text)
    if not func_matches:
        return None
    last_function = func_matches[-1]
    last_func_pos = text.rfind(f'<function={last_function}>')
    text_after_last_func = text[last_func_pos:]
    params = dict(re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', text_after_last_func, re.DOTALL))
    return {'function': last_function, 'arguments': params}

def extract_summary(text: str) -> str:
    matches = re.findall(r'<summary>(.*?)</summary>', text, re.DOTALL)
    return matches[-1].strip() if matches else None

def clean_response(response):
    if response is None:
        return None
    # 1. Handle explicit return tool calls first
    if '<function=return>' in response:
        result = response.split('<function=return>')[-1]
        # Strip any closing </function> tag
        if '</function>' in result:
            result = result.split('</function>')[0]
        return result.strip()
    
    # 2. Robustly strip reasoning tags (like <seed:think> or <[thought]>)
    # This regex looks for any tag that ends with 'think' or 'thought'
    response = re.sub(r'<(seed:)?think>.*?</(seed:)?think>', '', response, flags=re.DOTALL)
    response = re.sub(r'<\[[^\]]*thought[^\]]*\]>.*?</\[[^\]]*thought[^\]]*\]>', '', response, flags=re.DOTALL)
    
    # 3. Fallback: If tags are unclosed or messy, find the first <function=
    if '<function=' in response:
        return response[response.find('<function='):]
        
    return response.strip()


async def process_item(
        item: DataProto,
        context: TaskContext,
        LLMClass=CallLLM,
) -> DataProto:
    start_time = time.time()
    # Generate request_id based on item UID and agent type for consistent comparison
    item_uid = item.non_tensor_batch['uid'][0] if 'uid' in item.non_tensor_batch else str(uuid.uuid4())
    # Use run_id from context if available, otherwise generate a new one
    run_id = getattr(context, 'run_id', uuid.uuid4().hex[:8])  # Short unique identifier for this run
    request_id = f"{item_uid}_fold_agent_{run_id}"
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
    env = EnvClass(config, tokenizer, ability, request_id=request_id, run_id=run_id)

    try:
        await env.init_env(item)
    except Exception as e:
        logger.error(f"[Error] during environment init: {str(e)}")

    user_prompt, agent_config = await env.get_data(item, context)
    workflow = item.non_tensor_batch['extra_info'][0].get('workflow', None) or getattr(config.plugin, "workflow",
                                                                                       "search")
    user_prompt = create_chat(env.instance_info['problem_statement'], workflow, item)

    branch_prompt = BRANCH_MESSAGE_SEARCH if 'search' in workflow else BRANCH_MESSAGE
    summary_prompt = SUMMARY_PROMPT_SEARCH if 'search' in workflow else SUMMARY_PROMPT_CODE

    max_turn = agent_config.get("max_turn", 64)
    max_session = getattr(config.plugin, "max_session", 5)
    if not is_train:
        max_session = getattr(config.plugin, "val_max_session", max_session)
    session_timeout = getattr(config.plugin, "session_timeout", 90 * 60)
    process_reward = getattr(config.plugin, "process_reward", None)
    if process_reward is not None and isinstance(process_reward, str) and process_reward.lower() == "none":
        process_reward = None
    max_traj = getattr(config.plugin, "max_traj", None)
    enable_summary = getattr(config.plugin, "enable_summary", False)

    host = context.server_host
    port = context.server_port

    # Add request ID to meta_info for LLM client
    meta_info = agent_config.get("meta_info", {})
    meta_info['request_id'] = request_id
    meta_info['run_id'] = run_id
    llm_client = LLMClass(host, port, tokenizer, config, meta_info=meta_info, agent_type="fold_agent")
    logger.debug(f'[REQUEST {request_id}] LLM client initialized')

    prompt_turn = len(user_prompt)
    agent = dict()
    agent['main'] = Agent(llm_client, user_prompt, tokenizer, config, prompt_turn=prompt_turn, agent_type="main")
    branches = []
    branch_tasks = {}
    branch_return = {}
    init_len = len(agent['main'].context())
    current = 'main'
    session_start_time = time.time()
    iteration = 0
    mask_rollout = True  # If True then no grad update on this traj
    session_message = []
    # Global execution registry to track tool calls across agents
    global_execution_registry = set()
    # Track information stalls
    information_stalls = []
    # Track branch returns for correlation analysis
    branch_returns = []
    # Track tool call statistics
    total_calls_count = 0
    redundant_calls_count = 0
    while iteration < max_turn:
        if time.time() - session_start_time > session_timeout:
            logger.info('[SESSION] Session Timeout')
            break

        iteration += 1

        if enable_summary and len(agent[current].context()) - init_len > config.response_length * 0.95:  # summary
            if len(agent) >= max_session:
                logger.info('[SESSION] Session OOC after session %d', len(agent))
                break
            agent[current].rollback(k=2)  # rollback last turn
            agent[current].append({'role': 'assistant', 'content': "", })
            agent[current].append({'role': 'user', 'content': summary_prompt})
            session_message.append({'role': 'user', 'content': summary_prompt})
            response = await agent[current].step()
            session_message.append({'role': 'assistant', 'content': response})
            if response is None:
                break
            summary = extract_summary(response) or response
            next_session_prompt = (
                f"For this question, you have already made the following progress in previous session, "
                f"summarized as follow:\n\n{summary}\n\nNow continue work on it.")
            current = current + '+'
            agent[current] = Agent(llm_client, user_prompt, tokenizer, config, prompt_turn=prompt_turn)
            agent[current].append({'role': 'assistant', 'content': ""})
            agent[current].append({'role': 'user', 'content': next_session_prompt})
            session_message.append({'role': 'user', 'content': next_session_prompt})

        logger.debug(f'[REQUEST {request_id}] Calling LLM for main agent step {iteration}')
        response = await agent['main'].step()
        logger.debug(f'[REQUEST {request_id}] LLM response received for main agent step {iteration}: {response[:100]}...' if response else f'[REQUEST {request_id}] LLM response was None')

        if response is None:
            break

        # Check for information stall phrases in the response
        stall_patterns = [
            r"i don't have the info",
            r"i need to check again",
            r"the previous summary was unclear",
            r"i don't have the information",
            r"i need more information",
            r"i'm not sure",
            r"i can't determine",
            r"i don't know",
            r"the information is not available",
            r"i need to verify",
            r"i need to confirm",
            r"this is unclear",
            r"this is confusing"
        ]
        combined_pattern = re.compile('|'.join(stall_patterns), re.IGNORECASE)
        matches = combined_pattern.findall(response)
        if matches:
            # Record information stall
            stall_info = {
                'iteration': iteration,
                'timestamp': time.time(),
                'matched_phrases': matches,
                'response_excerpt': response[:200],
                'has_folded_info': len(branch_returns) > 0
            }
            information_stalls.append(stall_info)
            logger.info(f'[INFO_STALL] {request_id} Iteration {iteration}: {matches}')

        session_message.append({'role': 'assistant', 'content': response})
        cleaned_response = clean_response(response)
        fn_call = extract_fn_call(cleaned_response)
        if fn_call is not None and fn_call['function'] == 'branch':
            if len(branches) + 1 > max_session:
                observation = f"You've already reached the limit of {len(branches)} branch calls. Continue working independently."
            else:
                description = fn_call['arguments'].get('description', 'Agent')
                message_to_branch = fn_call['arguments'].get('prompt', 'Empty prompt')
                branch_count = len(branches)
                agent_name = f"#{branch_count}-" + description.replace(' ', '_')
                context_length = len(agent['main'].context())
                # Enhanced logging with more details
                logger.info(f'[BRANCH] {description} | Agent: {agent_name} | Context length: {context_length} | Branch count: {branch_count + 1}')
                logger.debug('[BRANCH] %s %d', description, len(agent['main'].context()))
                # Log branch event to database
                await log_event(
                    event_type='branch',
                    request_id=request_id,
                    run_id=run_id,
                    description=description,
                    agent_name=agent_name,
                    context_length=context_length,
                    branch_count=branch_count + 1
                )
                # print(message_to_branch)
                branches.append(agent_name)
                branch_tasks[agent_name] = message_to_branch
                history = agent['main'].messages()
                agent[agent_name] = Agent(llm_client, history, tokenizer, config, prompt_turn=prompt_turn, agent_type="branch", agent_name=agent_name)
                branch_prompt_formatted = branch_prompt.format(message=message_to_branch)
                agent[agent_name].append({'role': 'user', 'content': branch_prompt_formatted})
                logger.debug(f'[REQUEST {request_id}] Calling branch agent {agent_name} react')
                
                agent_return = await agent[agent_name].react(
                    partial(run_action, env, request_id=request_id),
                    max_turn=max_turn,
                    max_tokens=getattr(config.plugin, "branch_len", None),
                    session_timeout=session_timeout - time.time() + session_start_time,
                    should_continue=lambda resp: '<function=return>' not in resp,
                    safe_finish=lambda
                        x: "You are in branch mode and cannot branch task or finish the task. Use the `return` tool to go back to the main agent." if '<function=finish>' in x or '<function=branch>' in x else None,
                    summary_prompt="The context limit has been exceeded for the branch. Please finish the sub task directly and clearly state the progress made and the pending jobs of the sub task. Only summarize the sub task progress, using the return tool.",
                    observation_prompt=f"* You are now in branch mode: {description}. Conduct the sub task based on instruction, and when you complete the assigned sub task, use return tool to return, do not perform action beyond the assigned sub task.",
                )
                logger.debug(f'[REQUEST {request_id}] Branch agent {agent_name} react completed')
                iteration += agent_return['iteration']
                last_response = agent_return['last_response']
                session_message.extend(agent[agent_name].messages()[len(history):])
                
                # Record branch return information for analysis
                branch_returns.append({
                    'agent_name': agent_name,
                    'timestamp': time.time(),
                    'iteration': iteration,
                    'context_length': len(agent[agent_name].context())
                })
                fn_call = extract_fn_call(clean_response(last_response))
                branch_message = None
                if fn_call is not None and fn_call['function'] == 'return':
                    if 'message' in fn_call['arguments']:
                        branch_message = fn_call['arguments'].get('message', 'Empty message')
                        branch_message = f'Branch has finished its task, the returned message is:\n\n{branch_message}'
                    # Enhanced logging for return tool calls
                    logger.info(f'[RETURN] {description} | Agent: {agent_name} | Context length: {len(agent[agent_name].context())}')
                    logger.debug(f'[RETURN] Return message: {branch_message}' if branch_message else f'[RETURN] Return without message')
                    # Log return event to database
                    await log_event(
                        event_type='return',
                        request_id=request_id,
                        run_id=run_id,
                        description=description,
                        agent_name=agent_name,
                        context_length=len(agent[agent_name].context()),
                        branch_message=branch_message,
                        return_type='explicit'
                    )
                elif fn_call is not None and fn_call['function'] == 'finish':
                    if 'message' in fn_call['arguments']:
                        branch_message = fn_call['arguments'].get('message', 'Empty message')
                        branch_message = f'Branch has finished its task, the returned message is:\n\n{branch_message}'
                    # Enhanced logging for finish function (also treated as branch return)
                    logger.info(f'[RETURN] {description} | Agent: {agent_name} | Context length: {len(agent[agent_name].context())}')
                    logger.debug(f'[RETURN] Return message (via finish): {branch_message}' if branch_message else f'[RETURN] Return without message (via finish)')
                    # Log return event to database
                    await log_event(
                        event_type='return',
                        request_id=request_id,
                        run_id=run_id,
                        description=description,
                        agent_name=agent_name,
                        context_length=len(agent[agent_name].context()),
                        branch_message=branch_message,
                        return_type='finish'
                    )
                if branch_message is None:
                    branch_message = f'Branch has finished its task. The last message was:\n\n{clean_response(last_response)}'
                    # Enhanced logging for implicit return (no explicit return/finish call)
                    logger.info(f'[RETURN] {description} | Agent: {agent_name} | Context length: {len(agent[agent_name].context())}')
                    logger.debug(f'[RETURN] Implicit return without explicit function call')
                    # Log return event to database
                    await log_event(
                        event_type='return',
                        request_id=request_id,
                        run_id=run_id,
                        description=description,
                        agent_name=agent_name,
                        context_length=len(agent[agent_name].context()),
                        branch_message=branch_message,
                        return_type='implicit'
                    )
                
                # Extract and add all tool calls from branch message history to global registry
                for msg in agent[agent_name].messages():
                    if msg.get('role') == 'assistant':
                        fn_call = extract_fn_call(msg.get('content', ''))
                        if fn_call:
                            tool_signature = f"{fn_call['function']}({str(fn_call['arguments'])})"
                            global_execution_registry.add(tool_signature)
                            logger.debug(f'[BRANCH_TOOL_ADDED] {agent_name}: {tool_signature}')
                            
                observation = branch_message
                branch_return[agent_name] = observation
                # print(observation)
        else:
            # Extract and check for redundant tool calls in main agent
            fn_call = extract_fn_call(cleaned_response)
            if fn_call:
                tool_signature = f"{fn_call['function']}({str(fn_call['arguments'])})"
                # Check for redundant tool execution
                if tool_signature in global_execution_registry:
                    logger.warning(f'[REDUNDANT_EXECUTION] {request_id} Main Agent re-executing redundant tool: {tool_signature}')
                    await log_event(
                        event_type='REDUNDANT_EXECUTION',
                        request_id=request_id,
                        run_id=run_id,
                        tool_signature=tool_signature,
                        iteration=iteration,
                        agent_type='main',
                        timestamp=time.time()
                    )
                    redundant_calls_count += 1
                else:
                    global_execution_registry.add(tool_signature)
                    total_calls_count += 1
            
            observation = await run_action(env, cleaned_response, request_id=request_id)
            if observation is None:
                mask_rollout = False
                break

        if agent['main'].chat[-1]['role'] == 'user':
            logger.error('[ROLE ERROR]')
            logger.error('%s', agent['main'].chat[-1])
            agent['main'].append({'role': 'assistant', 'content': str(response)})

        if process_reward:
            observation = truncate_text(observation, max_lines=100, merge_repeat=True, merge_num=4)
        # print(observation)
        agent['main'].append({'role': 'user', 'content': observation})
        session_message.append({'role': 'user', 'content': observation})

    env.stats['session_time'] = time.time() - session_start_time
    # Calculate inference time - time up to reward calculation
    inference_time = time.time() - start_time
    env.stats['inference_time'] = inference_time

    # Log inference complete event to database
    await log_event(
        event_type='inference_complete',
        request_id=request_id,
        run_id=run_id,
        inference_time=inference_time,
        session_time=env.stats['session_time'],
        traj_num=len(agent),
        main_turn=len(agent['main'].messages())
    )

    logger.info('[TASK] Task Finish, Start Reward')
    try:
        score_msg, reward, reward_dict = await asyncio.wait_for(
            env.get_reward(item, agent['main'].messages(), context), timeout=60 * 10)
        score = (score_msg, reward)
        logger.debug(f'Reward score: {score}')
        
        # Log reward evaluation to database
        try:
            # Get difficulty if available
            difficulty = None
            if 'extra_info' in item.non_tensor_batch and item.non_tensor_batch['extra_info']:
                extra_info = item.non_tensor_batch['extra_info'][0]
                if 'difficulty' in extra_info:
                    difficulty = extra_info['difficulty']
            elif 'difficulty' in item.non_tensor_batch:
                difficulty = item.non_tensor_batch['difficulty'][0]
            
            # Get question if available
            question = None
            if hasattr(env, 'instance_info') and env.instance_info:
                question = env.instance_info.get('problem_statement', 'unknown')
            
            # Get judge model if available
            judge_model = os.getenv("JUDGE_OPENAI_MODEL", "unknown")
            
            await log_event(
                event_type='reward_evaluation_complete',
                request_id=request_id,
                run_id=run_id,
                question=question,
                reward_score=reward,
                judge_openai_model=judge_model,
                difficulty=difficulty
            )
        except Exception as e:
            logger.error(f"[Error] Logging reward evaluation: {e}")
    except Exception as e:
        logger.error(f"[Error] Getting reward: {e}")
        score, reward_dict = ("", 0), {"ans_reward": 0.0, "format_reward": 0.0, "ref_reward": 0.0}

    outs = []
    env.stats['get_final_score'] = score[1]
    env.stats['traj_num'] = len(agent)
    env.stats['main_len'] = min(len(agent['main'].context()) - init_len, config.response_length)
    env.stats['total_token'] = len(tokenizer.encode(print_chat(user_prompt + session_message)))
    env.stats['main_turn'] = len(agent['main'].messages())
    env.stats['is_branch'] = int(len(agent) > 1)
    env.stats['branch_success'] = int(int(len(agent) > 1) * score[1])
    env.stats['use_all_branch'] = int(len(branches) + 1 > max_session)
    
    # Add metrics for information stalls and bad folds
    env.stats['information_stall_count'] = len(information_stalls)
    env.stats['bad_fold_detection_count'] = 0  # Will be updated from log events
    env.stats['branch_return_count'] = len(branch_returns)
    
    # Log information stall analysis
    if information_stalls:
        stalls_after_folds = sum(1 for stall in information_stalls if stall['has_folded_info'])
        logger.info(f'[STALL_ANALYSIS] {request_id} Total stalls: {len(information_stalls)}, Stalls after folds: {stalls_after_folds}')
        await log_event(
            event_type='stall_analysis',
            request_id=request_id,
            run_id=run_id,
            total_stalls=len(information_stalls),
            stalls_after_folds=stalls_after_folds,
            branch_return_count=len(branch_returns),
            stalls=information_stalls
        )
    
    # Print redundant execution summary
    logger.info(f'[REDUNDANT_EXECUTION_SUMMARY] {request_id} Total Redundant Calls: {redundant_calls_count} out of {total_calls_count + redundant_calls_count} total calls')
    print(f'Total Redundant Calls: {redundant_calls_count} out of {total_calls_count + redundant_calls_count} total calls')

    if getattr(env, 'is_finish', False) or getattr(env, 'finish', False):
        mask_rollout = False
    if score[1] > 0:
        mask_rollout = False

    is_finish = getattr(env, 'is_finish', False) or getattr(env, 'finish', False)
    if getattr(config.plugin, "must_finish", None):
        if not is_finish:
            score = ('', 0)

    if process_reward and is_train:
        mask_rollout = False
        env.stats['concise_main'] = int(len(agent['main'].context()) - init_len <= config.response_length * 0.5)
        if 'cjk' in process_reward:
            env.stats['is_cjk'] = 0
            for name in agent:
                for i, turn in enumerate(agent[name].chat):
                    if is_weird(str(turn)):
                        logger.error('[CJK ERROR]')
                        logger.error('%s', turn)
                        env.stats['is_cjk'] = 1
                        agent[name].set_process_reward(i, -1)
                        if 'flat' in process_reward:
                            agent[name].set_cache('reward', 0)
        if score[1] > 0:
            # Check main
            if len(agent['main'].context()) - init_len > config.response_length * 0.5:
                bad_turn = [i for i, turn in enumerate(agent['main'].messages()) if
                            '<function=branch>' not in str(turn) and '<function=finish>' not in str(turn)]
                agent['main'].set_process_reward(bad_turn, -1)

            if len(agent) == 1:
                agent['main'].set_process_reward('all', -1)
                if 'flat' in process_reward:
                    agent['main'].set_cache('reward', 1 - 1)

            # Scope check
            if 'scope' in process_reward:
                env.stats['scope_judge'] = 1
                for name in branches:
                    assigned_task = branch_tasks[name]
                    return_message = branch_return[name]
                    is_focus, justification = await judge_scope(assigned_task, return_message)
                    if is_focus < 0:  # scope check, skip summary turn
                        logger.error(f'[FOCUS] Branch beyond focus: //{name}//. {justification}')
                        agent[name].set_process_reward([i for i in range(len(agent[name].chat) - 1)], -0.2)
                        if 'flat' in process_reward:
                            agent[name].set_cache('reward', 1 - 0.2)
                        env.stats['scope_judge'] = 0
                    elif is_focus > 0:
                        agent[name].set_process_reward([i for i in range(len(agent[name].chat) - 1)], 0.2)
                        if 'flat' in process_reward:
                            agent[name].set_cache('reward', 1 + 0.2)
            # Tool call error
            for name in branches:
                for i, turn in enumerate(agent[name].chat):
                    ERR_MARKERS = (
                        'Failed to validate tool call',
                        'Failed to parse tool call',
                        'You are in branch mode and cannot branch task or finish the task.',
                        'No function call was detected in the model response',
                        '[Error] The "search" function requires a "query" argument',
                        '[Error] The "open_page" function requires either a "docid" or a "url".',
                        '[Error] The function',
                    )
                    if any(m in str(turn) for m in ERR_MARKERS):
                        agent[name].set_process_reward(i - 1, -1)
        else:
            is_finish = getattr(env, 'is_finish', False) or getattr(env, 'finish', False)
            if 'drop_fail' in process_reward:
                if not is_finish:
                    for name in branches:
                        if 'cjk' in process_reward:
                            should_drop = True
                            for i, turn in enumerate(agent[name].chat):
                                if is_weird(str(turn)):
                                    agent[name].set_process_reward(i, -2)
                                    if 'flat' in process_reward:
                                        agent[name].set_cache('reward', -1)
                                    should_drop = False
                            if should_drop:
                                agent.pop(name)
                        else:
                            agent.pop(name)  # drop all branch if not finish (overlong mask)
            # Scope check + reward
            if 'reward_scope' in process_reward:
                env.stats['scope_judge'] = 1
                for name in branches:
                    assigned_task = branch_tasks[name]
                    return_message = branch_return[name]
                    is_focus, justification = await judge_scope(assigned_task, return_message)
                    if is_focus < 0:  # scope check, skip summary turn
                        logger.error(f'[FOCUS] Branch beyond focus: //{name}//. {justification}')
                        agent[name].set_process_reward([i for i in range(len(agent[name].chat) - 1)], -0.2)
                        if 'flat' in process_reward:
                            agent[name].set_cache('reward', 0 - 0.2)
                        env.stats['scope_judge'] = 0
                    elif is_focus > 0:
                        agent[name].set_process_reward([i for i in range(len(agent[name].chat) - 1)], 0.2)
                        if 'flat' in process_reward:
                            agent[name].set_cache('reward', 0 + 0.2)

    for name in agent if is_train else ['main']:  # in eval only return main agent
        out = await agent[name].dataproto()
        messages = agent[name].messages()
        if process_reward is not None and 'flat' in process_reward and 'reward' in agent[name].info_cache:
            out = await env.update_dataproto(out, item, messages, ('', agent[name].info_cache['reward']), reward_dict,
                                             tag=name, metrics=agent[name].get_metrics())
        else:
            out = await env.update_dataproto(out, item, messages, score, reward_dict,
                                             tag=name, metrics=agent[name].get_metrics())
        out.batch['is_overlong'] = torch.Tensor([mask_rollout])
        session_message_str = print_chat(session_message)
        out.non_tensor_batch['message_str'] = np.array([session_message_str], dtype=object)
        meta_info = f"N: {len(agent)} | {name}"
        out.non_tensor_batch['meta_info'] = np.array([meta_info], dtype=object)
        outs.append(copy.deepcopy(out))

    if max_traj is not None and len(outs) > max_traj:
        idx = [0] + sorted(random.sample(range(1, len(outs)), k=max_traj - 1))
        outs = [outs[i] for i in idx]

    try:
        end_time = time.time()
        full_completion_time = end_time - start_time
        # Use inference_time instead of full completion time to exclude reward calculation
        completion_time = inference_time
        for out in outs:
            if 'extra_data' not in out.non_tensor_batch:
                out.non_tensor_batch['extra_data'] = np.array([{}], dtype=object)
            if 'stats' not in out.non_tensor_batch['extra_data'][0]:
                out.non_tensor_batch['extra_data'][0]['stats'] = {}
            out.non_tensor_batch['extra_data'][0]['stats']['completion_time'] = completion_time
            out.non_tensor_batch['extra_data'][0]['stats']['full_completion_time'] = full_completion_time
            out.non_tensor_batch['extra_data'][0]['stats']['request_id'] = request_id
        res = DataProto.concat(outs)
        logger.info(f'[REQUEST {request_id}] process_item inference completed in {completion_time:.2f} seconds (full time including reward: {full_completion_time:.2f} seconds)')
        return res
    except Exception as e:
        breakpoint()
        return


# @register_handler("agent/fold_agent")
# class ReActAgent(AsyncAgent):
#     async def __call__(self, item: DataProto, context: TaskContext, **kwargs):
#         return await process_single_batch(item, context)
