import logging
import os
import torch
import numpy as np
from typing import Any, Union

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    AgentLoopMetrics,
    register,
)
from verl import DataProto
from agents.react_agent import process_item
from agents.utils import CallLLM, VERLNativeLLM, TaskContext

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("react_agent")
class ReactAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True

        logger.info("Initializing ReactAgentLoop class")

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.config = config

    async def run(
        self, sampling_params: dict[str, Any], **kwargs
    ) -> Union[AgentLoopOutput, list[AgentLoopOutput]]:
        item = DataProto()
        item.non_tensor_batch = kwargs
        
        item.meta_info = {
            'generation_kwargs': sampling_params,
            'uid': kwargs.get('uid', None)
        }

        server_host = getattr(self, 'server_host', 'localhost')
        server_port = getattr(self, 'server_port', 8000)
        
        server_manager = getattr(self, 'server_manager', None)
        
        context = TaskContext(
            config=self.config,
            global_step=kwargs.get('global_step', 0),
            server_host=server_host,
            server_port=server_port,
            is_train=kwargs.get('is_train', True),
            run_id=kwargs.get('run_id', 'unknown'),
            tokenizer=self.tokenizer,
            server_manager=server_manager
        )
        
        LLMClass = VERLNativeLLM if server_manager else CallLLM
        rollout_results = await process_item(item, context, LLMClass=LLMClass)

        return self._convert_dataproto_to_agentloopoutput(rollout_results, kwargs)

    def _convert_dataproto_to_agentloopoutput(
        self, 
        data_proto: DataProto, 
        kwargs: dict
    ) -> AgentLoopOutput:
        batch = data_proto.batch
        
        prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        response_length = self.config.actor_rollout_ref.rollout.response_length
        
        input_ids = batch['input_ids'][0]
        attention_mask = batch['attention_mask'][0]
        
        valid_indices = torch.where(attention_mask == 1)[0]
        
        prompt_ids = input_ids[:prompt_length].tolist()
        response_ids = input_ids[prompt_length:prompt_length + response_length].tolist()
        
        response_mask = [1] * len(response_ids)
        
        response_logprobs = None
        if 'rollout_behavior_log_probs' in batch:
            log_probs = batch['rollout_behavior_log_probs'][0]
            response_logprobs = log_probs[:len(response_ids)].tolist()
        
        metrics = AgentLoopMetrics(
            generate_sequences=0.0,
            tool_calls=0.0
        )
        
        reward_score = None
        if 'extra_data' in data_proto.non_tensor_batch:
            extra_data = data_proto.non_tensor_batch['extra_data']
            if isinstance(extra_data, np.ndarray):
                if extra_data.ndim == 0:
                    extra_data = extra_data.item()
                else:
                    extra_data = extra_data[0]
            if isinstance(extra_data, dict):
                reward_score = extra_data.get('reward', None)
        
        num_turns = 1
        if 'extra_data' in data_proto.non_tensor_batch:
            extra_data = data_proto.non_tensor_batch['extra_data']
            if isinstance(extra_data, np.ndarray):
                if extra_data.ndim == 0:
                    extra_data = extra_data.item()
                else:
                    extra_data = extra_data[0]
            if isinstance(extra_data, dict) and 'stats' in extra_data:
                num_turns = extra_data['stats'].get('total_turns', 1)
        
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            num_turns=num_turns,
            metrics=metrics,
            reward_score=reward_score,
            extra_fields={}
        )
