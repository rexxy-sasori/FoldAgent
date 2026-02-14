import logging
import os
from typing import Any, Union

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl import DataProto
from agents.react_agent import process_item
from agents.utils import CallLLM, TaskContext

# Import react_agent_loop to register it with VERL
import scripts.rl_training.react_agent_loop

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
        item = DataProto.from_dict(non_tensors=kwargs)

        server_host = getattr(self, 'server_host', 'localhost')
        server_port = getattr(self, 'server_port', 8000)
        
        meta_info = {
            'generation_kwargs': sampling_params.get('generation_kwargs', {}),
            'uid': kwargs.get('uid', None)
        }

        llm_client = CallLLM(
            host=server_host,
            port=server_port,
            tokenizer=self.tokenizer,
            config=self.config.actor_rollout_ref.rollout,
            meta_info=meta_info,
            agent_type="react_agent"
        )
        context = TaskContext(
            config=self.config,
            global_step=kwargs.get('global_step', 0),
            server_host=server_host,
            server_port=server_port,
            is_train=kwargs.get('is_train', True),
            run_id=kwargs.get('run_id', 'unknown'),
            tokenizer=self.tokenizer
        )
        rollout_results = await process_item(item, context)

        return rollout_results


def main():
    from verl.trainer.main_ppo import main as verl_main
    verl_main()


if __name__ == "__main__":
    main()
