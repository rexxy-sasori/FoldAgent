#!/bin/bash

set -e
set -x

export WANDB_API_KEY=$WANDB_API_KEY
export WANDB_BASE_URL=$(echo "$WANDB_BASE_URL" | tr -d '`' | xargs)
export WANDB_ENTITY=rexxy-sasori
export WANDB_PROJECT=deepsearch_rlhf

wandb status

export WANDB_DIR=/root/rl-training/wandb
export LOG_DIR=/root/rl-training/logs
export CHECKPOINT_DIR=/root/rl-training/checkpoints
export TRANSFORMERS_VERBOSITY=info
export HYDRA_FULL_ERROR=1

export PYTHONPATH=/app:/app/agents:/app/verl_deepsearch:$PYTHONPATH

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=true

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

export LOCAL_SEARCH_URL=${LOCAL_SEARCH_URL:-http://search-server.liuyunxin:8000}

export JUDGE_OPENAI_API_KEY=${JUDGE_OPENAI_API_KEY}
export JUDGE_OPENAI_BASE_URL=${JUDGE_OPENAI_BASE_URL:-https://lonlie.plus7.plus/v1}
export JUDGE_OPENAI_MODEL=${JUDGE_OPENAI_MODEL:-gpt-4.1}
export JUDGE_OPENAI_URL=${JUDGE_OPENAI_URL:-https://lonlie.plus7.plus/v1/chat/completions}

mkdir -p "$WANDB_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

PROMPT_LENGTH=16384
RESPONSE_LENGTH=32768
MAX_LENGTH=49152
MODEL_PATH=ByteDance-Seed/Seed-OSS-36B-Instruct

TRAIN_DATA_PATH=/root/rl-training/data/bc_train.parquet
TEST_DATA_PATH=/root/rl-training/data/bc_test.parquet

if [ ! -f "$TRAIN_DATA_PATH" ]; then
  echo "Copying training data from built-in directory..."
  mkdir -p "$(dirname "$TRAIN_DATA_PATH")"
  cp -r /app/data/* "$(dirname "$TRAIN_DATA_PATH")/"
fi

echo "Using training data: $TRAIN_DATA_PATH"
echo "Using LOCAL_SEARCH_URL: $LOCAL_SEARCH_URL"
echo "Using JUDGE_OPENAI_MODEL: $JUDGE_OPENAI_MODEL"
echo "SGLang RadixAttention: ENABLED (do NOT disable for performance)"

python3 scripts/rl_training/main.py \
algorithm.adv_estimator=grpo \
algorithm.mask_overlong=False \
actor_rollout_ref.rollout.agent.default_agent_loop=react_agent \
actor_rollout_ref.rollout.agent.agent_loop_config_path=/app/verl_deepsearch/deepsearch_training/configs/react_agent.yaml \
actor_rollout_ref.rollout.name=sglang \
actor_rollout_ref.rollout.mode=async \
actor_rollout_ref.rollout.calculate_log_probs=True \
actor_rollout_ref.rollout.engine_kwargs.sglang.context_length=49152 \
actor_rollout_ref.rollout.engine_kwargs.sglang.allow_auto_truncate=true \
actor_rollout_ref.rollout.engine_kwargs.sglang.disable_radix_attention=false \
actor_rollout_ref.rollout.engine_kwargs.sglang.enable_chunked_prefill=true \
actor_rollout_ref.rollout.engine_kwargs.sglang.max_running_requests=256 \
actor_rollout_ref.rollout.engine_kwargs.sglang.max_total_tokens=1000000 \
actor_rollout_ref.rollout.plugin.workflow=search \
actor_rollout_ref.rollout.plugin.max_turns=64 \
actor_rollout_ref.rollout.plugin.val_max_turns=200 \
actor_rollout_ref.rollout.plugin.session_timeout=5400 \
actor_rollout_ref.rollout.plugin.enable_summary=false \
actor_rollout_ref.rollout.plugin.max_tool_response_length=1000 \
actor_rollout_ref.rollout.plugin.tool_response_truncate_side=right \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.rollout.prompt_length=$PROMPT_LENGTH \
actor_rollout_ref.rollout.response_length=$RESPONSE_LENGTH \
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_LENGTH \
actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
actor_rollout_ref.rollout.n=4 \
actor_rollout_ref.rollout.agent.num_workers=4 \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
data.train_files=$TRAIN_DATA_PATH \
data.val_files=$TEST_DATA_PATH \
data.train_batch_size=64 \
data.max_prompt_length=$PROMPT_LENGTH \
data.max_response_length=$RESPONSE_LENGTH \
data.return_raw_chat=True \
actor_rollout_ref.actor.ppo_mini_batch_size=16 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_LENGTH \
actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=$MAX_LENGTH \
actor_rollout_ref.rollout.dtype=bfloat16 \
actor_rollout_ref.rollout.do_sample=True \
reward_model.type=judge \
reward_model.judge_openai_api_key=$JUDGE_OPENAI_API_KEY \
reward_model.judge_openai_base_url=$JUDGE_OPENAI_BASE_URL \
reward_model.judge_openai_model=$JUDGE_OPENAI_MODEL \
reward_model.judge_openai_url=$JUDGE_OPENAI_URL \
reward_model.max_retries=3 \
reward_model.timeout=30 \
reward_model.semaphore_limit=50 \
++reward_model.reward_manager=judge \
reward_model.use_reward_loop=True \
+ray_kwargs.runtime_env.env_vars.PYTHONPATH=/app:/app/agents:/app/verl_deepsearch:$PYTHONPATH \
trainer.val_before_train=False \
trainer.val_only=False \
trainer.n_gpus_per_node=4 \
trainer.nnodes=1 \
trainer.total_training_steps=1000 \
trainer.test_freq=10 \
trainer.save_freq=10 \
trainer.max_actor_ckpt_to_keep=1 \
trainer.max_critic_ckpt_to_keep=1 \
trainer.project_name=deepsearch_rlhf \
trainer.experiment_name=react_agent_judge_reward \
trainer.logger=wandb
