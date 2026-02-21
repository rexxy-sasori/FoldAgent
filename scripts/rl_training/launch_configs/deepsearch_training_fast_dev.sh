#!/bin/bash

set -e
set -x

export WANDB_API_KEY=$WANDB_API_KEY
export WANDB_BASE_URL=$(echo "$WANDB_BASE_URL" | tr -d '`' | xargs)
export WANDB_ENTITY=rexxy-sasori
export WANDB_PROJECT=deepsearch_rlhf

export WANDB_DIR=/root/rl-training/wandb
export LOG_DIR=/root/rl-training/logs
export CHECKPOINT_DIR=/root/rl-training/checkpoints
export TRANSFORMERS_VERBOSITY=info
export HYDRA_FULL_ERROR=1

export PYTHONPATH=/app:/app/agents:/app/verl_deepsearch:$PYTHONPATH

export NCCL_P2P_LEVEL=PIX
export NCCL_P2P_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export TOKENIZERS_PARALLELISM=true

export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_ATTENTION_BACKEND=flashinfer
export SGLANG_DISABLE_CUDA_GRAPH=1
export SGLANG_ENABLE_JIT_DEEPGEMM=0
export SGLANG_TORCH_COMPILE_MAX_BS=128
export SGLANG_ENABLE_PREFIX_CACHING=1
export SGLANG_WORKER_THREADS=8

unset CUDA_DEVICE_MAX_CONNECTIONS

export LOCAL_SEARCH_URL=${LOCAL_SEARCH_URL:-http://search-server.liuyunxin:8000}

export JUDGE_OPENAI_API_KEY=${JUDGE_OPENAI_API_KEY}
export JUDGE_OPENAI_BASE_URL=${JUDGE_OPENAI_BASE_URL:-https://lonlie.plus7.plus/v1}
export JUDGE_OPENAI_MODEL=${JUDGE_OPENAI_MODEL:-gpt-4.1}
export JUDGE_OPENAI_URL=${JUDGE_OPENAI_URL:-https://lonlie.plus7.plus/v1/chat/completions}

mkdir -p "$WANDB_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

PROMPT_LENGTH=4096
RESPONSE_LENGTH=8192
MAX_LENGTH=24576
MODEL_PATH=ByteDance-Seed/Seed-OSS-36B-Instruct

TRAIN_DATA_PATH=/root/rl-training/data/bc_train_with_system.parquet
TEST_DATA_PATH=/root/rl-training/data/bc_test_with_system.parquet

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
algorithm.mask_overlong=True \
actor_rollout_ref.rollout.agent.default_agent_loop=react_agent \
actor_rollout_ref.rollout.agent.agent_loop_config_path=/app/verl_deepsearch/deepsearch_training/configs/react_agent.yaml \
actor_rollout_ref.rollout.name=sglang \
actor_rollout_ref.rollout.mode=async \
actor_rollout_ref.rollout.disable_log_stats=False \
actor_rollout_ref.rollout.prometheus.enable=True \
actor_rollout_ref.rollout.calculate_log_probs=True \
actor_rollout_ref.rollout.enforce_eager=False \
actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
actor_rollout_ref.rollout.multi_turn.enable=True \
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
+actor_rollout_ref.rollout.plugin.workflow=search \
+actor_rollout_ref.rollout.plugin.max_turn=12 \
+actor_rollout_ref.rollout.plugin.val_max_turn=12 \
+actor_rollout_ref.rollout.plugin.session_timeout=300 \
+actor_rollout_ref.rollout.plugin.enable_summary=false \
+actor_rollout_ref.rollout.plugin.max_tool_response_length=1000 \
+actor_rollout_ref.rollout.plugin.tool_response_truncate_side=right \
+actor_rollout_ref.rollout.engine_kwargs.sglang.context_length=$MAX_LENGTH \
+actor_rollout_ref.rollout.engine_kwargs.sglang.allow_auto_truncate=true \
+actor_rollout_ref.rollout.engine_kwargs.sglang.max_running_requests=64 \
+actor_rollout_ref.rollout.engine_kwargs.sglang.chunked_prefill_size=4096 \
+actor_rollout_ref.rollout.engine_kwargs.sglang.trust_remote_code=true \
+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
actor_rollout_ref.actor.strategy="fsdp2" \
actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32000 \
actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
actor_rollout_ref.actor.entropy_checkpointing=True \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.rollout.prompt_length=$PROMPT_LENGTH \
actor_rollout_ref.rollout.response_length=$RESPONSE_LENGTH \
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_LENGTH \
actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
actor_rollout_ref.rollout.n=2 \
actor_rollout_ref.rollout.agent.num_workers=16 \
actor_rollout_ref.actor.use_dynamic_bsz=True \
+actor_rollout_ref.ref.use_dynamic_bsz=True \
data.train_files=$TRAIN_DATA_PATH \
data.val_files=$TEST_DATA_PATH \
data.train_batch_size=64 \
data.max_prompt_length=$PROMPT_LENGTH \
data.max_response_length=$RESPONSE_LENGTH \
data.return_raw_chat=True \
+data.add_uid=True \
+data.val_batch_size=128 \
actor_rollout_ref.actor.ppo_mini_batch_size=64 \
actor_rollout_ref.actor.fsdp_config.param_offload=True \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_LENGTH \
actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=$MAX_LENGTH \
actor_rollout_ref.rollout.dtype=bfloat16 \
actor_rollout_ref.rollout.do_sample=True \
+reward_model.type=judge \
+reward_model.judge_openai_api_key=$JUDGE_OPENAI_API_KEY \
+reward_model.judge_openai_base_url=$JUDGE_OPENAI_BASE_URL \
+reward_model.judge_openai_model=$JUDGE_OPENAI_MODEL \
+reward_model.judge_openai_url=$JUDGE_OPENAI_URL \
+reward_model.max_retries=3 \
+reward_model.timeout=120 \
+reward_model.semaphore_limit=100 \
reward_model.reward_manager=judge \
reward_model.use_reward_loop=True \
+ray_kwargs.runtime_env.env_vars.PYTHONPATH=/app:/app/agents:/app/verl_deepsearch:$PYTHONPATH \
+ray_kwargs.runtime_env.env_vars.VERL_AUTO_PADDING=TRUE \
trainer.val_before_train=True \
trainer.val_only=False \
trainer.n_gpus_per_node=4 \
trainer.nnodes=1 \
trainer.total_training_steps=100 \
trainer.test_freq=10 \
trainer.save_freq=10 \
trainer.max_actor_ckpt_to_keep=1 \
trainer.max_critic_ckpt_to_keep=1 \
trainer.project_name=deepsearch_rlhf \
trainer.experiment_name=react_agent_judge_reward_fast_dev \
trainer.logger=console