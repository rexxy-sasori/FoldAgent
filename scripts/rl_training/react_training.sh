#!/bin/bash

export WANDB_DIR=/root/rl-training/wandb
export LOG_DIR=/root/rl-training/logs
export CHECKPOINT_DIR=/root/rl-training/checkpoints
export TRANSFORMERS_VERBOSITY=info
export HYDRA_FULL_ERROR=1

# Set PYTHONPATH for Docker environment
# Docker WORKDIR is /app, so we use /app instead of local path
export PYTHONPATH=/app:/app/agents:$PYTHONPATH


# H200 GPU optimizations
# H200 has 141GB HBM3 memory and high compute capability
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=true

mkdir -p "$WANDB_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

# Model configuration for H200 GPUs
# H200 has 141GB HBM3 memory, can handle large models
PROMPT_LENGTH=16384
RESPONSE_LENGTH=32768
MAX_LENGTH=49152
MODEL_PATH=ByteDance-Seed/Seed-OSS-36B-Instruct

# Data configuration
TRAIN_DATA_PATH=/root/rl-training/data/bc_train.parquet
TEST_DATA_PATH=/root/rl-training/data/bc_test.parquet

# Copy data if not exists
if [ ! -f "$TRAIN_DATA_PATH" ]; then
  echo "Copying training data from built-in directory..."
  mkdir -p "$(dirname "$TRAIN_DATA_PATH")"
  cp -r /app/data/* "$(dirname "$TRAIN_DATA_PATH")/"
fi

# Skip preprocessing for react_agent
# react_agent uses system prompts from agents/prompts.py, not from verl_tool_adaptor
echo "Using training data: $TRAIN_DATA_PATH"

python3 scripts/rl_training/react_training.py \
algorithm.adv_estimator=grpo \
actor_rollout_ref.rollout.agent.default_agent_loop=react_agent \
actor_rollout_ref.rollout.agent.agent_loop_config_path=/app/scripts/rl_training/agent_loop_config.yaml \
actor_rollout_ref.rollout.name=sglang \
actor_rollout_ref.rollout.mode=async \
actor_rollout_ref.rollout.calculate_log_probs=True \
+actor_rollout_ref.rollout.plugin.workflow=search \
+actor_rollout_ref.rollout.plugin.max_turn=20 \
+actor_rollout_ref.rollout.plugin.val_max_turn=20 \
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
data.train_batch_size=16 \
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
trainer.val_before_train=False \
trainer.val_only=False \
trainer.n_gpus_per_node=4 \
trainer.nnodes=1 \
trainer.total_training_steps=100 \
trainer.test_freq=10 \
trainer.save_freq=10 \
trainer.max_actor_ckpt_to_keep=1 \
trainer.max_critic_ckpt_to_keep=1 \
trainer.project_name=react_agent_training \
trainer.experiment_name=seed_oss_36b_instruct_react_h200 \
trainer.logger=console