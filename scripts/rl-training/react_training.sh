#!/bin/bash

export WANDB_DIR=/root/react-training/wandb
export LOG_DIR=/root/react-training/logs
export CHECKPOINT_DIR=/root/react-training/checkpoints
export TRANSFORMERS_VERBOSITY=info
export HYDRA_FULL_ERROR=1
export SEARCH_SERVER_URL=${SEARCH_SERVER_URL:-"http://localhost:8000"}

mkdir -p "$WANDB_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$CHECKPOINT_DIR"

PROMPT_LENGTH=16384
RESPONSE_LENGTH=32768
MAX_LENGTH=49152
MODEL_PATH=ByteDance-Seed/Seed-OSS-36B-Instruct
TRAIN_DATA_PATH=/root/react-training/data/bc_train.parquet
TEST_DATA_PATH=/root/react-training/data/bc_test.parquet

if [ ! -f "$TRAIN_DATA_PATH" ]; then
echo "Copying training data from built-in directory..."
mkdir -p "$(dirname "$TRAIN_DATA_PATH")"
cp -r /app/data/* "$(dirname "$TRAIN_DATA_PATH")/"
fi

echo "Preprocessing training data with custom system prompt..."
python3 scripts/preprocess_data_with_system_prompt.py \
  --input "$TRAIN_DATA_PATH" \
  --output "$(dirname "$TRAIN_DATA_PATH")/bc_train_with_system_prompt.parquet" \
  --system_prompt /app/envs/verl_tool_adaptor/system_prompt_template.txt

TRAIN_DATA_PATH=$(dirname "$TRAIN_DATA_PATH")/bc_train_with_system_prompt.parquet
echo "Using preprocessed data: $TRAIN_DATA_PATH"

pip install transformers==4.57.1 

python3 scripts/rl-training/react_training.py \
algorithm.adv_estimator=foldgrpo \
actor_rollout_ref.rollout.agent.default_agent_loop=react_agent \
actor_rollout_ref.rollout.name=vllm \
actor_rollout_ref.rollout.mode=async \
actor_rollout_ref.rollout.calculate_log_probs=True \
actor_rollout_ref.rollout.plugin.workflow=search \
actor_rollout_ref.rollout.plugin.max_turn=20 \
actor_rollout_ref.rollout.plugin.val_max_turn=20 \
data.apply_chat_template_kwargs.add_generation_prompt=True \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.rollout.prompt_length=$PROMPT_LENGTH \
actor_rollout_ref.rollout.response_length=$RESPONSE_LENGTH \
actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_LENGTH \
actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
actor_rollout_ref.rollout.n=4 \
actor_rollout_ref.rollout.agent.num_workers=1 \
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
actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_LENGTH \
actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=$MAX_LENGTH \
actor_rollout_ref.rollout.dtype=bfloat16 \
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
trainer.experiment_name=seed_oss_36b_instruct_react
