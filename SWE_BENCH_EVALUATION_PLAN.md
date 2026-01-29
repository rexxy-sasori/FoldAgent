# SWE-bench Evaluation Plan for FoldAgent

## Overview
This plan outlines the steps to evaluate FoldAgent (Context-Folding) on the SWE-bench benchmark, focusing on code repair tasks that require long-horizon reasoning.

## Prerequisites
1. **Branch Setup**: Created `swe-bench` branch (already completed)
2. **Dependencies**: Ensure all required dependencies are installed
3. **Data**: Download and prepare SWE-bench datasets

## Step 1: Environment Setup

### 1.1 Install Required Dependencies
```bash
pip install -r requirements.txt
pip install datasets gitpython tqdm filelock  # For repo cloning
```

### 1.2 Download SWE-bench Repositories
Use the existing `download_repo.py` script to clone SWE-bench and SWE-Gym repositories:

```bash
python scripts/download_repo.py
```

This will create:
- `gym_data/`: Snapshots per SWE-bench instance
- `_repo_cache/`: Bare/partial repos (shared among workers)

## Step 2: Create SWE-bench Evaluation Script

### 2.1 Create `eval_swebench.py`
Create a new evaluation script based on `eval_bc.py` but adapted for SWE-bench tasks:

Key components to include:
- SWE-bench dataset loading
- Code repair task handling
- Evaluation metrics specific to SWE-bench
- Integration with FoldAgent workflow

### 2.2 Configure Agent Workflows
Adapt the existing agent workflows for SWE-bench:

- **ReAct Agent**: `workflow=search`
- **Fold Agent**: `workflow=search_branch` (Context-Folding)
- **Summary Agent**: `workflow=search, enable_summary`

## Step 3: Task Adaptation

### 3.1 SWE-bench Task Format
Modify the agent input format to handle SWE-bench tasks, which typically include:
- Repository context
- Issue description
- Test cases
- Expected behavior

### 3.2 Code Search Integration
Leverage the existing search server for code retrieval during evaluation:

```bash
cd envs && python search_server.py \
  --model Qwen/Qwen3-Embedding-8B \
  --corpus <swe-bench-code-corpus> \
  --host 0.0.0.0 \
  --port 8000
```

## Step 4: Evaluation Execution

### 4.1 Run Evaluation with GPT Models
```bash
export OPENAI_API_KEY='your-key'

python scripts/eval_swebench.py \
  --model_name gpt-5-nano \
  --num_workers 64 \
  --workflow search_branch \
  --prompt_length 16384 \
  --response_length 32768 \
  --max_turn 200 \
  --val_max_turn 200 \
  --max_session 10 \
  --val_max_session 10 \
  --output_dir results/swebench
```

### 4.2 Run Evaluation with Local LLMs
```bash
# Start vLLM server
vllm serve ByteDance-Seed/Seed-OSS-36B-Instruct --port 8001 --max-model-len 131072

# Run evaluation
export OPENAI_API_KEY='dummy'
export OPENAI_BASE_URL='http://localhost:8001/v1'

python scripts/eval_swebench.py \
  --model_name ByteDance-Seed/Seed-OSS-36B-Instruct \
  --num_workers 32 \
  --workflow search_branch \
  --prompt_length 16384 \
  --response_length 32768 \
  --max_turn 200 \
  --val_max_turn 200 \
  --max_session 10 \
  --val_max_session 10 \
  --output_dir results/swebench_local
```

## Step 5: Result Analysis

### 5.1 Evaluate Success Rate
- Measure the percentage of SWE-bench tasks successfully completed
- Compare performance across different agent workflows

### 5.2 Evaluate Efficiency
- Measure completion time per task
- Analyze token usage and context management

### 5.3 Evaluate Code Quality
- Assess the quality of generated code fixes
- Check for test case passing rates

### 5.4 Compare with Baselines
- Compare FoldAgent performance with ReAct and other baselines
- Analyze the impact of context-folding on performance

## Step 6: Reporting

### 6.1 Generate Detailed Report
Create a comprehensive report including:
- Evaluation methodology
- Results across different models and workflows
- Analysis of failure cases
- Insights into context-folding effectiveness for code repair

### 6.2 Visualizations
Generate visualizations to illustrate:
- Success rates by task difficulty
- Performance comparison between agents
- Token usage and context management patterns

## Expected Outcomes

1. **Performance Metrics**: Quantified results showing FoldAgent's effectiveness on SWE-bench
2. **Insights**: Understanding of how context-folding improves code repair capabilities
3. **Best Practices**: Identified optimal configurations for code repair tasks
4. **Future Directions**: Areas for improvement in the context-folding approach

## Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Setup | 1-2 days | Environment setup and data preparation |
| Implementation | 3-5 days | Create evaluation script and adapt agents |
| Evaluation | 3-7 days | Run evaluations with different models |
| Analysis | 2-3 days | Analyze results and generate reports |
| Finalization | 1-2 days | Polish report and documentation |

## Conclusion

This evaluation plan will systematically assess FoldAgent's performance on SWE-bench, providing valuable insights into the effectiveness of context-folding for code repair tasks. The results will contribute to the ongoing research on scaling long-horizon LLM agents and their applications in software engineering.