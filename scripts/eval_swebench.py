#!/usr/bin/env python3
import asyncio
import argparse
import json
import numpy as np
import pandas as pd
import sys
import warnings
import logging
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import os

# Set up run-specific output directory
run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_id = f"run_{run_timestamp}"
run_dir = Path('/root') / run_timestamp
run_dir.mkdir(parents=True, exist_ok=True)
log_filename = run_dir / f'eval_swebench.log'

# Read logging configuration from environment variables
log_level = os.environ.get('LOG_LEVEL', 'DEBUG').upper()
log_format = os.environ.get('LOG_FORMAT', f'%(asctime)s - {run_id} - %(name)s - %(levelname)s - %(message)s')

logging.basicConfig(
    level=getattr(logging, log_level, logging.DEBUG),
    format=log_format,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_filename))
    ]
)
logger = logging.getLogger('eval_swebench')

warnings.filterwarnings('ignore', message='.*fast tokenizer.*')

from omegaconf import OmegaConf
from transformers import AutoTokenizer
from agents.utils import CallAPI, TaskContext
from verl import DataProto


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate agents on SWE-bench benchmark')
    parser.add_argument('--data_path', default='data/swebench_test.parquet',
                        help='Path to test data parquet file (default: data/swebench_test.parquet)')
    parser.add_argument('--output_dir', default='results/swebench',
                        help='Directory to save evaluation results (default: results/swebench)')
    parser.add_argument('--prompt_length', type=int, default=16384,
                        help='Maximum prompt length in tokens (default: 16384)')
    parser.add_argument('--response_length', type=int, default=32768,
                        help='Maximum response length in tokens (default: 32768)')
    parser.add_argument('--workflow', default='search_branch',
                        help='Agent workflow: "search" for ReAct, "search_branch" for Context-Folding (default: search_branch)')
    parser.add_argument('--max_turn', type=int, default=200,
                        help='Maximum turns during training (default: 200)')
    parser.add_argument('--val_max_turn', type=int, default=200,
                        help='Maximum turns during validation/evaluation (default: 200)')
    parser.add_argument('--max_session', type=int, default=10,
                        help='Maximum branch sessions for Context-Folding during training (default: 10)')
    parser.add_argument('--val_max_session', type=int, default=10,
                        help='Maximum branch sessions for Context-Folding during validation (default: 10)')
    parser.add_argument('--model_name', default='gpt-5-nano',
                        help='Model name for API (e.g., gpt-5-nano, gpt-4o, or vLLM model path) (default: gpt-5-nano)')
    parser.add_argument('--num_workers', type=int, default=64,
                        help='Number of parallel evaluation workers (default: 64)')
    parser.add_argument('--local_search_url', default='http://localhost:8000',
                        help='URL of the local search server (default: http://localhost:8000)')
    parser.add_argument('--enable_summary', action='store_true',
                        help='Enable summary mode (use with workflow=search for Summary agent)')
    parser.add_argument('--track-log-prob', action='store_true',
                        help='Track log probabilities from LLM responses for variance analysis (default: False)')
    parser.add_argument('--use-swegym', action='store_true',
                        help='Include SWE-Gym dataset in evaluation (default: False)')
    return parser.parse_args()


async def eval_one(row, config, tokenizer, model_name, run_id):
    instance_id = row['instance_id']
    item_logger = logging.getLogger(f'eval_swebench.item-{instance_id}')
    item_logger.info(f"Starting evaluation for instance_id={instance_id}")
    
    item_logger.debug(f"Creating TaskContext with model: {model_name}")
    context = TaskContext(config=config, global_step=0, server_host=model_name,
                          server_port=0, is_train=False, run_id=run_id, tokenizer=tokenizer)

    item_logger.debug(f"Creating DataProto for instance")
    item = DataProto()
    item.non_tensor_batch = {
        'ability': np.array(["code_repair"], dtype=object),
        'extra_info': np.array([{
            'instance_id': instance_id,
            'repo': row['repo'],
            'base_commit': row['base_commit'],
            'patch': row.get('patch', ''),
            'problem_statement': row.get('problem_statement', ''),
            'test_patch': row.get('test_patch', ''),
            'test_filename': row.get('test_filename', ''),
            'test_command': row.get('test_command', ''),
        }], dtype=object),
        'uid': np.array([instance_id], dtype=object),
        'reward_model': np.array(["swebench_test"], dtype=object),
    }
    item.meta_info = {'generation_kwargs': {}, 'max_turn': config.actor_rollout_ref.rollout.plugin.val_max_turn}

    # Get workflow from config and select the appropriate agent
    workflow = config.actor_rollout_ref.rollout.plugin.workflow
    item_logger.info(f"Using workflow: {workflow}")
    
    if workflow == 'search':
        from agents.react_agent import process_item as react_process_item
        item_logger.info(f"Calling react_process_item for evaluation")
        output = await react_process_item(item, context, CallAPI)
    else:
        from agents.fold_agent import process_item as fold_process_item
        item_logger.info(f"Calling fold_process_item for evaluation")
        output = await fold_process_item(item, context, CallAPI)
    item_logger.info(f"process_item completed")

    score = output.non_tensor_batch.get('extra_data', [{}])[0].get('stats', {}).get('score', 0) if output else 0
    completion_time = output.non_tensor_batch.get('extra_data', [{}])[0].get('stats', {}).get('completion_time', 0) if output else 0
    request_id = output.non_tensor_batch.get('extra_data', [{}])[0].get('stats', {}).get('request_id', '') if output else ''
    status = 'success' if output else 'failed'
    repo = row.get('repo', 'unknown')
    
    result = {
        'instance_id': instance_id,
        'repo': repo,
        'score': score,
        'status': status,
        'completion_time': completion_time,
        'request_id': request_id
    }
    
    item_logger.info(f"Evaluation result: instance_id={instance_id}, repo={repo}, status={status}, score={score}, request_id={request_id}")
    return result


async def worker(worker_id, rows, args, pbar, shared_scores, run_id):
    worker_logger = logging.getLogger(f'eval_swebench.worker-{worker_id}')
    worker_logger.info(f"Initializing worker {worker_id} with {len(rows)} items")
    
    worker_logger.info(f"Loading tokenizer for model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    worker_logger.info(f"Tokenizer loaded successfully")

    config = OmegaConf.create({
        'actor_rollout_ref': {'rollout': {
            'prompt_length': args.prompt_length,
            'response_length': args.response_length,
            'plugin': {
                'workflow': args.workflow,
                'max_turn': args.max_turn,
                'val_max_turn': args.val_max_turn,
                'max_session': args.max_session,
                'val_max_session': args.val_max_session,
                'session_timeout': 5400,
                'process_reward': None,
                'max_traj': None,
                'must_finish': False,
                'double_check': False,
                'must_search': False,
                'enable_summary': args.enable_summary,
                'track_log_prob': args.track_log_prob
            }
        }}
    })

    results = []
    for idx, row in enumerate(rows):
        instance_id = row['instance_id']
        worker_logger.info(f"Processing item {idx+1}/{len(rows)}: instance_id={instance_id}")
        
        result = await eval_one(row, config, tokenizer, args.model_name, run_id)
        
        worker_logger.info(f"Completed item {idx+1}/{len(rows)}: instance_id={result['instance_id']}, status={result['status']}, score={result['score']}")
        
        results.append(result)
        shared_scores.append(result['score'])
        avg_score = np.mean(shared_scores)
        pbar.set_postfix({'avg_score': f"{avg_score:.3f}", 'id': result['instance_id']})
        pbar.update(1)

    return results


def load_swebench_data(args):
    from datasets import load_dataset
    
    logger.info("Loading SWE-bench Verified dataset")
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    
    if args.use_swegym:
        logger.info("Loading SWE-Gym dataset")
        ds_swegym = load_dataset("SWE-Gym/SWE-Gym", split="train")
        ds = ds + ds_swegym
    
    logger.info(f"Successfully loaded {len(ds)} items")
    return ds


def main():
    logger.info(f"Starting SWE-bench evaluation script - {run_id}")
    logger.info(f"Logging to file: {log_filename}")
    
    args = parse_args()
    logger.info(f"Parsed arguments: {args}")
    
    os.environ["LOCAL_SEARCH_URL"] = args.local_search_url
    logger.info(f"Set LOCAL_SEARCH_URL to {args.local_search_url}")

    # Load data
    logger.info("Loading SWE-bench datasets")
    dataset = load_swebench_data(args)
    df = pd.DataFrame(dataset)
    logger.info(f"Successfully loaded {len(df)} items")

    # Split for workers
    logger.info(f"Splitting data into {args.num_workers} chunks")
    chunk_size = len(df) // args.num_workers
    chunks = [df.iloc[i*chunk_size:(i+1)*chunk_size if i < args.num_workers-1 else len(df)]
              for i in range(args.num_workers)]
    logger.info(f"Created {len(chunks)} chunks with sizes: {[len(c) for c in chunks]}")

    # Run workers with progress bar
    logger.info(f"Initializing evaluation with {args.num_workers} workers")
    logger.info(f"Using workflow: {args.workflow}, model: {args.model_name}")
    logger.info(f"Configuration: prompt_length={args.prompt_length}, response_length={args.response_length}")
    logger.info(f"Max turns: {args.max_turn}, max sessions: {args.max_session}")
    
    async def run_all():
        shared_scores = []
        with tqdm(total=len(df), desc="Evaluating", unit="item") as pbar:
            logger.info("Starting worker tasks")
            tasks = [worker(i, [chunks[i].iloc[j] for j in range(len(chunks[i]))], args, pbar, shared_scores, run_id)
                     for i in range(args.num_workers)]
            logger.info(f"Launched {len(tasks)} worker tasks")
            return await asyncio.gather(*tasks)

    all_results = asyncio.run(run_all())
    results = [r for worker_results in all_results for r in worker_results]

    # Summary overall
    avg_score = np.mean([r['score'] for r in results])
    success_count = sum(r['status']=='success' for r in results)
    
    # Completion time statistics
    completion_times = [r['completion_time'] for r in results]
    avg_time = np.mean(completion_times)
    median_time = np.median(completion_times)
    min_time = np.min(completion_times)
    max_time = np.max(completion_times)
    std_time = np.std(completion_times)
    
    logger.info(f"{'='*60}")
    logger.info(f"{'='*60}")
    logger.info(f"Overall - Avg Score: {avg_score:.4f}, Success: {success_count}/{len(results)}")
    logger.info(f"Completion Time Statistics:")
    logger.info(f"  Average: {avg_time:.2f} seconds")
    logger.info(f"  Median: {median_time:.2f} seconds")
    logger.info(f"  Min: {min_time:.2f} seconds")
    logger.info(f"  Max: {max_time:.2f} seconds")
    logger.info(f"  Std: {std_time:.2f} seconds")

    # Summary by repository
    from collections import defaultdict
    by_repo = defaultdict(list)
    for r in results:
        by_repo[r['repo']].append(r['score'])

    logger.info(f"\nBy Repository:")
    for repo in sorted(by_repo.keys()):
        scores = by_repo[repo]
        repo_avg = np.mean(scores)
        repo_count = len(scores)
        logger.info(f"  {repo}: {repo_avg:.4f} ({repo_count} items)")

    # Save
    logger.info(f"Saving results...")
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    
    # Save results in both the run directory and the specified output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_filename = f"results_{run_id}_{timestamp}.json"
    
    # Save in run-specific directory
    run_results_file = run_dir / results_filename
    logger.info(f"Dumping results to run directory: {run_results_file}")
    
    # Save in specified output directory
    output_file = Path(args.output_dir) / results_filename
    logger.info(f"Dumping results to output directory: {output_file}")
    
    logger.info(f"Creating summary statistics")
    summary_by_repo = {repo: {'avg_score': float(np.mean(scores)), 'count': len(scores)}
                         for repo, scores in by_repo.items()}
    
    results_data = {
        'run_id': run_id,
        'log_file': str(log_filename),
        'run_dir': str(run_dir),
        'avg_score': avg_score,
        'by_repo': summary_by_repo,
        'results': results
    }
    
    # Write to both locations
    json.dump(results_data, open(run_results_file, 'w'), indent=2)
    json.dump(results_data, open(output_file, 'w'), indent=2)
    
    logger.info(f"Results saved successfully to {output_file}")


if __name__ == "__main__":
    main()