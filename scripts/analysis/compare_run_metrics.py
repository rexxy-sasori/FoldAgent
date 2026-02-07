#!/usr/bin/env python3
"""
Comprehensive analysis script to compare two specified run_ids across multiple difficulty levels.

Calculates and presents the following metrics for each difficulty level:
1) Success rate: defined as the percentage of successful operations relative to total operations
2) New tokens computed: calculated as total_prompt tokens minus cached tokens
3) New tokens per item: computed as new tokens divided by the number of items processed

Reads DATABASE_URL configuration from deployment/observability/logdb/postgresql-secrets.yaml
"""
import os
import sys
import json
import csv
import argparse
import asyncio
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path for importing db_client
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from agents.db_client import get_event_db

async def read_database_url() -> str:
    """
    Read DATABASE_URL from environment variable.
    
    Returns:
        str: The DATABASE_URL configuration
    """
    # Check environment variable
    db_url = os.environ.get('DATABASE_URL')
    if db_url:
        return db_url
    else:
        raise ValueError("DATABASE_URL environment variable is not specified")

async def validate_run_ids(db, run_id1: str, run_id2: str) -> Tuple[bool, bool]:
    """
    Validate that both run_ids exist in the database.
    
    Args:
        db: Database client instance
        run_id1: First run ID to validate
        run_id2: Second run ID to validate
    
    Returns:
        Tuple[bool, bool]: (run_id1_exists, run_id2_exists)
    """
    try:
        events1 = await db.get_events_by_run_id(run_id1)
        events2 = await db.get_events_by_run_id(run_id2)
        
        return len(events1) > 0, len(events2) > 0
    except Exception as e:
        print(f"Error validating run IDs: {e}")
        return False, False

async def analyze_run(db, run_id: str) -> Dict[str, Any]:
    """
    Analyze a single run and calculate metrics by difficulty level and item.
    
    Args:
        db: Database client instance
        run_id: Run ID to analyze
    
    Returns:
        Dict[str, Any]: Metrics organized by difficulty level
    """
    try:
        events = await db.get_events_by_run_id(run_id)
        
        # Organize events by difficulty level
        difficulty_metrics = {}
        # Also organize events by item ID for per-item analysis
        item_metrics = {}
        
        for event in events:
            event_data = event.get('event_data', {})
            event_type = event.get('event_type', '')
            request_id = event.get('request_id', '')
            
            # Extract item ID from request_id (first part before '_')
            item_id = request_id.split('_')[0] if '_' in request_id else 'unknown'
            
            # Get difficulty from various sources or default to unknown
            difficulty = 'unknown'
            # Try to get difficulty from event data if present
            if event_data.get('difficulty'):
                difficulty = event_data.get('difficulty', 'unknown')
            # Try to infer difficulty from ability if present
            elif event_data.get('ability'):
                difficulty = event_data.get('ability', 'unknown')
            
            if difficulty not in difficulty_metrics:
                difficulty_metrics[difficulty] = {
                    'total_operations': 0,
                    'successful_operations': 0,
                    'total_prompt_tokens': 0,
                    'cached_tokens': 0,
                    'items_processed': 0,
                    'total_reward': 0.0,
                    'items_with_reward_1': 0,
                    'reward_evaluations': 0,
                    'total_duration': 0.0,
                    'llm_responses': 0,
                    'unique_items': set()
                }
            
            # Add item_id to unique_items set (exclude 'unknown' items)
            if item_id != 'unknown':
                difficulty_metrics[difficulty]['unique_items'].add(item_id)
            
            # Initialize item metrics if not present
            if item_id not in item_metrics:
                item_metrics[item_id] = {
                    'total_prompt_tokens': 0,
                    'cached_tokens': 0,
                    'total_reward': 0.0,
                    'reward_score': 0.0,
                    'total_duration': 0.0,
                    'llm_responses': 0,
                    'reward_evaluations': 0
                }
            
            # Update metrics based on event type and data
            metrics = difficulty_metrics[difficulty]
            item_metric = item_metrics[item_id]
            
            # Track token metrics from llm_response events
            if event_type == 'llm_response':
                prompt_tokens = event_data.get('prompt_tokens', 0)
                cached_tokens = event_data.get('cached_tokens', 0)
                duration = event_data.get('duration', 0.0)
                
                metrics['total_prompt_tokens'] += prompt_tokens
                metrics['cached_tokens'] += cached_tokens
                metrics['total_duration'] += duration
                metrics['llm_responses'] += 1
                
                item_metric['total_prompt_tokens'] += prompt_tokens
                item_metric['cached_tokens'] += cached_tokens
                item_metric['total_duration'] += duration
                item_metric['llm_responses'] += 1
            
            # Track reward information from reward_evaluation_complete events
            if event_type == 'reward_evaluation_complete':
                reward_score = event_data.get('reward_score', 0.0)
                metrics['total_reward'] += reward_score
                metrics['reward_evaluations'] += 1
                # Count items with reward=1
                if reward_score == 1.0:
                    metrics['items_with_reward_1'] += 1
                
                item_metric['total_reward'] += reward_score
                item_metric['reward_score'] = reward_score  # Store the final reward
                item_metric['reward_evaluations'] += 1
            
            # Track operations and success
            # Consider inference_complete as a successful operation
            if event_type == 'inference_complete':
                metrics['total_operations'] += 1
                metrics['successful_operations'] += 1
            # Consider branch and return events as operations
            elif event_type in ['branch', 'return']:
                metrics['total_operations'] += 1
                metrics['successful_operations'] += 1
            # Consider llm_response as a successful operation
            elif event_type == 'llm_response':
                metrics['total_operations'] += 1
                metrics['successful_operations'] += 1
            # Consider tool-related events
            elif event_type in ['search', 'open_page', 'finish']:
                metrics['total_operations'] += 1
                # Assume success unless there's an error in the observation
                observation = event_data.get('observation', '')
                if 'Error' not in observation and 'error' not in observation:
                    metrics['successful_operations'] += 1
            # Consider reward_evaluation_complete as an operation
            elif event_type == 'reward_evaluation_complete':
                metrics['total_operations'] += 1
                metrics['successful_operations'] += 1
            
            # Track items processed (default to 1 if not specified)
            metrics['items_processed'] += event_data.get('items_processed', 1)
        
        # Calculate derived metrics
        for difficulty, metrics in difficulty_metrics.items():
            # Calculate success rate
            if metrics['total_operations'] > 0:
                metrics['success_rate'] = (
                    metrics['successful_operations'] / metrics['total_operations'] * 100
                )
            else:
                metrics['success_rate'] = 0.0
            
            # Calculate new tokens computed
            metrics['new_tokens_computed'] = (
                metrics['total_prompt_tokens'] - metrics['cached_tokens']
            )
            
            # Calculate unique items count
            unique_items_count = len(metrics['unique_items'])
            
            # Calculate new tokens per item
            if unique_items_count > 0:
                metrics['new_tokens_per_item'] = (
                    metrics['new_tokens_computed'] / unique_items_count
                )
            elif metrics['items_processed'] > 0:
                metrics['new_tokens_per_item'] = (
                    metrics['new_tokens_computed'] / metrics['items_processed']
                )
            else:
                metrics['new_tokens_per_item'] = 0.0
            
            # Calculate operations per item
            if unique_items_count > 0:
                metrics['operations_per_item'] = (
                    metrics['total_operations'] / unique_items_count
                )
            elif metrics['items_processed'] > 0:
                metrics['operations_per_item'] = (
                    metrics['total_operations'] / metrics['items_processed']
                )
            else:
                metrics['operations_per_item'] = 0.0
            
            # Calculate average reward per evaluation
            if metrics['reward_evaluations'] > 0:
                metrics['average_reward'] = (
                    metrics['total_reward'] / metrics['reward_evaluations']
                )
            else:
                metrics['average_reward'] = 0.0
            
            # Calculate average duration per LLM response
            if metrics['llm_responses'] > 0:
                metrics['average_duration_per_llm'] = (
                    metrics['total_duration'] / metrics['llm_responses']
                )
            else:
                metrics['average_duration_per_llm'] = 0.0
            
            # Remove unique_items set since it's not JSON serializable
            if 'unique_items' in metrics:
                del metrics['unique_items']
        
        # Add item metrics to the result for possible per-item analysis
        difficulty_metrics['_item_metrics'] = item_metrics
        
        return difficulty_metrics
    except Exception as e:
        print(f"Error analyzing run {run_id}: {e}")
        return {}

async def generate_output(metrics1: Dict[str, Any], metrics2: Dict[str, Any], 
                          run_id1: str, run_id2: str, output_format: str):
    """
    Generate structured output in the specified format.
    
    Args:
        metrics1: Metrics for first run
        metrics2: Metrics for second run
        run_id1: First run ID
        run_id2: Second run ID
        output_format: Output format (csv, json, or both)
    """
    # Get all unique difficulty levels, excluding '_item_metrics'
    all_difficulties = sorted(set([d for d in list(metrics1.keys()) + list(metrics2.keys()) if d != '_item_metrics']))
    
    # Extract item metrics for per-item analysis
    item_metrics1 = metrics1.get('_item_metrics', {})
    item_metrics2 = metrics2.get('_item_metrics', {})
    
    # Prepare data for output
    output_data = []
    for difficulty in all_difficulties:
            data = {
                'difficulty': difficulty,
                f'{run_id1}_success_rate': metrics1.get(difficulty, {}).get('success_rate', 0.0),
                f'{run_id2}_success_rate': metrics2.get(difficulty, {}).get('success_rate', 0.0),
                f'{run_id1}_new_tokens_computed': metrics1.get(difficulty, {}).get('new_tokens_computed', 0),
                f'{run_id2}_new_tokens_computed': metrics2.get(difficulty, {}).get('new_tokens_computed', 0),
                f'{run_id1}_new_tokens_per_item': metrics1.get(difficulty, {}).get('new_tokens_per_item', 0.0),
                f'{run_id2}_new_tokens_per_item': metrics2.get(difficulty, {}).get('new_tokens_per_item', 0.0),
                f'{run_id1}_operations_per_item': metrics1.get(difficulty, {}).get('operations_per_item', 0.0),
                f'{run_id2}_operations_per_item': metrics2.get(difficulty, {}).get('operations_per_item', 0.0),
                f'{run_id1}_items_with_reward_1': metrics1.get(difficulty, {}).get('items_with_reward_1', 0),
                f'{run_id2}_items_with_reward_1': metrics2.get(difficulty, {}).get('items_with_reward_1', 0),
                f'{run_id1}_total_reward': metrics1.get(difficulty, {}).get('total_reward', 0.0),
                f'{run_id2}_total_reward': metrics2.get(difficulty, {}).get('total_reward', 0.0),
                f'{run_id1}_reward_evaluations': metrics1.get(difficulty, {}).get('reward_evaluations', 0),
                f'{run_id2}_reward_evaluations': metrics2.get(difficulty, {}).get('reward_evaluations', 0),
                f'{run_id1}_average_reward': metrics1.get(difficulty, {}).get('average_reward', 0.0),
                f'{run_id2}_average_reward': metrics2.get(difficulty, {}).get('average_reward', 0.0),
                f'{run_id1}_average_duration_per_llm': metrics1.get(difficulty, {}).get('average_duration_per_llm', 0.0),
                f'{run_id2}_average_duration_per_llm': metrics2.get(difficulty, {}).get('average_duration_per_llm', 0.0),
                f'{run_id1}_total_operations': metrics1.get(difficulty, {}).get('total_operations', 0),
                f'{run_id2}_total_operations': metrics2.get(difficulty, {}).get('total_operations', 0),
                f'{run_id1}_items_processed': metrics1.get(difficulty, {}).get('items_processed', 0),
                f'{run_id2}_items_processed': metrics2.get(difficulty, {}).get('items_processed', 0),
            }
            output_data.append(data)
    
    # Generate JSON output
    if output_format in ['json', 'both']:
        json_output = {
            'run_ids': {
                'run_id1': run_id1,
                'run_id2': run_id2
            },
            'metrics': output_data,
            'item_analysis': {
                'run_id1_items': len(item_metrics1),
                'run_id2_items': len(item_metrics2),
                'common_items': len([item_id for item_id in item_metrics1 if item_id in item_metrics2])
            }
        }
        
        with open(f'run_comparison_{run_id1}_{run_id2}.json', 'w') as f:
            json.dump(json_output, f, indent=2)
        print(f"JSON output saved to run_comparison_{run_id1}_{run_id2}.json")
    
    # Generate CSV output
    if output_format in ['csv', 'both']:
        if output_data:
            fieldnames = ['difficulty']
            # Add metric fields in order
            metric_types = ['success_rate', 'new_tokens_computed', 'new_tokens_per_item', 
                           'operations_per_item', 'items_with_reward_1', 'total_reward', 
                           'reward_evaluations', 'average_reward', 'average_duration_per_llm', 
                           'total_operations', 'items_processed']
            for run_id in [run_id1, run_id2]:
                for metric in metric_types:
                    fieldnames.append(f'{run_id}_{metric}')
            
            with open(f'run_comparison_{run_id1}_{run_id2}.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(output_data)
            print(f"CSV output saved to run_comparison_{run_id1}_{run_id2}.csv")
        else:
            print("No data available for CSV output")
    
    # Print summary to console
    print("\nRun Comparison Summary:")
    print(f"Comparing run_id1: {run_id1} vs run_id2: {run_id2}")
    print("=" * 160)
    print(f"{'Difficulty':<15} {'Success Rate (%)':<20} {'New Tokens':<15} {'Tokens/Item':<15} {'Ops/Item':<10} {'Reward=1':<10} {'Avg LLM Dur':<10}")
    print("-" * 160)
    
    for difficulty in all_difficulties:
        success_rate1 = metrics1.get(difficulty, {}).get('success_rate', 0.0)
        success_rate2 = metrics2.get(difficulty, {}).get('success_rate', 0.0)
        new_tokens1 = metrics1.get(difficulty, {}).get('new_tokens_computed', 0)
        new_tokens2 = metrics2.get(difficulty, {}).get('new_tokens_computed', 0)
        tokens_per_item1 = metrics1.get(difficulty, {}).get('new_tokens_per_item', 0.0)
        tokens_per_item2 = metrics2.get(difficulty, {}).get('new_tokens_per_item', 0.0)
        ops_per_item1 = metrics1.get(difficulty, {}).get('operations_per_item', 0.0)
        ops_per_item2 = metrics2.get(difficulty, {}).get('operations_per_item', 0.0)
        items_with_reward_1_1 = metrics1.get(difficulty, {}).get('items_with_reward_1', 0)
        items_with_reward_1_2 = metrics2.get(difficulty, {}).get('items_with_reward_1', 0)
        avg_duration1 = metrics1.get(difficulty, {}).get('average_duration_per_llm', 0.0)
        avg_duration2 = metrics2.get(difficulty, {}).get('average_duration_per_llm', 0.0)
        
        print(f"{difficulty:<15} "
              f"{run_id1}: {success_rate1:.2f}% | {run_id2}: {success_rate2:.2f}% | "
              f"{run_id1}: {new_tokens1:<7} | {run_id2}: {new_tokens2:<7} | "
              f"{run_id1}: {tokens_per_item1:.2f} | {run_id2}: {tokens_per_item2:.2f} | "
              f"{run_id1}: {ops_per_item1:.2f} | {run_id2}: {ops_per_item2:.2f} | "
              f"{run_id1}: {items_with_reward_1_1:<5} | {run_id2}: {items_with_reward_1_2:<5} | "
              f"{run_id1}: {avg_duration1:.2f}s | {run_id2}: {avg_duration2:.2f}s")
    
    # Print per-item analysis
    print("\nPer-Item Analysis:")
    print("=" * 120)
    print(f"{'Item ID':<10} {'Metric':<25} {'Run 1':<20} {'Run 2':<20} {'Delta'}")
    print("-" * 120)
    
    # Get all unique item IDs
    all_item_ids = sorted(set(list(item_metrics1.keys()) + list(item_metrics2.keys())), key=lambda x: int(x) if x.isdigit() else x)
    
    for item_id in all_item_ids:
        # Skip unknown items
        if item_id == 'unknown':
            continue
        
        # Get item metrics
        item1 = item_metrics1.get(item_id, {})
        item2 = item_metrics2.get(item_id, {})
        
        # Skip if no data for either run
        if not item1 and not item2:
            continue
        
        # Calculate metrics for each item
        def calc_item_metrics(item):
            p = item.get('total_prompt_tokens', 0)
            c = item.get('cached_tokens', 0)
            reward = item.get('reward_score', 0.0)
            duration = item.get('total_duration', 0.0)
            llm_responses = item.get('llm_responses', 0)
            
            cache_ratio = (c / p * 100) if p > 0 else 0
            avg_duration = (duration / llm_responses) if llm_responses > 0 else 0
            
            return p, c, cache_ratio, reward, duration, avg_duration, llm_responses
        
        p1, c1, cache1, reward1, dur1, avg_dur1, llm1 = calc_item_metrics(item1)
        p2, c2, cache2, reward2, dur2, avg_dur2, llm2 = calc_item_metrics(item2)
        
        # Print item metrics
        print(f"{item_id:<10} | {'Avg LLM Duration (s)':<25} | {avg_dur1:>19.3f} | {avg_dur2:>19.3f} | {avg_dur2 - avg_dur1:>+7.3f}")
        print(f"{'':<10} | {'Final Reward Score':<25} | {reward1:>19.3f} | {reward2:>19.3f} | {'WIN' if reward2 > reward1 else 'LOSS' if reward2 < reward1 else 'SAME'}")
        print(f"{'':<10} | {'Total Duration / Cache%':<25} | {dur1:>10.3f}s / {cache1:>6.1f}% | {dur2:>10.3f}s / {cache2:>6.1f}% | {cache2 - cache1:>+7.1f}%")
        print(f"{'':<10} | {'LLM Responses':<25} | {llm1:>19} | {llm2:>19} | {llm2 - llm1:>+7}")
        print("-" * 120)

async def main():
    """
    Main function to parse arguments, connect to database, and run analysis.
    """
    parser = argparse.ArgumentParser(description='Compare metrics between two run IDs')
    parser.add_argument('run_id1', help='First run ID for comparison')
    parser.add_argument('run_id2', help='Second run ID for comparison')
    parser.add_argument('--format', choices=['csv', 'json', 'both'], default='both',
                        help='Output format (default: both)')
    
    args = parser.parse_args()
    
    try:
        # Read database URL from environment variable
        db_url = await read_database_url()
        
        # Initialize database client
        db = get_event_db(db_url=db_url)
        
        # Validate run IDs
        print(f"Validating run IDs: {args.run_id1} and {args.run_id2}...")
        run1_exists, run2_exists = await validate_run_ids(db, args.run_id1, args.run_id2)
        
        if not run1_exists:
            print(f"Error: Run ID {args.run_id1} not found in database")
            sys.exit(1)
        
        if not run2_exists:
            print(f"Error: Run ID {args.run_id2} not found in database")
            sys.exit(1)
        
        print("Run IDs validated successfully!")
        
        # Analyze both runs
        print(f"Analyzing run {args.run_id1}...")
        metrics1 = await analyze_run(db, args.run_id1)
        
        print(f"Analyzing run {args.run_id2}...")
        metrics2 = await analyze_run(db, args.run_id2)
        
        # Generate output
        await generate_output(metrics1, metrics2, args.run_id1, args.run_id2, args.format)
        
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
