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

def load_difficulty_mapping(data_path: str = 'data/bc_test.parquet') -> Dict[str, str]:
    """
    Load difficulty mapping from parquet file.
    
    Args:
        data_path: Path to the parquet file containing test data
    
    Returns:
        Dict[str, str]: Mapping from instance_id to difficulty level
    """
    try:
        import pandas as pd
        df = pd.read_parquet(data_path)
        difficulty_map = {}
        
        for _, row in df.iterrows():
            instance_id = row['extra_info'].get('instance_id', 'unknown')
            data_source = row.get('data_source', 'unknown')
            
            # Normalize difficulty names (handle typo: 'meduim' -> 'medium')
            difficulty = 'unknown'
            if 'easy' in str(data_source).lower():
                difficulty = 'easy'
            elif 'medium' in str(data_source).lower() or 'meduim' in str(data_source).lower():
                difficulty = 'medium'
            elif 'hard' in str(data_source).lower():
                difficulty = 'hard'
            
            difficulty_map[instance_id] = difficulty
        
        print(f"Loaded difficulty mapping for {len(difficulty_map)} items from {data_path}")
        return difficulty_map
    except ImportError:
        print("Warning: pandas not available, skipping difficulty mapping")
        return {}
    except Exception as e:
        print(f"Warning: Could not load difficulty mapping from {data_path}: {e}")
        return {}

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

async def analyze_run(db, run_id: str, difficulty_map: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Analyze a single run and calculate metrics by difficulty level and item.
    
    Args:
        db: Database client instance
        run_id: Run ID to analyze
        difficulty_map: Optional mapping from instance_id to difficulty level
    
    Returns:
        Dict[str, Any]: Metrics organized by difficulty level
    """
    if difficulty_map is None:
        difficulty_map = {}
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
            
            # Get difficulty from difficulty_map first, then fallback to event data
            difficulty = difficulty_map.get(item_id, 'unknown')
            if difficulty == 'unknown':
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

def generate_primary_comparison_table(metrics1: Dict[str, Any], metrics2: Dict[str, Any], 
                                      run_id1: str, run_id2: str) -> List[Dict[str, Any]]:
    """
    Generate primary comparison table with operations delta (new tokens) by difficulty level.
    
    Args:
        metrics1: Metrics for first run
        metrics2: Metrics for second run
        run_id1: First run ID
        run_id2: Second run ID
    
    Returns:
        List[Dict[str, Any]]: Comparison data organized by difficulty level
    """
    all_difficulties = sorted(set([d for d in list(metrics1.keys()) + list(metrics2.keys()) if d != '_item_metrics']))
    
    comparison_data = []
    
    for difficulty in all_difficulties:
        m1 = metrics1.get(difficulty, {})
        m2 = metrics2.get(difficulty, {})
        
        new_tokens1 = m1.get('new_tokens_computed', 0)
        new_tokens2 = m2.get('new_tokens_computed', 0)
        operations_delta = new_tokens2 - new_tokens1
        
        data = {
            'difficulty': difficulty,
            'react_new_tokens': new_tokens1,
            'folding_new_tokens': new_tokens2,
            'operations_delta': operations_delta,
            'react_total_operations': m1.get('total_operations', 0),
            'folding_total_operations': m2.get('total_operations', 0),
            'react_items_processed': m1.get('items_processed', 0),
            'folding_items_processed': m2.get('items_processed', 0),
            'react_success_rate': m1.get('success_rate', 0.0),
            'folding_success_rate': m2.get('success_rate', 0.0)
        }
        comparison_data.append(data)
    
    return comparison_data

def generate_outcome_analysis_table(item_metrics1: Dict[str, Any], item_metrics2: Dict[str, Any],
                                     difficulty_map: Dict[str, str]) -> Dict[str, Dict[str, int]]:
    """
    Generate detailed outcome analysis table organized by outcome categories and difficulty levels.
    
    Categories:
    - Loss to Win: items that changed from loss (reward_score < 1) to win (reward_score == 1)
    - Win to Loss: items that changed from win (reward_score == 1) to loss (reward_score < 1)
    - Win to Win: items that remained win (reward_score == 1 in both runs)
    - Loss to Loss: items that remained loss (reward_score < 1 in both runs)
    
    Args:
        item_metrics1: Item metrics for first run
        item_metrics2: Item metrics for second run
        difficulty_map: Mapping from instance_id to difficulty level
    
    Returns:
        Dict[str, Dict[str, int]]: Outcome categories with difficulty breakdowns
    """
    outcome_categories = {
        'Loss to Win': {'easy': 0, 'medium': 0, 'hard': 0, 'unknown': 0, 'total': 0},
        'Win to Loss': {'easy': 0, 'medium': 0, 'hard': 0, 'unknown': 0, 'total': 0},
        'Win to Win': {'easy': 0, 'medium': 0, 'hard': 0, 'unknown': 0, 'total': 0},
        'Loss to Loss': {'easy': 0, 'medium': 0, 'hard': 0, 'unknown': 0, 'total': 0}
    }
    
    # Get common items (items present in both runs)
    common_items = set(item_metrics1.keys()) & set(item_metrics2.keys())
    
    for item_id in common_items:
        # Skip unknown items
        if item_id == 'unknown':
            continue
        
        # Get reward scores
        reward1 = item_metrics1.get(item_id, {}).get('reward_score', 0.0)
        reward2 = item_metrics2.get(item_id, {}).get('reward_score', 0.0)
        
        # Determine win/loss status (reward_score == 1 is a win)
        is_win1 = reward1 == 1.0
        is_win2 = reward2 == 1.0
        
        # Get difficulty
        difficulty = difficulty_map.get(item_id, 'unknown').lower()
        if difficulty not in ['easy', 'medium', 'hard']:
            difficulty = 'unknown'
        
        # Categorize based on outcome transition
        if not is_win1 and is_win2:
            category = 'Loss to Win'
        elif is_win1 and not is_win2:
            category = 'Win to Loss'
        elif is_win1 and is_win2:
            category = 'Win to Win'
        else:
            category = 'Loss to Loss'
        
        # Update counts
        outcome_categories[category][difficulty] += 1
        outcome_categories[category]['total'] += 1
    
    return outcome_categories

def generate_detailed_item_analysis(item_metrics1: Dict[str, Any], item_metrics2: Dict[str, Any],
                                   difficulty_map: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Generate detailed item analysis for each outcome category, including token deltas per item ranked in decreasing order.
    
    Categories:
    - Loss to Win: items that changed from loss (reward_score < 1) to win (reward_score == 1)
    - Win to Loss: items that changed from win (reward_score == 1) to loss (reward_score < 1)
    - Win to Win: items that remained win (reward_score == 1 in both runs)
    - Loss to Loss: items that remained loss (reward_score < 1 in both runs)
    
    Args:
        item_metrics1: Item metrics for first run
        item_metrics2: Item metrics for second run
        difficulty_map: Mapping from instance_id to difficulty level
    
    Returns:
        Dict[str, List[Dict[str, Any]]]: Outcome categories with detailed item analysis
    """
    # Initialize categories
    categories = {
        'Loss to Win': [],
        'Win to Loss': [],
        'Win to Win': [],
        'Loss to Loss': []
    }
    
    # Get common items (items present in both runs)
    common_items = set(item_metrics1.keys()) & set(item_metrics2.keys())
    
    for item_id in common_items:
        # Skip unknown items
        if item_id == 'unknown':
            continue
        
        # Get reward scores
        reward1 = item_metrics1.get(item_id, {}).get('reward_score', 0.0)
        reward2 = item_metrics2.get(item_id, {}).get('reward_score', 0.0)
        
        # Determine win/loss status (reward_score == 1 is a win)
        is_win1 = reward1 == 1.0
        is_win2 = reward2 == 1.0
        
        # Get difficulty
        difficulty = difficulty_map.get(item_id, 'unknown').lower()
        if difficulty not in ['easy', 'medium', 'hard']:
            difficulty = 'unknown'
        
        # Categorize based on outcome transition
        if not is_win1 and is_win2:
            category = 'Loss to Win'
        elif is_win1 and not is_win2:
            category = 'Win to Loss'
        elif is_win1 and is_win2:
            category = 'Win to Win'
        else:
            category = 'Loss to Loss'
        
        # Get item metrics
        item1 = item_metrics1.get(item_id, {})
        item2 = item_metrics2.get(item_id, {})
        
        # Calculate new tokens for each item (total_prompt_tokens - cached_tokens)
        new_tokens1 = item1.get('total_prompt_tokens', 0) - item1.get('cached_tokens', 0)
        new_tokens2 = item2.get('total_prompt_tokens', 0) - item2.get('cached_tokens', 0)
        
        # Calculate token delta
        token_delta = new_tokens2 - new_tokens1
        
        # Create item analysis entry
        item_analysis = {
            'item_id': item_id,
            'difficulty': difficulty,
            'run1_new_tokens': new_tokens1,
            'run2_new_tokens': new_tokens2,
            'token_delta': token_delta,
            'run1_reward': reward1,
            'run2_reward': reward2,
            'run1_llm_responses': item1.get('llm_responses', 0),
            'run2_llm_responses': item2.get('llm_responses', 0)
        }
        
        # Add to category
        categories[category].append(item_analysis)
    
    # Sort items in each category by token delta in decreasing order
    for category in categories:
        categories[category].sort(key=lambda x: x['token_delta'], reverse=True)
    
    return categories


def generate_outcome_comparison_table(item_metrics1: Dict[str, Any], item_metrics2: Dict[str, Any],
                                       difficulty_map: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """
    Generate outcome-based comparison table with operations delta organized by outcome categories.
    
    Categories:
    - Loss to Win: items that changed from loss (reward_score < 1) to win (reward_score == 1)
    - Win to Loss: items that changed from win (reward_score == 1) to loss (reward_score < 1)
    - Win to Win: items that remained win (reward_score == 1 in both runs)
    - Loss to Loss: items that remained loss (reward_score < 1 in both runs)
    
    Args:
        item_metrics1: Item metrics for first run
        item_metrics2: Item metrics for second run
        difficulty_map: Mapping from instance_id to difficulty level
    
    Returns:
        Dict[str, Dict[str, Any]]: Outcome categories with metrics and deltas
    """
    outcome_data = {
        'Loss to Win': {
            'item_ids': [],
            'run1_new_tokens': 0,
            'run2_new_tokens': 0,
            'run1_total_operations': 0,
            'run2_total_operations': 0,
            'run1_items_processed': 0,
            'run2_items_processed': 0,
            'run1_success_rate': 0.0,
            'run2_success_rate': 0.0,
            'easy': 0, 'medium': 0, 'hard': 0, 'unknown': 0
        },
        'Win to Loss': {
            'item_ids': [],
            'run1_new_tokens': 0,
            'run2_new_tokens': 0,
            'run1_total_operations': 0,
            'run2_total_operations': 0,
            'run1_items_processed': 0,
            'run2_items_processed': 0,
            'run1_success_rate': 0.0,
            'run2_success_rate': 0.0,
            'easy': 0, 'medium': 0, 'hard': 0, 'unknown': 0
        },
        'Win to Win': {
            'item_ids': [],
            'run1_new_tokens': 0,
            'run2_new_tokens': 0,
            'run1_total_operations': 0,
            'run2_total_operations': 0,
            'run1_items_processed': 0,
            'run2_items_processed': 0,
            'run1_success_rate': 0.0,
            'run2_success_rate': 0.0,
            'easy': 0, 'medium': 0, 'hard': 0, 'unknown': 0
        },
        'Loss to Loss': {
            'item_ids': [],
            'run1_new_tokens': 0,
            'run2_new_tokens': 0,
            'run1_total_operations': 0,
            'run2_total_operations': 0,
            'run1_items_processed': 0,
            'run2_items_processed': 0,
            'run1_success_rate': 0.0,
            'run2_success_rate': 0.0,
            'easy': 0, 'medium': 0, 'hard': 0, 'unknown': 0
        }
    }
    
    # Get common items (items present in both runs)
    common_items = set(item_metrics1.keys()) & set(item_metrics2.keys())
    
    for item_id in common_items:
        # Skip unknown items
        if item_id == 'unknown':
            continue
        
        # Get reward scores
        reward1 = item_metrics1.get(item_id, {}).get('reward_score', 0.0)
        reward2 = item_metrics2.get(item_id, {}).get('reward_score', 0.0)
        
        # Determine win/loss status (reward_score == 1 is a win)
        is_win1 = reward1 == 1.0
        is_win2 = reward2 == 1.0
        
        # Get difficulty
        difficulty = difficulty_map.get(item_id, 'unknown').lower()
        if difficulty not in ['easy', 'medium', 'hard']:
            difficulty = 'unknown'
        
        # Categorize based on outcome transition
        if not is_win1 and is_win2:
            category = 'Loss to Win'
        elif is_win1 and not is_win2:
            category = 'Win to Loss'
        elif is_win1 and is_win2:
            category = 'Win to Win'
        else:
            category = 'Loss to Loss'
        
        # Get item metrics
        item1 = item_metrics1.get(item_id, {})
        item2 = item_metrics2.get(item_id, {})
        
        # Calculate new tokens for each item (total_prompt_tokens - cached_tokens)
        new_tokens1 = item1.get('total_prompt_tokens', 0) - item1.get('cached_tokens', 0)
        new_tokens2 = item2.get('total_prompt_tokens', 0) - item2.get('cached_tokens', 0)
        
        # Estimate operations (use LLM responses as a proxy)
        operations1 = item1.get('llm_responses', 0)
        operations2 = item2.get('llm_responses', 0)
        
        # Update category data
        outcome_data[category]['item_ids'].append(item_id)
        outcome_data[category]['run1_new_tokens'] += new_tokens1
        outcome_data[category]['run2_new_tokens'] += new_tokens2
        outcome_data[category]['run1_total_operations'] += operations1
        outcome_data[category]['run2_total_operations'] += operations2
        outcome_data[category]['run1_items_processed'] += 1
        outcome_data[category]['run2_items_processed'] += 1
        outcome_data[category][difficulty] += 1
    
    # Calculate derived metrics for each category
    for category in outcome_data:
        data = outcome_data[category]
        item_count = len(data['item_ids'])
        
        # Calculate operations delta
        data['operations_delta'] = data['run2_new_tokens'] - data['run1_new_tokens']
        
        # Calculate success rates (percentage of items with reward == 1)
        if item_count > 0:
            if 'Win' in category:
                data['run1_success_rate'] = 100.0 if 'to Win' in category else 0.0
                data['run2_success_rate'] = 100.0 if 'to Win' in category or category == 'Win to Win' else 0.0
            else:
                data['run1_success_rate'] = 0.0
                data['run2_success_rate'] = 0.0
    
    return outcome_data

async def generate_output(metrics1: Dict[str, Any], metrics2: Dict[str, Any], 
                          run_id1: str, run_id2: str, output_format: str, difficulty_map: Dict[str, str] = None):
    """
    Generate structured output in the specified format.
    
    Args:
        metrics1: Metrics for first run
        metrics2: Metrics for second run
        run_id1: First run ID
        run_id2: Second run ID
        output_format: Output format (csv, json, or both)
        difficulty_map: Optional mapping from instance_id to difficulty level
    """
    # Use 'react' and 'folding' labels instead of full run IDs for better readability
    run1_label = 'react'
    run2_label = 'folding'
    
    if difficulty_map is None:
        difficulty_map = {}
    # Get all unique difficulty levels, excluding '_item_metrics'
    all_difficulties = sorted(set([d for d in list(metrics1.keys()) + list(metrics2.keys()) if d != '_item_metrics']))
    
    # Extract item metrics for per-item analysis
    item_metrics1 = metrics1.get('_item_metrics', {})
    item_metrics2 = metrics2.get('_item_metrics', {})
    
    # Generate detailed item analysis by category
    detailed_item_analysis = generate_detailed_item_analysis(item_metrics1, item_metrics2, difficulty_map)
    
    # Prepare data for output
    output_data = []
    for difficulty in all_difficulties:
            data = {
                'difficulty': difficulty,
                f'{run1_label}_success_rate': metrics1.get(difficulty, {}).get('success_rate', 0.0),
                f'{run2_label}_success_rate': metrics2.get(difficulty, {}).get('success_rate', 0.0),
                f'{run1_label}_new_tokens_computed': metrics1.get(difficulty, {}).get('new_tokens_computed', 0),
                f'{run2_label}_new_tokens_computed': metrics2.get(difficulty, {}).get('new_tokens_computed', 0),
                f'{run1_label}_new_tokens_per_item': metrics1.get(difficulty, {}).get('new_tokens_per_item', 0.0),
                f'{run2_label}_new_tokens_per_item': metrics2.get(difficulty, {}).get('new_tokens_per_item', 0.0),
                f'{run1_label}_operations_per_item': metrics1.get(difficulty, {}).get('operations_per_item', 0.0),
                f'{run2_label}_operations_per_item': metrics2.get(difficulty, {}).get('operations_per_item', 0.0),
                f'{run1_label}_items_with_reward_1': metrics1.get(difficulty, {}).get('items_with_reward_1', 0),
                f'{run2_label}_items_with_reward_1': metrics2.get(difficulty, {}).get('items_with_reward_1', 0),
                f'{run1_label}_total_reward': metrics1.get(difficulty, {}).get('total_reward', 0.0),
                f'{run2_label}_total_reward': metrics2.get(difficulty, {}).get('total_reward', 0.0),
                f'{run1_label}_reward_evaluations': metrics1.get(difficulty, {}).get('reward_evaluations', 0),
                f'{run2_label}_reward_evaluations': metrics2.get(difficulty, {}).get('reward_evaluations', 0),
                f'{run1_label}_average_reward': metrics1.get(difficulty, {}).get('average_reward', 0.0),
                f'{run2_label}_average_reward': metrics2.get(difficulty, {}).get('average_reward', 0.0),
                f'{run1_label}_average_duration_per_llm': metrics1.get(difficulty, {}).get('average_duration_per_llm', 0.0),
                f'{run2_label}_average_duration_per_llm': metrics2.get(difficulty, {}).get('average_duration_per_llm', 0.0),
                f'{run1_label}_total_operations': metrics1.get(difficulty, {}).get('total_operations', 0),
                f'{run2_label}_total_operations': metrics2.get(difficulty, {}).get('total_operations', 0),
                f'{run1_label}_items_processed': metrics1.get(difficulty, {}).get('items_processed', 0),
                f'{run2_label}_items_processed': metrics2.get(difficulty, {}).get('items_processed', 0),
            }
            output_data.append(data)
    
    # Generate primary comparison and outcome analysis data for output
    primary_comparison = generate_primary_comparison_table(metrics1, metrics2, run_id1, run_id2)
    outcome_analysis = generate_outcome_analysis_table(item_metrics1, item_metrics2, difficulty_map)
    outcome_comparison = generate_outcome_comparison_table(item_metrics1, item_metrics2, difficulty_map)
    
    # Generate JSON output
    if output_format in ['json', 'both']:
        json_output = {
            'run_ids': {
                'react': run_id1,
                'folding': run_id2
            },
            'metrics': output_data,
            'primary_comparison': primary_comparison,
            'outcome_analysis': outcome_analysis,
            'outcome_comparison': outcome_comparison,
            'detailed_item_analysis': detailed_item_analysis,
            'item_analysis': {
                'react_items': len(item_metrics1),
                'folding_items': len(item_metrics2),
                'common_items': len([item_id for item_id in item_metrics1 if item_id in item_metrics2])
            }
        }
        
        with open(f'run_comparison_{run1_label}_{run2_label}.json', 'w') as f:
            json.dump(json_output, f, indent=2)
        print(f"JSON output saved to run_comparison_{run1_label}_{run2_label}.json")
    
    # Generate CSV output
    if output_format in ['csv', 'both']:
        if output_data:
            # Main metrics CSV
            fieldnames = ['difficulty']
            metric_types = ['success_rate', 'new_tokens_computed', 'new_tokens_per_item', 
                           'operations_per_item', 'items_with_reward_1', 'total_reward', 
                           'reward_evaluations', 'average_reward', 'average_duration_per_llm', 
                           'total_operations', 'items_processed']
            for run_label in [run1_label, run2_label]:
                for metric in metric_types:
                    fieldnames.append(f'{run_label}_{metric}')
            
            with open(f'run_comparison_{run1_label}_{run2_label}.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(output_data)
            print(f"CSV output saved to run_comparison_{run1_label}_{run2_label}.csv")
            
            # Primary comparison CSV
            primary_fieldnames = ['difficulty', f'{run1_label}_new_tokens', f'{run2_label}_new_tokens', 
                                   'operations_delta', f'{run1_label}_total_operations', f'{run2_label}_total_operations',
                                   f'{run1_label}_items_processed', f'{run2_label}_items_processed',
                                   f'{run1_label}_success_rate', f'{run2_label}_success_rate']
            with open(f'run_comparison_{run1_label}_{run2_label}_primary.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=primary_fieldnames)
                writer.writeheader()
                writer.writerows(primary_comparison)
            print(f"Primary comparison CSV saved to run_comparison_{run1_label}_{run2_label}_primary.csv")
            
            # Outcome analysis CSV
            outcome_fieldnames = ['outcome_category', 'easy', 'medium', 'hard', 'unknown', 'total']
            with open(f'run_comparison_{run1_label}_{run2_label}_outcome.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=outcome_fieldnames)
                writer.writeheader()
                for category in ['Loss to Win', 'Win to Loss', 'Win to Win', 'Loss to Loss']:
                    counts = outcome_analysis[category]
                    writer.writerow({
                        'outcome_category': category,
                        'easy': counts['easy'],
                        'medium': counts['medium'],
                        'hard': counts['hard'],
                        'unknown': counts['unknown'],
                        'total': counts['total']
                    })
            print(f"Outcome analysis CSV saved to run_comparison_{run1_label}_{run2_label}_outcome.csv")
            
            # Outcome comparison CSV
            outcome_comp_fieldnames = ['outcome_category', f'{run1_label}_new_tokens', f'{run2_label}_new_tokens',
                                         'operations_delta', f'{run1_label}_total_operations', f'{run2_label}_total_operations',
                                         f'{run1_label}_items_processed', f'{run2_label}_items_processed',
                                         f'{run1_label}_success_rate', f'{run2_label}_success_rate',
                                         'easy', 'medium', 'hard', 'unknown']
            with open(f'run_comparison_{run1_label}_{run2_label}_outcome_comparison.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=outcome_comp_fieldnames)
                writer.writeheader()
                for category in ['Loss to Win', 'Win to Loss', 'Win to Win', 'Loss to Loss']:
                    data = outcome_comparison[category]
                    writer.writerow({
                        'outcome_category': category,
                        f'{run1_label}_new_tokens': data['run1_new_tokens'],
                        f'{run2_label}_new_tokens': data['run2_new_tokens'],
                        'operations_delta': data['operations_delta'],
                        f'{run1_label}_total_operations': data['run1_total_operations'],
                        f'{run2_label}_total_operations': data['run2_total_operations'],
                        f'{run1_label}_items_processed': data['run1_items_processed'],
                        f'{run2_label}_items_processed': data['run2_items_processed'],
                        f'{run1_label}_success_rate': data['run1_success_rate'],
                        f'{run2_label}_success_rate': data['run2_success_rate'],
                        'easy': data['easy'],
                        'medium': data['medium'],
                        'hard': data['hard'],
                        'unknown': data['unknown']
                    })
            print(f"Outcome comparison CSV saved to run_comparison_{run1_label}_{run2_label}_outcome_comparison.csv")
            
            # Detailed item analysis CSV
            detailed_fieldnames = ['category', 'item_id', 'difficulty', 'run1_new_tokens', 'run2_new_tokens', 'token_delta', 'run1_reward', 'run2_reward', 'run1_llm_responses', 'run2_llm_responses']
            with open(f'run_comparison_{run1_label}_{run2_label}_detailed_items.csv', 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=detailed_fieldnames)
                writer.writeheader()
                for category, items in detailed_item_analysis.items():
                    for item in items:
                        writer.writerow({
                            'category': category,
                            'item_id': item['item_id'],
                            'difficulty': item['difficulty'],
                            'run1_new_tokens': item['run1_new_tokens'],
                            'run2_new_tokens': item['run2_new_tokens'],
                            'token_delta': item['token_delta'],
                            'run1_reward': item['run1_reward'],
                            'run2_reward': item['run2_reward'],
                            'run1_llm_responses': item['run1_llm_responses'],
                            'run2_llm_responses': item['run2_llm_responses']
                        })
            print(f"Detailed item analysis CSV saved to run_comparison_{run1_label}_{run2_label}_detailed_items.csv")
        else:
            print("No data available for CSV output")
    
    # Print summary to console
    print("\nRun Comparison Summary:")
    print(f"Comparing run_id1: {run_id1} vs run_id2: {run_id2}")
    print("=" * 160)
    
    # Define column headers and widths for summary
    summary_headers = [
        ("Difficulty", 12),
        (f"{run1_label} Success", 18),
        (f"{run2_label} Success", 18),
        (f"{run1_label} Tokens", 18),
        (f"{run2_label} Tokens", 18),
        (f"{run1_label} Items", 12),
        (f"{run2_label} Items", 12),
        (f"{run1_label} Dur", 12),
        (f"{run2_label} Dur", 12)
    ]
    
    # Print header row with proper spacing
    header_row = ""
    for header, width in summary_headers:
        header_row += f"{header:<{width}} | "
    print(header_row.rstrip(" | "))
    
    # Print separator line
    separator = ""
    for _, width in summary_headers:
        separator += "-" * width + "-+-"
    print(separator.rstrip("-+-"))
    
    # Print data rows
    for difficulty in all_difficulties:
        success_rate1 = metrics1.get(difficulty, {}).get('success_rate', 0.0)
        success_rate2 = metrics2.get(difficulty, {}).get('success_rate', 0.0)
        new_tokens1 = metrics1.get(difficulty, {}).get('new_tokens_computed', 0)
        new_tokens2 = metrics2.get(difficulty, {}).get('new_tokens_computed', 0)
        items_with_reward_1_1 = metrics1.get(difficulty, {}).get('items_with_reward_1', 0)
        items_with_reward_1_2 = metrics2.get(difficulty, {}).get('items_with_reward_1', 0)
        avg_duration1 = metrics1.get(difficulty, {}).get('average_duration_per_llm', 0.0)
        avg_duration2 = metrics2.get(difficulty, {}).get('average_duration_per_llm', 0.0)
        
        row_data = [
            difficulty,
            f"{success_rate1:.2f}%",
            f"{success_rate2:.2f}%",
            f"{new_tokens1:,}",
            f"{new_tokens2:,}",
            f"{items_with_reward_1_1:,}",
            f"{items_with_reward_1_2:,}",
            f"{avg_duration1:.2f}s",
            f"{avg_duration2:.2f}s"
        ]
        
        data_row = ""
        for i, (data, (_, width)) in enumerate(zip(row_data, summary_headers)):
            if i == 0:
                data_row += f"{data:<{width}} | "
            else:
                data_row += f"{data:>{width}} | "
        print(data_row.rstrip(" | "))
    
    # Print per-item analysis
    print("\nPer-Item Analysis:")
    print("=" * 140)
    
    # Define column headers and widths for per-item analysis
    item_headers = [
        ("Item ID", 10),
        ("Difficulty", 10),
        ("Metric", 25),
        ("Run 1", 20),
        ("Run 2", 20),
        ("Delta", 10)
    ]
    
    # Print header row with proper spacing
    header_row = ""
    for header, width in item_headers:
        header_row += f"{header:<{width}} | "
    print(header_row.rstrip(" | "))
    
    # Print separator line
    separator = ""
    for _, width in item_headers:
        separator += "-" * width + "-+-"
    print(separator.rstrip("-+-"))
    
    # Get all unique item IDs
    all_item_ids = sorted(set(list(item_metrics1.keys()) + list(item_metrics2.keys())), key=lambda x: int(x) if x.isdigit() else x)
    
    for item_id in all_item_ids:
        # Skip unknown items
        if item_id == 'unknown':
            continue
        
        # Get difficulty for this item
        difficulty = difficulty_map.get(item_id, 'unknown')
        
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
        
        # Print item metrics with proper formatting
        metrics = [
            ("Avg LLM Duration (s)", f"{avg_dur1:.3f}", f"{avg_dur2:.3f}", f"{avg_dur2 - avg_dur1:>+7.3f}"),
            ("Final Reward Score", f"{reward1:.3f}", f"{reward2:.3f}", f"{'WIN' if reward2 > reward1 else 'LOSS' if reward2 < reward1 else 'SAME':>7}"),
            ("Total Duration / Cache%", f"{dur1:.3f}s / {cache1:.1f}%", f"{dur2:.3f}s / {cache2:.1f}%", f"{cache2 - cache1:>+7.1f}%"),
            ("LLM Responses", str(llm1), str(llm2), f"{llm2 - llm1:>+7}")
        ]
        
        for i, (metric_name, run1_val, run2_val, delta_val) in enumerate(metrics):
            if i == 0:
                row_data = [item_id, difficulty, metric_name, run1_val, run2_val, delta_val]
            else:
                row_data = ["", "", metric_name, run1_val, run2_val, delta_val]
            
            data_row = ""
            for j, (data, (_, width)) in enumerate(zip(row_data, item_headers)):
                if j < 3:  # Left-align first three columns
                    data_row += f"{data:<{width}} | "
                else:  # Right-align remaining columns
                    data_row += f"{data:>{width}} | "
            print(data_row.rstrip(" | "))
        
        # Print separator line after each item
        print(separator.rstrip("-+-"))
    
    # Generate and print primary comparison table
    print("\n" + "=" * 140)
    print("PRIMARY COMPARISON TABLE - Operations Delta (New Tokens)")
    print("=" * 140)
    
    # Define column headers and widths for primary comparison
    primary_headers = [
        ("Difficulty", 10),
        (f"{run1_label} Tokens", 18),
        (f"{run2_label} Tokens", 18),
        ("Delta", 18),
        (f"{run1_label} Ops", 12),
        (f"{run2_label} Ops", 12),
        (f"{run1_label} Items", 12),
        (f"{run2_label} Items", 12)
    ]
    
    # Print header row with proper spacing
    header_row = ""
    for header, width in primary_headers:
        header_row += f"{header:<{width}} | "
    print(header_row.rstrip(" | "))
    
    # Print separator line
    separator = ""
    for _, width in primary_headers:
        separator += "-" * width + "-+-"
    print(separator.rstrip("-+-"))
    
    # Print data rows
    for row in primary_comparison:
            delta_str = f"{row['operations_delta']:+,}" if row['operations_delta'] != 0 else "0"
            row_data = [
                row['difficulty'],
                f"{row['react_new_tokens']:,}",
                f"{row['folding_new_tokens']:,}",
                delta_str,
                f"{row['react_total_operations']:,}",
                f"{row['folding_total_operations']:,}",
                f"{row['react_items_processed']:,}",
                f"{row['folding_items_processed']:,}"
            ]
            
            data_row = ""
            for i, (data, (_, width)) in enumerate(zip(row_data, primary_headers)):
                # Right-align numeric columns (except first column)
                if i == 0:
                    data_row += f"{data:<{width}} | "
                else:
                    data_row += f"{data:>{width}} | "
            print(data_row.rstrip(" | "))
    
    print("=" * 140)
    
    # Generate and print outcome analysis table
    print("\n" + "=" * 140)
    print("SECONDARY ANALYSIS TABLE - Outcome Categories by Difficulty")
    print("=" * 140)
    
    # Define column headers and widths for outcome analysis
    outcome_headers = [
        ("Outcome Category", 20),
        ("Easy", 10),
        ("Medium", 10),
        ("Hard", 10),
        ("Unknown", 10),
        ("Total", 10)
    ]
    
    # Print header row with proper spacing
    header_row = ""
    for header, width in outcome_headers:
        header_row += f"{header:<{width}} | "
    print(header_row.rstrip(" | "))
    
    # Print separator line
    separator = ""
    for _, width in outcome_headers:
        separator += "-" * width + "-+-"
    print(separator.rstrip("-+-"))
    
    # Print data rows
    for category in ['Loss to Win', 'Win to Loss', 'Win to Win', 'Loss to Loss']:
        counts = outcome_analysis[category]
        row_data = [
            category,
            str(counts['easy']),
            str(counts['medium']),
            str(counts['hard']),
            str(counts['unknown']),
            str(counts['total'])
        ]
        
        data_row = ""
        for i, (data, (_, width)) in enumerate(zip(row_data, outcome_headers)):
            if i == 0:
                data_row += f"{data:<{width}} | "
            else:
                data_row += f"{data:>{width}} | "
        print(data_row.rstrip(" | "))
    
    print("=" * 140)
    
    # Generate and print outcome comparison table
    print("\n" + "=" * 140)
    print("OUTCOME COMPARISON TABLE - Operations Delta by Outcome Category")
    print("=" * 140)
    
    # Define column headers and widths for outcome comparison
    outcome_comp_headers = [
        ("Outcome Category", 20),
        (f"{run1_label} Tokens", 18),
        (f"{run2_label} Tokens", 18),
        ("Delta", 18),
        (f"{run1_label} Ops", 12),
        (f"{run2_label} Ops", 12),
        ("Items", 10),
        ("Easy", 8),
        ("Med", 8),
        ("Hard", 8)
    ]
    
    # Print header row with proper spacing
    header_row = ""
    for header, width in outcome_comp_headers:
        header_row += f"{header:<{width}} | "
    print(header_row.rstrip(" | "))
    
    # Print separator line
    separator = ""
    for _, width in outcome_comp_headers:
        separator += "-" * width + "-+-"
    print(separator.rstrip("-+-"))
    
    # Print data rows
    for category in ['Loss to Win', 'Win to Loss', 'Win to Win', 'Loss to Loss']:
        data = outcome_comparison[category]
        delta_str = f"{data['operations_delta']:+,}" if data['operations_delta'] != 0 else "0"
        row_data = [
            category,
            f"{data['run1_new_tokens']:,}",
            f"{data['run2_new_tokens']:,}",
            delta_str,
            f"{data['run1_total_operations']:,}",
            f"{data['run2_total_operations']:,}",
            str(len(data['item_ids'])),
            str(data['easy']),
            str(data['medium']),
            str(data['hard'])
        ]
        
        data_row = ""
        for i, (data_item, (_, width)) in enumerate(zip(row_data, outcome_comp_headers)):
            if i == 0:
                data_row += f"{data_item:<{width}} | "
            else:
                data_row += f"{data_item:>{width}} | "
        print(data_row.rstrip(" | "))
    
    print("=" * 140)
    
    # Generate and print detailed item analysis by category
    print("\n" + "=" * 180)
    print("DETAILED ITEM ANALYSIS - Token Deltas by Outcome Category")
    print("=" * 180)
    
    # Define column headers and widths for detailed item analysis
    detailed_headers = [
        ("Category", 15),
        ("Item ID", 10),
        ("Difficulty", 10),
        (f"{run1_label} Tokens", 15),
        (f"{run2_label} Tokens", 15),
        ("Token Delta", 15),
        (f"{run1_label} Reward", 12),
        (f"{run2_label} Reward", 12),
        (f"{run1_label} LLM", 10),
        (f"{run2_label} LLM", 10)
    ]
    
    # Print header row with proper spacing
    header_row = ""
    for header, width in detailed_headers:
        header_row += f"{header:<{width}} | "
    print(header_row.rstrip(" | "))
    
    # Print separator line
    separator = ""
    for _, width in detailed_headers:
        separator += "-" * width + "-+-"
    print(separator.rstrip("-+-"))
    
    # Print detailed item analysis for each category
    for category, items in detailed_item_analysis.items():
        if items:
            for i, item in enumerate(items):
                row_data = [
                    category if i == 0 else "",
                    item['item_id'],
                    item['difficulty'],
                    f"{item['run1_new_tokens']:,}",
                    f"{item['run2_new_tokens']:,}",
                    f"{item['token_delta']:+,}",
                    f"{item['run1_reward']:.1f}",
                    f"{item['run2_reward']:.1f}",
                    str(item['run1_llm_responses']),
                    str(item['run2_llm_responses'])
                ]
                
                data_row = ""
                for j, (data, (_, width)) in enumerate(zip(row_data, detailed_headers)):
                    if j < 3:
                        data_row += f"{data:<{width}} | "
                    else:
                        data_row += f"{data:>{width}} | "
                print(data_row.rstrip(" | "))
            print(separator.rstrip("-+-"))
        else:
            row_data = [category, "-", "-", "-", "-", "-", "-", "-", "-", "-"]
            data_row = ""
            for j, (data, (_, width)) in enumerate(zip(row_data, detailed_headers)):
                if j < 3:
                    data_row += f"{data:<{width}} | "
                else:
                    data_row += f"{data:>{width}} | "
            print(data_row.rstrip(" | "))
            print(separator.rstrip("-+-"))
    
    print("=" * 180)

async def main():
    """
    Main function to parse arguments, connect to database, and run analysis.
    """
    parser = argparse.ArgumentParser(description='Compare metrics between two run IDs')
    parser.add_argument('run_id1', help='First run ID for comparison (React run)')
    parser.add_argument('run_id2', help='Second run ID for comparison (Folding run)')
    parser.add_argument('--format', choices=['csv', 'json', 'both'], default='both',
                        help='Output format (default: both)')
    parser.add_argument('--data_path', default='data/bc_test.parquet',
                        help='Path to test data parquet file (default: data/bc_test.parquet)')
    
    args = parser.parse_args()
    
    try:
        # Load difficulty mapping from parquet file
        difficulty_map = load_difficulty_mapping(args.data_path)
        
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
        
        # Analyze both runs with difficulty mapping
        print(f"Analyzing run {args.run_id1}...")
        metrics1 = await analyze_run(db, args.run_id1, difficulty_map)
        
        print(f"Analyzing run {args.run_id2}...")
        metrics2 = await analyze_run(db, args.run_id2, difficulty_map)
        
        # Generate output
        await generate_output(metrics1, metrics2, args.run_id1, args.run_id2, args.format, difficulty_map)
        
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
