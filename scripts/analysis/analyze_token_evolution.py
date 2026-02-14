#!/usr/bin/env python3
"""
Script to analyze how cached tokens vs total tokens evolve at each LLM response event round
for both react and folding runs.
"""

import argparse
import asyncio
import json
import csv
import os
import sys
from typing import Dict, Any, List, Tuple
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

async def fetch_run_metrics(db, run_id: str) -> Dict[str, Any]:
    """
    Fetch metrics for a specific run, including token usage at each LLM event round.
    
    Args:
        db: Database client instance
        run_id: Run ID to fetch metrics for
    
    Returns:
        Dict[str, Any]: Run metrics organized by item and round
    """
    try:
        # Fetch all events for this run
        events = await db.get_events_by_run_id(run_id)
        
        # Organize metrics by item and round
        run_metrics = {
            "run_id": run_id,
            "items": {}
        }
        
        # Track round numbers for each item
        item_rounds = {}
        
        for event in events:
            # Extract event data
            event_type = event.get('event_type', '')
            event_data = event.get('event_data', {})
            request_id = event.get('request_id', '')
            
            # Extract item ID from request_id (first part before '_')
            item_id = request_id.split('_')[0] if '_' in request_id else 'unknown'
            
            # Skip non-LLM events
            if event_type != "llm_response":
                continue
            
            # Skip unknown items
            if item_id == 'unknown':
                continue
            
            # Initialize item data if not exists
            if item_id not in run_metrics["items"]:
                run_metrics["items"][item_id] = {
                    "rounds": [],
                    "total_rounds": 0
                }
                item_rounds[item_id] = 0
            
            # Increment round for this item
            item_rounds[item_id] += 1
            round_num = item_rounds[item_id]
            
            # Extract token usage from event data
            total_prompt_tokens = event_data.get("prompt_tokens", 0)
            cached_tokens = event_data.get("cached_tokens", 0)
            new_tokens = total_prompt_tokens - cached_tokens
            
            # Calculate cache ratio
            cache_ratio = (cached_tokens / total_prompt_tokens * 100) if total_prompt_tokens > 0 else 0
            
            # Store round data
            round_data = {
                "round": round_num,
                "total_prompt_tokens": total_prompt_tokens,
                "cached_tokens": cached_tokens,
                "new_tokens": new_tokens,
                "cache_ratio": cache_ratio,
                "timestamp": event.get('created_at')
            }
            
            run_metrics["items"][item_id]["rounds"].append(round_data)
            run_metrics["items"][item_id]["total_rounds"] = round_num
        
        return run_metrics
        
    except Exception as e:
        print(f"Error fetching run metrics: {e}")
        return {
            "run_id": run_id,
            "items": {}
        }

def generate_token_evolution_analysis(react_metrics: Dict[str, Any], folding_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate analysis of token evolution for both runs.
    
    Args:
        react_metrics: Metrics for react run
        folding_metrics: Metrics for folding run
    
    Returns:
        Dict[str, Any]: Analysis results
    """
    analysis = {
        "summary": {
            "react_run_id": react_metrics.get("run_id"),
            "folding_run_id": folding_metrics.get("run_id"),
            "common_items": 0,
            "total_items_react": len(react_metrics.get("items", {})),
            "total_items_folding": len(folding_metrics.get("items", {}))
        },
        "items": {}
    }
    
    # Get common items
    react_items = set(react_metrics.get("items", {}).keys())
    folding_items = set(folding_metrics.get("items", {}).keys())
    common_items = react_items.intersection(folding_items)
    analysis["summary"]["common_items"] = len(common_items)
    
    # Analyze each common item
    for item_id in common_items:
        react_item = react_metrics.get("items", {}).get(item_id, {})
        folding_item = folding_metrics.get("items", {}).get(item_id, {})
        
        item_analysis = {
            "react_rounds": react_item.get("total_rounds", 0),
            "folding_rounds": folding_item.get("total_rounds", 0),
            "react_data": react_item.get("rounds", []),
            "folding_data": folding_item.get("rounds", [])
        }
        
        analysis["items"][item_id] = item_analysis
    
    return analysis

def generate_summary_tables(analysis: Dict[str, Any]) -> None:
    """
    Generate summary tables from the analysis.
    
    Args:
        analysis: Analysis results
    """
    print("=" * 160)
    print("TOKEN EVOLUTION ANALYSIS SUMMARY")
    print("=" * 160)
    print(f"React Run ID: {analysis['summary']['react_run_id']}")
    print(f"Folding Run ID: {analysis['summary']['folding_run_id']}")
    print(f"Total Items (React): {analysis['summary']['total_items_react']}")
    print(f"Total Items (Folding): {analysis['summary']['total_items_folding']}")
    print(f"Common Items: {analysis['summary']['common_items']}")
    print("=" * 160)
    
    # Print item-level summary (no detailed table)
    print("\nITEM-LEVEL SUMMARY")
    print("=" * 160)
    
    for item_id, item_data in analysis['items'].items():
        react_rounds = item_data.get('react_rounds', 0)
        folding_rounds = item_data.get('folding_rounds', 0)
        print(f"Item ID: {item_id} | React Rounds: {react_rounds} | Folding Rounds: {folding_rounds}")
    
    print("=" * 160)

def generate_csv_output(analysis: Dict[str, Any], output_prefix: str) -> None:
    """
    Generate CSV output from the analysis.
    
    Args:
        analysis: Analysis results
        output_prefix: Prefix for output files
    """
    # Generate item-level token evolution CSV
    item_csv_path = f"{output_prefix}_token_evolution.csv"
    
    with open(item_csv_path, 'w', newline='') as f:
        fieldnames = [
            'item_id', 'run', 'round',
            'total_prompt_tokens', 'cached_tokens', 'new_tokens', 'cache_ratio'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for item_id, item_data in analysis['items'].items():
            # Write React data
            for round_data in item_data.get('react_data', []):
                row = {
                    'item_id': item_id,
                    'run': 'react',
                    'round': round_data['round'],
                    'total_prompt_tokens': round_data['total_prompt_tokens'],
                    'cached_tokens': round_data['cached_tokens'],
                    'new_tokens': round_data['new_tokens'],
                    'cache_ratio': round_data['cache_ratio']
                }
                writer.writerow(row)
            
            # Write Folding data
            for round_data in item_data.get('folding_data', []):
                row = {
                    'item_id': item_id,
                    'run': 'folding',
                    'round': round_data['round'],
                    'total_prompt_tokens': round_data['total_prompt_tokens'],
                    'cached_tokens': round_data['cached_tokens'],
                    'new_tokens': round_data['new_tokens'],
                    'cache_ratio': round_data['cache_ratio']
                }
                writer.writerow(row)
    
    print(f"Token evolution CSV saved to {item_csv_path}")

def generate_json_output(analysis: Dict[str, Any], output_prefix: str) -> None:
    """
    Generate JSON output from the analysis.
    
    Args:
        analysis: Analysis results
        output_prefix: Prefix for output files
    """
    json_path = f"{output_prefix}_token_evolution.json"
    
    with open(json_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Token evolution JSON saved to {json_path}")

def generate_token_visualization(analysis: Dict[str, Any], item_id: str, output_prefix: str, output_dir: str = None) -> None:
    """
    Generate data visualization for token evolution of a specific item.
    
    Args:
        analysis: Analysis results
        item_id: Item ID to visualize
        output_prefix: Prefix for output files
        output_dir: Optional directory to save the visualization
    """
    try:
        # Get item data
        item_data = analysis['items'].get(item_id)
        if not item_data:
            print(f"Item ID {item_id} not found in analysis results")
            return
        
        # Extract data for each run
        react_data = item_data.get('react_data', [])
        folding_data = item_data.get('folding_data', [])
        
        # Prepare data for plotting
        # React data
        react_rounds = [r['round'] for r in react_data]
        react_cache_percentage = [r['cache_ratio'] for r in react_data]
        
        # Folding data
        folding_rounds = [r['round'] for r in folding_data]
        folding_cache_percentage = [r['cache_ratio'] for r in folding_data]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot React data if available
        if react_data:
            # Plot React cache percentage
            plt.plot(react_rounds, react_cache_percentage, 'r-', marker='o', label='React Cache %', linewidth=2)
        
        # Plot Folding data if available
        if folding_data:
            # Plot Folding cache percentage
            plt.plot(folding_rounds, folding_cache_percentage, 'b-', marker='s', label='Folding Cache %', linewidth=2)
        
        # Configure plot
        plt.title(f'Cache Percentage Evolution for Item {item_id}\nReact vs Folding Runs', fontsize=16, fontweight='bold')
        plt.xlabel('LLM Response Round', fontsize=14)
        plt.ylabel('Cache Percentage (%)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper left', fontsize=12)
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x:.1f}%'))
        
        # Set y-axis limits to 0-100%
        plt.ylim(0, 100)
        
        # Set x-axis ticks to whole numbers for both runs
        all_rounds = sorted(set(react_rounds + folding_rounds))
        if all_rounds:
            plt.xticks(all_rounds)
        
        # Adjust layout
        plt.tight_layout()
        
        # Determine output path
        if output_dir:
            output_path = os.path.join(output_dir, f"{item_id}_token_evolution.png")
        else:
            output_path = f"{output_prefix}_token_evolution_{item_id}.png"
        
        # Save plot
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Token evolution visualization saved to {output_path}")
        
    except Exception as e:
        print(f"Error generating token visualization for item {item_id}: {e}")

async def main():
    """
    Main function to parse arguments, connect to database, and run analysis.
    """
    parser = argparse.ArgumentParser(description='Analyze token evolution at each LLM event round')
    parser.add_argument('react_run_id', help='React run ID (first position)')
    parser.add_argument('folding_run_id', help='Folding run ID (second position)')
    parser.add_argument('--format', choices=['csv', 'json', 'both'], default='both',
                        help='Output format (default: both)')
    parser.add_argument('--visualize', type=str, default=None,
                        help='Item ID to generate token evolution visualization for')
    
    args = parser.parse_args()
    
    try:
        # Get database client
        db_url = await read_database_url()
        db = get_event_db(db_url)
        
        # Fetch metrics for both runs
        print(f"Fetching metrics for react run: {args.react_run_id}")
        react_metrics = await fetch_run_metrics(db, args.react_run_id)
        
        print(f"Fetching metrics for folding run: {args.folding_run_id}")
        folding_metrics = await fetch_run_metrics(db, args.folding_run_id)
        
        # Generate analysis
        print("Generating token evolution analysis...")
        analysis = generate_token_evolution_analysis(react_metrics, folding_metrics)
        
        # Generate summary tables
        generate_summary_tables(analysis)
        
        # Generate output files
        output_prefix = f"token_evolution_{args.react_run_id}_{args.folding_run_id}"
        
        if args.format in ['csv', 'both']:
            generate_csv_output(analysis, output_prefix)
        
        if args.format in ['json', 'both']:
            generate_json_output(analysis, output_prefix)
        
        # Generate visualization if item ID is provided
        if args.visualize:
            if args.visualize == 'all':
                # Create directory for all visualizations
                output_dir = f"token_evolution_{args.react_run_id}_{args.folding_run_id}_visualizations"
                os.makedirs(output_dir, exist_ok=True)
                print(f"Creating visualizations for all items in directory: {output_dir}")
                
                # Generate visualization for each item ID
                item_ids = list(analysis['items'].keys())
                total_items = len(item_ids)
                print(f"Found {total_items} items to visualize")
                
                for i, item_id in enumerate(item_ids, 1):
                    print(f"Generating visualization {i}/{total_items} for item {item_id}...")
                    generate_token_visualization(analysis, item_id, output_prefix, output_dir)
                
                print(f"All visualizations completed. Saved to: {output_dir}")
            else:
                print(f"Generating token evolution visualization for item {args.visualize}...")
                generate_token_visualization(analysis, args.visualize, output_prefix)
            
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
