#!/usr/bin/env python3
"""
Filter training data based on difficulty level for diagnostic purposes.

This script filters a dataset to include only items of specified difficulty levels
(easy, medium, hard) to help diagnose model performance on easier tasks.
"""

import argparse
import json
import pandas as pd
from pathlib import Path


def analyze_dataset(df, title="Dataset"):
    """Analyze and print comprehensive statistics about the dataset."""
    print(f"\n{title} Statistics:")
    print("-" * 50)
    print(f"Total samples: {len(df)}")
    
    # Difficulty distribution
    difficulty_counts = {}
    for level in ['easy', 'medium', 'hard']:
        if level == 'medium':
            count = len(df[df['data_source'].apply(lambda x: 'medium' in str(x) or 'meduim' in str(x))])
        else:
            count = len(df[df['data_source'].apply(lambda x: level in str(x))])
        difficulty_counts[level] = count
    
    print("\nDifficulty Distribution:")
    for level, count in difficulty_counts.items():
        percentage = count / len(df) * 100 if len(df) > 0 else 0
        print(f"  {level}: {count} ({percentage:.2f}%)")
    
    # Additional stats if messages column exists
    if 'messages' in df.columns:
        # Estimate context lengths if possible
        avg_msg_count = df['messages'].apply(lambda x: len(x) if isinstance(x, list) else 0).mean()
        print(f"\nAverage number of messages per sample: {avg_msg_count:.2f}")
        
        # Check for system messages
        has_system = df['messages'].apply(
            lambda x: isinstance(x, list) and len(x) > 0 and x[0].get('role') == 'system' if isinstance(x, list) else False
        ).sum()
        print(f"Samples with system messages: {has_system} ({has_system/len(df)*100:.2f}%)")
    
    # Data source distribution
    print(f"\nTop 10 Data Sources:")
    data_source_counts = df['data_source'].value_counts().head(10)
    for source, count in data_source_counts.items():
        print(f"  {source}: {count}")
    
    print()


def filter_by_difficulty(input_path, output_path, difficulty_levels, stats_only=False):
    """
    Filter dataset based on difficulty levels.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        difficulty_levels: List of difficulty levels to keep ['easy', 'medium', 'hard', 'all']
        stats_only: If True, only show statistics without saving filtered data
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} examples")
    
    # Analyze original dataset
    analyze_dataset(df, "ORIGINAL DATASET")
    
    if stats_only:
        return df
    
    # Determine which rows to keep based on difficulty levels
    if 'all' in difficulty_levels or len(difficulty_levels) == 0:
        # Keep all rows
        filtered_df = df.copy()
        print("\nKeeping all difficulty levels (no filtering applied)")
    else:
        # Create a mask for rows to keep
        mask = pd.Series([False] * len(df), dtype=bool)
        
        for level in difficulty_levels:
            if level == 'medium':
                # Handle both 'medium' and 'meduim' (typo in data)
                level_mask = df['data_source'].apply(lambda x: 'medium' in str(x) or 'meduim' in str(x))
            else:
                level_mask = df['data_source'].apply(lambda x: level in str(x))
            mask = mask | level_mask
        
        filtered_df = df[mask].copy()
        
        print(f"\nDifficulty levels kept: {', '.join(difficulty_levels)}")
        print(f"Filtered dataset size: {len(filtered_df)} ({len(filtered_df)/len(df)*100:.2f}% of original)")
    
    # Analyze filtered dataset
    if not stats_only:
        analyze_dataset(filtered_df, "FILTERED DATASET")
    
    if not stats_only:
        print(f"\nSaving filtered dataset to {output_path}...")
        filtered_df.to_parquet(output_path, index=False)
        print(f"Saved successfully!")
    
    return filtered_df


def main():
    parser = argparse.ArgumentParser(
        description='Filter training data based on difficulty level for diagnostic purposes'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input parquet file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output parquet file path (default: input_filtered.parquet)'
    )
    parser.add_argument(
        '--keep',
        type=str,
        required=True,
        help='Difficulty levels to keep (comma-separated): easy,medium,hard,all'
    )
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Show statistics only, do not save filtered data'
    )
    
    args = parser.parse_args()
    
    # Parse difficulty levels
    difficulty_levels = [level.strip().lower() for level in args.keep.split(',')]
    
    # Validate difficulty levels
    valid_levels = {'easy', 'medium', 'hard', 'all'}
    for level in difficulty_levels:
        if level not in valid_levels:
            print(f"Error: Invalid difficulty level '{level}'. Valid options: {', '.join(valid_levels)}")
            return 1
    
    if args.output is None:
        input_path = Path(args.input)
        args.output = str(input_path.with_name(f"{input_path.stem}_filtered{input_path.suffix}"))
    
    print("=" * 80)
    print("Dataset Difficulty Filter Script")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Difficulty levels to keep: {', '.join(difficulty_levels)}")
    print(f"Stats only: {args.stats_only}")
    print()
    
    # Filter data
    filtered_df = filter_by_difficulty(args.input, args.output, difficulty_levels, args.stats_only)
    
    print()
    print("=" * 80)
    print("Filtering complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()