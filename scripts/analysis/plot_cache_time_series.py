#!/usr/bin/env python3
"""
Script to plot cache time series data from the ideal prefix cache analysis.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Plot cache time series data')
    parser.add_argument('--csv-file', type=str, required=True, help='Path to the CSV file with time series data')
    parser.add_argument('--output-dir', type=str, default='./plots', help='Directory to save plots')
    parser.add_argument('--title', type=str, default='Ideal Prefix Cache Analysis', help='Plot title prefix')
    return parser.parse_args()


def load_data(csv_file):
    """Load and preprocess the time series data"""
    df = pd.read_csv(csv_file)
    
    # Parse timestamps automatically
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
    
    # Convert boolean strings to actual booleans
    if df['is_cache_hit'].dtype == 'object':
        df['is_cache_hit'] = df['is_cache_hit'].map({'True': True, 'False': False})
    
    return df


def plot_cache_hits_over_time(df, output_dir, title_prefix):
    """Plot cache hits and misses over time"""
    plt.figure(figsize=(12, 6))
    
    # Plot hits and misses
    hits = df[df['is_cache_hit']]
    misses = df[~df['is_cache_hit']]
    
    plt.scatter(hits['timestamp'], hits['prompt_index'], color='green', label='Cache Hit', alpha=0.6, s=20)
    plt.scatter(misses['timestamp'], misses['prompt_index'], color='red', label='Cache Miss', alpha=0.6, s=20)
    
    plt.title(f'{title_prefix}: Cache Hits vs Misses Over Time')
    plt.xlabel('Time')
    plt.ylabel('Prompt Index')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate()
    
    # Save the plot
    output_file = os.path.join(output_dir, 'cache_hits_over_time.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cache hits plot to: {output_file}")


def plot_cached_percentage_over_time(df, output_dir, title_prefix):
    """Plot cached percentage over time"""
    plt.figure(figsize=(12, 6))
    
    plt.scatter(df['timestamp'], df['cached_percentage'], color='blue', alpha=0.5, s=20)
    plt.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='100% Cached')
    
    plt.title(f'{title_prefix}: Cached Token Percentage Over Time')
    plt.xlabel('Time')
    plt.ylabel('Cached Percentage (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate()
    
    # Save the plot
    output_file = os.path.join(output_dir, 'cached_percentage_over_time.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cached percentage plot to: {output_file}")


def plot_cumulative_hit_rate(df, output_dir, title_prefix):
    """Plot cumulative hit rate over time"""
    plt.figure(figsize=(12, 6))
    
    # Calculate cumulative hit rate
    df['cumulative_hits'] = df['is_cache_hit'].cumsum()
    df['cumulative_total'] = range(1, len(df) + 1)
    df['cumulative_hit_rate'] = (df['cumulative_hits'] / df['cumulative_total']) * 100
    
    plt.plot(df['timestamp'], df['cumulative_hit_rate'], color='purple', linewidth=2)
    plt.axhline(y=df['cumulative_hit_rate'].iloc[-1], color='orange', linestyle='--', alpha=0.7, 
                label=f'Final Hit Rate: {df["cumulative_hit_rate"].iloc[-1]:.2f}%')
    
    plt.title(f'{title_prefix}: Cumulative Cache Hit Rate Over Time')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Hit Rate (%)')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate()
    
    # Save the plot
    output_file = os.path.join(output_dir, 'cumulative_hit_rate.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cumulative hit rate plot to: {output_file}")


def plot_cached_tokens_vs_total(df, output_dir, title_prefix):
    """Plot cached tokens vs total tokens per prompt"""
    plt.figure(figsize=(12, 6))
    
    # Filter out rows where cached_tokens might be NaN
    df_valid = df.dropna(subset=['cached_tokens', 'token_count'])
    
    plt.scatter(df_valid['token_count'], df_valid['cached_tokens'], color='teal', alpha=0.5, s=20)
    plt.plot([0, df_valid['token_count'].max()], [0, df_valid['token_count'].max()], 
             color='red', linestyle='--', alpha=0.7, label='100% Cached Line')
    
    plt.title(f'{title_prefix}: Cached Tokens vs Total Tokens per Prompt')
    plt.xlabel('Total Tokens')
    plt.ylabel('Cached Tokens')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    # Save the plot
    output_file = os.path.join(output_dir, 'cached_vs_total_tokens.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cached vs total tokens plot to: {output_file}")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from: {args.csv_file}")
    df = load_data(args.csv_file)
    
    print(f"Data loaded successfully. Total prompts: {len(df)}")
    print(f"Cache hit rate: {df['is_cache_hit'].mean() * 100:.2f}%")
    print(f"Average cached percentage: {df['cached_percentage'].mean():.2f}%")
    
    # Generate plots
    plot_cache_hits_over_time(df, args.output_dir, args.title)
    plot_cached_percentage_over_time(df, args.output_dir, args.title)
    plot_cumulative_hit_rate(df, args.output_dir, args.title)
    plot_cached_tokens_vs_total(df, args.output_dir, args.title)
    
    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
