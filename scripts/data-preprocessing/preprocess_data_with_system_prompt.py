#!/usr/bin/env python3
"""
Preprocess training data to inject custom system prompt for VERL tool_agent.

This script reads the system prompt template and prepends it to all training examples
in the dataset as a system message.
"""

import argparse
import json
import pandas as pd
from pathlib import Path


def load_system_prompt(template_path):
    """Load system prompt from template file."""
    with open(template_path, 'r') as f:
        system_prompt = f.read()
    return system_prompt


def preprocess_parquet(input_path, output_path, system_prompt, replace=False):
    """
    Preprocess parquet file to inject system prompt.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file
        system_prompt: System prompt string to inject
        replace: If True, replace existing system prompts; if False, skip them
    """
    print(f"Loading data from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} examples")
    
    modified_count = 0
    skipped_count = 0
    
    for idx, row in df.iterrows():
        messages = None
        
        if 'messages' in row and row['messages'] is not None:
            messages = row['messages']
            if isinstance(messages, str):
                messages = json.loads(messages)
        elif 'raw_prompt' in row and row['raw_prompt'] is not None:
            raw_prompt = row['raw_prompt']
            if isinstance(raw_prompt, str):
                messages = json.loads(raw_prompt)
            else:
                messages = raw_prompt
        elif 'prompt' in row and row['prompt'] is not None:
            prompt_data = row['prompt']
            if isinstance(prompt_data, str):
                messages = json.loads(prompt_data)
            elif hasattr(prompt_data, 'tolist'):
                messages = prompt_data.tolist()
            else:
                messages = prompt_data
        else:
            print(f"Warning: No messages found in row {idx}, skipping...")
            skipped_count += 1
            continue
        
        if not messages or not isinstance(messages, list):
            print(f"Warning: Invalid messages format in row {idx}, skipping...")
            skipped_count += 1
            continue
        
        if messages and messages[0].get('role') == 'system':
            if replace:
                messages[0]['content'] = system_prompt
                print(f"Row {idx}: Replaced existing system prompt")
            else:
                print(f"Row {idx}: System prompt already exists, skipping...")
                skipped_count += 1
                continue
        else:
            system_message = {
                'role': 'system',
                'content': system_prompt
            }
            messages = [system_message] + messages
            print(f"Row {idx}: Added new system prompt")
        
        if 'messages' in row:
            df.at[idx, 'messages'] = messages
        elif 'raw_prompt' in row:
            df.at[idx, 'raw_prompt'] = json.dumps(messages)
        elif 'prompt' in row:
            df.at[idx, 'prompt'] = messages
        
        modified_count += 1
    
    print(f"\nModified {modified_count} examples")
    print(f"Skipped {skipped_count} examples")
    print(f"Saving to {output_path}...")
    df.to_parquet(output_path, index=False)
    print(f"Saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess training data to inject custom system prompt for VERL tool_agent'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='/root/rl-training/data/bc_train.parquet',
        help='Input parquet file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/root/rl-training/data/bc_train_with_system_prompt.parquet',
        help='Output parquet file path'
    )
    parser.add_argument(
        '--system_prompt',
        type=str,
        default='/app/envs/verl_tool_adaptor/system_prompt_template.txt',
        help='System prompt template file path'
    )
    parser.add_argument(
        '--replace',
        action='store_true',
        help='Replace existing system prompts instead of skipping them'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("VERL System Prompt Injection Script")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"System Prompt: {args.system_prompt}")
    print(f"Replace existing: {args.replace}")
    print()
    
    # Load system prompt
    print("Loading system prompt template...")
    system_prompt = load_system_prompt(args.system_prompt)
    print(f"System prompt length: {len(system_prompt)} characters")
    print(f"First 100 chars: {system_prompt[:100]}...")
    print()
    
    # Preprocess data
    preprocess_parquet(args.input, args.output, system_prompt, replace=args.replace)
    
    print()
    print("=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
