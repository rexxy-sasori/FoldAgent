#!/usr/bin/env python3
import json

# Test configuration matrix for sglang server and eval_bc combinations

def generate_test_configs():
    # Define parameter options
    scheduling_policies = ['lpm', 'fcfs']
    context_lengths = [49152, 130172]
    workflows = ['search', 'search_branch']
    
    # Generate all combinations
    configs = []
    for policy in scheduling_policies:
        for context_len in context_lengths:
            for workflow in workflows:
                config = {
                    'scheduling_policy': policy,
                    'context_length': context_len,
                    'workflow': workflow
                }
                configs.append(config)
    
    return configs

if __name__ == "__main__":
    configs = generate_test_configs()
    # Output JSON format for bash script parsing
    print(json.dumps(configs, indent=2))
