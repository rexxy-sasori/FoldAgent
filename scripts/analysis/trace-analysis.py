import json
from collections import Counter

def analyze_vllm_traces_detailed(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    results = []
    token_counts = []
    
    for trace in data.get('data', []):
        if not trace['spans']: continue
        
        main_span = trace['spans'][0]
        tags = {tag['key']: tag['value'] for tag in main_span['tags']}
        
        p_tokens = int(tags.get('gen_ai.usage.prompt_tokens', 0))
        if p_tokens > 0:
            token_counts.append(p_tokens)
            results.append({
                "trace_id": trace['traceID'],
                "queue_s": float(tags.get('gen_ai.latency.time_in_queue', 0)),
                "prompt_tokens": p_tokens
            })

    # Create Histogram Bins (e.g., increments of 10k tokens)
    bin_size = 10000 
    histogram = Counter((t // bin_size) * bin_size for t in token_counts)
    
    print("--- TOKEN LENGTH HISTOGRAM ---")
    for bin_start in sorted(histogram.keys()):
        count = histogram[bin_start]
        bar = "█" * (count * 40 // len(results)) # Scale bar to 40 chars
        print(f"{bin_start:6} - {bin_start + bin_size:6} tokens: {bar} ({count})")

    # Correlate Token Length with Queue Time
    long_context_queue = [r['queue_s'] for r in results if r['prompt_tokens'] > 50000]
    avg_long_queue = sum(long_context_queue) / len(long_context_queue) if long_context_queue else 0
    
    print(f"\nAverage Queue Wait for >50k tokens: {avg_long_queue:.2f}s")

analyze_vllm_traces_detailed('/Users/rexsasori/Downloads/data.json')