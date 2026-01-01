import re
import json
import argparse
import logging
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LogPatterns:
    """Regex patterns for extracting relevant information from your logs"""
    EVAL_BC_LOG = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - ([\w.-]+) - (\w+) - (.+)$')
    CALLAPI_REQUEST = re.compile(r'\[CallAPI Request \((\w+)\)\] URL: ([^,]+), Request: (.+)$')

class RadixNode:
    def __init__(self, block_data, parent=None):
        self.block_data = block_data
        self.parent = parent
        self.children = {} 

    def is_leaf(self):
        return len(self.children) == 0

class SGLangRealisticCache:
    """
    Simulates SGLang Block-based Radix Cache with:
    1. Chat Templates (Real Token IDs)
    2. Block-level matching (PagedAttention)
    3. First-Miss Logic (Requests compute then cache)
    """
    def __init__(self, vram_gb, model_name, block_size=16):
        logger.info(f"Loading {model_name}...")
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.block_size = block_size
        
        # Calculate KV cache size
        layers = self.config.num_hidden_layers
        kv_heads = getattr(self.config, "num_key_value_heads", self.config.num_attention_heads)
        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        
        self.bytes_per_token = 2 * layers * kv_heads * head_dim * 2 
        self.bytes_per_block = self.bytes_per_token * block_size
        self.capacity_blocks = int((vram_gb * 1024**3) // self.bytes_per_block)
        
        self.root = RadixNode(None)
        self.block_count = 0
        self.leaf_lru = OrderedDict() 

    def _add_node(self, block_data, parent):
        if block_data in parent.children:
            return parent.children[block_data]
        new_node = RadixNode(block_data, parent)
        parent.children[block_data] = new_node
        if parent in self.leaf_lru:
            del self.leaf_lru[parent]
        self.leaf_lru[new_node] = True
        self.block_count += 1
        return new_node

    def evict(self):
        while self.block_count > self.capacity_blocks and self.leaf_lru:
            oldest_leaf, _ = self.leaf_lru.popitem(last=False)
            parent = oldest_leaf.parent
            if parent:
                del parent.children[oldest_leaf.block_data]
                self.block_count -= 1
                if parent.is_leaf() and parent != self.root:
                    self.leaf_lru[parent] = True

    def process_request(self, messages):
        # 1. Apply Chat Template (Aligns with real SGLang input)
        try:
            full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tokens = self.tokenizer.encode(full_prompt, add_special_tokens=False)
        except Exception:
            full_prompt = "".join([m.get('content', '') for m in messages])
            tokens = self.tokenizer.encode(full_prompt)

        if not tokens: return None

        # 2. Block-level Chunks
        blocks = [tuple(tokens[i:i + self.block_size]) 
                  for i in range(0, len(tokens) - (len(tokens) % self.block_size), self.block_size)]
        
        # 3. Match BEFORE committing (Simulation of "First-Time Miss")
        matched_blocks = 0
        curr = self.root
        for b in blocks:
            if b in curr.children:
                curr = curr.children[b]
                matched_blocks += 1
                if curr.is_leaf() and curr != self.root:
                    self.leaf_lru.move_to_end(curr)
            else:
                break
        
        hit_rate = (matched_blocks * self.block_size) / len(tokens)
        
        # 4. Commit to cache AFTER calculation
        insert_curr = curr
        for i in range(matched_blocks, len(blocks)):
            insert_curr = self._add_node(blocks[i], insert_curr)
        
        self.evict()
        
        return {
            "total": len(tokens),
            "hit_rate": hit_rate,
            "vram_usage": (self.block_count / self.capacity_blocks) * 100
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vram-gb", type=float, default=40.0)
    parser.add_argument("--output", default="realistic_analysis.csv")
    args = parser.parse_args()

    cache = SGLangRealisticCache(args.vram_gb, args.model)
    results = []

    with open(args.log_file, 'r') as f:
        lines = f.readlines()
        
    pbar = tqdm(lines, desc="Analyzing")
    for line in pbar:
        m = LogPatterns.EVAL_BC_LOG.match(line)
        if not m: continue
        req_m = LogPatterns.CALLAPI_REQUEST.match(m.group(4))
        if not req_m: continue
        
        try:
            data = json.loads(req_m.group(3))
            stats = cache.process_request(data.get('messages', []))
            if stats:
                stats['timestamp'] = m.group(1)
                results.append(stats)
                pbar.set_postfix({"Req_HR": f"{stats['hit_rate']:.1%}"})
        except Exception: continue

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    print(f"\nFinal Avg Per-Request Hit Rate: {df['hit_rate'].mean():.2%}")

if __name__ == "__main__":
    main()