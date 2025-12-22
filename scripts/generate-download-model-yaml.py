#!/usr/bin/env python3
"""
General script to generate a Kubernetes YAML file for downloading Hugging Face models
using HF mirror in China.

Usage:
python scripts/generate-download-model-yaml.py <model-path> [--output <yaml-file>]

Example:
python scripts/generate-download-model-yaml.py ByteDance-Seed/Seed-OSS-36B-Instruct
python scripts/generate-download-model-yaml.py ByteDance-Seed/Seed-OSS-36B-Instruct --output custom-download.yaml
"""

import argparse
import sys
import yaml
from pathlib import Path

def generate_download_model_yaml(model_path, output_file=None):
    """Generate Kubernetes YAML for downloading a Hugging Face model."""
    
    # Validate model path format
    if '/' not in model_path:
        print(f"Error: Invalid model path format '{model_path}'. Expected format: 'org/model-name'")
        return False
    
    # Create output file path if not provided
    if output_file is None:
        model_name = model_path.replace('/', '-').lower()
        output_file = f"download-model-{model_name}.yaml"
    
    # YAML template with placeholders
    yaml_template = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": f"download-model-{model_path.replace('/', '-').lower()}",
            "namespace": "liuyunxin"
        },
        "spec": {
            "containers": [
                {
                    "name": "download-model",
                    "image": "harbor.xa.xshixun.com:7443/hanfeigeng/vllm/vllm-openai:v0.10.2-linux-amd64",
                    "command": ["bash", "-c"],
                    "args": [
                        f"set -e\n"
                        f"# Install huggingface_hub if not already installed\n"
                        f"echo \"Installing huggingface_hub...\"\n"
                        f"pip install -U huggingface_hub==0.36.0\n"
                        f"\n"
                        f"# Create cache directory if it doesn't exist\n"
                        f"mkdir -p /root/.cache/huggingface\n"
                        f"\n"
                        f"# Download the model\n"
                        f"echo \"Downloading model {model_path} using HF mirror...\"\n"
                        f"huggingface-cli download --resume-download --local-dir-use-symlinks False {model_path}\n"
                        f"\n"
                        f"# Verify the download\n"
                        f"echo \"\\nVerifying model download...\"\n"
                        f"MODEL_DIR=\"/root/.cache/huggingface/hub/models--{model_path.replace('/', '--')}\"\n"
                        f"if [ -d \"$MODEL_DIR\" ]; then\n"
                        f"  echo \"✅ Model directory exists: $MODEL_DIR\"\n"
                        f"  echo \"Number of files downloaded: $(find $MODEL_DIR -type f | wc -l)\"\n"
                        f"  echo \"\\nListing model files:\"\n"
                        f"  ls -la $MODEL_DIR\n"
                        f"else\n"
                        f"  echo \"❌ Model directory not found\"\n"
                        f"  exit 1\n"
                        f"fi\n"
                        f"\n"
                        f"# Keep the pod running for inspection\n"
                        f"echo \"\\nModel download completed successfully!\"\n"
                        f"echo \"Pod will sleep indefinitely for further inspection.\"\n"
                        f"sleep infinity"
                    ],
                    "volumeMounts": [
                        {
                            "name": "gpfshome",
                            "mountPath": "/root"
                        }
                    ],
                    "env": [
                        {"name": "CUDA_VISIBLE_DEVICES", "value": "0"},
                        {"name": "http_proxy", "value": "http://192.168.3.226:7890"},
                        {"name": "https_proxy", "value": "http://192.168.3.226:7890"},
                        {"name": "HTTP_PROXY", "value": "http://192.168.3.226:7890"},
                        {"name": "HTTPS_PROXY", "value": "http://192.168.3.226:7890"},
                        {"name": "APT_PROXY", "value": "http://192.168.3.241:3142"},
                        {"name": "PIP_INDEX_URL", "value": "http://192.168.12.70:9181/repository/pypi-proxy/simple/"},
                        {"name": "PIP_TRUSTED_HOST", "value": "192.168.12.70"},
                        {"name": "HF_ENDPOINT", "value": "https://hf-mirror.com"}
                    ],
                    "resources": {
                        "limits": {
                            "memory": "64Gi",
                            "cpu": "32"
                        },
                        "requests": {
                            "memory": "32Gi",
                            "cpu": "16"
                        }
                    }
                }
            ],
            "volumes": [
                {
                    "name": "gpfshome",
                    "persistentVolumeClaim": {
                        "claimName": "pvc-gpfshome-liuyunxin"
                    }
                }
            ]
        }
    }
    
    # Write YAML to file
    try:
        with open(output_file, 'w') as f:
            yaml.dump(yaml_template, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Successfully generated YAML file: {output_file}")
        print(f"\nTo deploy this pod, run:")
        print(f"  kubectl apply -f {output_file}")
        print(f"\nTo check the download progress:")
        print(f"  kubectl logs -n liuyunxin {yaml_template['metadata']['name']} -f")
        print(f"\nTo clean up after download:")
        print(f"  kubectl delete -f {output_file}")
        
        return True
    except Exception as e:
        print(f"Error writing YAML file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Generate Kubernetes YAML for downloading Hugging Face models")
    parser.add_argument('model_path', help='Hugging Face model path (e.g., ByteDance-Seed/Seed-OSS-36B-Instruct)')
    parser.add_argument('--output', '-o', help='Output YAML file path')
    
    args = parser.parse_args()
    
    # Generate the YAML file
    success = generate_download_model_yaml(args.model_path, args.output)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
