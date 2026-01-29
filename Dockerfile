# Use specified base image from harbor registry
FROM harbor.xa.xshixun.com:7443/hanfeigeng/vllm/vllm-openai:v0.13.0-linux-amd64

# Install git for repository cloning
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy only the verl package directory for installation
COPY external/verl/ ./external/verl/

# Install verl package without editable mode
RUN pip3 install ./external/verl

# Install additional Python dependencies
RUN pip3 install fastapi uvicorn transformers numpy pandas tqdm omegaconf torch huggingface_hub==0.36.0 sqlalchemy asyncpg psycopg2-binary aiosqlite datasets gitpython filelock
RUN pip3 install flash-attn --no-build-isolation 

# Clean up unnecessary files after installation
# RUN rm -rf /app/external/verl \
#     && pip3 cache purge \
#     && rm -rf /root/.cache/pip \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# Copy only the necessary application code
COPY envs/ ./envs/
COPY agents/ ./agents/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Create necessary directories for SWE-bench evaluation
RUN mkdir -p gym_data _repo_cache

# Expose ports for services
EXPOSE 8000

# Default command to start search server
CMD ["python3", "envs/search_server.py"]