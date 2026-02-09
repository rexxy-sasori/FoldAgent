# Use your specific base image
FROM harbor.xa.xshixun.com:7443/hanfeigeng/vllm/vllm-openai:v0.13.0-linux-amd64

# Combine system installs and cleanup to keep the layer small
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set env vars to prevent cache bloat and ensure logs flush immediately
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install heavy, stable dependencies first. 
# This ensures that changing your local code doesn't trigger a 10-minute re-download.
RUN pip3 install --no-cache-dir \
    fastapi uvicorn transformers numpy pandas tqdm omegaconf \
    torch huggingface_hub==0.36.0 sqlalchemy asyncpg \
    psycopg2-binary aiosqlite datasets gitpython filelock

# Flash-attn is kept separate as it often requires specific build isolation settings
RUN pip3 install flash-attn --no-build-isolation 

# Install your local package, then immediately remove the source to save space
COPY external/verl/ ./external/verl/
RUN pip3 install ./external/verl && rm -rf ./external/verl

# Copy application code last (these change most frequently)
COPY envs/ ./envs/
COPY agents/ ./agents/
COPY scripts/ ./scripts/
COPY data/ ./data/

EXPOSE 8000

CMD ["python3", "envs/search_server.py"]