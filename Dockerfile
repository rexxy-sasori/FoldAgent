# Use specified base image from harbor registry
FROM harbor.xa.xshixun.com:7443/llm-course/verl:vemlp-th2.4.0-cu124-vllm0.6.3-ray2.10-te1.7-v0.0.3

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy only the verl package directory for installation
COPY external/verl/ ./external/verl/

# Install verl package without editable mode
RUN pip3 install ./external/verl

# Install additional Python dependencies
RUN pip3 install fastapi uvicorn transformers numpy pandas tqdm omegaconf

# Clean up unnecessary files after installation
RUN rm -rf /app/external/verl \
    && pip3 cache purge \
    && rm -rf /root/.cache/pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy only the necessary application code
COPY envs/ ./envs/
COPY agents/ ./agents/
COPY scripts/ ./scripts/
COPY data/ ./data/

# Expose ports for services
EXPOSE 8000

# Default command to start search server
CMD ["python3", "envs/search_server.py"]