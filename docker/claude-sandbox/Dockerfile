# Sandbox image for running Claude Code CLI inside Docker.
# Build:  docker build -t rgb-agent/claude-sandbox:latest .

FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git npm \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI globally via npm
RUN npm install -g @anthropic-ai/claude-code

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash sandbox

USER sandbox
WORKDIR /workspace

ENTRYPOINT ["claude"]
