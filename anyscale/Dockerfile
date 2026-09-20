# Training image for Anyscale Jobs: Anyscale's Ray base with CUDA 12.8 and
# Python 3.12 (the repo pins >=3.12,<3.13), plus uv and git. The repo itself is
# shipped as the job's working_dir, and train.sh runs `uv sync` at start.
FROM anyscale/ray:2.49.0-slim-py312-cu128

USER root
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && mv /root/.local/bin/uv /usr/local/bin/uv
USER ray
