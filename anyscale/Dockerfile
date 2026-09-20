# Training image with the repo's Python environment already synced, so a job
# starts training instead of spending 5 minutes on `uv sync` on a cold node.
# Build once (the lockfile is the cache key; rebuild when uv.lock changes):
#   anyscale image build -f scripts/anyscale/Dockerfile --name microduck-train --ray-version 2.58.0
# The venv lives at /home/ray/venv; UV_PROJECT_ENVIRONMENT points uv at it so
# the job's `uv sync --frozen` is a no-op check and `uv run` uses it. git is
# already in the base image (the bam dependency installs from GitHub).
FROM anyscale/ray:2.58.0-py313-cu130

RUN pip install --quiet --user uv
ENV PATH="/home/ray/.local/bin:${PATH}"
ENV UV_PROJECT_ENVIRONMENT=/home/ray/venv
ENV UV_HTTP_TIMEOUT=600
# Anyscale image builds refuse COPY from a local directory, so the lockfile
# comes from the public fork's headstand branch instead.
RUN git clone --depth 1 -b headstand https://github.com/zachgarner/microduck_rl /home/ray/lockfile-src \
 && cd /home/ray/lockfile-src && uv sync --frozen --no-install-project
