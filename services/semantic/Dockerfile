FROM python:3.12-slim
WORKDIR /app

# Non-root runtime. python:3.12-slim's default user is uid 0, so without this a
# container escape or an RCE through any dependency starts as root holding
# DATABASE_URL and the OpenAI/Anthropic/Mistral keys — and this image has by far
# the largest dependency surface of the seven (deepeval pulls in a whole
# evaluation stack). A *numeric* USER (below) is also what a Kubernetes
# `runAsNonRoot` admission policy needs — it cannot resolve a name to prove the
# user is not uid 0, and refuses to schedule the pod.
#
# /app and everything COPY'd into it deliberately stay root-owned and
# world-readable: no service writes to disk at runtime (there is no
# open(..., "w"), os.makedirs or tempfile anywhere under services/), so the
# running process can read its own code and cannot rewrite it. deepeval is the
# one exception — see DEEPEVAL_CACHE_FOLDER below.
#
# Declared before the pip layer so it caches: this never changes.
RUN groupadd --gid 10001 dunetrace \
 && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/dunetrace \
            --shell /usr/sbin/nologin dunetrace

COPY services/semantic/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# audit Finding 17 (HIGH): explainer_svc imports `dunetrace` (the SDK), so the
# sdk-py package must be present + on PYTHONPATH — it was missing here, which
# crash-looped the worker. Matches services/alerts/Dockerfile. (Authorized during
# audit to enable Phase 3 semantic testing.)
COPY packages/schemas-py/ packages/schemas-py/
COPY packages/sdk-py/ packages/sdk-py/
COPY services/explainer/ services/explainer/
COPY services/semantic/ services/semantic/

# Default sampling rates and evaluator config, baked in.
#
# semantic_svc/config_loader.py reads SEMANTIC_SAMPLING_CONFIG (default
# /app/docs/config/semantic-sampling.yml) and SEMANTIC_EVALUATORS_YML (default
# /app/docs/config/semantic-evaluators.yml). Both compose files bind-mount the
# host copies over these paths — those mounts are now genuine OVERRIDES of
# working defaults rather than the only source of the files. It matters because
# Docker silently creates a *directory* when a bind-mount source is missing: an
# operator who ran only docker-compose.ghcr.yml ("no build step required")
# without the repo got empty directories mounted here and the loaders'
# try/except swallowed the IsADirectoryError, so every agent ran on the built-in
# sampling rate and no evaluator ever asked for a second opinion. With real
# files here those mounts instead fail loudly ("cannot mount directory onto
# file"), and an image run outside compose is correctly configured.
COPY docs/config/semantic-sampling.yml /app/docs/config/semantic-sampling.yml
COPY docs/config/semantic-evaluators.yml /app/docs/config/semantic-evaluators.yml

ENV PYTHONPATH=/app/packages/schemas-py:/app/packages/sdk-py:/app/services/explainer:/app/services/semantic
# Set explicitly rather than inherited from /etc/passwd, so `~` expansion is a
# property of this image and not of the runtime's user lookup.
ENV HOME=/home/dunetrace
# deepeval creates its hidden state directory at IMPORT time, unguarded and
# relative to the process CWD: `os.makedirs(HIDDEN_DIR, exist_ok=True)` runs at
# module scope in deepeval/telemetry.py, and every evaluator in this service
# does `from deepeval.metrics import ...` at module scope. /app is root-owned,
# so as uid 10001 that raises PermissionError before the worker's first line.
# DEEPEVAL_CACHE_FOLDER is deepeval's own override for that path
# (deepeval/constants.py) — point it at the runtime user's writable home.
ENV DEEPEVAL_CACHE_FOLDER=/home/dunetrace/.deepeval
# As uid 10001 CPython cannot write __pycache__ into the root-owned tree and
# silently gives up on every import. Make that explicit instead of implicit.
ENV PYTHONDONTWRITEBYTECODE=1
# Build identity. Every service resolves APP_VERSION, else GIT_COMMIT, else a
# hardcoded fallback, and reports the result as dunetrace_build_info{service,
# version} and in GET /health. Nothing used to supply either, so every
# published image claimed the fallback version forever.
# .github/workflows/docker-publish.yml passes --build-arg APP_VERSION=<release
# tag>; an empty value here is harmless (the getenv chain treats "" as unset).
# Declared last on purpose: a version bump must not invalidate the pip layer.
ARG APP_VERSION=""
ENV APP_VERSION=${APP_VERSION}

# This worker has no API of its own; /health, /ready and /metrics are served by
# dunetrace_schemas.metrics' stdlib server on METRICS_PORT (semantic_svc/config.py
# default 9103, matching both compose files). EXPOSE publishes nothing by itself.
EXPOSE 9103

# Readiness baked into the image, so the published artifact carries a probe
# outside compose (plain `docker run`, ECS, Nomad). Both compose files define
# their own healthcheck on the same endpoint, which overrides this one.
# This worker is opt-in: with SEMANTIC_WORKER_ENABLED false it logs one line and
# exits 0, and a stopped container is never probed — `Exited (0)` stays its
# normal disabled state, not "unhealthy".
# 127.0.0.1, not localhost: the server binds IPv4 and `localhost` resolves to
# ::1 first on hosts whose /etc/hosts lists it that way.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:9103/ready', timeout=8)"]

USER 10001:10001

CMD ["python", "-m", "semantic_svc.worker"]
