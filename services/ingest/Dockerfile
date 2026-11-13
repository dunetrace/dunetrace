FROM python:3.12-slim
WORKDIR /app

# Non-root runtime. python:3.12-slim's default user is uid 0, so without this a
# container escape or an RCE through any dependency starts as root holding
# DATABASE_URL and ADMIN_API_KEY. A *numeric* USER (below) is also what a
# Kubernetes `runAsNonRoot` admission policy needs — it cannot resolve a name to
# prove the user is not uid 0, and refuses to schedule the pod.
#
# /app and everything COPY'd into it deliberately stay root-owned and
# world-readable: no service writes to disk at runtime (there is no
# open(..., "w"), os.makedirs or tempfile anywhere under services/), so the
# running process can read its own code and cannot rewrite it.
#
# Declared before the pip layer so it caches: this never changes.
RUN groupadd --gid 10001 dunetrace \
 && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/dunetrace \
            --shell /usr/sbin/nologin dunetrace

COPY services/ingest/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY packages/schemas-py/ packages/schemas-py/
COPY services/ingest/ services/ingest/

# Default detector thresholds, baked in.
#
# ingest_svc/config.py reads DETECTOR_CONFIG (default /app/detectors.yml) and
# serves the merged result to every SDK at GET /v1/detector-config. Both compose
# files bind-mount the host's ./detectors.yml over this path — that mount is now
# a genuine OVERRIDE of a working default rather than the only source of the
# file. It matters because Docker silently creates a *directory* when a
# bind-mount source is missing: an operator who ran only docker-compose.ghcr.yml
# ("no build step required") without the repo got an empty directory mounted at
# /app/detectors.yml, the loader's try/except swallowed the IsADirectoryError,
# and every detector ran on class defaults with one log line as the only trace.
# With a real file here that mount instead fails loudly ("cannot mount directory
# onto file"), and an image run outside compose is correctly configured.
COPY detectors.yml /app/detectors.yml

ENV PYTHONPATH=/app/packages/schemas-py:/app/services/ingest
# Set explicitly rather than inherited from /etc/passwd, so `~` expansion is a
# property of this image and not of the runtime's user lookup.
ENV HOME=/home/dunetrace
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

# The port this image serves. EXPOSE publishes nothing by itself — it is what
# `docker run -P`, ECS and every image inspector read to learn the port.
EXPOSE 8001

# Readiness baked into the image, so the published artifact carries a probe
# outside compose (plain `docker run`, ECS, Nomad). Both compose files define
# their own healthcheck on the same endpoint, which overrides this one.
# 127.0.0.1, not localhost: the server binds IPv4 and `localhost` resolves to
# ::1 first on hosts whose /etc/hosts lists it that way.
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/ready', timeout=4)"]

USER 10001:10001

CMD ["uvicorn", "ingest_svc.main:app", "--host", "0.0.0.0", "--port", "8001"]
