FROM python:3.12-slim
WORKDIR /app

# Non-root runtime. python:3.12-slim's default user is uid 0, so without this a
# container escape or an RCE through any dependency starts as root holding
# DATABASE_URL, DUNETRACE_MASTER_KEY (the key every stored integration
# credential is encrypted with), the LLM provider keys and the GitHub App
# private key. A *numeric* USER (below) is also what a Kubernetes
# `runAsNonRoot` admission policy needs — it cannot resolve a name to prove the
# user is not uid 0, and refuses to schedule the pod.
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

COPY services/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY packages/schemas-py/ packages/schemas-py/
COPY packages/sdk-py/ packages/sdk-py/
COPY services/explainer/ services/explainer/
COPY services/api/ services/api/

# Default voice cost rates, baked in.
#
# api_svc/voice_pricing.py reads VOICE_PRICING_CONFIG (default
# /app/voice-pricing.yml) and already falls back to the same values compiled in,
# so this file is not load-bearing the way detectors.yml is for ingest — but
# both compose files bind-mount ./voice-pricing.yml here, and a missing host
# file makes Docker create a *directory* at the mount source and mount it over
# the path. Shipping the real file turns that silent degradation into a loud
# "cannot mount directory onto file", and keeps the published image's rates
# visible (and editable) instead of only living in Python.
COPY voice-pricing.yml /app/voice-pricing.yml

ENV PYTHONPATH=/app/packages/schemas-py:/app/packages/sdk-py:/app/services/explainer:/app/services/api
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
EXPOSE 8002

# Readiness baked into the image, so the published artifact carries a probe
# outside compose (plain `docker run`, ECS, Nomad). Both compose files define
# their own healthcheck on the same endpoint, which overrides this one.
# 127.0.0.1, not localhost: the server binds IPv4 and `localhost` resolves to
# ::1 first on hosts whose /etc/hosts lists it that way.
HEALTHCHECK --interval=10s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8002/ready', timeout=4)"]

USER 10001:10001

CMD ["uvicorn", "api_svc.main:app", "--host", "0.0.0.0", "--port", "8002"]
