FROM python:3.12-slim
WORKDIR /app

# Non-root runtime. python:3.12-slim's default user is uid 0, so without this a
# container escape or an RCE through any dependency starts as root holding
# DATABASE_URL and DUNETRACE_MASTER_KEY — the key both of this image's workers
# exist to decrypt every stored Langfuse/LangSmith/Braintrust/ElevenLabs
# credential with. A *numeric* USER (below) is also what a Kubernetes
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

# requirements BEFORE any source. This was the only one of the seven images that
# copied packages/schemas-py/ first, so every edit anywhere under that package
# invalidated the pip layer and forced a full reinstall of cryptography,
# asyncpg, httpx and pydantic — twice per release, since the image is built for
# amd64 and arm64 and the non-native half runs under QEMU.
COPY services/integrations/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY packages/schemas-py/ packages/schemas-py/
COPY services/integrations/ services/integrations/

ENV PYTHONPATH=/app/packages/schemas-py:/app/services/integrations
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

# BOTH ports, because this one image runs two different workers depending on
# `command`: integrations_svc.worker on INTEGRATIONS_METRICS_PORT (9104) and
# integrations_svc.elevenlabs_worker on ELEVENLABS_METRICS_PORT (9105). They
# share one Settings class, which is why there are two env vars rather than one
# METRICS_PORT. EXPOSE publishes nothing by itself.
EXPOSE 9104
EXPOSE 9105

# Readiness baked into the image, so the published artifact carries a probe
# outside compose (plain `docker run`, ECS, Nomad). Both compose files define
# their own healthcheck, which overrides this one.
#
# It probes BOTH ports and passes if either answers, rather than hardcoding the
# CMD's 9104: only one worker runs per container and 127.0.0.1 is that
# container's own loopback, so at most one of the two can ever respond — which
# makes "either" exactly right for an image whose entrypoint is chosen at run
# time. Hardcoding 9104 would have reported every elevenlabs container
# permanently unhealthy unless its operator also replaced --health-cmd.
#
# Both workers are opt-in: with their *_WORKER_ENABLED flag false they log one
# line and exit 0, and a stopped container is never probed — `Exited (0)` stays
# their normal disabled state, not "unhealthy".
#
# 127.0.0.1, not localhost: the server binds IPv4 and `localhost` resolves to
# ::1 first on hosts whose /etc/hosts lists it that way.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${INTEGRATIONS_METRICS_PORT:-9104}/ready', timeout=4)" \
   || python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${ELEVENLABS_METRICS_PORT:-9105}/ready', timeout=4)"

USER 10001:10001

CMD ["python", "-m", "integrations_svc.worker"]
