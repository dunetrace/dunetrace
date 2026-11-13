# explainer_svc is a library, not a runtime service.
# This Dockerfile is for running its tests in isolation only.
# In production it is COPY'd into the alerts and api service images.
FROM python:3.12-slim
WORKDIR /app

# Non-root, like the six runtime images. Nothing here holds a credential, but a
# test image that runs as uid 0 is still a root shell one `docker run` away, and
# keeping all seven identical means nobody has to remember which is which.
# A *numeric* USER (below) is what a Kubernetes `runAsNonRoot` admission policy
# needs — it cannot resolve a name to prove the user is not uid 0.
RUN groupadd --gid 10001 dunetrace \
 && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/dunetrace \
            --shell /usr/sbin/nologin dunetrace

COPY services/explainer/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY packages/schemas-py/ packages/schemas-py/
COPY packages/sdk-py/ packages/sdk-py/
COPY services/explainer/ services/explainer/
ENV PYTHONPATH=/app/packages/sdk-py:/app/services/explainer
ENV HOME=/home/dunetrace
# /app is root-owned, so as uid 10001 neither CPython's __pycache__ nor pytest's
# .pytest_cache can be written there. CPython gives up silently; pytest prints a
# warning on every run, so turn its cache off rather than leave the noise. (The
# only thing lost is --lf/--ff, which nothing in this image's CMD uses.)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTEST_ADDOPTS="-p no:cacheprovider"

# No EXPOSE and no HEALTHCHECK: this image serves nothing. Its CMD runs a test
# suite to completion and exits, so a readiness probe would have nothing to ask.
USER 10001:10001

CMD ["python", "-m", "pytest", "services/explainer/tests/", "-v"]
