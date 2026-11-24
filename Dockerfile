FROM ghcr.io/astral-sh/uv:0.13.9-python3.13-trixie-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --locked --no-dev --no-install-project

COPY gfjproxy ./gfjproxy

ARG GFJPROXY_BRANCH
ARG GFJPROXY_VERSION

ENV GFJPROXY_BRANCH=${GFJPROXY_BRANCH} \
    GFJPROXY_VERSION=${GFJPROXY_VERSION} \
    PATH="/app/.venv/bin:$PATH" \
    PORT=5000 \
    PYTHONUNBUFFERED=1

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/health', timeout=4).read()"

SHELL ["/bin/sh", "-c"]

CMD exec gunicorn -b 0.0.0.0:${PORT} -k gevent -w 5 -t ${GFJPROXY_PROCESS_TIMEOUT:-300} "gfjproxy.app:create_app()"
