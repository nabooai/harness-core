# harness-core — the read-API + dashboard server, containerized.
#
#   docker build -t harness-core .
#   docker run --rm -p 8077:8077 -v "$PWD/runs:/data/local/runs" harness-core
#   # then open http://localhost:8077 — point it at run dirs by mounting them under
#   # $HARNESS_RUNS_BASE (/data), as <label>/runs/, or set HARNESS_RUNS_ROOTS explicitly:
#   docker run --rm -p 8077:8077 -e HARNESS_RUNS_ROOTS="fdav14=/data/fdav14/runs" \
#     -v "$PWD/fdav14/runs:/data/fdav14/runs" harness-core

FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

# Copy the project and install it + the `server` extra into a .venv from the locked
# resolution (reproducible). Dev tooling (the `dev` group) is excluded from the image.
COPY . .
RUN uv sync --frozen --no-dev --extra server

ENV PATH="/app/.venv/bin:$PATH" \
    HARNESS_RUNS_BASE=/data \
    PYTHONUNBUFFERED=1

# Run dirs are mounted here (auto-discovers every <label>/runs/ beneath it).
VOLUME ["/data"]
EXPOSE 8077

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8077/healthz').status==200 else 1)"

CMD ["harness-core", "server", "--host", "0.0.0.0", "--port", "8077"]
