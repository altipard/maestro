# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir ".[server,openai,anthropic,google,grpc]"


FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
  tini ca-certificates \
  && rm -rf /var/lib/apt/lists/*

COPY --from=base /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=base /usr/local/bin /usr/local/bin
COPY --from=base /app/src /app/src

WORKDIR /app

EXPOSE 8080

ENTRYPOINT ["tini", "--"]
CMD ["python", "-m", "maestro", "--config", "/config.yaml"]
