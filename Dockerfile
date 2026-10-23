# Şirket mirror'ı / UBI python için: --build-arg BASE_IMAGE=...
ARG BASE_IMAGE=python:3.13-slim
FROM ${BASE_IMAGE}

# İç Nexus/Artifactory için: --build-arg PIP_INDEX_URL=... --build-arg PIP_TRUSTED_HOST=...
ARG PIP_INDEX_URL
ARG PIP_TRUSTED_HOST
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    MEM0_DIR=/tmp/.mem0 \
    MEM0_TELEMETRY=False

# UBI tabanlı imajda apt yoksa: microdnf install -y git ca-certificates
USER root
RUN apt-get update \
    && apt-get install -y git ca-certificates --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# OpenShift rastgele UID: grup 0 yazabilmeli
RUN mkdir -p /app/data \
    && chgrp -R 0 /app && chmod -R g=u /app

USER 1001
EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port 8080 --proxy-headers --forwarded-allow-ips='*' --workers ${WORKERS:-1} --loop uvloop --http httptools"]
