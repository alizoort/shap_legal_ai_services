# syntax=docker/dockerfile:1.4

FROM python:3.12-slim AS shared-packages

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PROGRESS_BAR=off

WORKDIR /build

COPY shap_contracts/ ./shap_contracts/
COPY shap_service_runtime/ ./shap_service_runtime/
COPY scripts/ ./scripts/
COPY shared_packages/versions.toml ./shared_packages/versions.toml

RUN --mount=type=cache,target=/root/.cache/pip \
    apt-get update && apt-get install -y --no-install-recommends build-essential && \
    python -m pip install --upgrade pip setuptools wheel build && \
    python scripts/build_shared_packages.py --output-dir /build/shared_packages

FROM python:3.12-slim AS dependencies

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_PROGRESS_BAR=off

WORKDIR /app

COPY --from=shared-packages /build/shared_packages/dist/shap-contracts/*.whl /tmp/wheels/
COPY --from=shared-packages /build/shared_packages/dist/shap-service-runtime/*.whl /tmp/wheels/
COPY services/gateway_service/requirements.txt ./requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install /tmp/wheels/shap_contracts-*.whl && \
    python -m pip install /tmp/wheels/shap_service_runtime-*.whl && \
    python -m pip install -r requirements.txt

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

COPY --from=dependencies /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin
COPY . .

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

EXPOSE 8000

CMD ["uvicorn", "services.gateway_service.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
