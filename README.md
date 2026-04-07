# SHAP Legal AI Services

Minimal backend sibling scaffold for SHAP Legal AI. This repository mirrors the reference service-local workflow and layout, but only includes:

- `services/gateway_service`
- `services/legal_ai_service`
- `shap_service_runtime`
- `shap_contracts`

## Local quality gates

```bash
ruff check .
mypy .
python3 -m unittest discover -s tests
python3 -m unittest discover -s services/gateway_service/tests
python3 -m unittest discover -s services/legal_ai_service/tests
```

## Shared package workflow

```bash
python3 scripts/build_shared_packages.py --output-dir dist/shared-packages
python3 scripts/validate_shared_package_artifacts.py --dist-root dist/shared-packages/dist
./scripts/smoke_compose_stack.sh
```

## Local runtime

```bash
docker compose up --build
```

The local stack exposes:

- gateway: `http://127.0.0.1:8000`
- legal_ai_service: `http://127.0.0.1:8103`
- postgres: `127.0.0.1:5437`
