# legal_ai_service

Stateless FastAPI service for the SHAP Legal AI employment compliance POC.

## Main routes

- `GET /legal-ai/ping`
- `GET /legal-ai/model-summary`
- `POST /legal-ai/analyze`

## Dataset and training

```bash
python3 -m services.legal_ai_service.scripts.export_synthetic_dataset
python3 -m services.legal_ai_service.scripts.train_compliance_model
```

## Local quality gates

```bash
ruff check .
mypy .
python3 -m unittest discover -s tests
```
