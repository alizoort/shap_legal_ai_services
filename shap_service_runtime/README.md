# shap-service-runtime

Versioned shared runtime package source for SHAP Legal AI services.

This package is the source of truth for:
- shared environment/config parsing helpers
- signed user-context helpers for the gateway boundary
- RabbitMQ topology helpers for optional async flows
- generic Alembic/database helpers
- HTTP observability and readiness helpers

## Configuration

### MessagingSettings

RabbitMQ messaging topology and retry policy configuration.

```python
from shap_service_runtime.config import get_messaging_settings

settings = get_messaging_settings()
print(settings.url)
print(settings.exchange)
print(settings.legal_ai_events_queue)
```

Environment variables:
- `RABBITMQ_URL`
- `RABBITMQ_EXCHANGE`
- `RABBITMQ_LEGAL_AI_EVENTS_QUEUE`
- `RABBITMQ_PUBLISH_BATCH_SIZE`
- `RABBITMQ_PREFETCH_COUNT`
- `RABBITMQ_RETRY_DELAYS_MS`
- `RABBITMQ_INBOX_MAX_RETRIES`

### AuthSettings

Gateway user-context signing and validation configuration.

```python
from shap_service_runtime.config import get_auth_settings

settings = get_auth_settings()
print(settings.secret)
print(settings.max_age_seconds)
```

### ObservabilitySettings

CORS and readiness/metrics configuration.

```python
from shap_service_runtime.config import get_observability_settings

settings = get_observability_settings()
origins = settings.get_cors_origins()
regex = settings.get_cors_origin_regex()
```

### Legacy CommonSettings

`CommonSettings` remains available for service bootstraps that mirror the reference repo style.

```python
from shap_service_runtime.config import get_common_settings

settings = get_common_settings()
print(settings.app_name)
print(settings.rabbitmq_exchange)
```
