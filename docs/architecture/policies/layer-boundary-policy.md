# Layer Boundary Policy

## Dependency Direction
Allowed dependency direction for new code:
1. `app -> application -> domain`
2. `app -> infrastructure` only for wiring and adapter composition.
3. `infrastructure -> application/domain` to implement ports.

Disallowed:
- `domain -> application`
- `domain -> app`
- `domain -> infrastructure`
- `application -> app`

## Framework Boundary Rules
- `domain` cannot import:
  - `fastapi`
  - `sqlalchemy`
  - `httpx`
  - `celery`
- `application` cannot raise `fastapi.HTTPException`.
- Transport-layer exceptions are only allowed in `app/controllers`.

## Data Access Rules
- Application layer should work through ports/interfaces for repositories and external systems.
- ORM models are infrastructure concerns; do not expose ORM objects as API contracts.
- Cross-service direct DB reads/writes are forbidden.

## Error Mapping Rules
- Domain/application layers raise typed domain/application exceptions.
- Controllers translate exceptions to HTTP response codes and payloads.
- Messaging consumers map processing outcomes to inbox status transitions.

## Migration Policy
- Existing legacy files are allowed during migration.
- Net-new boundary violations are not allowed.
- Touched legacy files must be progressively aligned.

## Enforcement
- Tests:
  - `tests/test_import_boundaries.py`
  - `tests/test_service_db_isolation.py`
  - `tests/test_layer_fastapi_boundary_guard.py`

