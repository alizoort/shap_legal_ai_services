# Microservice Bootstrap Checklist

## 1. Service Skeleton
- [ ] Create service package under `services/<name>_service/`.
- [ ] Add `app/main.py` with `/health` and router wiring.
- [ ] Add `application`, `domain`, and `infrastructure` folders.
- [ ] Add `__init__.py` files for import clarity.

## 2. Config and Runtime
- [ ] Add required settings in `shap_service_runtime/config.py`.
- [ ] Validate critical runtime config at startup.
- [ ] Add docker-compose service definition and port mapping.

## 3. Persistence
- [ ] Provision service-specific DB URL.
- [ ] Add SQLAlchemy models and database module.
- [ ] Add migration baseline at:
  - `services/<name>_service/infrastructure/db/migrations/versions/0001_initial_schema.py`
- [ ] Add DB isolation tests if ownership changes.

## 4. Contracts
- [ ] Define request/response DTOs.
- [ ] Define or reuse event contracts in `shap_contracts/events.py`.
- [ ] Add contract tests for API/event behavior.

## 5. Messaging (if applicable)
- [ ] Add outbox table and publisher.
- [ ] Add inbox table and consumer.
- [ ] Implement retry + DLQ semantics and idempotency keys.
- [ ] Add topology and retry behavior tests.

## 6. Security
- [ ] Enforce signed user-context headers if required.
- [ ] Add permission checks at controller boundary.
- [ ] Ensure no bearer-token validation outside gateway.

## 7. Quality Gates
- [ ] Add unit tests for domain/application behavior.
- [ ] Add integration tests for critical flows.
- [ ] Run lint, type checks, and tests before merge.

## 8. Instruction Artifacts
- [ ] Add instruction manifest JSON.
- [ ] Generate instruction markdown from manifest.
- [ ] Commit manifest + generated instructions with implementation.
