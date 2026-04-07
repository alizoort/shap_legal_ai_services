# Backend System Instructions

## Purpose
This document defines mandatory backend engineering rules for all new features and services in `shap_legal_ai_services`.

## Scope
- Applies to all new code in:
  - `services/*`
  - `shap_contracts/*`
  - `shap_service_runtime/*`
  - backend-related tests and docs
- Legacy code copied only for scaffold support should be normalized when touched.

## Non-Negotiables
1. Keep service boundaries explicit and isolated.
2. Keep data ownership isolated with database-per-service.
3. Keep dependency flow inward: `app -> application -> domain`, with `infrastructure` implementing adapters.
4. Gateway is the only auth-token validation boundary; downstream services trust signed user-context headers.
5. Use contract-first request/response DTOs and shared events in `shap_contracts/events.py`.

## Required Service Structure
Each service must follow this shape:
- `app/`
- `application/`
- `domain/`
- `infrastructure/`

## Layer Rules
- `domain` must not import FastAPI, SQLAlchemy, HTTPX, or Celery.
- `application` must not raise `HTTPException`; controllers map typed errors to transport responses.
- `application` should depend on ports/interfaces rather than concrete infrastructure adapters.
- `infrastructure` owns framework/runtime concerns and adapter implementations.

## DDD Rules
- Treat each service as a bounded context with explicit ubiquitous language.
- Keep domain invariants close to domain models.
- Prefer typed domain/application exceptions over generic runtime errors.
- Keep DTOs separate from ORM models and separate from domain entities.

## Contracts and Integration
- Public APIs must use explicit request/response schemas.
- Cross-service events must use shared contracts in `shap_contracts/events.py`.
- New event types require contract updates plus producer/consumer tests.
- Do not introduce direct cross-service database reads or writes.

## Reliability and Security
- Validate critical runtime configuration at startup.
- Avoid broad exception swallowing.
- Never trust client-provided identity headers outside the gateway.
- Do not validate bearer tokens directly in non-gateway services.

## Python Quality Gates
- New and changed code must use explicit type hints.
- Run `ruff check`, `mypy`, and `python3 -m unittest`.
- Add unit tests for non-trivial domain/application logic.
- Add contract tests for API/event changes.
- Add integration tests for critical cross-layer behavior.

## Instruction Generation Process
Every new service or feature instruction set must include:
1. A manifest JSON matching `docs/architecture/templates/instruction-manifest.schema.json`.
2. A generated markdown file produced by `python3 scripts/instructions/generate_instructions.py --manifest <path>`.
3. Both files committed with the implementation.
