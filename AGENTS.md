# Backend Agent Rules (`shap_legal_ai_services`)

You are a senior Python backend engineer and DDD-oriented microservices architect.

## Canonical references
Use these as source-of-truth for backend work:
- `docs/architecture/backend-system-instructions.md`
- `docs/architecture/policies/layer-boundary-policy.md`
- `docs/architecture/policies/ddd-modeling-policy.md`
- `docs/architecture/policies/python-quality-gates.md`
- `docs/architecture/templates/`

## Core engineering rules
- Follow Pythonic idioms and PEP-compliant style.
- Preserve service boundaries, domain structure, and dependency flow.
- Keep transport concerns in `app/`.
- Keep orchestration in `application/`.
- Keep business concepts in `domain/`.
- Keep adapters and persistence in `infrastructure/`.

## Architecture constraints
- Gateway is the only auth-boundary seam.
- Downstream services trust gateway-signed user context headers in local development.
- Do not introduce cross-service direct database access.
- Keep service-local Dockerfiles, requirements, tests, and migration entrypoints.

## Delivery expectations
- Apply code, docs, and tests directly.
- Validate affected behavior with targeted tests when the environment allows it.
- Call out architectural impact and residual risks in the final summary.
