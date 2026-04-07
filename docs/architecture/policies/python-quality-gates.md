# Python Quality Gates

## Applicability
- Mandatory for new backend modules.
- Mandatory for touched areas of existing modules.

## Coding Standards
- Python 3.12-compatible code.
- Explicit type hints for function signatures and key variables.
- Prefer small cohesive functions over large multi-purpose routines.
- Use clear names that reflect domain intent.

## Error Handling
- Use typed exceptions for known business failures.
- Avoid unbounded `except Exception` unless:
  - behavior is explicitly defensive, and
  - error is logged with context, and
  - fallback behavior is deterministic.

## Testing Standards
- Unit tests required for non-trivial domain/application logic.
- Contract tests required for event/API changes.
- Integration tests required for cross-service behavior changes.

## Tooling Targets
- Lint: `ruff check`
- Type checks: `mypy`
- Tests: `python3 -m unittest`

## Pull Request Checklist
1. Types added for new/changed code.
2. Error paths and failure modes tested.
3. Contracts updated (if API/event changed).
4. Architecture boundaries respected.
5. Generated feature/service instruction document included.

