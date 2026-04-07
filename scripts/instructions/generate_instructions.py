#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class ManifestValidationError(ValueError):
    pass


REQUIRED_TOP_LEVEL_FIELDS = (
    "title",
    "scope",
    "bounded_context",
    "use_cases",
    "commands",
    "queries",
    "events_emitted",
    "events_consumed",
    "api_endpoints",
    "data_ownership",
    "dependencies",
    "security_constraints",
    "observability",
    "test_plan",
)

ALLOWED_SCOPE_VALUES = {"feature", "service", "microservice"}
ALLOWED_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE"}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestValidationError(f"Manifest not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestValidationError(f"Invalid JSON in manifest: {exc}") from exc
    if not isinstance(data, dict):
        raise ManifestValidationError("Manifest root must be a JSON object")
    return data


def _require_non_empty_string(
    errors: list[str], manifest: dict[str, Any], key: str
) -> None:
    value = manifest.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"'{key}' must be a non-empty string")


def _require_string_list(errors: list[str], manifest: dict[str, Any], key: str) -> None:
    value = manifest.get(key)
    if not isinstance(value, list):
        errors.append(f"'{key}' must be a list of strings")
        return
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            errors.append(f"'{key}[{index}]' must be a non-empty string")


def _validate_api_endpoints(errors: list[str], manifest: dict[str, Any]) -> None:
    endpoints = manifest.get("api_endpoints")
    if not isinstance(endpoints, list):
        errors.append("'api_endpoints' must be a list")
        return
    for index, endpoint in enumerate(endpoints):
        if not isinstance(endpoint, dict):
            errors.append(f"'api_endpoints[{index}]' must be an object")
            continue
        method = endpoint.get("method")
        path = endpoint.get("path")
        purpose = endpoint.get("purpose")
        if method not in ALLOWED_METHODS:
            errors.append(
                f"'api_endpoints[{index}].method' must be one of {sorted(ALLOWED_METHODS)}"
            )
        if not isinstance(path, str) or not path.strip():
            errors.append(f"'api_endpoints[{index}].path' must be a non-empty string")
        if not isinstance(purpose, str) or not purpose.strip():
            errors.append(f"'api_endpoints[{index}].purpose' must be a non-empty string")


def _validate_data_ownership(errors: list[str], manifest: dict[str, Any]) -> None:
    ownership = manifest.get("data_ownership")
    if not isinstance(ownership, dict):
        errors.append("'data_ownership' must be an object")
        return
    for key in ("service_database", "cross_service_access"):
        value = ownership.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"'data_ownership.{key}' must be a non-empty string")
    entities = ownership.get("owned_entities")
    if not isinstance(entities, list) or not entities:
        errors.append("'data_ownership.owned_entities' must be a non-empty list")
        return
    for index, item in enumerate(entities):
        if not isinstance(item, str) or not item.strip():
            errors.append(
                f"'data_ownership.owned_entities[{index}]' must be a non-empty string"
            )


def _validate_observability(errors: list[str], manifest: dict[str, Any]) -> None:
    observability = manifest.get("observability")
    if not isinstance(observability, dict):
        errors.append("'observability' must be an object")
        return
    for key in ("logs", "metrics", "alerts"):
        value = observability.get(key)
        if not isinstance(value, list):
            errors.append(f"'observability.{key}' must be a list")
            continue
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                errors.append(
                    f"'observability.{key}[{index}]' must be a non-empty string"
                )


def _validate_test_plan(errors: list[str], manifest: dict[str, Any]) -> None:
    test_plan = manifest.get("test_plan")
    if not isinstance(test_plan, dict):
        errors.append("'test_plan' must be an object")
        return
    for key in ("unit", "contract", "integration", "acceptance"):
        value = test_plan.get(key)
        if not isinstance(value, list):
            errors.append(f"'test_plan.{key}' must be a list")
            continue
        for index, item in enumerate(value):
            if not isinstance(item, str) or not item.strip():
                errors.append(f"'test_plan.{key}[{index}]' must be a non-empty string")


def validate_manifest(manifest: dict[str, Any]) -> None:
    errors: list[str] = []
    for field in REQUIRED_TOP_LEVEL_FIELDS:
        if field not in manifest:
            errors.append(f"Missing required field: '{field}'")

    _require_non_empty_string(errors, manifest, "title")
    _require_non_empty_string(errors, manifest, "bounded_context")
    _require_string_list(errors, manifest, "use_cases")
    _require_string_list(errors, manifest, "commands")
    _require_string_list(errors, manifest, "queries")
    _require_string_list(errors, manifest, "events_emitted")
    _require_string_list(errors, manifest, "events_consumed")
    _require_string_list(errors, manifest, "dependencies")
    _require_string_list(errors, manifest, "security_constraints")

    scope = manifest.get("scope")
    if scope not in ALLOWED_SCOPE_VALUES:
        errors.append(f"'scope' must be one of {sorted(ALLOWED_SCOPE_VALUES)}")

    _validate_api_endpoints(errors, manifest)
    _validate_data_ownership(errors, manifest)
    _validate_observability(errors, manifest)
    _validate_test_plan(errors, manifest)

    if errors:
        joined = "\n".join(f"- {message}" for message in errors)
        raise ManifestValidationError(f"Manifest validation failed:\n{joined}")


def _to_bullets(items: list[str]) -> str:
    if not items:
        return "- (none)"
    return "\n".join(f"- {item}" for item in items)


def _format_endpoints(endpoints: list[dict[str, str]]) -> str:
    if not endpoints:
        return "- (none)"
    rows: list[str] = []
    for endpoint in endpoints:
        rows.append(
            f"- `{endpoint['method']} {endpoint['path']}`: {endpoint['purpose']}"
        )
    return "\n".join(rows)


def generate_instruction_markdown(manifest: dict[str, Any]) -> str:
    title = manifest["title"].strip()
    scope = manifest["scope"]
    bounded_context = manifest["bounded_context"].strip()
    description = manifest.get("description", "")
    description_text = description.strip() if isinstance(description, str) else ""

    data_ownership = manifest["data_ownership"]
    observability = manifest["observability"]
    test_plan = manifest["test_plan"]

    lines = [
        f"# {title} ({scope.capitalize()} Instructions)",
        "",
        "## Summary",
        f"- Bounded context: `{bounded_context}`",
        f"- Scope: `{scope}`",
    ]
    if description_text:
        lines.append(f"- Description: {description_text}")

    lines.extend(
        [
            "",
            "## Use Cases",
            _to_bullets(manifest["use_cases"]),
            "",
            "## Command and Query Model",
            "### Commands",
            _to_bullets(manifest["commands"]),
            "",
            "### Queries",
            _to_bullets(manifest["queries"]),
            "",
            "## API Endpoints",
            _format_endpoints(manifest["api_endpoints"]),
            "",
            "## Event Contracts",
            "### Events Emitted",
            _to_bullets(manifest["events_emitted"]),
            "",
            "### Events Consumed",
            _to_bullets(manifest["events_consumed"]),
            "",
            "## Data Ownership",
            f"- Service database: `{data_ownership['service_database']}`",
            "- Owned entities:",
            _to_bullets(data_ownership["owned_entities"]),
            f"- Cross-service access policy: {data_ownership['cross_service_access']}",
            "",
            "## Dependencies",
            _to_bullets(manifest["dependencies"]),
            "",
            "## Security Constraints",
            _to_bullets(manifest["security_constraints"]),
            "",
            "## Observability",
            "### Logs",
            _to_bullets(observability["logs"]),
            "",
            "### Metrics",
            _to_bullets(observability["metrics"]),
            "",
            "### Alerts",
            _to_bullets(observability["alerts"]),
            "",
            "## Test Plan",
            "### Unit",
            _to_bullets(test_plan["unit"]),
            "",
            "### Contract",
            _to_bullets(test_plan["contract"]),
            "",
            "### Integration",
            _to_bullets(test_plan["integration"]),
            "",
            "### Acceptance",
            _to_bullets(test_plan["acceptance"]),
            "",
        ]
    )

    rollout = manifest.get("rollout")
    if isinstance(rollout, list):
        lines.extend(["## Rollout", _to_bullets([str(item) for item in rollout]), ""])

    rollback = manifest.get("rollback")
    if isinstance(rollback, list):
        lines.extend(["## Rollback", _to_bullets([str(item) for item in rollback]), ""])

    return "\n".join(lines).rstrip() + "\n"


def _default_output_path(manifest_path: Path) -> Path:
    if manifest_path.suffix.lower() == ".json":
        return manifest_path.with_suffix(".instructions.md")
    return manifest_path.parent / f"{manifest_path.name}.instructions.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate backend instruction markdown from manifest JSON."
    )
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON file.")
    parser.add_argument(
        "--output",
        required=False,
        help="Optional output markdown path. Defaults to manifest sibling file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_path = (
        Path(args.output).resolve()
        if args.output
        else _default_output_path(manifest_path)
    )

    try:
        manifest = _load_json(manifest_path)
        validate_manifest(manifest)
        rendered = generate_instruction_markdown(manifest)
    except ManifestValidationError as exc:
        print(str(exc))
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote instructions to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
