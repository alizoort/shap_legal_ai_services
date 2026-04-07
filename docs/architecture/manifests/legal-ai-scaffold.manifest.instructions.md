# Legal AI Scaffold (Service Instructions)

## Summary
- Bounded context: `legal_ai`
- Scope: `service`
- Description: Reserve the gateway, contracts, and service seams for the initial SHAP Legal AI proof of concept.

## Use Cases
- Reserve the legal-ai gateway route and downstream service boundary.
- Expose a placeholder legal-ai endpoint that proves the scaffold is wired correctly.
- Keep contracts and runtime helpers ready for the later vertical-slice implementation.

## Command and Query Model
### Commands
- Refresh scaffold status metadata.

### Queries
- Read scaffold readiness metadata.

## API Endpoints
- `GET /legal-ai/ping`: Return placeholder legal-ai scaffold metadata.

## Event Contracts
### Events Emitted
- PlaceholderReserved

### Events Consumed
- (none)

## Data Ownership
- Service database: `legal_ai_db`
- Owned entities:
- reserved legal-ai scaffold state
- Cross-service access policy: No direct cross-service database reads or writes.

## Dependencies
- gateway_service
- shap_service_runtime
- shap_contracts
- postgres

## Security Constraints
- Gateway remains the only auth-token validation boundary.
- Downstream services trust signed user-context headers in local development mode.
- Critical runtime configuration must be validated at startup.

## Observability
### Logs
- startup validation
- migration/readiness failures

### Metrics
- health and readiness endpoints
- HTTP request counters and latency

### Alerts
- (none)

## Test Plan
### Unit
- Placeholder scaffold service response

### Contract
- Gateway reserved route shape

### Integration
- Docker Compose boot and health checks

### Acceptance
- Gateway forwards /legal-ai/ping to legal_ai_service
