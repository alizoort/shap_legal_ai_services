# DDD Modeling Policy

## Bounded Context
- Each service is a bounded context with explicit ownership of:
  - data model
  - invariants
  - behavior
  - contracts it publishes/consumes.

## Ubiquitous Language
- Use business terms consistently across:
  - domain entities
  - DTOs
  - endpoint names
  - event contracts.
- Avoid ambiguous aliases for core identifiers.

## Domain Model Guidance
- Entities:
  - Have stable identity and lifecycle.
  - Enforce core invariants.
- Value Objects:
  - Represent immutable concepts without identity.
- Domain Services:
  - Encapsulate business behavior that does not belong to one entity.

## Application Layer Guidance
- Orchestrates use cases and transactions.
- Coordinates ports:
  - repositories
  - event publishers
  - external service clients.
- Avoid embedding transport or framework logic in use cases.

## Infrastructure Layer Guidance
- Implements technical adapters and persistence concerns.
- Maintains mapping between domain/application models and storage models.
- Must not redefine domain truth; only persist and retrieve it.

## Contracts and Events
- Treat public HTTP schemas and shared event envelopes as stable contracts.
- Evolve contracts via additive changes and explicit compatibility strategy.
- New events require idempotency strategy and consumer failure-mode definition.

