# Architecture notes

## Boundary of responsibility

The harness owns execution semantics, not model intelligence:

| Layer | Owns | Must not own |
|---|---|---|
| Orchestrator | DAG readiness, task transitions, recovery | External side effects |
| Agent | Role reasoning, deciding what result/tool is needed | Direct filesystem/API access |
| ToolGateway | Policy, approval, atomic claim, execution, replay | Task dependency decisions |
| Store | Durable state, audit log, artifact links | Business logic |
| ContextAssembler | Role visibility and token budget | Source-of-truth state |

## State transitions

```text
Workflow: pending -> running -> succeeded
                         |----> waiting_approval -> running
                         `----> failed

Task: pending -> running -> succeeded
                    |----> pending (bounded retry)
                    |----> waiting_approval -> pending
                    `----> failed

Tool call: [new] -> running -> succeeded -> replay
                 |      `--> failed (read/idempotent write may retry)
                 |      `--> uncertain (non-idempotent: never blind retry)
                 `--> waiting_approval -> running
```

The database is the source of truth. In-memory agent messages are disposable.

## Crash windows

There is no general exactly-once guarantee across a local database and an arbitrary
external service. The gateway therefore makes the guarantee explicit:

- Before execution: a transactional claim prevents two workers from starting the
  same call concurrently.
- After success is persisted: repeated calls replay the saved result.
- Crash during a read or idempotent write: the expired lease may be reclaimed.
- Crash during a non-idempotent write: the call becomes uncertain and must be
  reconciled using the provider's idempotency key or an operator check.

This is intentionally safer than claiming exactly-once behavior that the external
system cannot support.

## LangGraph integration decision

LangGraph is optional. If adopted, use it as the outer graph/runtime and keep a
single owner for each concern:

- LangGraph node = adapter around one role Agent.
- LangGraph checkpointer = conversational/node continuation state.
- This store = authoritative task, approval, artifact, and side-effect ledger.
- Every LangGraph tool node still calls `ToolGateway.execute`.

Do not mirror task status independently in both systems. Either derive graph state
from this store, or replace the orchestrator completely and retain only the gateway,
store, and context layers.

