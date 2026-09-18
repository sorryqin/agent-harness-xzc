import pytest

from agent_harness.domain import (
    ApprovalRequired, CallConflictError, RiskLevel, UnsafeRetryError,
)
from agent_harness.store import SQLiteStore
from agent_harness.tools import ToolGateway, ToolRegistry, ToolSpec


def setup_gateway(tmp_path, risk, handler):
    store = SQLiteStore(tmp_path / "state.db")
    workflow_id = store.create_workflow("test tools")
    registry = ToolRegistry()
    registry.register(ToolSpec("tool", "test tool", risk, handler=handler))
    return store, workflow_id, ToolGateway(store, registry)


def test_success_is_replayed_without_duplicate_side_effect(tmp_path):
    effects = []
    store, workflow_id, gateway = setup_gateway(
        tmp_path, RiskLevel.IDEMPOTENT_WRITE, lambda args: effects.append(args) or {"ok": True}
    )
    kwargs = dict(session_id=store.session_id(workflow_id), workflow_id=workflow_id,
                  task_id="task", tool_name="tool", arguments={"x": 1},
                  purpose="test", call_id="call-1")

    assert gateway.execute(**kwargs) == {"ok": True}
    assert gateway.execute(**kwargs) == {"ok": True}
    assert effects == [{"x": 1}]


def test_non_idempotent_call_requires_approval_before_side_effect(tmp_path):
    effects = []
    store, workflow_id, gateway = setup_gateway(
        tmp_path, RiskLevel.NON_IDEMPOTENT_WRITE,
        lambda args: effects.append(args) or {"sent": True},
    )
    kwargs = dict(session_id=store.session_id(workflow_id), workflow_id=workflow_id,
                  task_id="task", tool_name="tool", arguments={"text": "hello"},
                  purpose="notify customer", call_id="call-send")

    with pytest.raises(ApprovalRequired) as caught:
        gateway.execute(**kwargs)
    assert effects == []

    store.decide_approval(caught.value.approval_id, True, "reviewer")
    assert gateway.execute(**kwargs) == {"sent": True}
    assert len(effects) == 1


def test_call_id_cannot_be_reused_with_different_arguments(tmp_path):
    store, workflow_id, gateway = setup_gateway(
        tmp_path, RiskLevel.READ_ONLY, lambda args: args
    )
    common = dict(session_id=store.session_id(workflow_id), workflow_id=workflow_id,
                  task_id="task", tool_name="tool", purpose="read", call_id="same")
    gateway.execute(arguments={"x": 1}, **common)
    with pytest.raises(CallConflictError):
        gateway.execute(arguments={"x": 2}, **common)


def test_atomic_claim_allows_only_one_running_owner(tmp_path):
    store, workflow_id, gateway = setup_gateway(tmp_path, RiskLevel.READ_ONLY, lambda args: args)
    fingerprint = gateway.fingerprint({"x": 1})
    with store.transaction() as conn:
        now = "2026-01-01T00:00:00+00:00"
        conn.execute(
            """INSERT INTO tool_calls
            (call_id, session_id, workflow_id, task_id, tool_name, risk, args_fingerprint,
             args_json, purpose, status, attempt, lease_owner, lease_until, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', 1, 'other', 9999999999, ?, ?)""",
            ("busy", store.session_id(workflow_id), workflow_id, "task", "tool",
             RiskLevel.READ_ONLY, fingerprint, '{"x":1}', "read", now, now),
        )
    with pytest.raises(RuntimeError, match="already running"):
        gateway.execute(session_id=store.session_id(workflow_id), workflow_id=workflow_id,
                        task_id="task", tool_name="tool", arguments={"x": 1},
                        purpose="read", call_id="busy")


def test_non_idempotent_failure_becomes_uncertain_and_is_not_retried(tmp_path):
    effects = []

    def ambiguous_failure(args):
        effects.append(args)
        raise TimeoutError("response lost after send")

    store, workflow_id, gateway = setup_gateway(
        tmp_path, RiskLevel.NON_IDEMPOTENT_WRITE, ambiguous_failure
    )
    kwargs = dict(session_id=store.session_id(workflow_id), workflow_id=workflow_id,
                  task_id="task", tool_name="tool", arguments={"text": "hello"},
                  purpose="notify", call_id="ambiguous")
    with pytest.raises(ApprovalRequired) as caught:
        gateway.execute(**kwargs)
    store.decide_approval(caught.value.approval_id, True, "reviewer")
    with pytest.raises(TimeoutError):
        gateway.execute(**kwargs)
    with pytest.raises(UnsafeRetryError):
        gateway.execute(**kwargs)
    assert len(effects) == 1
