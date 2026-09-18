import pytest

from agent_harness.agents import ToolCallingAgent
from agent_harness.context import ContextAssembler
from agent_harness.domain import ApprovalRequired, ModelToolCall, ModelTurn, RiskLevel, Task
from agent_harness.store import SQLiteStore
from agent_harness.tools import ToolGateway, ToolRegistry, ToolSpec


class FakeModel:
    def __init__(self):
        self.calls = 0

    def next_turn(self, *, system, messages, tools):
        self.calls += 1
        if any(message.get("role") == "tool" for message in messages):
            return ModelTurn("notification complete", assistant_message={
                "role": "assistant", "content": "notification complete",
            })
        return ModelTurn("", [ModelToolCall("provider-1", "send", {"text": "hello"})], {
            "role": "assistant", "content": None,
            "tool_calls": [{
                "id": "provider-1", "type": "function",
                "function": {"name": "send", "arguments": '{"text":"hello"}'},
            }],
        })


def test_agent_model_decision_and_tool_result_survive_approval_pause(tmp_path):
    effects = []
    store = SQLiteStore(tmp_path / "state.db")
    workflow_id = store.create_workflow("notify")
    task = Task("notify", workflow_id, "notify", "operator", "send notification")
    store.add_task(task)
    registry = ToolRegistry()
    registry.register(ToolSpec(
        "send", "send a notification", RiskLevel.NON_IDEMPOTENT_WRITE,
        handler=lambda args: effects.append(args) or {"sent": True},
    ))
    gateway = ToolGateway(store, registry)
    model = FakeModel()
    agent = ToolCallingAgent("You are an operator", model, gateway, registry, store)
    context = ContextAssembler(store).assemble(workflow_id, task)

    with pytest.raises(ApprovalRequired) as caught:
        agent.run(task, context)
    assert model.calls == 1
    assert effects == []

    store.decide_approval(caught.value.approval_id, True, "reviewer")
    result = agent.run(task, context)
    assert result.output["text"] == "notification complete"
    assert model.calls == 2  # step 0 came from checkpoint; only step 1 called the model
    assert effects == [{"text": "hello"}]
