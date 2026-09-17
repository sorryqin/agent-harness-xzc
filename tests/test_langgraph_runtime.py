from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent_harness.demo import build_demo, build_langgraph_demo_components
from agent_harness.domain import ModelToolCall, ModelTurn, RiskLevel, Task, TaskStatus
from agent_harness.langgraph_runtime import build_agent_graph, build_workflow_graph
from agent_harness.store import SQLiteStore
from agent_harness.tools import ToolGateway, ToolRegistry, ToolSpec


class ApprovalModel:
    def next_turn(self, *, system, messages, tools):
        if any(message.get("role") == "tool" for message in messages):
            text = "message delivery verified"
            return ModelTurn(text, assistant_message={"role": "assistant", "content": text})
        return ModelTurn(
            "",
            [ModelToolCall("provider-send", "message.send", {"text": "hello"}, "notify")],
            {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": "provider-send", "type": "function",
                    "function": {"name": "message.send", "arguments": '{"text":"hello"}'},
                }],
            },
        )


def test_multi_agent_parent_graph_runs_role_subgraphs(tmp_path):
    store, workflow_id, _ = build_demo(tmp_path / "business.db")
    gateway, models, prompts = build_langgraph_demo_components(store)
    saver = InMemorySaver()
    graph = build_workflow_graph(
        store=store, models=models, role_prompts=prompts,
        registry=gateway.registry, gateway=gateway, checkpointer=saver,
        allowed_tools={"analyst": {"repo.search"}, "backend": {"issue.label"}, "tester": set()},
    )
    config = {"configurable": {"thread_id": workflow_id}}

    result = graph.invoke({"workflow_id": workflow_id}, config)

    assert result["workflow_status"] == "succeeded"
    assert len(result["deliveries"]) == 3
    assert all(task.status == TaskStatus.SUCCEEDED for task in store.list_tasks(workflow_id))
    assert len(store.list_memories(f"workflow:{workflow_id}")) == 3
    assert len(list(graph.get_state_history(config))) > 3


def test_agent_interrupts_before_non_idempotent_tool_and_resumes(tmp_path):
    effects = []
    store = SQLiteStore(tmp_path / "business.db")
    workflow_id = store.create_workflow("send a notification")
    task = Task("notify", workflow_id, "notify", "operator", "send safely")
    store.add_task(task)
    assert store.start_task(task.id)

    registry = ToolRegistry()
    registry.register(ToolSpec(
        "message.send", "send message", RiskLevel.NON_IDEMPOTENT_WRITE,
        handler=lambda args: effects.append(args) or {"message_id": "m-1"},
    ))
    gateway = ToolGateway(store, registry)
    saver = InMemorySaver()
    graph = build_agent_graph(
        store=store,
        models={"operator": ApprovalModel()},
        role_prompts={"operator": "operate safely"},
        registry=registry,
        gateway=gateway,
        checkpointer=saver,
    )
    config = {"configurable": {"thread_id": workflow_id}}

    paused = graph.invoke({"workflow_id": workflow_id, "task_id": task.id}, config)
    assert paused["__interrupt__"][0].value["kind"] == "tool_approval"
    assert effects == []
    assert store.get_task(task.id).status == TaskStatus.WAITING_APPROVAL

    completed = graph.invoke(
        Command(resume={"action": "approve", "actor": "reviewer"}), config
    )
    assert completed["agent_status"] == "succeeded"
    assert effects == [{"text": "hello"}]
    assert store.get_task(task.id).status == TaskStatus.SUCCEEDED


def test_parent_workflow_resumes_nested_agent_interrupt(tmp_path):
    effects = []
    store = SQLiteStore(tmp_path / "business.db")
    workflow_id = store.create_workflow("nested approval")
    store.add_task(Task("notify", workflow_id, "notify", "operator", "send safely"))
    registry = ToolRegistry()
    registry.register(ToolSpec(
        "message.send", "send message", RiskLevel.NON_IDEMPOTENT_WRITE,
        handler=lambda args: effects.append(args) or {"message_id": "m-1"},
    ))
    gateway = ToolGateway(store, registry)
    graph = build_workflow_graph(
        store=store,
        models={"operator": ApprovalModel()},
        role_prompts={"operator": "operate safely"},
        registry=registry,
        gateway=gateway,
        checkpointer=InMemorySaver(),
        allowed_tools={"operator": {"message.send"}},
    )
    config = {"configurable": {"thread_id": workflow_id}}

    paused = graph.invoke({"workflow_id": workflow_id}, config)
    assert paused["__interrupt__"][0].value["task_id"] == "notify"
    assert effects == []

    completed = graph.invoke(
        Command(resume={"action": "approve", "actor": "reviewer"}), config
    )
    assert completed["workflow_status"] == "succeeded"
    assert completed["deliveries"]["notify"]["status"] == "succeeded"
    assert effects == [{"text": "hello"}]


def test_context_compression_is_an_explicit_checkpointed_node(tmp_path):
    store = SQLiteStore(tmp_path / "business.db")
    workflow_id = store.create_workflow("large context")
    task = Task("large", workflow_id, "large", "worker", "x" * 3000)
    store.add_task(task)
    assert store.start_task(task.id)
    registry = ToolRegistry()
    gateway = ToolGateway(store, registry)

    class FinalModel:
        def next_turn(self, *, system, messages, tools):
            return ModelTurn("done", assistant_message={"role": "assistant", "content": "done"})

    graph = build_agent_graph(
        store=store, models={"worker": FinalModel()}, role_prompts={"worker": "work"},
        registry=registry, gateway=gateway, context_budget=120,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": workflow_id}}
    result = graph.invoke({"workflow_id": workflow_id, "task_id": task.id}, config)

    assert result["compression_applied"] is True
    assert result["agent_status"] == "succeeded"
