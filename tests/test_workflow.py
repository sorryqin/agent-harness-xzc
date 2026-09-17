from agent_harness.agents import DemoAgent
from agent_harness.context import ContextAssembler
from agent_harness.domain import Task, TaskStatus, WorkflowStatus
from agent_harness.orchestrator import Orchestrator
from agent_harness.store import SQLiteStore


def test_dag_executes_in_dependency_order(tmp_path):
    store = SQLiteStore(tmp_path / "state.db")
    workflow_id = store.create_workflow("ship safely")
    store.add_task(Task("analysis", workflow_id, "analyze", "analyst", "analyze"))
    store.add_task(Task("build", workflow_id, "build", "backend", "build", depends_on=["analysis"]))
    store.add_task(Task("test", workflow_id, "test", "tester", "test", depends_on=["build"]))
    orchestrator = Orchestrator(store, {
        "analyst": DemoAgent(), "backend": DemoAgent(), "tester": DemoAgent(),
    })

    assert orchestrator.run(workflow_id) == WorkflowStatus.SUCCEEDED
    assert all(task.status == TaskStatus.SUCCEEDED for task in store.list_tasks(workflow_id))
    started = [e["task_id"] for e in store.recent_events(workflow_id, 100) if e["event_type"] == "task.started"]
    assert started == ["analysis", "build", "test"]


def test_context_is_role_scoped_and_budgeted(tmp_path):
    store = SQLiteStore(tmp_path / "state.db")
    workflow_id = store.create_workflow("goal")
    store.add_task(Task("a", workflow_id, "a", "analyst", "A" * 500))
    task = Task("b", workflow_id, "b", "backend", "B", depends_on=["a"])
    store.add_task(task)
    store.finish_task("a", {"result": "x" * 1000})
    context = ContextAssembler(store, budget_tokens=300).assemble(workflow_id, task)

    assert "system_constraints" in context["layers"]
    assert "current_task" in context["layers"]
    assert context["estimated_tokens"] <= 340  # required layers may slightly exceed budget

