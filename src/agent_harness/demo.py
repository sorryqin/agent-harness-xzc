from __future__ import annotations

import uuid
from pathlib import Path

from .agents import DemoAgent
from .context import ContextAssembler
from .domain import RiskLevel, Task
from .orchestrator import Orchestrator
from .store import SQLiteStore
from .tools import ToolGateway, ToolRegistry, ToolSpec


def build_demo(db_path: str | Path = "harness.db") -> tuple[SQLiteStore, str, Orchestrator]:
    store = SQLiteStore(db_path)
    workflow_id = store.create_workflow("分析变更影响，完成实现方案并给出测试结论")
    tasks = [
        Task(
            id=f"task_analysis_{uuid.uuid4().hex[:6]}", workflow_id=workflow_id,
            title="需求与影响分析", role="analyst",
            instructions="识别改动范围、风险和验收标准。",
        ),
    ]
    tasks.append(Task(
        id=f"task_backend_{uuid.uuid4().hex[:6]}", workflow_id=workflow_id,
        title="后端方案设计", role="backend",
        instructions="基于影响分析设计实现方案。", depends_on=[tasks[0].id],
    ))
    tasks.append(Task(
        id=f"task_test_{uuid.uuid4().hex[:6]}", workflow_id=workflow_id,
        title="测试验证", role="tester",
        instructions="根据分析与实现方案生成测试结论。",
        depends_on=[tasks[0].id, tasks[1].id],
    ))
    for task in tasks:
        store.add_task(task)
    agents = {role: DemoAgent() for role in ("analyst", "backend", "tester")}
    return store, workflow_id, Orchestrator(store, agents, ContextAssembler(store, budget_tokens=1200))


def build_example_gateway(store: SQLiteStore) -> ToolGateway:
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="repo.search", description="Search repository text", risk=RiskLevel.READ_ONLY,
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        handler=lambda args: {"query": args["query"], "matches": []},
    ))
    registry.register(ToolSpec(
        name="issue.label", description="Set labels to an exact desired list",
        risk=RiskLevel.IDEMPOTENT_WRITE,
        input_schema={
            "type": "object",
            "properties": {
                "issue": {"type": "integer"},
                "labels": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["issue", "labels"],
        },
        handler=lambda args: {"issue": args["issue"], "labels": args["labels"]},
    ))
    registry.register(ToolSpec(
        name="message.send", description="Send an external message",
        risk=RiskLevel.NON_IDEMPOTENT_WRITE,
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        handler=lambda args: {"message_id": f"msg_{uuid.uuid4().hex[:8]}", "text": args["text"]},
    ))
    return ToolGateway(store, registry)


def build_langgraph_demo_components(store: SQLiteStore):
    from .langgraph_runtime import ScriptedModel

    gateway = build_example_gateway(store)
    models = {
        "analyst": ScriptedModel("analyst", "repo.search"),
        "backend": ScriptedModel("backend", "issue.label"),
        "tester": ScriptedModel("tester"),
    }
    prompts = {
        "analyst": "你是影响分析 Agent。所有结论必须引用可追踪证据。",
        "backend": "你是后端开发 Agent。输出必须满足依赖任务的验收标准。",
        "tester": "你是测试验证 Agent。只根据可观察结果给出结论。",
    }
    return gateway, models, prompts
