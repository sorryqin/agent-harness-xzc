from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from jsonschema import ValidationError, validate as validate_json_schema

from .agents import ToolCallingModel
from .context import ContextAssembler, ContextLayer, estimate_tokens
from .domain import ModelToolCall, ModelTurn, RiskLevel, TaskStatus, WorkflowStatus
from .store import SQLiteStore, dumps
from .tools import ToolGateway, ToolRegistry


class AgentGraphState(TypedDict, total=False):
    workflow_id: str
    task_id: str
    role: str
    messages: list[dict[str, Any]]
    raw_context_layers: list[dict[str, Any]]
    raw_context_tokens: int
    context: dict[str, Any]
    compression_applied: bool
    model_step: int
    model_turn: dict[str, Any]
    validation_error: str | None
    pending_tool: dict[str, Any] | None
    approval: dict[str, Any] | None
    output: dict[str, Any] | None
    memory_candidates: list[dict[str, Any]]
    agent_status: str
    error: str | None


class WorkflowGraphState(AgentGraphState, total=False):
    task_order: list[str]
    deliveries: dict[str, dict[str, Any]]
    workflow_status: str
    workflow_error: str | None
    final_report: dict[str, Any] | None


def _stable_call_id(task_id: str, step: int, name: str, arguments: dict[str, Any]) -> str:
    fingerprint = hashlib.sha256(dumps(arguments).encode()).hexdigest()
    identity = f"{task_id}:{step}:{name}:{fingerprint}".encode()
    return f"call_{hashlib.sha256(identity).hexdigest()[:24]}"


def build_agent_graph(
    *,
    store: SQLiteStore,
    models: dict[str, ToolCallingModel],
    role_prompts: dict[str, str],
    registry: ToolRegistry,
    gateway: ToolGateway,
    context_budget: int = 1800,
    max_steps: int = 12,
    checkpointer: Any = None,
    allowed_tools: dict[str, set[str]] | None = None,
):
    """Build one reusable, checkpointable role-agent subgraph.

    The role changes the model, prompt, and allowed tool set; execution semantics
    remain identical across agents.
    """
    assembler = ContextAssembler(store, budget_tokens=context_budget)

    def role_tool_schemas(role: str) -> list[dict[str, Any]]:
        schemas = registry.schemas()
        if allowed_tools is None:
            return schemas
        allowed = allowed_tools.get(role, set())
        return [schema for schema in schemas if schema["name"] in allowed]

    def retrieve_context(state: AgentGraphState) -> dict[str, Any]:
        task = store.get_task(state["task_id"])
        layers = assembler.build_layers(state["workflow_id"], task)
        serialized = [asdict(layer) for layer in layers]
        return {
            "role": task.role,
            "raw_context_layers": serialized,
            "raw_context_tokens": sum(estimate_tokens(layer.content) for layer in layers),
        }

    def route_context(state: AgentGraphState) -> Literal["compact_context", "assemble_context"]:
        return "compact_context" if state["raw_context_tokens"] > context_budget else "assemble_context"

    def _deserialize_layers(state: AgentGraphState) -> list[ContextLayer]:
        return [ContextLayer(**layer) for layer in state["raw_context_layers"]]

    def compact_context(state: AgentGraphState) -> dict[str, Any]:
        packed = assembler.pack(_deserialize_layers(state))
        return {"context": packed, "compression_applied": True}

    def assemble_context(state: AgentGraphState) -> dict[str, Any]:
        layers = _deserialize_layers(state)
        return {
            "context": {
                "layers": {layer.name: layer.content for layer in layers if layer.content},
                "estimated_tokens": sum(estimate_tokens(layer.content) for layer in layers),
                "omitted": [],
            },
            "compression_applied": False,
        }

    def call_llm(state: AgentGraphState) -> dict[str, Any]:
        task = store.get_task(state["task_id"])
        step = state.get("model_step", 0)
        if step >= max_steps:
            return {
                "agent_status": "failed",
                "error": f"agent exceeded max_steps={max_steps}",
                "model_turn": {"text": "", "tool_calls": [], "assistant_message": {}},
            }
        messages = list(state.get("messages", []))
        if not messages:
            messages.append({
                "role": "user",
                "content": json.dumps({
                    "task": task.instructions,
                    "acceptance_criteria": task.input.get("acceptance_criteria", []),
                    "context": state["context"],
                }, ensure_ascii=False),
            })

        saved = store.get_agent_step(task.id, step)
        if saved is None:
            model = models[task.role]
            turn = model.next_turn(
                system=role_prompts[task.role],
                messages=messages,
                tools=role_tool_schemas(task.role),
            )
            saved = {
                "text": turn.text,
                "assistant_message": turn.assistant_message,
                "tool_calls": [asdict(call) for call in turn.tool_calls],
            }
            store.save_agent_step(task.id, step, saved)

        assistant = saved.get("assistant_message") or {
            "role": "assistant", "content": saved.get("text", "")
        }
        return {
            "messages": [*messages, assistant],
            "model_turn": saved,
            "model_step": step + 1,
            "validation_error": None,
            "pending_tool": None,
        }

    def validate_decision(state: AgentGraphState) -> dict[str, Any]:
        if state.get("agent_status") == "failed":
            return {}
        turn = state["model_turn"]
        calls = turn.get("tool_calls", [])
        error: str | None = None
        if len(calls) > 1:
            error = "only one tool call is allowed per auditable agent step"
        elif calls:
            call = calls[0]
            schemas = {schema["name"]: schema for schema in role_tool_schemas(state["role"])}
            if call.get("name") not in schemas:
                error = f"unknown tool: {call.get('name')}"
            elif not isinstance(call.get("arguments"), dict):
                error = "tool arguments must be an object"
            else:
                try:
                    validate_json_schema(
                        instance=call["arguments"],
                        schema=schemas[call["name"]]["input_schema"],
                    )
                except ValidationError as exc:
                    error = f"tool arguments violate schema: {exc.message}"
        elif not str(turn.get("text", "")).strip():
            error = "model returned neither a tool call nor a deliverable"

        if error:
            correction = {
                "role": "user",
                "content": f"Decision rejected by deterministic validator: {error}. Return a corrected decision.",
            }
            return {
                "validation_error": error,
                "messages": [*state.get("messages", []), correction],
                "pending_tool": None,
            }
        if calls:
            call = calls[0]
            call_id = _stable_call_id(
                state["task_id"], state["model_step"] - 1, call["name"], call["arguments"]
            )
            return {
                "validation_error": None,
                "pending_tool": {**call, "call_id": call_id},
            }
        return {
            "validation_error": None,
            "output": {
                "role": state["role"],
                "text": turn["text"].strip(),
                "steps": state["model_step"],
                "compression_applied": state.get("compression_applied", False),
            },
        }

    def route_decision(state: AgentGraphState) -> Literal["llm", "policy", "verify", "fail"]:
        if state.get("agent_status") == "failed" or state.get("model_step", 0) >= max_steps and state.get("validation_error"):
            return "fail"
        if state.get("validation_error"):
            return "llm"
        if state.get("pending_tool"):
            return "policy"
        return "verify"

    def policy_check(state: AgentGraphState) -> dict[str, Any]:
        request = dict(state["pending_tool"] or {})
        spec = registry.get(request["name"])
        request["risk"] = spec.risk.value
        request["requires_approval"] = spec.risk in gateway.require_approval
        return {"pending_tool": request}

    def route_policy(state: AgentGraphState) -> Literal["approval", "tool"]:
        return "approval" if state["pending_tool"]["requires_approval"] else "tool"

    def approval_node(state: AgentGraphState) -> dict[str, Any]:
        request = state["pending_tool"]
        assert request is not None
        status = gateway.authorization_status(
            session_id=store.session_id(state["workflow_id"]),
            workflow_id=state["workflow_id"],
            task_id=state["task_id"],
            tool_name=request["name"],
            arguments=request["arguments"],
            purpose=request.get("purpose", "agent requested tool call"),
            call_id=request["call_id"],
        )
        if status["status"] == "approved":
            return {"approval": status}
        if status["status"] == "rejected":
            return {"approval": status, "agent_status": "failed", "error": "tool approval rejected"}

        store.wait_task_for_approval(state["task_id"])
        response = interrupt({
            "kind": "tool_approval",
            "approval_id": status["approval_id"],
            "task_id": state["task_id"],
            "tool": request["name"],
            "risk": request["risk"],
            "arguments": request["arguments"],
            "purpose": request.get("purpose", "agent requested tool call"),
        })
        action = response.get("action") if isinstance(response, dict) else response
        approved = action in ("approve", "approved", True)
        store.decide_approval(
            str(status["approval_id"]), approved,
            response.get("actor", "human") if isinstance(response, dict) else "human",
            response.get("note", "") if isinstance(response, dict) else "",
        )
        return {
            "approval": {"status": "approved" if approved else "rejected", **status},
            **({} if approved else {"agent_status": "failed", "error": "tool approval rejected"}),
        }

    def route_approval(state: AgentGraphState) -> Literal["tool", "fail"]:
        return "fail" if state.get("agent_status") == "failed" else "tool"

    def execute_tool(state: AgentGraphState) -> dict[str, Any]:
        request = state["pending_tool"]
        assert request is not None
        result = gateway.execute(
            session_id=store.session_id(state["workflow_id"]),
            workflow_id=state["workflow_id"],
            task_id=state["task_id"],
            tool_name=request["name"],
            arguments=request["arguments"],
            purpose=request.get("purpose", "agent requested tool call"),
            call_id=request["call_id"],
        )
        provider_id = request.get("provider_call_id", request["call_id"])
        return {
            "messages": [*state.get("messages", []), {
                "role": "tool", "tool_call_id": provider_id, "content": dumps(result),
            }],
            "pending_tool": None,
            "approval": None,
        }

    def verify_deliverable(state: AgentGraphState) -> dict[str, Any]:
        output = state.get("output") or {}
        text = str(output.get("text", "")).strip()
        if text:
            return {"agent_status": "verified", "validation_error": None}
        return {
            "validation_error": "deliverable text is empty",
            "messages": [*state.get("messages", []), {
                "role": "user", "content": "Deliverable validation failed: produce a non-empty result.",
            }],
        }

    def route_verification(state: AgentGraphState) -> Literal["extract_memory", "llm", "fail"]:
        if state.get("agent_status") == "verified":
            return "extract_memory"
        return "fail" if state.get("model_step", 0) >= max_steps else "llm"

    def extract_memory(state: AgentGraphState) -> dict[str, Any]:
        output = state["output"] or {}
        content = str(output.get("text", ""))
        candidate = {
            "key": f"task_outcome:{state['task_id']}",
            "content": content[:1000],
            "confidence": 1.0,
            "source_task_id": state["task_id"],
        }
        return {"memory_candidates": [candidate]}

    def commit_memory(state: AgentGraphState) -> dict[str, Any]:
        for candidate in state.get("memory_candidates", []):
            if candidate["confidence"] < 0.8 or not candidate["content"].strip():
                continue
            store.put_memory(
                namespace=f"workflow:{state['workflow_id']}",
                memory_key=candidate["key"],
                content=candidate["content"],
                source_task_id=candidate["source_task_id"],
                confidence=candidate["confidence"],
            )
        return {}

    def finalize(state: AgentGraphState) -> dict[str, Any]:
        output = state["output"] or {}
        store.finish_task(state["task_id"], output)
        return {"agent_status": "succeeded", "error": None}

    def fail(state: AgentGraphState) -> dict[str, Any]:
        error = state.get("error") or state.get("validation_error") or "agent failed"
        task = store.get_task(state["task_id"])
        if task.status != TaskStatus.FAILED:
            store.fail_task(task.id, error, retryable=False)
        return {"agent_status": "failed", "error": error}

    builder = StateGraph(AgentGraphState)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("compact_context", compact_context)
    builder.add_node("assemble_context", assemble_context)
    builder.add_node("llm", call_llm)
    builder.add_node("validate", validate_decision)
    builder.add_node("policy", policy_check)
    builder.add_node("approval", approval_node)
    builder.add_node("tool", execute_tool)
    builder.add_node("verify", verify_deliverable)
    builder.add_node("extract_memory", extract_memory)
    builder.add_node("commit_memory", commit_memory)
    builder.add_node("finalize", finalize)
    builder.add_node("fail", fail)

    builder.add_edge(START, "retrieve_context")
    builder.add_conditional_edges("retrieve_context", route_context)
    builder.add_edge("compact_context", "llm")
    builder.add_edge("assemble_context", "llm")
    builder.add_edge("llm", "validate")
    builder.add_conditional_edges("validate", route_decision)
    builder.add_conditional_edges("policy", route_policy)
    builder.add_conditional_edges("approval", route_approval)
    builder.add_edge("tool", "retrieve_context")
    builder.add_conditional_edges("verify", route_verification)
    builder.add_edge("extract_memory", "commit_memory")
    builder.add_edge("commit_memory", "finalize")
    builder.add_edge("finalize", END)
    builder.add_edge("fail", END)
    return builder.compile(checkpointer=checkpointer, name="role_agent")


def build_workflow_graph(
    *,
    store: SQLiteStore,
    models: dict[str, ToolCallingModel],
    role_prompts: dict[str, str],
    registry: ToolRegistry,
    gateway: ToolGateway,
    checkpointer: Any,
    context_budget: int = 1800,
    allowed_tools: dict[str, set[str]] | None = None,
):
    """Build the parent scheduler graph around the reusable agent subgraph."""
    agent_graph = build_agent_graph(
        store=store,
        models=models,
        role_prompts=role_prompts,
        registry=registry,
        gateway=gateway,
        context_budget=context_budget,
        allowed_tools=allowed_tools,
    )

    def validate_plan(state: WorkflowGraphState) -> dict[str, Any]:
        workflow_id = state["workflow_id"]
        tasks = store.list_tasks(workflow_id)
        ids = {task.id for task in tasks}
        if not tasks:
            return {"workflow_status": "failed", "workflow_error": "workflow has no tasks"}
        for task in tasks:
            missing = set(task.depends_on) - ids
            if missing:
                return {
                    "workflow_status": "failed",
                    "workflow_error": f"task {task.id} has missing dependencies: {sorted(missing)}",
                }
            if task.role not in models or task.role not in role_prompts:
                return {
                    "workflow_status": "failed",
                    "workflow_error": f"no agent configured for role {task.role}",
                }

        indegree = {task.id: len(task.depends_on) for task in tasks}
        dependents: dict[str, list[str]] = {task.id: [] for task in tasks}
        for task in tasks:
            for dependency in task.depends_on:
                dependents[dependency].append(task.id)
        queue = sorted(task_id for task_id, degree in indegree.items() if degree == 0)
        order: list[str] = []
        while queue:
            current = queue.pop(0)
            order.append(current)
            for dependent in dependents[current]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    queue.append(dependent)
                    queue.sort()
        if len(order) != len(tasks):
            return {"workflow_status": "failed", "workflow_error": "task graph contains a cycle"}
        store.set_workflow_status(workflow_id, WorkflowStatus.RUNNING)
        return {
            "task_order": order,
            "deliveries": state.get("deliveries", {}),
            "workflow_status": "running",
            "workflow_error": None,
        }

    def route_plan(state: WorkflowGraphState) -> Literal["schedule", "workflow_fail"]:
        return "workflow_fail" if state.get("workflow_status") == "failed" else "schedule"

    def schedule(state: WorkflowGraphState) -> dict[str, Any]:
        tasks = store.list_tasks(state["workflow_id"])
        if any(task.status == TaskStatus.FAILED for task in tasks):
            return {"workflow_status": "failed", "workflow_error": "one or more tasks failed"}
        ready = store.ready_tasks(state["workflow_id"])
        if ready:
            order = {task_id: index for index, task_id in enumerate(state["task_order"])}
            task = min(ready, key=lambda item: order[item.id])
            if not store.start_task(task.id):
                return {"workflow_status": "running"}
            return {
                "task_id": task.id,
                "role": task.role,
                "messages": [],
                "raw_context_layers": [],
                "raw_context_tokens": 0,
                "context": {},
                "compression_applied": False,
                "model_step": 0,
                "model_turn": {},
                "validation_error": None,
                "pending_tool": None,
                "approval": None,
                "output": None,
                "memory_candidates": [],
                "agent_status": "running",
                "error": None,
                "workflow_status": "dispatch",
            }
        if tasks and all(task.status == TaskStatus.SUCCEEDED for task in tasks):
            return {"workflow_status": "complete"}
        return {
            "workflow_status": "failed",
            "workflow_error": "no runnable task; dependency graph is blocked",
        }

    def route_schedule(state: WorkflowGraphState) -> Literal["agent", "complete", "workflow_fail"]:
        return {
            "dispatch": "agent",
            "complete": "complete",
            "failed": "workflow_fail",
        }.get(state.get("workflow_status", "failed"), "workflow_fail")

    def collect_delivery(state: WorkflowGraphState) -> dict[str, Any]:
        deliveries = dict(state.get("deliveries", {}))
        deliveries[state["task_id"]] = {
            "status": state.get("agent_status"),
            "role": state.get("role"),
            "output": state.get("output"),
            "error": state.get("error"),
        }
        return {"deliveries": deliveries}

    def route_delivery(state: WorkflowGraphState) -> Literal["schedule", "workflow_fail"]:
        return "workflow_fail" if state.get("agent_status") == "failed" else "schedule"

    def complete(state: WorkflowGraphState) -> dict[str, Any]:
        store.set_workflow_status(state["workflow_id"], WorkflowStatus.SUCCEEDED)
        report = {
            "workflow_id": state["workflow_id"],
            "status": "succeeded",
            "task_order": state["task_order"],
            "deliveries": state.get("deliveries", {}),
        }
        return {"workflow_status": "succeeded", "final_report": report}

    def workflow_fail(state: WorkflowGraphState) -> dict[str, Any]:
        store.set_workflow_status(state["workflow_id"], WorkflowStatus.FAILED)
        return {"workflow_status": "failed", "final_report": {
            "workflow_id": state["workflow_id"],
            "status": "failed",
            "error": state.get("workflow_error") or state.get("error"),
            "deliveries": state.get("deliveries", {}),
        }}

    builder = StateGraph(WorkflowGraphState)
    builder.add_node("validate_plan", validate_plan)
    builder.add_node("schedule", schedule)
    builder.add_node("agent", agent_graph)
    builder.add_node("collect_delivery", collect_delivery)
    builder.add_node("complete", complete)
    builder.add_node("workflow_fail", workflow_fail)
    builder.add_edge(START, "validate_plan")
    builder.add_conditional_edges("validate_plan", route_plan)
    builder.add_conditional_edges("schedule", route_schedule)
    builder.add_edge("agent", "collect_delivery")
    builder.add_conditional_edges("collect_delivery", route_delivery)
    builder.add_edge("complete", END)
    builder.add_edge("workflow_fail", END)
    return builder.compile(checkpointer=checkpointer, name="multi_agent_workflow")


class ScriptedModel:
    """Deterministic model used to exercise the real graph without API credentials."""

    def __init__(self, role: str, tool_name: str | None = None) -> None:
        self.role = role
        self.tool_name = tool_name

    def next_turn(
        self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
    ) -> ModelTurn:
        has_tool_result = any(message.get("role") == "tool" for message in messages)
        if self.tool_name and not has_tool_result:
            arguments = {"query": "authentication call sites"} if self.tool_name == "repo.search" else {"issue": 42, "labels": ["agent-reviewed"]}
            provider_id = f"provider_{self.role}_1"
            return ModelTurn(
                text="",
                tool_calls=[ModelToolCall(provider_id, self.tool_name, arguments, f"{self.role} evidence")],
                assistant_message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": provider_id,
                        "type": "function",
                        "function": {"name": self.tool_name, "arguments": dumps(arguments)},
                    }],
                },
            )
        text = {
            "analyst": "影响范围已确认：认证入口、会话校验和相关测试需要同步更新。",
            "backend": "实现方案已完成：隔离变更边界，保留兼容层并增加审计事件。",
            "tester": "验证完成：依赖路径、异常恢复和回归用例均已覆盖。",
        }.get(self.role, f"{self.role} task completed")
        return ModelTurn(text=text, assistant_message={"role": "assistant", "content": text})
