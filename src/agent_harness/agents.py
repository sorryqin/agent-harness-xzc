from __future__ import annotations

import json
import os
import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .domain import AgentResult, ModelToolCall, ModelTurn, Task
from .store import SQLiteStore, dumps
from .tools import ToolGateway, ToolRegistry


class Agent(Protocol):
    def run(self, task: Task, context: dict[str, Any]) -> AgentResult: ...


class ModelClient(Protocol):
    def complete(self, *, system: str, prompt: str) -> str: ...


class ToolCallingModel(Protocol):
    def next_turn(
        self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
    ) -> ModelTurn: ...


@dataclass(slots=True)
class LLMAgent:
    role_prompt: str
    client: ModelClient

    def run(self, task: Task, context: dict[str, Any]) -> AgentResult:
        answer = self.client.complete(
            system=self.role_prompt,
            prompt=json.dumps({"task": task.instructions, "context": context}, ensure_ascii=False),
        )
        return AgentResult(output={"text": answer, "role": task.role})


class OpenAIClient:
    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model or os.getenv("AGENT_MODEL", "gpt-4.1-mini")

    def complete(self, *, system: str, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            instructions=system,
            input=prompt,
        )
        return response.output_text


class AnthropicClient:
    def __init__(self, model: str | None = None) -> None:
        from anthropic import Anthropic
        self.client = Anthropic()
        self.model = model or os.getenv("AGENT_MODEL", "claude-sonnet-4-20250514")

    def complete(self, *, system: str, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model, system=system, max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAIToolCallingModel:
    """OpenAI Chat Completions adapter for the provider-neutral agent loop."""

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model or os.getenv("AGENT_MODEL", "gpt-4.1-mini")

    def next_turn(
        self, *, system: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]],
    ) -> ModelTurn:
        openai_tools = [{
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        } for tool in tools]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, *messages],
            tools=openai_tools or None,
        )
        message = response.choices[0].message
        assistant_message = message.model_dump(exclude_none=True)
        calls = [ModelToolCall(
            provider_call_id=call.id,
            name=call.function.name,
            arguments=json.loads(call.function.arguments),
        ) for call in (message.tool_calls or [])]
        return ModelTurn(
            text=message.content or "", tool_calls=calls, assistant_message=assistant_message
        )


@dataclass(slots=True)
class ToolCallingAgent:
    """Checkpointed model/tool loop shared by all role agents."""

    role_prompt: str
    model: ToolCallingModel
    gateway: ToolGateway
    registry: ToolRegistry
    store: SQLiteStore
    max_steps: int = 12

    def run(self, task: Task, context: dict[str, Any]) -> AgentResult:
        messages: list[dict[str, Any]] = [{
            "role": "user",
            "content": json.dumps({"task": task.instructions, "context": context}, ensure_ascii=False),
        }]
        session_id = self.store.session_id(task.workflow_id)

        for step in range(self.max_steps):
            saved = self.store.get_agent_step(task.id, step)
            if saved is None:
                turn = self.model.next_turn(
                    system=self.role_prompt,
                    messages=messages,
                    tools=self.registry.schemas(),
                )
                saved = {
                    "text": turn.text,
                    "assistant_message": turn.assistant_message,
                    "tool_calls": [asdict(call) for call in turn.tool_calls],
                }
                self.store.save_agent_step(task.id, step, saved)
            turn = ModelTurn(
                text=saved["text"],
                assistant_message=saved["assistant_message"],
                tool_calls=[ModelToolCall(**call) for call in saved["tool_calls"]],
            )
            messages.append(turn.assistant_message or {"role": "assistant", "content": turn.text})
            if not turn.tool_calls:
                return AgentResult(output={"text": turn.text, "role": task.role, "steps": step + 1})

            for index, call in enumerate(turn.tool_calls):
                fingerprint = self.gateway.fingerprint(call.arguments)
                call_id = self._stable_call_id(task.id, step, index, call.name, fingerprint)
                result = self.gateway.execute(
                    session_id=session_id,
                    workflow_id=task.workflow_id,
                    task_id=task.id,
                    tool_name=call.name,
                    arguments=call.arguments,
                    purpose=call.purpose,
                    call_id=call_id,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.provider_call_id,
                    "content": dumps(result),
                })
        raise RuntimeError(f"agent exceeded max_steps={self.max_steps}")

    @staticmethod
    def _stable_call_id(task_id: str, step: int, index: int, name: str, fingerprint: str) -> str:
        identity = f"{task_id}:{step}:{index}:{name}:{fingerprint}".encode()
        return f"call_{hashlib.sha256(identity).hexdigest()[:24]}"


class DemoAgent:
    """Deterministic role agent used by the local demo and tests."""

    def run(self, task: Task, context: dict[str, Any]) -> AgentResult:
        dependency_count = len(task.depends_on)
        return AgentResult(output={
            "role": task.role,
            "decision": f"Completed: {task.title}",
            "dependency_count": dependency_count,
            "context_tokens": context["estimated_tokens"],
        })
