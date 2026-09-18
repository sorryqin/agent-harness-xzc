# Resumable Multi-Agent Harness

一个面向代码影响分析、方案设计与测试验证的轻量多 Agent 编排内核。它刻意不把 LangGraph 作为核心依赖，用少量可读代码展示简历中真正关键的工程机制：持久化状态机、DAG 调度、工具副作用治理、调用级 Checkpoint 与分层上下文。

## 架构

```text
                     ┌─────────────────────────────┐
需求 ──> Workflow ──>│ DAG Orchestrator            │
                     │ analyst -> backend -> tester│
                     └──────────────┬──────────────┘
                                    │ role-scoped context
                     ┌──────────────▼──────────────┐
                     │ Context Assembler            │
                     │ constraints / state / trace  │
                     │ summary / artifact index     │
                     └──────────────────────────────┘

Agent tool request ──> permission ──> checkpoint/claim ──> local or MCP
                           │                 │
                           ▼                 ▼
                       approval          replay/audit

All state ───────────────────────────────> SQLite + event log
```

## 已实现能力

- 按 `analyst / backend / tester` 角色拆分 DAG 子任务，依赖满足后才可执行。
- 工作流、任务、审批、工具调用、产物和事件统一持久化，进程重启后再次 `run` 即可续跑。
- 工具统一经过 `ToolGateway`，支持本地函数与 MCP 路由。
- 工具分为只读、幂等写、非幂等写；默认对非幂等写先审批、后执行。
- `session_id + call_id + 参数指纹` 校验调用身份；成功结果直接回放，避免重复副作用。
- 模型每一步决策先写入 `agent_steps` Checkpoint；审批暂停后恢复时不重新采样旧步骤。
- 租约实现原子抢占；非幂等调用结果不确定时拒绝盲目重试。
- 上下文按系统约束、当前任务、依赖输出、产物索引、近期轨迹、历史摘要分层，并按预算裁剪。
- OpenAI / Anthropic 文本适配器、OpenAI function calling 适配器，以及无需 API Key 的确定性 Demo Agent。

## 快速开始

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
agent-harness --db demo.db demo
```

查看持久化状态：

```bash
agent-harness --db demo.db show <workflow_id>
```

审批高风险调用：

```bash
agent-harness --db demo.db approve <approval_id> --actor reviewer
```

## 核心语义

### 工具调用安全顺序

1. 查找工具定义并确定风险等级。
2. 对高风险调用创建/校验审批记录；未批准时绝不触碰 handler 或 MCP。
3. 在 `BEGIN IMMEDIATE` 事务中按 `call_id` 原子抢占。
4. 已成功调用返回持久化结果；参数指纹冲突直接拒绝。
5. 本地或 MCP 调用完成后写入结果与审计事件。
6. 非幂等调用若在执行中失联，标记为 `uncertain`，要求人工对账，而不是重试。

### 是否需要 LangGraph

不需要。Harness 关注的是执行语义和安全边界，LangGraph 是一种图编排实现。这个项目自己实现核心层更适合用于面试展示；若生产团队已经使用 LangGraph，可把它放在外层：节点调用这里的 Agent/ToolGateway，checkpoint 仍以本项目的调用账本为最终事实来源。不要同时维护两套相互竞争的任务状态机。

## 下一步可扩展

- 将单进程轮询替换为 Redis/Postgres 队列与 `SKIP LOCKED` worker。
- 增加真正的模型 tool-use 循环和 JSON Schema 参数校验。
- 接入 OpenTelemetry，将 event log 映射到 trace/span。
- 增加 Git worktree 隔离、patch artifact 和测试报告解析。
- 提供 FastAPI 控制面，用于审批、暂停、恢复和执行图查看。
