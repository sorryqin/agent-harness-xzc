# Resumable Multi-Agent Harness

一个面向代码影响分析、方案设计与测试验证的多 Agent 编排内核。项目同时提供自研轻量 Orchestrator 和 LangGraph 两层执行图，底层共享同一套工具安全、业务状态与审计机制。

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

LangGraph 版本采用两层结构：

```text
Workflow Graph
  validate_plan -> schedule -> role_agent subgraph -> collect -> schedule

Role Agent Subgraph
  retrieve_context -> compact/assemble -> llm -> validate
      -> policy -> approval(interrupt) -> tool -> loop
      -> verify -> extract_memory -> commit_memory -> finalize
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
- LangGraph 父图、可复用角色子图、SQLite Checkpointer 和 Human-in-the-loop interrupt。
- 每个角色独立工具白名单；LLM 输出与 ToolGateway 边界均执行 JSON Schema 参数校验。

## 快速开始

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
agent-harness --db demo.db demo
agent-harness --db graph-demo.db langgraph-demo
```

LangGraph Demo 会使用独立的 `graph-demo.db.checkpoints` 保存图状态；业务状态仍保存在
`graph-demo.db`。两者分别承担执行恢复和业务事实记录。

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

### LangGraph 与 Harness 的边界

LangGraph 负责节点编排、图状态 Checkpoint、条件路由和人工中断。Harness Store 负责任务、审批、产物、长期记忆和工具副作用账本。工具节点只能调用 `ToolGateway`，不能直接访问本地或 MCP handler。详细设计见 [`docs/langgraph-design.md`](docs/langgraph-design.md)。

当前 SQLite 版本为了保证演示语义清晰，父图按 DAG 就绪顺序串行调度；生产环境切换到 Postgres 后，可将同一批无依赖冲突的任务通过 `Send` 扇出，并使用独立子图 namespace 和确定性 reducer 汇总结果。

## 下一步可扩展

- 将单进程轮询替换为 Redis/Postgres 队列与 `SKIP LOCKED` worker。
- 增加真正的模型 tool-use 循环和 JSON Schema 参数校验。
- 接入 OpenTelemetry，将 event log 映射到 trace/span。
- 增加 Git worktree 隔离、patch artifact 和测试报告解析。
- 提供 FastAPI 控制面，用于审批、暂停、恢复和执行图查看。
