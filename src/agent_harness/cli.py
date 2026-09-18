from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .demo import build_demo, build_langgraph_demo_components
from .langgraph_runtime import build_workflow_graph
from .store import SQLiteStore


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Resumable multi-agent harness")
    parser.add_argument("--db", default="harness.db")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="Create and run a deterministic demo workflow")
    langgraph_demo = sub.add_parser("langgraph-demo", help="Run the two-level LangGraph workflow")
    langgraph_demo.add_argument("--checkpoint-db")
    resume = sub.add_parser("langgraph-resume", help="Resume a LangGraph human interrupt")
    resume.add_argument("workflow_id")
    resume.add_argument("--action", choices=("approve", "reject"), required=True)
    resume.add_argument("--actor", default="human")
    resume.add_argument("--note", default="")
    resume.add_argument("--checkpoint-db")
    show = sub.add_parser("show", help="Show durable workflow state")
    show.add_argument("workflow_id")
    approve = sub.add_parser("approve", help="Approve a pending tool call")
    approve.add_argument("approval_id")
    approve.add_argument("--actor", default="human")
    args = parser.parse_args()

    if args.command == "demo":
        store, workflow_id, orchestrator = build_demo(Path(args.db))
        status = orchestrator.run(workflow_id)
        print(json.dumps({
            "workflow_id": workflow_id, "status": status,
            "tasks": [task.__dict__ if hasattr(task, "__dict__") else {
                "id": task.id, "role": task.role, "status": task.status, "output": task.output,
            } for task in store.list_tasks(workflow_id)],
        }, ensure_ascii=False, indent=2, default=str))
    elif args.command == "langgraph-demo":
        store, workflow_id, _ = build_demo(Path(args.db))
        gateway, models, prompts = build_langgraph_demo_components(store)
        checkpoint_path = args.checkpoint_db or f"{args.db}.checkpoints"
        connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
        try:
            graph = build_workflow_graph(
                store=store, models=models, role_prompts=prompts,
                registry=gateway.registry, gateway=gateway,
                checkpointer=SqliteSaver(connection),
                allowed_tools={"analyst": {"repo.search"}, "backend": {"issue.label"}, "tester": set()},
            )
            result = graph.invoke(
                {"workflow_id": workflow_id},
                {"configurable": {"thread_id": workflow_id}},
            )
            print(json.dumps({
                "workflow_id": workflow_id,
                "checkpoint_db": checkpoint_path,
                "status": result.get("workflow_status"),
                "report": result.get("final_report"),
                "interrupts": [item.value for item in result.get("__interrupt__", ())],
            }, ensure_ascii=False, indent=2, default=str))
        finally:
            connection.close()
    elif args.command == "langgraph-resume":
        store = SQLiteStore(args.db)
        gateway, models, prompts = build_langgraph_demo_components(store)
        checkpoint_path = args.checkpoint_db or f"{args.db}.checkpoints"
        connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
        try:
            graph = build_workflow_graph(
                store=store, models=models, role_prompts=prompts,
                registry=gateway.registry, gateway=gateway,
                checkpointer=SqliteSaver(connection),
                allowed_tools={"analyst": {"repo.search"}, "backend": {"issue.label"}, "tester": set()},
            )
            result = graph.invoke(
                Command(resume={"action": args.action, "actor": args.actor, "note": args.note}),
                {"configurable": {"thread_id": args.workflow_id}},
            )
            print(json.dumps({
                "workflow_id": args.workflow_id,
                "status": result.get("workflow_status"),
                "report": result.get("final_report"),
                "interrupts": [item.value for item in result.get("__interrupt__", ())],
            }, ensure_ascii=False, indent=2, default=str))
        finally:
            connection.close()
    elif args.command == "show":
        store = SQLiteStore(args.db)
        print(json.dumps({
            "workflow": store.get_workflow(args.workflow_id),
            "tasks": [{"id": t.id, "title": t.title, "status": t.status, "output": t.output, "error": t.error}
                      for t in store.list_tasks(args.workflow_id)],
            "approvals": store.list_approvals(args.workflow_id),
            "artifacts": store.list_artifacts(args.workflow_id),
            "events": store.recent_events(args.workflow_id),
        }, ensure_ascii=False, indent=2, default=str))
    elif args.command == "approve":
        SQLiteStore(args.db).decide_approval(args.approval_id, True, args.actor)
        print(f"approved {args.approval_id}")


if __name__ == "__main__":
    main()
