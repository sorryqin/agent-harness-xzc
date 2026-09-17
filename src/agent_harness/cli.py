from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .demo import build_demo
from .store import SQLiteStore


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Resumable multi-agent harness")
    parser.add_argument("--db", default="harness.db")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="Create and run a deterministic demo workflow")
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
