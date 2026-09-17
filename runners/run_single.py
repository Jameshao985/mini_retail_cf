"""Inspect one task trajectory with a scripted control agent."""

from __future__ import annotations

import argparse
import json

from agents import BadAgentAlwaysCancel, BadAgentChecksButIgnores, ScriptedGoodAgent

from .common import execute_tasks, load_tasks


AGENTS = {
    "scripted": ScriptedGoodAgent,
    "always_cancel": BadAgentAlwaysCancel,
    "checks_but_ignores": BadAgentChecksButIgnores,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id")
    parser.add_argument("--agent", choices=sorted(AGENTS), default="scripted")
    args = parser.parse_args()
    task = next((item for item in load_tasks() if item["task_id"] == args.task_id), None)
    if task is None:
        raise SystemExit(f"Unknown task_id: {args.task_id}")
    print(json.dumps(execute_tasks(AGENTS[args.agent](), [task])[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

