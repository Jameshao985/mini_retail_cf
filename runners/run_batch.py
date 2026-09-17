"""Run scripted controls across all counterfactual tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents import BadAgentAlwaysCancel, BadAgentChecksButIgnores, ScriptedGoodAgent

from .common import execute_tasks, load_tasks, persist_results


AGENTS = {
    "scripted": ScriptedGoodAgent,
    "always_cancel": BadAgentAlwaysCancel,
    "checks_but_ignores": BadAgentChecksButIgnores,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=sorted(AGENTS), default="scripted")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tasks-path", type=Path)
    parser.add_argument("--label")
    args = parser.parse_args()
    tasks = load_tasks(args.tasks_path)
    if args.limit is not None:
        tasks = tasks[: args.limit]
    agent = AGENTS[args.agent]()
    episodes = execute_tasks(agent, tasks)
    label = args.label or (args.agent if args.limit is None else f"{args.agent}_limit{args.limit}")
    print(json.dumps(persist_results(label, episodes), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
