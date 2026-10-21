"""Over-calling premise check: how often an agent calls a tool on tasks it solves without one.

uv run python -m eval.overcall --agents L S --sets probe e2_train  -> eval/results/overcall.json
Only ANSWER-labeled tasks count (the agent succeeded on every no-tool trial). Counts are trials and
tasks, with Wilson 95% intervals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from whetherman.labeling import wilson

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "fixtures" / "labels"


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["label"] == "ANSWER"]
    trials = sum(r["tool_trials"] for r in ans)
    called = sum(r["calls_tool_when_offered"] for r in ans)
    tasks_called = sum(r["calls_tool_when_offered"] > 0 for r in ans)
    return {
        "n_tasks": len(rows),
        "labels": {k: sum(r["label"] == k for r in rows) for k in ("ACT", "ANSWER", "ambiguous")},
        "answer_tasks": len(ans),
        "with_tool_trials_on_answer": trials,
        "trials_calling_tool": called,
        "trial_rate": called / trials if trials else None,
        "trial_rate_ci": list(wilson(called, trials)) if trials else None,
        "answer_tasks_with_any_call": tasks_called,
        "task_rate": tasks_called / len(ans) if ans else None,
        "task_rate_ci": list(wilson(tasks_called, len(ans))) if ans else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", default=["L", "S"])
    ap.add_argument("--sets", nargs="+", default=["probe", "e2_train"])
    a = ap.parse_args()
    res: dict = {}
    for ag in a.agents:
        for s in a.sets:
            p = LABELS / ag / f"{s}.jsonl"
            if p.exists():
                rows = [json.loads(x) for x in p.open()]
                src = ROOT / "data" / "fixtures" / "w2t" / f"{s}.jsonl"
                if src.exists():  # drop labels for tasks since removed from the set (dedup)
                    keep = {json.loads(x)["id"] for x in src.open()}
                    rows = [r for r in rows if r["task_id"] in keep]
                res[f"{ag}/{s}"] = summarize(rows)
    out = ROOT / "eval" / "results" / "overcall.json"
    out.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f"{k}: labels {v['labels']}; tool called on {v['trials_calling_tool']}/{v['with_tool_trials_on_answer']} "
              f"ANSWER trials, {v['answer_tasks_with_any_call']}/{v['answer_tasks']} ANSWER tasks")


if __name__ == "__main__":
    main()
