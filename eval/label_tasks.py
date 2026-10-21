"""Label task sets per agent with the shared label_necessity (resumable).

uv run python -m eval.label_tasks --agent S --sets w2t/e2_train w2t/probe w2t/e2_test
Writes data/fixtures/labels/<agent>/<set-name>.jsonl (one NecessityLabel per task).
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval import spend
from whetherman.agents import agent_config, make_client
from whetherman.labeling import label_necessity, load_config
from whetherman.tasks import load_tasks

FIX = Path(__file__).resolve().parents[1] / "data" / "fixtures"


INFRA = ("HTTPStatusError", "ReadTimeout", "ConnectTimeout", "ConnectError", "RemoteProtocolError",
         "WriteTimeout", "PoolTimeout", "ReadError")


def infra_error(label: dict) -> bool:
    """True if any episode failed for transport/provider reasons rather than agent behavior."""
    return any((ep.get("error") or "").startswith(INFRA) for ep in label.get("episodes", []))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--est-usd-per-task", type=float, default=0.002)
    a = ap.parse_args()
    cfg = load_config()
    acfg = agent_config(a.agent)
    hosted = acfg["provider"] != "ollama"
    for s in a.sets:
        tasks = load_tasks(FIX / f"{s}.jsonl")[: a.limit]
        out = FIX / "labels" / a.agent / f"{Path(s).name}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        rows = [json.loads(x) for x in out.open()] if out.exists() else []
        bad = [r for r in rows if infra_error(r)]
        if bad:  # an infrastructure failure is not an agent failure: drop and relabel
            print(f"  dropping {len(bad)} labels with infrastructure errors", flush=True)
            rows = [r for r in rows if not infra_error(r)]
            out.write_text("".join(json.dumps(r) + "\n" for r in rows))
        done = {r["task_id"] for r in rows}
        todo = [t for t in tasks if t.id not in done]
        if hosted:
            spend.check_cap("hosted", len(todo) * a.est_usd_per_task)
        print(f"[{a.agent}] {s}: {len(todo)} to label ({len(done)} done)", flush=True)
        t0, cost = time.time(), 0.0

        def one(t):
            client = make_client(a.agent)
            try:
                return label_necessity(t, client, cfg, keep_episodes=True)
            finally:
                client.close()

        n = failed = 0
        try:
            with ThreadPoolExecutor(acfg.get("parallel", 1)) as ex, out.open("a") as f:
                for n, lab in enumerate(ex.map(one, todo), 1):
                    cost += lab.cost_usd
                    d = lab.to_dict()
                    if infra_error(d):
                        failed += 1
                        continue
                    f.write(json.dumps(d) + "\n")
                    f.flush()
                    if n % 10 == 0 or n == len(todo):
                        print(f"  {n}/{len(todo)} {time.time() - t0:.0f}s ${cost:.4f}", flush=True)
        finally:  # record partial spend too, so an interrupted run is still accounted for
            if hosted and cost:
                spend.record("hosted", f"label {a.agent} {s}", cost, tasks=n)
        if failed:
            print(f"  {failed} tasks hit infrastructure errors and were not written; rerun to retry", flush=True)


if __name__ == "__main__":
    main()
