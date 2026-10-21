"""Fit the experiment profiles (profiles/<agent>.json) through the same `calibrate` path users run.

uv run python -m eval.calibrate_agents --agents L S [--backbone checkpoints/shared-v1]

The probe labels collected by eval.label_tasks (data/fixtures/labels/<agent>/{probe,probe_dev}.jsonl)
are seeded into calibrate's resumable label file, so no agent episode is run twice. Tasks missing a
label are labeled by calibrate itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from whetherman.agents import agent_config
from whetherman.calibrate.probe_pack import load_probe_pack
from whetherman.calibrate.run import calibrate
from whetherman.labeling import infra_error
from whetherman.providers.openai_compat import PROVIDERS

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data" / "fixtures" / "labels"
SEEDED = ROOT / "profiles" / "labels"


def seed_labels(agent: str) -> int:
    ids = {t.id for t in load_probe_pack().tasks}
    rows = [json.loads(x) for f in ("probe.jsonl", "probe_dev.jsonl") if (LABELS / agent / f).exists()
            for x in (LABELS / agent / f).open()]
    rows = [r for r in rows if r["task_id"] in ids and not infra_error(r)]
    SEEDED.mkdir(parents=True, exist_ok=True)
    (SEEDED / f"{agent}.probe.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", default=["L", "S"])
    ap.add_argument("--backbone", default="checkpoints/shared-v1")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    for ag in a.agents:
        cfg = agent_config(ag)
        n = seed_labels(ag)
        print(f"[{ag}] seeded {n} probe labels")
        calibrate(PROVIDERS[cfg["provider"]].base_url, cfg["model"], name=ag, backbone=a.backbone, yes=True,
                  parallel=cfg.get("parallel", 1), labels_dir=str(SEEDED), device=a.device)


if __name__ == "__main__":
    main()
