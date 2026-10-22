"""Record what was frozen before the held-out runs: code commit, checkpoint and profile hashes.

uv run python -m eval.freeze --write     -> eval/results/freeze.json
uv run python -m eval.freeze --check     -> exit 1 if anything frozen has changed since

The held-out scripts are run only while --check passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "results" / "freeze.json"
FROZEN = ["checkpoints/shared-v3/adapter_model.safetensors", "checkpoints/shared-v3/head.pt",
          "checkpoints/shared-v3/adapter_config.json", "checkpoints/identity-v3/adapter_model.safetensors",
          "checkpoints/identity-v3/head.pt", "profiles/L.json", "profiles/S.json", "profiles/U.json",
          "configs/labeling.yaml", "configs/agents.yaml", "data/fixtures/ab/ab_final.jsonl",
          "data/fixtures/e3/e3.jsonl", "data/fixtures/e1/test.jsonl", "data/fixtures/w2t/e2_test.jsonl",
          "data/fixtures/w2t/multihop_test.jsonl"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def snapshot() -> dict:
    return {f: sha(ROOT / f) for f in FROZEN}


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.write:
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout.strip()
        if dirty:
            sys.exit(f"commit first; tracked changes:\n{dirty}")
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        OUT.write_text(json.dumps({"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "code_commit": commit,
                                   "sha256": snapshot()}, indent=1))
        print(f"frozen at {commit[:10]}; wrote {OUT.relative_to(ROOT)}")
        return
    rec = json.loads(OUT.read_text())
    changed = [f for f, h in snapshot().items() if rec["sha256"].get(f) != h]
    if changed:
        sys.exit(f"frozen files changed since {rec['frozen_at']}: {changed}")
    print(f"freeze intact ({rec['frozen_at']}, code {rec['code_commit'][:10]})")


if __name__ == "__main__":
    main()
