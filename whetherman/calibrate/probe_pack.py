"""Probe packs: tasks + oracles + executable tools/simulators that `calibrate` needs.

Schema (pack.json): name, version, task_files (JSONL paths relative to the pack), fixtures.
Each task: id, messages, tools (registered tool names), oracle, optional context/appropriate_tool.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from whetherman.tasks import Task
from whetherman.tools.registry import REGISTRY, load_all

DEFAULT_PACK = Path(__file__).resolve().parents[2] / "data" / "fixtures" / "probe_pack"
ORACLE_TYPES = {"answer", "contains", "state", "question", "decline", "all"}


class ProbePackError(ValueError):
    pass


@dataclass
class ProbePack:
    root: Path
    manifest: dict
    tasks: list[Task]
    sha256: str


def _hash_tree(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        files = sorted(x for x in p.rglob("*") if x.is_file() and "__pycache__" not in x.parts) if p.is_dir() else [p]
        for f in files:
            h.update(str(f.name).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def _validate_oracle(o: dict | None, tid: str) -> None:
    if not isinstance(o, dict) or o.get("type") not in ORACLE_TYPES:
        raise ProbePackError(f"task {tid}: missing or unknown oracle {o!r}")
    if o["type"] == "answer" and "gold" not in o:
        raise ProbePackError(f"task {tid}: answer oracle without gold")
    if o["type"] == "all":
        for sub in o.get("of", []):
            _validate_oracle(sub, tid)


def load_probe_pack(path: str | Path | None = None) -> ProbePack:
    """Load and validate. Fails loudly before any agent call is made."""
    load_all()
    root = Path(path or DEFAULT_PACK).resolve()
    mf = root / "pack.json"
    if not mf.exists():
        raise ProbePackError(f"no pack.json in {root}")
    manifest = json.loads(mf.read_text())
    tasks: list[Task] = []
    files = []
    for rel in manifest.get("task_files", []):
        f = (root / rel).resolve()
        if not f.exists():
            raise ProbePackError(f"task file not found: {rel}")
        files.append(f)
        for i, line in enumerate(f.read_text().splitlines()):
            if not line.strip():
                continue
            d = json.loads(line)
            tid = d.get("id") or f"{rel}:{i}"
            _validate_oracle(d.get("oracle"), tid)
            if not d.get("messages"):
                raise ProbePackError(f"task {tid}: no messages")
            for name in d.get("tools", []):
                if name not in REGISTRY:
                    raise ProbePackError(f"task {tid}: tool {name!r} has no implementation or simulator")
            d.setdefault("context", {})
            d["context"].setdefault("pack_root", str(root))
            tasks.append(Task.from_dict(d))
    if not tasks:
        raise ProbePackError("probe pack has no tasks")
    if len({t.id for t in tasks}) != len(tasks):
        raise ProbePackError("duplicate task ids in probe pack")
    sha = _hash_tree(files + [root / x for x in manifest.get("fixtures", [])] + [mf])
    return ProbePack(root=root, manifest=manifest, tasks=tasks, sha256=sha)
