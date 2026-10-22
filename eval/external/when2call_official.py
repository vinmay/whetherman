"""When2Call official MCQ evaluation (NVIDIA/When2Call), self-run on frozen routers. Held-out: run once.

uv run python -m eval.external.when2call_official --routers shared untuned qwen_prompted

The official MCQ evaluation scores a language model's log-likelihood of each answer option with
LM-Eval-Harness. Whetherman's routers are classifiers, not answer generators, so each router's predicted
class is mapped to the matching option (ACT->tool_call, ANSWER->direct, ASK->request_for_info,
DECLINE->cannot_answer) and written as LM-Eval-Harness samples ({"doc", "macro_f1": [gold, pred], "acc"}).
Scoring then uses the official code: macro-F1 exactly as LM-Eval-Harness aggregates it (sklearn
f1_score, average="macro"), accuracy, and NVIDIA's `additional_metrics.py` (confusion matrix and tool
hallucination rate), imported from the repository at the pinned commit. acc_norm (length-normalized
log-likelihood) has no classifier equivalent and is not reported.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from eval import ladder

ROOT = Path(__file__).resolve().parents[2]
REPO = "https://github.com/NVIDIA/When2Call.git"
COMMIT = "ecc8d42388e91ab37e7e737d48e16e8ecea3d1dc"
SRC = ROOT / "data" / "raw" / "external" / "When2Call"
OUT = ROOT / "eval" / "external" / "when2call"
TO_OPTION = {"ACT": "tool_call", "ANSWER": "direct", "ASK": "request_for_info", "DECLINE": "cannot_answer"}


def fetch() -> None:
    if not (SRC / ".git").exists():
        SRC.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", REPO, str(SRC)], check=True)
    subprocess.run(["git", "-C", str(SRC), "checkout", "-q", COMMIT], check=True)


def decision_points(items: list[dict]) -> list[dict]:
    sys.path.insert(0, str(ROOT / "data"))
    from prepare import convert_schema  # same BFCL-style -> OpenAI conversion as E1

    return [{"id": r["uuid"], "messages": [{"role": "user", "content": r["question"]}],
             "tools": [convert_schema(t) for t in r["tools"]]} for r in items]


def confusion(official, path: Path, golds: list[int], preds: list[int], names: list[str]) -> dict:
    """Official confusion matrix. It infers the label set from the data and fails when a class never
    occurs in gold or predictions ("direct" is never gold), so fall back to the same sklearn matrix with
    the four labels fixed, and say so."""
    try:
        return {"confusion_matrix": official.calculate_confusion_matrix(str(path), print_matrix=False)}
    except ValueError:
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(golds, preds, labels=list(range(len(names))))
        return {"confusion_matrix": {f"true:{g}": {f"pred:{p}": int(cm[i][j]) for j, p in enumerate(names)}
                                     for i, g in enumerate(names)},
                "confusion_matrix_note": "official function needs every class present; same matrix, labels fixed"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routers", nargs="+", default=["shared", "untuned", "qwen_prompted"])
    ap.add_argument("--shared-checkpoint", default="shared-v3")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--mcq-file", default=None, help="smoke test only (non-held-out file); nothing is saved")
    a = ap.parse_args()
    fetch()
    from sklearn.metrics import f1_score

    sys.path.insert(0, str(SRC / "evaluation" / "mcq" / "lm_eval_harness" / "when2call"))
    import additional_metrics  # NVIDIA's official extra metrics

    smoke = a.mcq_file is not None
    mcq = Path(a.mcq_file) if smoke else SRC / "data" / "test" / "when2call_test_mcq.jsonl"
    items = [json.loads(x) for x in mcq.open()]
    pts = decision_points(items)
    ladder.SHARED = a.shared_checkpoint
    (OUT / "smoke" if smoke else OUT).mkdir(parents=True, exist_ok=True)
    summary: dict = {"repo": REPO, "commit": COMMIT, "n": len(items), "scoring": "classifier class -> MCQ option",
                     "shared_checkpoint": a.shared_checkpoint, "routers": {}}
    cache: dict = {}
    for rn in a.routers:
        set_name = "smoke_mcq" if smoke else "when2call_mcq"
        raw = ladder.raw_outputs(rn, set_name, pts, a.device, cache)
        samples = []
        for r in items:
            names = list(r["answers"])
            logits = raw[r["uuid"]]
            pred_label = ["ACT", "ANSWER", "ASK", "DECLINE"][max(range(4), key=lambda i: logits[i])]
            gold, pred = names.index(r["correct_answer"]), names.index(TO_OPTION[pred_label])
            doc = {"uuid": r["uuid"], "source": r["source"], "correct_answer": r["correct_answer"],
                   "answers": {k: "" for k in names}, "tools": [json.loads(t)["name"] for t in r["tools"]]}
            samples.append({"doc": doc, "macro_f1": [gold, pred], "acc": float(gold == pred),
                            "router_label": pred_label})
        tag = f"_{a.shared_checkpoint}" if rn == "shared" else ""
        path = (OUT / "smoke" if smoke else OUT) / f"{rn}{tag}.samples.jsonl"
        path.write_text("".join(json.dumps(s) + "\n" for s in samples))
        golds = [s["macro_f1"][0] for s in samples]
        preds = [s["macro_f1"][1] for s in samples]
        summary["routers"][rn] = {
            "macro_f1": float(f1_score(golds, preds, average="macro")),
            "acc": sum(s["acc"] for s in samples) / len(samples),
            "hallucination_rate": additional_metrics.calculate_hallucination_rate(str(path), print_rate=False),
            **confusion(additional_metrics, path, golds, preds, list(items[0]["answers"])),
            "samples": str(path.relative_to(ROOT)),
        }
        m = summary["routers"][rn]
        print(f"{rn:14s} macro-F1 {m['macro_f1']:.3f} acc {m['acc']:.3f} hallucination {m['hallucination_rate']:.3f}")
    if not smoke:
        (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
