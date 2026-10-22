"""Reproduce every reported number, in order. Each step is resumable or cached, so reruns are cheap.

uv run python -m eval.run_all                 # everything (hosted steps need DIGITALOCEAN_INFERENCE_KEY)
uv run python -m eval.run_all --list
uv run python -m eval.run_all --only ladder_dev calibrate
uv run python -m eval.run_all --from train

Held-out steps (ladder_final, ab_final) are listed but skipped unless --final is passed: they run once,
after the backbone, profiles and code are frozen.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

PY = [sys.executable, "-m"]
LABEL_SETS = ["w2t/probe", "probe_pack/probe_dev", "w2t/probe_extra", "w2t/e2_train", "e3/e3_capability_tasks",
              "w2t/multihop_train", "w2t/e2_test", "w2t/multihop_test"]

LADDER = ["eval.ladder", "--shared-checkpoint", "shared-v3", "--identity-checkpoint", "identity-v3"]
FINAL_CONFIGS = ["none", "adapted", "shared", "rules", "qwen_prompted"]

STEPS: list[tuple[str, list[str], bool]] = [  # (name, command, held_out)
    ("prepare", [sys.executable, "data/prepare.py"], False),
    ("dedup", PY + ["eval.dedup"], False),
    ("label_S", PY + ["eval.label_tasks", "--agent", "S", "--sets", *LABEL_SETS], False),
    ("label_L", PY + ["eval.label_tasks", "--agent", "L", "--sets", *LABEL_SETS], False),
    ("overcall", PY + ["eval.overcall"], False),
    ("multistep", PY + ["training.multistep_data"], False),
    ("train", PY + ["training.train_backbone", "--out", "checkpoints/shared-v3", "--epochs", "3",
                    "--soft-disagreement"], False),
    ("train_identity", PY + ["training.train_backbone", "--out", "checkpoints/identity-v3", "--epochs", "3",
                             "--identity-from", "checkpoints/shared-v3"], False),
    ("calibrate", PY + ["eval.calibrate_agents", "--agents", "L", "S", "--backbone", "checkpoints/shared-v3"], False),
    ("latency", PY + ["eval.latency", "--device", "cuda", "--checkpoint", "checkpoints/shared-v3",
                      "--tag", "v3_cuda"], False),
    ("ladder_dev", PY + LADDER + ["--phase", "dev"], False),
    ("ab_dev_S", PY + ["agent.ab", "--agent", "S", "--suite", "ab_dev",
                       "--configs", "none", "adapted", "shared", "rules"], False),
    ("ab_dev_L", PY + ["agent.ab", "--agent", "L", "--suite", "ab_dev",
                       "--configs", "none", "adapted", "shared", "rules"], False),
    ("ladder_final", PY + LADDER + ["--phase", "final"], True),
    ("ab_final_S", PY + ["agent.ab", "--agent", "S", "--suite", "ab_final", "--configs", *FINAL_CONFIGS,
                         "profile:L"], True),
    ("ab_final_L", PY + ["agent.ab", "--agent", "L", "--suite", "ab_final", "--configs", *FINAL_CONFIGS,
                         "profile:S"], True),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--from", dest="start")
    ap.add_argument("--final", action="store_true", help="also run the held-out steps")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    names = [s[0] for s in STEPS]
    if a.list:
        for n, cmd, held in STEPS:
            print(f"{n:14s} {'[held-out] ' if held else ''}{' '.join(cmd[1:])}")
        return
    todo = STEPS
    if a.start:
        todo = STEPS[names.index(a.start):]
    if a.only:
        todo = [s for s in STEPS if s[0] in a.only]
    for n, cmd, held in todo:
        if held and not a.final:
            print(f"-- skip {n} (held-out; pass --final)")
            continue
        print(f"== {n}: {' '.join(cmd[1:])}", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
