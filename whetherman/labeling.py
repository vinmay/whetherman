"""The one shared tool-necessity labeling function: `label_necessity(task, agent)`.

Used both to build E2 labels and by `whetherman calibrate`. Never fork this logic.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from whetherman.episode import Agent, Episode, run_episode
from whetherman.tasks import Task

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "labeling.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    return yaml.safe_load(Path(path or CONFIG_PATH).read_text())


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class NecessityLabel:
    task_id: str
    label: str  # ACT | ANSWER | ambiguous
    stage: str  # screen | extended
    notool_success: int
    notool_trials: int
    tool_success: int
    tool_trials: int
    notool_ci: tuple[float, float]
    tool_ci: tuple[float, float]
    tool_calls_with_tools: list[int] = field(default_factory=list)  # calls per with-tool trial
    calls_tool_when_offered: int = 0  # with-tool trials where the agent called >= 1 tool
    cost_usd: float = 0.0
    wall_ms: float = 0.0
    episodes: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


RunFn = Callable[..., Episode]


def _decide(nt: int, nn: int, wt: int, wn: int, cfg: dict[str, Any], stage: str) -> str | None:
    if stage == "screen":
        if nt >= cfg["answer_min_success_screen"]:
            return "ANSWER"
        if nt <= cfg["act_max_notool_screen"] and wt >= cfg["act_min_tool_screen"]:
            return "ACT"
        return None
    if nt >= cfg["answer_min_success_full"]:
        return "ANSWER"
    if nt <= cfg["act_max_notool_full"] and wt >= cfg["act_min_tool_full"]:
        return "ACT"
    return "ambiguous"


def label_necessity(task: Task, agent: Agent, config: dict[str, Any] | None = None,
                    run_fn: RunFn = run_episode, keep_episodes: bool = False) -> NecessityLabel:
    """Screen 3+3 trials, extend mixed items to 5+5, then apply the committed rules."""
    cfg = config or load_config()
    no_eps: list[Episode] = []
    w_eps: list[Episode] = []

    def trials(n: int) -> None:
        for _ in range(n):
            i = len(no_eps)
            kw = {"temperature": cfg["temperature"], "seed": cfg["seed_base"] + i,
                  "max_steps": cfg["max_steps"]}
            no_eps.append(run_fn(agent, task, with_tools=False, **kw))
            w_eps.append(run_fn(agent, task, with_tools=True, **kw))

    trials(cfg["screening_trials"])
    succ = lambda eps: sum(e.success for e in eps)  # noqa: E731
    label = _decide(succ(no_eps), len(no_eps), succ(w_eps), len(w_eps), cfg, "screen")
    stage = "screen"
    if label is None:
        trials(cfg["extra_trials"])
        stage = "extended"
        label = _decide(succ(no_eps), len(no_eps), succ(w_eps), len(w_eps), cfg, "extended")
    z = cfg["wilson_z"]
    all_eps = no_eps + w_eps
    return NecessityLabel(
        task_id=task.id, label=label, stage=stage,
        notool_success=succ(no_eps), notool_trials=len(no_eps),
        tool_success=succ(w_eps), tool_trials=len(w_eps),
        notool_ci=wilson(succ(no_eps), len(no_eps), z), tool_ci=wilson(succ(w_eps), len(w_eps), z),
        tool_calls_with_tools=[e.n_tool_calls for e in w_eps],
        calls_tool_when_offered=sum(e.n_tool_calls > 0 for e in w_eps),
        cost_usd=sum(e.cost_usd for e in all_eps), wall_ms=sum(e.wall_ms for e in all_eps),
        episodes=[e.summary() for e in all_eps] if keep_episodes else [],
    )
