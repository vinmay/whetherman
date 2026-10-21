"""Load agent definitions from configs/agents.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from whetherman.providers.openai_compat import ChatClient

AGENTS_PATH = Path(__file__).resolve().parents[1] / "configs" / "agents.yaml"


def agent_config(name: str) -> dict[str, Any]:
    return yaml.safe_load(AGENTS_PATH.read_text())[name]


def make_client(name: str, **kw: Any) -> ChatClient:
    cfg = agent_config(name)
    return ChatClient(cfg["model"], provider=cfg["provider"], **kw)
