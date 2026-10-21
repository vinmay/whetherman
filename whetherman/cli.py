"""whetherman CLI: calibrate an agent, serve the gating proxy."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _load_dotenv() -> None:
    env = Path.cwd() / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def main(argv: list[str] | None = None) -> None:
    _load_dotenv()
    ap = argparse.ArgumentParser(prog="whetherman", description="Forecast whether your agent needs a tool.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="measure an agent on the probe pack and write its profile")
    c.add_argument("--agent-endpoint", required=True, help="OpenAI-compatible base URL, e.g. http://localhost:11434/v1")
    c.add_argument("--model", required=True)
    c.add_argument("--probe-pack", default=None, help="probe pack directory (default: bundled pack)")
    c.add_argument("--name", default=None, help="profile name (default: model slug)")
    c.add_argument("--backbone", default="checkpoints/shared-v1")
    c.add_argument("--api-key-env", default=None, help="env var holding the endpoint API key")
    c.add_argument("--parallel", type=int, default=1)
    c.add_argument("--labels-dir", default=None)
    c.add_argument("--out-dir", default="profiles")
    c.add_argument("--device", default="auto")
    c.add_argument("--yes", action="store_true", help="skip the cost confirmation")

    s = sub.add_parser("serve", help="run the gating proxy")
    s.add_argument("--profile", required=True, help="profile name or path (profiles/<name>.json)")
    s.add_argument("--upstream", required=True, help="agent's OpenAI-compatible base URL")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8088)
    s.add_argument("--backbone", default=None)
    s.add_argument("--device", default="auto")
    s.add_argument("--timeout-ms", type=float, default=3000)
    s.add_argument("--log", default=None)

    a = ap.parse_args(argv)
    if a.cmd == "calibrate":
        from whetherman.calibrate.run import calibrate

        calibrate(a.agent_endpoint, a.model, a.probe_pack, a.name, a.backbone, a.yes, a.parallel, a.labels_dir,
                  a.out_dir, a.device, os.environ.get(a.api_key_env) if a.api_key_env else None)
    elif a.cmd == "serve":
        from whetherman.proxy.server import serve

        prof = a.profile if a.profile.endswith(".json") else f"profiles/{a.profile}.json"
        serve(prof, a.upstream, a.host, a.port, a.backbone, a.device, a.log, a.timeout_ms)


if __name__ == "__main__":
    main()
