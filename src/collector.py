"""Localhost canary collector (demonstration stub).

SAFETY (SPEC Section 1):
- Binds to 127.0.0.1 ONLY. It refuses to start on any non-localhost host.
- It records whatever it is POSTed to a local JSONL log. In this project the only
  thing ever sent is the fixed benign canary string. The collector has no outbound
  network behavior and performs no action on the payload beyond logging it.
- Evaluation does NOT depend on this server (SPEC Section 7): attack success is
  detected from the model's generated tool-call text. This collector exists only to
  make the end-to-end demo concrete.

Usage:
    python -m src.collector                 # start server (reads configs/default.yaml)
    python -m src.collector --smoke          # start, self-POST the canary, verify, exit
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request

# Support both "python -m src.collector" and "python src/collector.py".
try:
    from src.config import abspath, load_config
except ModuleNotFoundError:  # pragma: no cover - path shim
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import abspath, load_config

LOCALHOST = {"127.0.0.1", "localhost", "::1"}


def create_app(log_path: Path) -> Flask:
    app = Flask(__name__)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    @app.route("/collect", methods=["POST"])
    def collect():
        payload = request.get_json(silent=True) or {}
        entry = {
            "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
            "remote_addr": request.remote_addr,
            "payload": payload,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return jsonify({"ok": True})

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"ok": True, "service": "canary-collector"})

    return app


def _assert_localhost(host: str) -> None:
    if host not in LOCALHOST:
        raise SystemExit(
            f"REFUSING to bind collector to non-localhost host {host!r}. "
            f"This stub is localhost-only by design (SPEC Section 1)."
        )


def run_server(host: str, port: int, log_path: Path) -> None:
    _assert_localhost(host)
    app = create_app(log_path)
    app.run(host=host, port=port, debug=False, use_reloader=False)


def _smoke(host: str, port: int, log_path: Path, canary: str) -> int:
    """Start the server in a thread, POST the canary, confirm a log line appears."""
    import requests

    _assert_localhost(host)
    app = create_app(log_path)
    before = log_path.read_text().count("\n") if log_path.exists() else 0

    t = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False),
        daemon=True,
    )
    t.start()

    base = f"http://{host}:{port}"
    for _ in range(50):
        try:
            if requests.get(f"{base}/health", timeout=0.5).ok:
                break
        except Exception:
            time.sleep(0.1)
    else:
        print("SMOKE FAIL: server did not come up")
        return 1

    resp = requests.post(f"{base}/collect", json={"payload": canary}, timeout=2)
    assert resp.ok and resp.json().get("ok") is True, resp.text
    time.sleep(0.2)

    after = log_path.read_text().count("\n") if log_path.exists() else 0
    last = log_path.read_text().splitlines()[-1]
    assert canary in last, f"canary not in last log line: {last}"
    print(f"SMOKE OK: health 200, POST 200, log grew {before}->{after}")
    print(f"  last line: {last}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Localhost canary collector stub")
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true", help="self-test and exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    host = cfg["collector"]["host"]
    port = int(cfg["collector"]["port"])
    log_path = abspath(cfg["collector"]["log_path"])
    canary = cfg["canary"]["token"]

    if args.smoke:
        return _smoke(host, port, log_path, canary)

    print(f"Canary collector on http://{host}:{port}  (log: {log_path})")
    run_server(host, port, log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
