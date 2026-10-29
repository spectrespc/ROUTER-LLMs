"""Shared helpers for the embedding pool demos (stdlib only)."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROUTER_DIR = Path(__file__).resolve().parents[1]
WORK_DIR = Path("/tmp/kilo/embed-demo")
EXAMPLES_DIR = ROUTER_DIR / "examples"
MOCK_SERVER = EXAMPLES_DIR / "mock_embedding_server.py"


def ensure_port_free(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise SystemExit(
                f"Port {port} is already in use ({exc}). Stop the process or change the demo port."
            ) from exc


def prepare_workspace(backends: dict, pools: dict) -> Path:
    """Copy the router into an isolated sandbox and write temporary configs."""
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)
    shutil.copy2(ROUTER_DIR / "router.py", WORK_DIR / "router.py")
    shutil.copy2(ROUTER_DIR / "embedding_pool.py", WORK_DIR / "embedding_pool.py")
    shutil.copytree(ROUTER_DIR / "templates", WORK_DIR / "templates")
    (WORK_DIR / "backends.json").write_text(json.dumps(backends, indent=2), encoding="utf-8")
    (WORK_DIR / "embedding_pools.json").write_text(json.dumps(pools, indent=2), encoding="utf-8")
    (WORK_DIR / ".env").write_text("API_KEY=demo-key\n", encoding="utf-8")
    return WORK_DIR


def start_mock(port: int, dimensions: int = 768, model: str = "nomic-embed-text-v1.5-Q8_0",
               fail: bool = False, slow: float = 0.0) -> subprocess.Popen:
    command = [
        sys.executable,
        str(MOCK_SERVER),
        "--port",
        str(port),
        "--dimensions",
        str(dimensions),
        "--model",
        model,
    ]
    if fail:
        command.append("--fail")
    if slow:
        command.extend(["--slow", str(slow)])
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wait_http(f"http://127.0.0.1:{port}/health", timeout=10.0)
    return process


def start_router(port: int, api_key: str = "demo-key") -> subprocess.Popen:
    env = dict(os.environ)
    env["API_KEY"] = api_key
    env["PYTHONUNBUFFERED"] = "1"
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "router:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]
    process = subprocess.Popen(
        command,
        cwd=str(WORK_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    wait_http(f"http://127.0.0.1:{port}/health", timeout=20.0)
    return process


def wait_http(url: str, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status < 500:
                    return
        except Exception as exc:  # noqa: BLE001 - startup polling
            last_error = exc
            time.sleep(0.15)
    raise SystemExit(f"Timed out waiting for {url}: {last_error}")


def request_json(url: str, method: str = "GET", body: dict | None = None,
                 api_key: str = "demo-key", timeout: float = 30.0):
    """Return (status_code, decoded JSON). HTTP errors are decoded, not raised."""
    data = None
    headers = {"Authorization": f"Bearer {api_key}"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", "replace")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"raw": raw}
        return exc.code, payload


def stop(*processes: subprocess.Popen | None) -> None:
    for process in processes:
        if process is None or process.poll() is not None:
            continue
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)