#!/usr/bin/env python3
"""Demo: transparent fallback, circuit breaker, cooldown and recovery.

Primary mock runs on 8127 (active), backup on 8128 (role=backup), router copy on
8123.  Everything happens in /tmp/kilo/embed-demo.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from demo_common import (  # noqa: E402
    ensure_port_free,
    prepare_workspace,
    request_json,
    start_mock,
    start_router,
    stop,
)

PRIMARY_PORT = 8127
BACKUP_PORT = 8128
ROUTER_PORT = 8123
MODEL = "nomic-embed-text-v1.5-Q8_0"
COOLDOWN = 3
REQUESTS = 8

BACKENDS = {
    "llama-embedded": f"http://127.0.0.1:{PRIMARY_PORT}",
    "llama-embedded-backup": f"http://127.0.0.1:{BACKUP_PORT}",
}

POOLS = {
    "defaults": {
        "failure_threshold": 2,
        "cooldown_seconds": COOLDOWN,
        "attempt_timeout_seconds": 3,
        "max_total_seconds": 10,
    },
    "default_pool": None,
    "pools": {
        "embed-768": {
            "embedding_model": "nomic-embed-text-v1.5",
            "dimensions": 768,
            "members": [
                {"backend": "llama-embedded", "model": MODEL, "weight": 1, "role": "active"},
                {"backend": "llama-embedded-backup", "model": MODEL, "weight": 1, "role": "backup"},
            ],
        }
    },
}


def ask(router_port: int, text: str):
    return request_json(
        f"http://127.0.0.1:{router_port}/v1/embeddings",
        method="POST",
        body={"model": "embed-768", "input": text},
    )


def main() -> None:
    for port in (PRIMARY_PORT, BACKUP_PORT, ROUTER_PORT):
        ensure_port_free(port)

    prepare_workspace(BACKENDS, POOLS)
    primary = None
    backup = None
    router = None
    vectors = 0
    responses = 0
    failures = 0
    try:
        primary = start_mock(PRIMARY_PORT, model=MODEL)
        backup = start_mock(BACKUP_PORT, model=MODEL)
        router = start_router(ROUTER_PORT)
        print("[demo] primary, backup and router are up")

        for index in range(3):
            status, payload = ask(ROUTER_PORT, f"healthy {index}")
            responses += 1
            if status != 200:
                failures += 1
                continue
            vectors += len(payload["data"][0]["embedding"]) == 768
            if index == 0:
                print(f"[demo] healthy request -> backend_used={payload['backend_used']}")
        print("[demo] primary serves all requests while healthy")

        # Simulate a broken primary while the process stays reachable.
        stop(primary)
        primary = start_mock(PRIMARY_PORT, model=MODEL, fail=True)
        print("[demo] primary now returns HTTP 500")

        for index in range(2):
            status, payload = ask(ROUTER_PORT, f"failover {index}")
            responses += 1
            if status != 200:
                failures += 1
                print(f"[demo] failover request failed: {status} {payload}")
                continue
            vectors += len(payload["data"][0]["embedding"]) == 768
            print(
                f"[demo] failover request {index}: backend_used={payload['backend_used']} "
                f"attempts={payload['attempts']} fallback_used={payload['fallback_used']}"
            )

        _, status_payload = request_json(
            f"http://127.0.0.1:{ROUTER_PORT}/admin/api/embedding/pools"
        )
        pool = status_payload["pools"][0]
        primary_status = next(m for m in pool["members"] if m["backend"] == "llama-embedded")
        print(
            f"[demo] breaker: state={primary_status['breaker']} "
            f"consecutive_failures={primary_status['consecutive_failures']} "
            f"open_for_seconds={primary_status['open_for_seconds']}"
        )
        if primary_status["breaker"] != "open":
            raise SystemExit("demo failed: primary breaker did not open")

        status, payload = ask(ROUTER_PORT, "while breaker is open")
        responses += 1
        if status != 200:
            failures += 1
        else:
            vectors += len(payload["data"][0]["embedding"]) == 768
            print(
                f"[demo] request with open breaker -> backend_used={payload['backend_used']} "
                f"attempts={payload['attempts']}"
            )

        # Restore the primary and wait for the cooldown.  The next request must
        # half-open the breaker and route back to the primary.
        stop(primary)
        primary = start_mock(PRIMARY_PORT, model=MODEL)
        print(f"[demo] primary restored; waiting {COOLDOWN + 0.7:.1f}s for cooldown")
        time.sleep(COOLDOWN + 0.7)

        status, payload = ask(ROUTER_PORT, "after cooldown")
        responses += 1
        if status != 200:
            failures += 1
            print(f"[demo] recovery request failed: {status} {payload}")
        else:
            vectors += len(payload["data"][0]["embedding"]) == 768
            print(
                f"[demo] recovery request -> backend_used={payload['backend_used']} "
                f"attempts={payload['attempts']}"
            )
            if payload["backend_used"] != "llama-embedded":
                raise SystemExit("demo failed: recovered primary did not receive traffic")

        _, status_payload = request_json(
            f"http://127.0.0.1:{ROUTER_PORT}/admin/api/embedding/pools"
        )
        pool = status_payload["pools"][0]
        primary_status = next(m for m in pool["members"] if m["backend"] == "llama-embedded")
        print(
            f"[demo] breaker after recovery: state={primary_status['breaker']} "
            f"consecutive_failures={primary_status['consecutive_failures']}"
        )

        # Fill the remaining budget with requests to prove there is no data loss.
        while responses < REQUESTS:
            status, payload = ask(ROUTER_PORT, f"final {responses}")
            responses += 1
            if status != 200:
                failures += 1
            else:
                vectors += len(payload["data"][0]["embedding"]) == 768

        print(f"\n[demo] responses={responses} failures={failures} vectors_768d={vectors}")
        if failures or vectors != responses:
            raise SystemExit("demo failed: missing or malformed vectors")
        print("[demo] all requests succeeded; no embedding was lost during failover")
    finally:
        stop(router, primary, backup)


if __name__ == "__main__":
    main()