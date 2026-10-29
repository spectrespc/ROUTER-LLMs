#!/usr/bin/env python3
"""Demo: smooth weighted round-robin across three healthy embedding backends.

Runs entirely in /tmp/kilo/embed-demo: mocks on 8127/8128/8129 and a copy of the
router on 8123.  Nothing is written into the working ROUTER directory.
"""

from __future__ import annotations

import collections
import sys
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

PORTS = {"a": 8127, "b": 8128, "c": 8129}
ROUTER_PORT = 8123
MODEL = "nomic-embed-text-v1.5-Q8_0"
REQUESTS = 30

BACKENDS = {
    "llama-embedded": f"http://127.0.0.1:{PORTS['a']}",
    "llama-embedded-2": f"http://127.0.0.1:{PORTS['b']}",
    "llama-embedded-backup": f"http://127.0.0.1:{PORTS['c']}",
}

POOLS = {
    "defaults": {
        "failure_threshold": 3,
        "cooldown_seconds": 3,
        "attempt_timeout_seconds": 5,
        "max_total_seconds": 15,
    },
    "default_pool": None,
    "pools": {
        "embed-768": {
            "embedding_model": "nomic-embed-text-v1.5",
            "dimensions": 768,
            # All three are active here so the SWRR weights are visible.
            # The failover demo shows the backup role.
            "members": [
                {"backend": "llama-embedded", "model": MODEL, "weight": 3, "role": "active"},
                {"backend": "llama-embedded-2", "model": MODEL, "weight": 1, "role": "active"},
                {"backend": "llama-embedded-backup", "model": MODEL, "weight": 1, "role": "active"},
            ],
        }
    },
}


def main() -> None:
    for port in (*PORTS.values(), ROUTER_PORT):
        ensure_port_free(port)

    prepare_workspace(BACKENDS, POOLS)
    mocks = []
    router = None
    try:
        for name, port in PORTS.items():
            mocks.append(start_mock(port, dimensions=768, model=MODEL))
            print(f"[demo] mock {name} -> http://127.0.0.1:{port}")
        router = start_router(ROUTER_PORT)
        print(f"[demo] router -> http://127.0.0.1:{ROUTER_PORT}")

        distribution: collections.Counter[str] = collections.Counter()
        dimensions_ok = 0
        failures = 0
        for index in range(REQUESTS):
            status, payload = request_json(
                f"http://127.0.0.1:{ROUTER_PORT}/v1/embeddings",
                method="POST",
                body={"model": "embed-768", "input": f"load balancing sample {index}"},
            )
            if status != 200:
                failures += 1
                print(f"[demo] request {index} failed: {status} {payload}")
                continue
            distribution[payload["backend_used"]] += 1
            if len(payload["data"][0]["embedding"]) == 768:
                dimensions_ok += 1

        print("\n[demo] distribution by backend_used:")
        width = max(len(name) for name in PORTS) + 2
        for name in distribution:
            count = distribution[name]
            bar = "#" * count
            print(f"  {name:<{width}} {count:>3}  {bar}")
        print(
            f"\n[demo] requests={REQUESTS} ok={REQUESTS - failures} failures={failures} "
            f"vectors_768d={dimensions_ok}"
        )
        print("[demo] expected with weights 3:1:1 over 30 requests: a=18, b=6, c=6")

        _, status_payload = request_json(
            f"http://127.0.0.1:{ROUTER_PORT}/admin/api/embedding/pools"
        )
        pool = status_payload["pools"][0]
        print(
            f"[demo] pool '{pool['name']}': enabled={pool['enabled']} "
            f"healthy={pool['members_healthy']}/{pool['members_total']} "
            f"open_breakers={pool['open_breakers']}"
        )
        if failures or dimensions_ok != REQUESTS:
            raise SystemExit("demo failed: missing or malformed vectors")
    finally:
        stop(router, *mocks)


if __name__ == "__main__":
    main()