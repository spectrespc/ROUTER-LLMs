#!/usr/bin/env python3
"""Minimal OpenAI-compatible embedding server for router demos (stdlib only).

Examples:
    python mock_embedding_server.py --port 8127 --dimensions 768 --model nomic-embed-text-v1.5
    python mock_embedding_server.py --port 8127 --fail
    python mock_embedding_server.py --port 8127 --slow 2.0

The vector for a text is deterministic (derived from SHA-256), so demos can
verify that retries/fallbacks return plausible embeddings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mock llama.cpp embedding server")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--dimensions", type=int, default=768)
    parser.add_argument("--model", default="nomic-embed-text-v1.5")
    parser.add_argument("--fail", action="store_true", help="answer every embedding request with HTTP 500")
    parser.add_argument("--slow", type=float, default=0.0, help="sleep N seconds before answering")
    parser.add_argument("--host", default="127.0.0.1")
    return parser.parse_args()


def make_vector(text: str, dimensions: int) -> list[float]:
    digest = hashlib.sha256(str(text).encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    vector = []
    for index in range(dimensions):
        value = math.sin((seed % 100003) * 0.0001 + index * 0.37)
        vector.append(round(value, 6))
    return vector


def main() -> None:
    args = parse_args()

    class Handler(BaseHTTPRequestHandler):
        server_version = "MockEmbedding/1.0"

        def log_message(self, fmt, *log_args):  # keep demo output readable
            pass

        def _send_json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802 - http.server API
            if self.path in ("/health", "/v1/health"):
                self._send_json(200, {"status": "ok"})
            elif self.path == "/props":
                self._send_json(200, {"embedding": True, "n_ctx": 2048})
            elif self.path in ("/v1/models", "/models"):
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": args.model,
                                "object": "model",
                                "created": int(time.time()),
                                "owned_by": "mock",
                            }
                        ],
                    },
                )
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802 - http.server API
            if self.path != "/v1/embeddings":
                self._send_json(404, {"error": "not found"})
                return

            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                self._send_json(400, {"error": {"message": "invalid JSON"}})
                return

            if args.slow > 0:
                time.sleep(args.slow)
            if args.fail:
                self._send_json(500, {"error": {"message": "mock backend is down"}})
                return

            raw_input = body.get("input", "")
            if isinstance(raw_input, list):
                texts = [str(item) for item in raw_input]
            else:
                texts = [str(raw_input)]

            data = [
                {
                    "object": "embedding",
                    "index": index,
                    "embedding": make_vector(text, args.dimensions),
                }
                for index, text in enumerate(texts)
            ]
            prompt_tokens = sum(max(1, len(text.split())) for text in texts)
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": data,
                    "model": args.model,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "total_tokens": prompt_tokens,
                    },
                },
            )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    mode = "FAIL" if args.fail else f"{args.dimensions}d"
    print(f"[mock] http://{args.host}:{args.port} model={args.model} mode={mode}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()