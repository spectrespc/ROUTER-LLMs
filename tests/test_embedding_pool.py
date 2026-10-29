"""Unit tests for embedding_pool.py and the pool path of /v1/embeddings.

No live servers are used: every test injects a fake async client factory.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from fastapi import HTTPException

ROUTER_DIR = Path(__file__).resolve().parents[1]
if str(ROUTER_DIR) not in sys.path:
    sys.path.insert(0, str(ROUTER_DIR))

from embedding_pool import EmbeddingPoolManager, normalize_model_name  # noqa: E402

BACKENDS = {
    "a": "http://a:8080",
    "b": "http://b:8080",
    "c": "http://c:8080",
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
class FakeResponse:
    def __init__(self, status_code: int = 200, payload=None, text: str | None = None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class FakeClient:
    def __init__(self, factory: "FakeFactory", timeout):
        self.factory = factory
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def post(self, url, json=None):
        self.factory.calls.append({"url": url, "body": json, "timeout": self.timeout})
        result = self.factory.responder(url, json)
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, Exception):
            raise result
        return result


class FakeFactory:
    def __init__(self, responder):
        self.responder = responder
        self.calls = []
        self.timeouts = []

    def __call__(self, timeout):
        self.timeouts.append(timeout)
        return FakeClient(self, timeout)


def embedding_payload(dimensions: int = 4, value: float = 0.1):
    return {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [value] * dimensions}],
        "model": "backend-model",
        "usage": {"prompt_tokens": 3, "total_tokens": 3},
    }


def config(
    members,
    *,
    embedding_model: str = "test-embed",
    dimensions: int = 4,
    defaults=None,
    pool_overrides=None,
    default_pool=None,
    extra_pools=None,
):
    pool = {
        "embedding_model": embedding_model,
        "dimensions": dimensions,
        "members": members,
    }
    pool.update(pool_overrides or {})
    pools = {"p": pool}
    pools.update(extra_pools or {})
    return {
        "defaults": defaults or {},
        "default_pool": default_pool,
        "pools": pools,
    }


def member(backend: str, model: str = "test-embed-q8", weight: int = 1, role: str = "active"):
    return {"backend": backend, "model": model, "weight": weight, "role": role}


def load_manager(tmp_path, cfg, factory=None, backends=BACKENDS, name="embedding_pools.json"):
    path = tmp_path / name
    path.write_text(json.dumps(cfg), encoding="utf-8")
    manager = EmbeddingPoolManager(path, factory or FakeFactory(lambda url, body: FakeResponse()))
    asyncio.run(manager.load(backends))
    return manager


def pool_of(manager, name="p"):
    pool = manager.get_pool(name)
    assert pool is not None
    return pool


async def call_execute(manager, body=None, cache=None, resolver=None):
    body = body or {"model": "p", "input": "hello"}
    resolver = resolver or (lambda model, backend: f"resolved-{backend}")
    return await manager.execute(
        pool_of(manager),
        body,
        cache,
        resolver,
        backends=BACKENDS,
        requested_model="p",
    )


# ----------------------------------------------------------------------
# 1. Smooth weighted round-robin
# ----------------------------------------------------------------------
def test_swrr_sequence_3_1_1(tmp_path):
    manager = load_manager(
        tmp_path,
        config(
            [
                member("a", weight=3),
                member("b", weight=1),
                member("c", weight=1),
            ]
        ),
    )
    pool = pool_of(manager)
    sequence = [manager.pick_chain(pool)[0][0].member.backend for _ in range(10)]
    assert sequence == ["a", "b", "a", "c", "a", "a", "b", "a", "c", "a"]
    assert sequence.count("a") == 6
    assert sequence.count("b") == 2
    assert sequence.count("c") == 2


def test_swrr_fallback_order_is_full_chain(tmp_path):
    manager = load_manager(
        tmp_path,
        config([member("a", weight=3), member("b", weight=1), member("c", weight=1)]),
    )
    chain, retry_after = manager.pick_chain(pool_of(manager))
    assert retry_after is None
    assert [candidate.member.backend for candidate in chain] == ["a", "b", "c"]


def test_backups_are_used_only_after_healthy_actives(tmp_path):
    manager = load_manager(
        tmp_path,
        config([member("a"), member("b", role="backup")]),
    )
    chain, _ = manager.pick_chain(pool_of(manager))
    assert [candidate.member.backend for candidate in chain] == ["a", "b"]
    assert chain[0].member.role == "active"
    assert chain[1].member.role == "backup"


# ----------------------------------------------------------------------
# 2. Circuit breaker
# ----------------------------------------------------------------------
def test_breaker_opens_after_threshold_and_rejects_with_retry_after(tmp_path):
    factory = FakeFactory(lambda url, body: FakeResponse(500, {"error": "boom"}))
    manager = load_manager(
        tmp_path,
        config([member("a")], pool_overrides={"failure_threshold": 3, "cooldown_seconds": 10}),
        factory,
    )

    for _ in range(3):
        with pytest.raises(HTTPException) as info:
            asyncio.run(call_execute(manager))
        assert info.value.status_code == 502

    state = manager._states[("p", "a")]
    assert state.consecutive_failures == 3
    assert state.open_until > 0

    with pytest.raises(HTTPException) as info:
        asyncio.run(call_execute(manager))
    assert info.value.status_code == 503
    assert info.value.detail["code"] == "no_healthy_members"
    assert "Retry-After" in (info.value.headers or {})


def test_half_open_allows_single_probe_and_success_closes_breaker(tmp_path):
    factory = FakeFactory(lambda url, body: FakeResponse(500, {"error": "boom"}))
    manager = load_manager(
        tmp_path,
        config([member("a")], pool_overrides={"failure_threshold": 1, "cooldown_seconds": 0.1}),
        factory,
    )

    with pytest.raises(HTTPException):
        asyncio.run(call_execute(manager))
    assert manager._states[("p", "a")].open_until > 0

    async def slow_success(url, body):
        await asyncio.sleep(0.1)
        return FakeResponse(200, embedding_payload(4))

    factory.responder = slow_success

    async def run_concurrently():
        await asyncio.sleep(0.12)
        return await asyncio.gather(
            call_execute(manager),
            call_execute(manager),
            return_exceptions=True,
        )

    results = asyncio.run(run_concurrently())
    successes = [item for item in results if not isinstance(item, Exception)]
    rejected = [item for item in results if isinstance(item, HTTPException)]
    assert len(successes) == 1
    assert len(rejected) == 1
    assert rejected[0].status_code == 503
    state = manager._states[("p", "a")]
    assert state.consecutive_failures == 0
    assert state.open_until == 0
    assert state.probe_inflight is False


# ----------------------------------------------------------------------
# 3-4. Fallback chain and client errors
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "failure",
    [
        httpx.ConnectError("connection refused"),
        httpx.ReadTimeout("read timeout"),
        FakeResponse(429, {"error": "rate limited"}),
        FakeResponse(503, {"error": "unavailable"}),
    ],
)
def test_fallback_to_backup_preserves_body(tmp_path, failure):
    def responder(url, body):
        if "//a:" in url:
            return failure
        return FakeResponse(200, embedding_payload(4))

    factory = FakeFactory(responder)
    manager = load_manager(
        tmp_path,
        config([member("a"), member("b")]),
        factory,
    )
    original = {"model": "p", "input": "unchanged text", "encoding_format": "float"}
    result = asyncio.run(call_execute(manager, body=original))

    assert result.status_code == 200
    assert result.backend_used == "b"
    assert result.attempts == 2
    assert result.fallback_used is True
    assert result.payload["backend_used"] == "b"
    assert result.payload["pool"] == "p"
    assert result.payload["attempts"] == 2
    assert result.payload["fallback_used"] is True

    assert len(factory.calls) == 2
    for call in factory.calls:
        assert call["body"]["input"] == "unchanged text"
        assert call["body"]["encoding_format"] == "float"
    assert factory.calls[0]["body"]["model"] == "resolved-a"
    assert factory.calls[1]["body"]["model"] == "resolved-b"


def test_400_is_returned_transparently_without_retry(tmp_path):
    factory = FakeFactory(
        lambda url, body: FakeResponse(400, {"error": {"message": "bad input", "type": "invalid_request"}})
    )
    manager = load_manager(tmp_path, config([member("a"), member("b")]), factory)
    result = asyncio.run(call_execute(manager))

    assert result.status_code == 400
    assert result.payload["error"]["message"] == "bad input"
    assert len(factory.calls) == 1
    assert manager._states[("p", "a")].consecutive_failures == 0


# ----------------------------------------------------------------------
# 5. Dimension guard
# ----------------------------------------------------------------------
def test_dimension_mismatch_is_a_member_failure_and_not_returned(tmp_path):
    def responder(url, body):
        if "//a:" in url:
            return FakeResponse(200, embedding_payload(3))
        return FakeResponse(200, embedding_payload(4))

    factory = FakeFactory(responder)
    manager = load_manager(
        tmp_path,
        config([member("a"), member("b")], dimensions=4),
        factory,
    )
    result = asyncio.run(call_execute(manager))

    assert result.status_code == 200
    assert result.backend_used == "b"
    assert result.attempts == 2
    assert len(result.payload["data"][0]["embedding"]) == 4
    assert manager._states[("p", "a")].consecutive_failures == 1
    assert "dimension_mismatch" in manager._states[("p", "a")].last_error


# ----------------------------------------------------------------------
# 6. Pool exhaustion and deadlines
# ----------------------------------------------------------------------
def test_all_attempts_failed_returns_502_aggregate(tmp_path):
    factory = FakeFactory(lambda url, body: FakeResponse(500, {"error": "down"}))
    manager = load_manager(
        tmp_path,
        config([member("a"), member("b")], pool_overrides={"failure_threshold": 5}),
        factory,
    )
    with pytest.raises(HTTPException) as info:
        asyncio.run(call_execute(manager))

    assert info.value.status_code == 502
    detail = info.value.detail
    assert detail["code"] == "all_attempts_failed"
    assert detail["pool"] == "p"
    assert [item["backend"] for item in detail["members"]] == ["a", "b"]


def test_deadline_exceeded_returns_504(tmp_path):
    async def slow_failure(url, body):
        await asyncio.sleep(0.6)
        return FakeResponse(500, {"error": "slow"})

    factory = FakeFactory(slow_failure)
    manager = load_manager(
        tmp_path,
        config(
            [member("a"), member("b")],
            pool_overrides={
                "max_total_seconds": 2.5,
                "attempt_timeout_seconds": 0.5,
                "failure_threshold": 5,
            },
        ),
        factory,
    )
    with pytest.raises(HTTPException) as info:
        asyncio.run(call_execute(manager))

    assert info.value.status_code == 504
    assert info.value.detail["code"] == "deadline_exceeded"
    assert len(factory.calls) == 1


def test_disabled_pool_returns_503_pool_disabled(tmp_path):
    manager = load_manager(
        tmp_path,
        config([{"backend": "missing", "model": "test-embed-q8"}]),
    )
    with pytest.raises(HTTPException) as info:
        asyncio.run(call_execute(manager))
    assert info.value.status_code == 503
    assert info.value.detail["code"] == "pool_disabled"


# ----------------------------------------------------------------------
# 7. Config validation
# ----------------------------------------------------------------------
def test_corrupt_config_keeps_previous_configuration(tmp_path):
    manager = load_manager(tmp_path, config([member("a")]))
    path = tmp_path / "embedding_pools.json"
    path.write_text("{not valid json", encoding="utf-8")

    reloaded = asyncio.run(manager.load(BACKENDS))

    assert reloaded is False
    assert manager.list_pools() == ["p"]
    assert manager.get_pool("p") is not None
    assert (tmp_path / "embedding_pools.json.corrupt").exists()
    assert manager.status(BACKENDS)["last_load_error"]


def test_unknown_backend_and_model_mismatch_are_rejected(tmp_path):
    manager = load_manager(
        tmp_path,
        config(
            [
                member("missing"),
                member("a", model="another-model"),
                member("b"),
            ]
        ),
    )
    pool = pool_of(manager)
    assert [item.backend for item in pool.members] == ["b"]
    reasons = {(item["backend"], item["reason"]) for item in pool.rejected}
    assert ("missing", "unknown_backend") in reasons
    assert ("a", "model_mismatch") in reasons


def test_pool_without_valid_members_is_disabled(tmp_path):
    manager = load_manager(
        tmp_path,
        config(
            [member("a", model="different-model")],
            extra_pools={
                "empty": {
                    "embedding_model": "test-embed",
                    "dimensions": 4,
                    "members": [],
                }
            },
        ),
    )
    pool = pool_of(manager)
    assert pool.enabled is False
    assert pool.disabled_reason == "no_valid_members"
    assert manager.get_pool("empty").enabled is False


def test_normalization_handles_dots_and_backend_suffix():
    assert normalize_model_name("nomic-embed-text-v1.5-Q4_K_M.gguf") == "nomic-embed-text-v1-5-q4_k_m"
    assert normalize_model_name("nomic-embed-text-v1.5@llama-embedded") == "nomic-embed-text-v1-5"


def test_quantized_member_model_is_compatible_with_pool_model(tmp_path):
    manager = load_manager(
        tmp_path,
        config([member("a", model="nomic-embed-text-v1.5-Q4_K_M")], embedding_model="nomic-embed-text-v1.5"),
    )
    assert len(pool_of(manager).members) == 1


def test_cached_model_mismatch_defers_candidate_but_keeps_it(tmp_path):
    manager = load_manager(tmp_path, config([member("a"), member("b")]))
    cache = {
        "a": {"status": "online", "models": [{"id": "other-model@a", "original_path": "other-model.gguf"}]},
        "b": {"status": "online", "models": [{"id": "test-embed-q8@b", "original_path": "test-embed-q8.gguf"}]},
    }
    chain, _ = manager.pick_chain(pool_of(manager), cache)
    assert [candidate.member.backend for candidate in chain] == ["b", "a"]
    assert chain[1].cache_mismatch == "cached_models_do_not_match"

    status = manager.status(BACKENDS, cache)
    members = {item["backend"]: item for item in status["pools"][0]["members"]}
    assert members["a"]["cache_hint"] == "cached_models_do_not_match"
    assert members["b"]["cache_hint"] is None


def test_pool_model_entries_and_health_summary(tmp_path):
    manager = load_manager(tmp_path, config([member("a"), member("b", role="backup")]))
    entries = manager.pool_model_entries()
    assert entries[0]["id"] == "p"
    assert entries[0]["pool"] is True
    assert entries[0]["dimensions"] == 4
    assert len(entries[0]["members"]) == 2

    summary = manager.health_summary()
    assert summary["pools"] == 1
    assert summary["members_healthy"] == 2
    assert summary["members_total"] == 2
    assert summary["details"]["p"]["enabled"] is True


# ----------------------------------------------------------------------
# 8. Reload by mtime and breaker reset
# ----------------------------------------------------------------------
def test_reload_keeps_state_for_surviving_members(tmp_path):
    def responder(url, body):
        if "//a:" in url:
            return FakeResponse(500, {"error": "down"})
        return FakeResponse(200, embedding_payload(4))

    factory = FakeFactory(responder)
    manager = load_manager(
        tmp_path,
        config([member("a"), member("b")], pool_overrides={"failure_threshold": 1, "cooldown_seconds": 60}),
        factory,
    )

    # a opens after the first failure; b succeeds first and then serves only
    # requests where a is already open.
    asyncio.run(call_execute(manager))
    assert manager._states[("p", "a")].open_until > 0
    asyncio.run(call_execute(manager))

    new_cfg = config(
        [member("a"), member("b"), member("c")],
        pool_overrides={"failure_threshold": 1, "cooldown_seconds": 60},
    )
    path = tmp_path / "embedding_pools.json"
    path.write_text(json.dumps(new_cfg), encoding="utf-8")
    future = path.stat().st_mtime + 10
    os.utime(path, (future, future))

    assert asyncio.run(manager.maybe_reload(BACKENDS)) is True
    assert [item.backend for item in pool_of(manager).members] == ["a", "b", "c"]
    assert manager._states[("p", "a")].open_until > 0
    assert manager._states[("p", "c")].consecutive_failures == 0


def test_reset_breaker_closes_member(tmp_path):
    factory = FakeFactory(lambda url, body: FakeResponse(500, {"error": "down"}))
    manager = load_manager(
        tmp_path,
        config([member("a")], pool_overrides={"failure_threshold": 1, "cooldown_seconds": 60}),
        factory,
    )
    with pytest.raises(HTTPException):
        asyncio.run(call_execute(manager))
    assert manager._states[("p", "a")].open_until > 0

    assert asyncio.run(manager.reset_breaker("p", "a")) is True
    assert manager._states[("p", "a")].open_until == 0
    assert manager._states[("p", "a")].consecutive_failures == 0
    assert asyncio.run(manager.reset_breaker("p", "unknown")) is False


# ----------------------------------------------------------------------
# 9. default_pool behaviour through router.route_embeddings
# ----------------------------------------------------------------------
class FakeRequest:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return dict(self._body)


def _import_router_in_tmp(tmp_path, monkeypatch, cfg):
    (tmp_path / "backends.json").write_text(json.dumps(BACKENDS), encoding="utf-8")
    (tmp_path / "embedding_pools.json").write_text(json.dumps(cfg), encoding="utf-8")
    (tmp_path / "templates").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("router", None)
    router = importlib.import_module("router")
    return router


def test_empty_model_routes_to_default_pool(tmp_path, monkeypatch):
    cfg = config([member("a")], default_pool="p")
    router = _import_router_in_tmp(tmp_path, monkeypatch, cfg)

    factory = FakeFactory(lambda url, body: FakeResponse(200, embedding_payload(4)))
    manager = EmbeddingPoolManager(tmp_path / "embedding_pools.json", factory)
    asyncio.run(manager.load(BACKENDS))
    router.embedding_pool_manager = manager
    router.MODELS_CACHE = {}

    async def fake_cache():
        return {}

    router.get_models_cache = fake_cache

    response = asyncio.run(
        router.route_embeddings(FakeRequest({"input": "hello"}), api_key="test")
    )
    assert response.status_code == 200
    payload = json.loads(response.body)
    assert payload["pool"] == "p"
    assert payload["backend_used"] == "a"


def test_empty_model_without_default_pool_still_returns_400(tmp_path, monkeypatch):
    cfg = config([member("a")])
    router = _import_router_in_tmp(tmp_path, monkeypatch, cfg)

    manager = EmbeddingPoolManager(
        tmp_path / "embedding_pools.json", FakeFactory(lambda url, body: FakeResponse())
    )
    asyncio.run(manager.load(BACKENDS))
    router.embedding_pool_manager = manager
    router.MODELS_CACHE = {}

    async def fake_cache():
        return {}

    router.get_models_cache = fake_cache

    with pytest.raises(HTTPException) as info:
        asyncio.run(router.route_embeddings(FakeRequest({"input": "hello"}), api_key="test"))
    assert info.value.status_code == 400


# ----------------------------------------------------------------------
# 10. Admin panel structure
# ----------------------------------------------------------------------
def test_admin_page_contains_pool_section(tmp_path, monkeypatch):
    cfg = config([member("a")])
    (tmp_path / "backends.json").write_text(json.dumps(BACKENDS), encoding="utf-8")
    (tmp_path / "embedding_pools.json").write_text(json.dumps(cfg), encoding="utf-8")
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    shutil.copy2(ROUTER_DIR / "templates" / "admin.html", templates_dir / "admin.html")

    monkeypatch.chdir(tmp_path)
    sys.modules.pop("router", None)
    router = importlib.import_module("router")

    async def no_cache_update():
        return None

    router.update_models_cache = no_cache_update

    from fastapi.testclient import TestClient

    with TestClient(router.app) as client:
        response = client.get("/admin")

    assert response.status_code == 200
    assert 'id="pool-section"' in response.text
    assert "/admin/api/embedding/pools" in response.text
    assert 'id="pool-reload-btn"' in response.text


# ----------------------------------------------------------------------
# 11. Admin API: pool and member management
# ----------------------------------------------------------------------
@contextmanager
def admin_client(tmp_path, monkeypatch, cfg=None):
    cfg = cfg if cfg is not None else config([member("a")])
    (tmp_path / "backends.json").write_text(json.dumps(BACKENDS), encoding="utf-8")
    (tmp_path / "embedding_pools.json").write_text(json.dumps(cfg), encoding="utf-8")
    templates_dir = tmp_path / "templates"
    templates_dir.mkdir()
    shutil.copy2(ROUTER_DIR / "templates" / "admin.html", templates_dir / "admin.html")

    monkeypatch.chdir(tmp_path)
    sys.modules.pop("router", None)
    router = importlib.import_module("router")
    router.API_KEY = "test"

    async def no_cache_update():
        return None

    async def empty_cache():
        return {}

    router.update_models_cache = no_cache_update
    router.get_models_cache = empty_cache

    from fastapi.testclient import TestClient

    with TestClient(router.app) as client:
        yield router, client


AUTH = {"Authorization": "Bearer test"}


def pool_status(client, name):
    response = client.get("/admin/api/embedding/pools", headers=AUTH)
    assert response.status_code == 200
    for pool in response.json()["pools"]:
        if pool["name"] == name:
            return pool
    return None


def test_admin_create_pool_then_add_member(tmp_path, monkeypatch):
    with admin_client(tmp_path, monkeypatch) as (_router, client):
        response = client.post(
            "/admin/api/embedding/pools",
            json={"name": "new-pool", "embedding_model": "test-embed", "dimensions": 4},
            headers=AUTH,
        )
        assert response.status_code == 200, response.text

        pool = pool_status(client, "new-pool")
        assert pool is not None
        assert pool["enabled"] is False
        assert pool["disabled_reason"] == "no_valid_members"

        response = client.post(
            "/admin/api/embedding/pools/new-pool/members",
            json=member("b", weight=2, role="backup"),
            headers=AUTH,
        )
        assert response.status_code == 200, response.text

        pool = pool_status(client, "new-pool")
        assert pool["enabled"] is True
        assert [m["backend"] for m in pool["members"]] == ["b"]
        assert pool["members"][0]["weight"] == 2
        assert pool["members"][0]["role"] == "backup"

        saved = json.loads((tmp_path / "embedding_pools.json").read_text(encoding="utf-8"))
        assert saved["pools"]["new-pool"]["members"][0]["backend"] == "b"


def test_admin_create_pool_with_initial_member(tmp_path, monkeypatch):
    with admin_client(tmp_path, monkeypatch) as (_router, client):
        response = client.post(
            "/admin/api/embedding/pools",
            json={
                "name": "embed-4",
                "embedding_model": "test-embed",
                "dimensions": 4,
                "members": [member("a", weight=3)],
            },
            headers=AUTH,
        )
        assert response.status_code == 200, response.text
        pool = pool_status(client, "embed-4")
        assert pool["enabled"] is True
        assert pool["members"][0]["backend"] == "a"
        assert pool["members"][0]["weight"] == 3


def test_admin_rejects_invalid_pool_and_member(tmp_path, monkeypatch):
    with admin_client(tmp_path, monkeypatch) as (_router, client):
        # Duplicate name.
        response = client.post(
            "/admin/api/embedding/pools",
            json={"name": "p", "embedding_model": "test-embed", "dimensions": 4},
            headers=AUTH,
        )
        assert response.status_code == 409

        # Invalid name.
        response = client.post(
            "/admin/api/embedding/pools",
            json={"name": "bad name", "embedding_model": "test-embed", "dimensions": 4},
            headers=AUTH,
        )
        assert response.status_code == 400

        # Duplicate backend inside a new pool.
        response = client.post(
            "/admin/api/embedding/pools",
            json={
                "name": "dup-pool",
                "embedding_model": "test-embed",
                "dimensions": 4,
                "members": [member("a"), member("a", weight=2)],
            },
            headers=AUTH,
        )
        assert response.status_code == 400
        assert "duplicate" in response.json()["detail"]

        # Unknown backend.
        response = client.post(
            "/admin/api/embedding/pools/p/members",
            json=member("unknown"),
            headers=AUTH,
        )
        assert response.status_code == 400
        assert "unknown_backend" in response.json()["detail"]

        # Model mismatch.
        response = client.post(
            "/admin/api/embedding/pools/p/members",
            json={"backend": "b", "model": "other-model"},
            headers=AUTH,
        )
        assert response.status_code == 400
        assert "model_mismatch" in response.json()["detail"]

        # Duplicate member.
        response = client.post(
            "/admin/api/embedding/pools/p/members",
            json=member("a"),
            headers=AUTH,
        )
        assert response.status_code == 409


def test_admin_update_and_remove_member_and_pool(tmp_path, monkeypatch):
    with admin_client(tmp_path, monkeypatch) as (_router, client):
        response = client.put(
            "/admin/api/embedding/pools/p/members/a",
            json={"weight": 5, "role": "backup"},
            headers=AUTH,
        )
        assert response.status_code == 200, response.text
        pool = pool_status(client, "p")
        assert pool["members"][0]["weight"] == 5
        assert pool["members"][0]["role"] == "backup"

        response = client.delete(
            "/admin/api/embedding/pools/p/members/a", headers=AUTH
        )
        assert response.status_code == 200, response.text
        pool = pool_status(client, "p")
        assert pool["members"] == []
        assert pool["enabled"] is False

        response = client.delete("/admin/api/embedding/pools/p", headers=AUTH)
        assert response.status_code == 200, response.text
        assert pool_status(client, "p") is None

        response = client.delete("/admin/api/embedding/pools/p", headers=AUTH)
        assert response.status_code == 404

        saved = json.loads((tmp_path / "embedding_pools.json").read_text(encoding="utf-8"))
        assert saved["pools"] == {}


def test_admin_delete_default_pool_clears_reference(tmp_path, monkeypatch):
    cfg = config([member("a")], default_pool="p")
    with admin_client(tmp_path, monkeypatch, cfg) as (_router, client):
        response = client.delete("/admin/api/embedding/pools/p", headers=AUTH)
        assert response.status_code == 200, response.text
        saved = json.loads((tmp_path / "embedding_pools.json").read_text(encoding="utf-8"))
        assert saved["default_pool"] is None