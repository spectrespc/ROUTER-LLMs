"""Embedding pool routing: smooth weighted round-robin with circuit breaker fallback.

This module is deliberately independent from ``router.py``: it receives backend
URLs, the models cache and a model resolver as arguments so it can be unit-tested
without a running application and without import cycles.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import httpx
from fastapi import HTTPException

logger = logging.getLogger(__name__)

DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_COOLDOWN_SECONDS = 30.0
DEFAULT_ATTEMPT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_TOTAL_SECONDS = 90.0
MIN_ATTEMPT_BUDGET_SECONDS = 2.0

VALID_ROLES = ("active", "backup")
POOL_NAME_RE = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9_-]{0,62}$")


class PoolConfigError(ValueError):
    """Invalid pool or member definition supplied to the admin API."""


class PoolNotFoundError(PoolConfigError):
    """The requested pool is not present in the configuration."""


class PoolExistsError(PoolConfigError):
    """A pool with the requested name already exists."""


class MemberExistsError(PoolConfigError):
    """The backend is already a member of the pool."""


class MemberNotFoundError(PoolConfigError):
    """The backend is not a member of the pool."""


def default_client_factory(timeout: Any) -> httpx.AsyncClient:
    """Return a real httpx client. Tests inject their own factory."""
    return httpx.AsyncClient(timeout=timeout)


def normalize_model_name(value: Any) -> str:
    """Normalize a model name the same way router.clean_model_name does.

    Dots become dashes (``v1.5`` -> ``v1-5``), ``.gguf`` and ``/models/`` are
    stripped, an ``@backend`` suffix is removed and matching is case-insensitive.
    """
    text = str(value or "").strip().lower()
    if "@" in text:
        text = text.rsplit("@", 1)[0]
    text = text.replace("/models/", "").replace(".gguf", "")
    text = text.replace(".", "-")
    return text.strip()


def models_are_compatible(pool_model: Any, member_model: Any) -> bool:
    """Prefix comparison in normalized form (different quants are compatible)."""
    left = normalize_model_name(pool_model)
    right = normalize_model_name(member_model)
    if not left or not right:
        return False
    return left.startswith(right) or right.startswith(left)


def _as_positive_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def _as_positive_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


@dataclass
class MemberConfig:
    backend: str
    model: str
    weight: int = 1
    role: str = "active"
    enabled: bool = True


@dataclass
class PoolConfig:
    name: str
    embedding_model: str
    dimensions: int
    members: List[MemberConfig] = field(default_factory=list)
    failure_threshold: int = DEFAULT_FAILURE_THRESHOLD
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS
    attempt_timeout_seconds: float = DEFAULT_ATTEMPT_TIMEOUT_SECONDS
    max_total_seconds: float = DEFAULT_MAX_TOTAL_SECONDS
    enabled: bool = True
    disabled_reason: Optional[str] = None
    rejected: List[Dict[str, Any]] = field(default_factory=list)

    def member(self, backend: str) -> Optional[MemberConfig]:
        for item in self.members:
            if item.backend == backend:
                return item
        return None


@dataclass
class MemberState:
    consecutive_failures: int = 0
    open_until: float = 0.0
    probe_inflight: bool = False
    swrr_current: float = 0.0
    total_requests: int = 0
    total_failures: int = 0
    last_error: Optional[str] = None
    last_failure_at: Optional[float] = None
    last_success_at: Optional[float] = None


@dataclass
class Candidate:
    member: MemberConfig
    state: MemberState
    is_probe: bool = False
    cache_mismatch: Optional[str] = None


@dataclass
class EmbeddingResult:
    payload: Dict[str, Any]
    pool: str
    backend_used: str
    model_served: str
    attempts: int
    fallback_used: bool
    status_code: int = 200


class EmbeddingPoolManager:
    """Loads pool config, selects members and executes embedding requests."""

    def __init__(
        self,
        path: Path | str = Path("embedding_pools.json"),
        client_factory: Callable[[Any], Any] = default_client_factory,
    ) -> None:
        self.path = Path(path)
        self._client_factory = client_factory
        self._lock = asyncio.Lock()
        self._config_lock = asyncio.Lock()
        self._pools: Dict[str, PoolConfig] = {}
        self._states: Dict[Tuple[str, str], MemberState] = {}
        self._defaults: Dict[str, float] = {
            "failure_threshold": DEFAULT_FAILURE_THRESHOLD,
            "cooldown_seconds": DEFAULT_COOLDOWN_SECONDS,
            "attempt_timeout_seconds": DEFAULT_ATTEMPT_TIMEOUT_SECONDS,
            "max_total_seconds": DEFAULT_MAX_TOTAL_SECONDS,
        }
        self._default_pool: Optional[str] = None
        self._mtime: Optional[float] = None
        self._loaded = False
        self._last_load_error: Optional[str] = None
        self._loaded_at: Optional[float] = None

    # ------------------------------------------------------------------
    # Loading and validation
    # ------------------------------------------------------------------
    @property
    def default_pool(self) -> Optional[str]:
        return self._default_pool

    @property
    def loaded(self) -> bool:
        return self._loaded

    def list_pools(self) -> List[str]:
        return list(self._pools.keys())

    def get_pool(self, name: str) -> Optional[PoolConfig]:
        return self._pools.get(str(name).strip())

    def pool_exists(self, name: str) -> bool:
        return str(name).strip() in self._pools

    def resolve_pool(self, model: Any) -> Optional[PoolConfig]:
        """Return the pool addressed by an exact model name, if any."""
        name = str(model or "").strip()
        if not name:
            return None
        return self._pools.get(name)

    async def load(self, backends: Optional[Dict[str, str]] = None) -> bool:
        """Read the config file and atomically swap the active configuration.

        A corrupt file is preserved as ``<name>.corrupt`` and the previous
        configuration stays active.  Member state survives a reload for the
        same ``(pool, backend)`` pair.
        """
        try:
            raw, mtime = self._read_config()
        except FileNotFoundError:
            logger.warning("Embedding pool config %s not found; pools are disabled.", self.path)
            async with self._lock:
                self._pools = {}
                self._states = {}
                self._default_pool = None
                self._mtime = None
                self._loaded = True
                self._last_load_error = "config file not found"
                self._loaded_at = time.time()
            return False
        except Exception as exc:  # malformed JSON / unreadable file
            logger.error("Failed to load %s: %s. Keeping previous configuration.", self.path, exc)
            self._preserve_corrupt_file()
            async with self._lock:
                self._mtime = None
                self._loaded = True
                self._last_load_error = str(exc)
            return False

        try:
            pools, defaults, default_pool = self._parse_config(raw, backends)
        except Exception as exc:
            logger.error("Invalid embedding pool config shape: %s. Keeping previous configuration.", exc)
            self._preserve_corrupt_file()
            async with self._lock:
                self._mtime = None
                self._loaded = True
                self._last_load_error = str(exc)
            return False

        async with self._lock:
            new_states: Dict[Tuple[str, str], MemberState] = {}
            for pool in pools.values():
                for member in pool.members:
                    key = (pool.name, member.backend)
                    new_states[key] = self._states.get(key, MemberState())
                    # A rejected/unknown backend can not keep a live state.
            self._pools = pools
            self._states = new_states
            self._defaults = defaults
            self._default_pool = default_pool
            self._mtime = mtime
            self._loaded = True
            self._last_load_error = None
            self._loaded_at = time.time()

        for pool in pools.values():
            if pool.name in (backends or {}):
                logger.warning(
                    "Embedding pool name '%s' collides with a backend name; the pool wins in routing.",
                    pool.name,
                )
            for rejected in pool.rejected:
                logger.warning(
                    "Pool '%s': rejected member %s (%s)",
                    pool.name,
                    rejected.get("backend"),
                    rejected.get("reason"),
                )
        logger.info("Embedding pools loaded: %s", ", ".join(pools) or "none")
        return True

    # ------------------------------------------------------------------
    # Configuration management (admin API)
    # ------------------------------------------------------------------
    def _read_raw_config(self, allow_missing: bool = False) -> Dict[str, Any]:
        try:
            data, _ = self._read_config()
        except FileNotFoundError:
            if not allow_missing:
                raise
            return {
                "defaults": dict(self._defaults),
                "default_pool": None,
                "pools": {},
            }
        return data

    def _write_raw_config(self, config: Dict[str, Any]) -> None:
        """Atomic write with an in-place fallback for single-file bind mounts."""
        data = json.dumps(config, indent=2, ensure_ascii=False)
        tmp_file = self.path.with_name(self.path.name + ".tmp")
        try:
            with open(tmp_file, "w", encoding="utf-8") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_file, self.path)
        except OSError as exc:
            # Single-file bind mount: rename over the mountpoint returns EBUSY.
            logger.warning(
                "Atomic write unavailable for %s: %s. Writing in place.", self.path, exc
            )
            try:
                tmp_file.unlink(missing_ok=True)
            except OSError:
                pass
            with open(self.path, "w", encoding="utf-8") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())

    @staticmethod
    def _validate_pool_name(name: Any) -> str:
        value = str(name or "").strip()
        if not POOL_NAME_RE.match(value):
            raise PoolConfigError(
                "Pool name must be 1-63 chars, start with alphanumeric/underscore, "
                "and contain only [a-zA-Z0-9_-]"
            )
        return value

    @staticmethod
    def _normalize_member(member: Any) -> Dict[str, Any]:
        if not isinstance(member, dict):
            raise PoolConfigError("Member must be an object")
        backend = str(member.get("backend", "")).strip()
        model = str(member.get("model", "")).strip()
        if not backend:
            raise PoolConfigError("member.backend is required")
        if not model:
            raise PoolConfigError("member.model is required")
        try:
            weight = int(member.get("weight", 1))
        except (TypeError, ValueError):
            raise PoolConfigError("member.weight must be a positive integer") from None
        if weight < 1:
            raise PoolConfigError("member.weight must be a positive integer")
        role = str(member.get("role", "active")).strip().lower() or "active"
        if role not in VALID_ROLES:
            raise PoolConfigError("member.role must be 'active' or 'backup'")
        enabled = member.get("enabled", True)
        if not isinstance(enabled, bool):
            raise PoolConfigError("member.enabled must be true or false")
        if not enabled:
            raise PoolConfigError(
                "member.enabled=false is not supported by the admin API; remove the member instead"
            )
        return {
            "backend": backend,
            "model": model,
            "weight": weight,
            "role": role,
            "enabled": True,
        }

    def _find_raw_member(
        self, entry: Dict[str, Any], backend: str
    ) -> Optional[Dict[str, Any]]:
        members = entry.get("members") or []
        if not isinstance(members, list):
            return None
        for item in members:
            if isinstance(item, dict) and str(item.get("backend", "")).strip() == backend:
                return item
        return None

    async def _commit(
        self,
        raw: Dict[str, Any],
        backends: Optional[Dict[str, str]],
        pool_name: str,
        expect_member: Optional[str] = None,
        allow_empty: bool = False,
    ) -> PoolConfig:
        """Validate the mutated config, persist it and reload the manager."""
        pools, _, _ = self._parse_config(raw, backends)
        pool = pools.get(pool_name)
        if pool is None:
            raise PoolConfigError(f"Pool '{pool_name}' is invalid after the change")
        if expect_member and pool.member(expect_member) is None:
            reason = next(
                (
                    item.get("reason")
                    for item in pool.rejected
                    if item.get("backend") == expect_member
                ),
                "invalid_member",
            )
            raise PoolConfigError(f"member '{expect_member}' rejected: {reason}")
        if not pool.members and not allow_empty:
            raise PoolConfigError("Pool must have at least one valid member")
        self._write_raw_config(raw)
        await self.load(backends)
        logger.info("Embedding pool config updated: %s", pool_name)
        return pool

    async def create_pool(
        self,
        name: Any,
        embedding_model: Any,
        dimensions: Any,
        members: Any = None,
        backends: Optional[Dict[str, str]] = None,
    ) -> None:
        pool_name = self._validate_pool_name(name)
        model = str(embedding_model or "").strip()
        if not model:
            raise PoolConfigError("embedding_model is required")
        try:
            dims = int(dimensions)
        except (TypeError, ValueError):
            raise PoolConfigError("dimensions must be a positive integer") from None
        if dims < 1:
            raise PoolConfigError("dimensions must be a positive integer")
        if members is None:
            members = []
        if not isinstance(members, list):
            raise PoolConfigError("members must be a list")
        normalized = [self._normalize_member(item) for item in members]
        seen_backends = set()
        for item in normalized:
            if item["backend"] in seen_backends:
                raise PoolConfigError(
                    f"duplicate member backend '{item['backend']}'"
                )
            seen_backends.add(item["backend"])

        async with self._config_lock:
            raw = self._read_raw_config(allow_missing=True)
            raw_pools = raw.setdefault("pools", {})
            if not isinstance(raw_pools, dict):
                raise PoolConfigError("'pools' must be an object")
            if pool_name in raw_pools:
                raise PoolExistsError(f"Pool '{pool_name}' already exists")
            raw_pools[pool_name] = {
                "embedding_model": model,
                "dimensions": dims,
                "members": normalized,
            }
            # Validate every requested member; _commit only checks one.
            pools, _, _ = self._parse_config(raw, backends)
            created = pools.get(pool_name)
            if created is None:
                raise PoolConfigError(f"Pool '{pool_name}' is invalid")
            for item in normalized:
                if created.member(item["backend"]) is None:
                    reason = next(
                        (
                            rejected.get("reason")
                            for rejected in created.rejected
                            if rejected.get("backend") == item["backend"]
                        ),
                        "invalid_member",
                    )
                    raise PoolConfigError(
                        f"member '{item['backend']}' rejected: {reason}"
                    )
            await self._commit(
                raw,
                backends,
                pool_name,
                allow_empty=not normalized,
            )

    async def delete_pool(
        self, name: Any, backends: Optional[Dict[str, str]] = None
    ) -> None:
        pool_name = str(name or "").strip()
        async with self._config_lock:
            raw = self._read_raw_config()
            raw_pools = raw.get("pools") or {}
            if not isinstance(raw_pools, dict) or pool_name not in raw_pools:
                raise PoolNotFoundError(f"Pool '{pool_name}' not found")
            del raw_pools[pool_name]
            if raw.get("default_pool") == pool_name:
                raw["default_pool"] = None
            self._write_raw_config(raw)
            await self.load(backends)
            logger.info("Embedding pool removed: %s", pool_name)

    async def add_member(
        self,
        pool_name: Any,
        member: Any,
        backends: Optional[Dict[str, str]] = None,
    ) -> None:
        name = str(pool_name or "").strip()
        normalized = self._normalize_member(member)
        async with self._config_lock:
            raw = self._read_raw_config()
            raw_pools = raw.get("pools") or {}
            entry = raw_pools.get(name) if isinstance(raw_pools, dict) else None
            if not isinstance(entry, dict):
                raise PoolNotFoundError(f"Pool '{name}' not found")
            raw_members = entry.setdefault("members", [])
            if not isinstance(raw_members, list):
                raise PoolConfigError(f"Pool '{name}' has invalid 'members'")
            if self._find_raw_member(entry, normalized["backend"]) is not None:
                raise MemberExistsError(
                    f"Backend '{normalized['backend']}' is already a member of pool '{name}'"
                )
            raw_members.append(normalized)
            await self._commit(raw, backends, name, expect_member=normalized["backend"])

    async def update_member(
        self,
        pool_name: Any,
        backend: Any,
        member: Any,
        backends: Optional[Dict[str, str]] = None,
    ) -> None:
        name = str(pool_name or "").strip()
        backend_name = str(backend or "").strip()
        if not isinstance(member, dict):
            raise PoolConfigError("Member must be an object")
        async with self._config_lock:
            raw = self._read_raw_config()
            raw_pools = raw.get("pools") or {}
            entry = raw_pools.get(name) if isinstance(raw_pools, dict) else None
            if not isinstance(entry, dict):
                raise PoolNotFoundError(f"Pool '{name}' not found")
            existing = self._find_raw_member(entry, backend_name)
            if existing is None:
                raise MemberNotFoundError(
                    f"Backend '{backend_name}' is not a member of pool '{name}'"
                )
            merged = dict(existing)
            for key in ("model", "weight", "role", "enabled"):
                if key in member:
                    merged[key] = member[key]
            merged["backend"] = backend_name
            normalized = self._normalize_member(merged)
            raw_members = entry.get("members") or []
            entry["members"] = [
                normalized
                if isinstance(item, dict)
                and str(item.get("backend", "")).strip() == backend_name
                else item
                for item in raw_members
            ]
            await self._commit(raw, backends, name, expect_member=backend_name)

    async def remove_member(
        self,
        pool_name: Any,
        backend: Any,
        backends: Optional[Dict[str, str]] = None,
    ) -> None:
        name = str(pool_name or "").strip()
        backend_name = str(backend or "").strip()
        async with self._config_lock:
            raw = self._read_raw_config()
            raw_pools = raw.get("pools") or {}
            entry = raw_pools.get(name) if isinstance(raw_pools, dict) else None
            if not isinstance(entry, dict):
                raise PoolNotFoundError(f"Pool '{name}' not found")
            raw_members = entry.get("members") or []
            remaining = [
                item
                for item in raw_members
                if not (
                    isinstance(item, dict)
                    and str(item.get("backend", "")).strip() == backend_name
                )
            ]
            if len(remaining) == len(raw_members):
                raise MemberNotFoundError(
                    f"Backend '{backend_name}' is not a member of pool '{name}'"
                )
            entry["members"] = remaining
            await self._commit(raw, backends, name, allow_empty=True)

    async def maybe_reload(self, backends: Optional[Dict[str, str]] = None) -> bool:
        """Reload when the file mtime changed. Cheap enough per request."""
        try:
            current = self.path.stat().st_mtime
        except OSError:
            return False
        if self._loaded and current == self._mtime:
            return False
        return await self.load(backends)

    def _read_config(self) -> Tuple[Dict[str, Any], float]:
        stat = self.path.stat()
        with open(self.path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("top-level JSON value must be an object")
        return data, stat.st_mtime

    def _preserve_corrupt_file(self) -> None:
        target = self.path.with_name(self.path.name + ".corrupt")
        try:
            os.replace(self.path, target)
        except OSError as exc:
            logger.warning("Could not preserve corrupt %s: %s", self.path, exc)

    def _parse_config(
        self,
        raw: Dict[str, Any],
        backends: Optional[Dict[str, str]],
    ) -> Tuple[Dict[str, PoolConfig], Dict[str, float], Optional[str]]:
        defaults = dict(self._defaults)
        raw_defaults = raw.get("defaults", {})
        if raw_defaults is not None and not isinstance(raw_defaults, dict):
            raise ValueError("'defaults' must be an object")
        if isinstance(raw_defaults, dict):
            if "failure_threshold" in raw_defaults:
                defaults["failure_threshold"] = _as_positive_int(
                    raw_defaults["failure_threshold"], defaults["failure_threshold"]
                )
            for key in ("cooldown_seconds", "attempt_timeout_seconds", "max_total_seconds"):
                if key in raw_defaults:
                    defaults[key] = _as_positive_float(raw_defaults[key], defaults[key])

        raw_pools = raw.get("pools", {})
        if raw_pools is None:
            raw_pools = {}
        if not isinstance(raw_pools, dict):
            raise ValueError("'pools' must be an object")

        pools: Dict[str, PoolConfig] = {}
        for name, entry in raw_pools.items():
            pool_name = str(name).strip()
            if not pool_name:
                logger.warning("Skipping embedding pool with an empty name")
                continue
            if not isinstance(entry, dict):
                logger.warning("Skipping embedding pool '%s': entry is not an object", pool_name)
                continue
            try:
                pools[pool_name] = self._parse_pool(pool_name, entry, defaults, backends)
            except ValueError as exc:
                logger.warning("Skipping embedding pool '%s': %s", pool_name, exc)

        default_pool = raw.get("default_pool")
        if default_pool is not None:
            default_pool = str(default_pool).strip() or None
        if default_pool and default_pool not in pools:
            logger.warning("default_pool '%s' does not exist; disabling the default.", default_pool)
            default_pool = None
        return pools, defaults, default_pool

    def _parse_pool(
        self,
        name: str,
        entry: Dict[str, Any],
        defaults: Dict[str, float],
        backends: Optional[Dict[str, str]],
    ) -> PoolConfig:
        embedding_model = str(entry.get("embedding_model", "")).strip()
        if not embedding_model:
            raise ValueError("'embedding_model' is required")
        dimensions = _as_positive_int(entry.get("dimensions"), 0)
        if not dimensions:
            raise ValueError("'dimensions' must be a positive integer")
        raw_members = entry.get("members", [])
        if not isinstance(raw_members, list):
            raise ValueError("'members' must be a list")

        pool = PoolConfig(
            name=name,
            embedding_model=embedding_model,
            dimensions=dimensions,
            failure_threshold=_as_positive_int(
                entry.get("failure_threshold"), defaults["failure_threshold"]
            ),
            cooldown_seconds=_as_positive_float(
                entry.get("cooldown_seconds"), defaults["cooldown_seconds"]
            ),
            attempt_timeout_seconds=_as_positive_float(
                entry.get("attempt_timeout_seconds"), defaults["attempt_timeout_seconds"]
            ),
            max_total_seconds=_as_positive_float(
                entry.get("max_total_seconds"), defaults["max_total_seconds"]
            ),
        )

        seen_backends = set()
        for raw_member in raw_members:
            if not isinstance(raw_member, dict):
                pool.rejected.append({"backend": None, "model": None, "reason": "member_not_object"})
                continue
            backend = str(raw_member.get("backend", "")).strip()
            model = str(raw_member.get("model", "")).strip()
            role = str(raw_member.get("role", "active")).strip().lower() or "active"
            enabled = raw_member.get("enabled", True)
            weight = _as_positive_int(raw_member.get("weight"), 1)
            reason = None

            if not backend:
                reason = "missing_backend"
            elif backend in seen_backends:
                reason = "duplicate_backend"
            elif backends is not None and backend not in backends:
                reason = "unknown_backend"
            elif not model:
                reason = "missing_model"
            elif not models_are_compatible(embedding_model, model):
                reason = "model_mismatch"
            elif role not in VALID_ROLES:
                reason = "invalid_role"
            elif not isinstance(enabled, bool):
                reason = "invalid_enabled"
            elif not enabled:
                reason = "member_disabled"

            if reason:
                pool.rejected.append({"backend": backend or None, "model": model or None, "reason": reason})
                continue

            seen_backends.add(backend)
            pool.members.append(
                MemberConfig(backend=backend, model=model, weight=weight, role=role, enabled=True)
            )

        if not pool.members:
            pool.enabled = False
            pool.disabled_reason = "no_valid_members"
        return pool

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def _state(self, pool_name: str, backend: str) -> MemberState:
        key = (pool_name, backend)
        state = self._states.get(key)
        if state is None:
            state = MemberState()
            self._states[key] = state
        return state

    @staticmethod
    def _cache_mismatch(member: MemberConfig, cache: Optional[Dict[str, Any]]) -> Optional[str]:
        if not cache:
            return None
        info = cache.get(member.backend)
        if not isinstance(info, dict):
            return None
        models = info.get("models") or []
        if not models:
            return None
        wanted = normalize_model_name(member.model)
        for cached in models:
            cached_id = cached.get("original_path") or cached.get("id") or ""
            candidate = normalize_model_name(cached_id)
            if candidate.startswith(wanted) or wanted.startswith(candidate):
                return None
        return "cached_models_do_not_match"

    def _swrr_order(
        self,
        pool: PoolConfig,
        members: Iterable[MemberConfig],
        now: float,
    ) -> List[MemberConfig]:
        """One SWRR step: advance all weights, rank, then penalize the winner.

        Ranking after the single advance produces a full fallback order while
        consuming exactly one round-robin step per request.
        """
        members = list(members)
        if not members:
            return []
        total = sum(max(1, member.weight) for member in members)
        ranked: List[Tuple[float, int, MemberConfig]] = []
        for index, member in enumerate(members):
            state = self._state(pool.name, member.backend)
            state.swrr_current += max(1, member.weight)
            ranked.append((state.swrr_current, index, member))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        winner = ranked[0][2]
        self._state(pool.name, winner.backend).swrr_current -= total
        return [item[2] for item in ranked]

    def _candidate(
        self,
        pool: PoolConfig,
        member: MemberConfig,
        cache: Optional[Dict[str, Any]],
        *,
        probe: bool = False,
    ) -> Candidate:
        return Candidate(
            member=member,
            state=self._state(pool.name, member.backend),
            is_probe=probe,
            cache_mismatch=self._cache_mismatch(member, cache),
        )

    def _build_chain(
        self,
        pool: PoolConfig,
        cache: Optional[Dict[str, Any]],
        now: float,
    ) -> Tuple[List[Candidate], Optional[float]]:
        """Build the ordered candidate chain. Caller must hold ``self._lock``."""
        chain: List[Candidate] = []
        healthy_by_role: Dict[str, List[MemberConfig]] = {"active": [], "backup": []}
        half_open: List[MemberConfig] = []
        for member in pool.members:
            if not member.enabled:
                continue
            state = self._state(pool.name, member.backend)
            if state.open_until <= 0:
                healthy_by_role[member.role].append(member)
            elif now >= state.open_until:
                if not state.probe_inflight:
                    half_open.append(member)

        for role in ("active", "backup"):
            for member in self._swrr_order(pool, healthy_by_role[role], now):
                chain.append(self._candidate(pool, member, cache))
            # A half-open probe belongs to its own tier: healthy actives first,
            # then active probes, then healthy backups, then backup probes.
            # This lets a recovered primary reclaim traffic while a healthy
            # backup is still available.
            for member in [item for item in half_open if item.role == role]:
                state = self._state(pool.name, member.backend)
                if state.probe_inflight:
                    continue
                state.probe_inflight = True
                chain.append(self._candidate(pool, member, cache, probe=True))

        # Give cached model mismatch candidates a lower priority but do not drop
        # them: the cache may be stale and real responses are authoritative.
        chain.sort(key=lambda candidate: 0 if candidate.cache_mismatch is None else 1)

        retry_after: Optional[float] = None
        if not chain:
            waits = [
                state.open_until - now
                for member in pool.members
                for state in [self._state(pool.name, member.backend)]
                if state.open_until > now
            ]
            if waits:
                retry_after = max(0.0, min(waits))
        return chain, retry_after

    def pick_chain(
        self,
        pool: PoolConfig,
        cache: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Candidate], Optional[float]]:
        """Public helper used by tests and status tooling (not thread-safe)."""
        return self._build_chain(pool, cache, time.monotonic())

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    @staticmethod
    def _error_detail(
        code: str,
        pool: PoolConfig,
        members: List[Dict[str, Any]],
        retry_after: Optional[float] = None,
    ) -> Dict[str, Any]:
        detail: Dict[str, Any] = {
            "type": "pool_error",
            "code": code,
            "pool": pool.name,
            "members": members,
        }
        if retry_after is not None:
            detail["retry_after_seconds"] = round(retry_after, 3)
        return detail

    async def execute(
        self,
        pool: PoolConfig,
        body: Dict[str, Any],
        cache: Optional[Dict[str, Any]],
        resolve_model: Callable[[str, str], str],
        backends: Optional[Dict[str, str]] = None,
        requested_model: Optional[str] = None,
    ) -> EmbeddingResult:
        """Route one embedding request through the pool with fallback."""
        if not pool.enabled:
            raise HTTPException(
                status_code=503,
                detail=self._error_detail(
                    "pool_disabled",
                    pool,
                    [{"backend": None, "error": pool.disabled_reason or "pool_disabled"}],
                ),
            )

        start = time.monotonic()
        async with self._lock:
            chain, retry_after = self._build_chain(pool, cache, start)

        if not chain:
            headers = None
            if retry_after is not None:
                headers = {"Retry-After": str(max(1, int(round(retry_after))))}
            raise HTTPException(
                status_code=503,
                headers=headers,
                detail=self._error_detail("no_healthy_members", pool, [], retry_after),
            )

        first_backend = chain[0].member.backend
        requested = requested_model if requested_model is not None else str(body.get("model", ""))
        errors: List[Dict[str, Any]] = []

        try:
            for index, candidate in enumerate(chain):
                member = candidate.member
                remaining = pool.max_total_seconds - (time.monotonic() - start)
                if remaining < MIN_ATTEMPT_BUDGET_SECONDS:
                    await self._release_probes(chain[index:])
                    raise HTTPException(
                        status_code=504,
                        detail=self._error_detail(
                            "deadline_exceeded",
                            pool,
                            errors,
                            max(0.0, pool.max_total_seconds - (time.monotonic() - start)),
                        ),
                    )

                attempt_timeout = min(pool.attempt_timeout_seconds, remaining)
                url = (backends or {}).get(member.backend)
                if not url:
                    # Backend disappeared after config load: treat as member failure.
                    error = "backend_not_configured"
                    await self._record_failure(pool, candidate, error, start)
                    errors.append({"backend": member.backend, "error": error})
                    continue

                try:
                    resolved_model = resolve_model(member.model, member.backend)
                except Exception as exc:  # resolver must never break routing
                    logger.warning("Model resolver failed for %s: %s", member.backend, exc)
                    resolved_model = member.model

                attempt_body = dict(body)
                attempt_body["model"] = resolved_model

                try:
                    response = await self._post_embeddings(url, attempt_body, attempt_timeout)
                    status_code = response.status_code
                    if status_code == 429 or status_code >= 500:
                        error = f"HTTP {status_code}: {self._response_text(response)}"
                    elif status_code >= 400:
                        # Client error: return it transparently, do not retry and
                        # do not count it against the member's circuit breaker.
                        payload = self._response_json(response)
                        await self._release_candidate(candidate)
                        await self._record_member_request(pool, candidate, success=False)
                        if not isinstance(payload, dict):
                            payload = {"error": self._response_text(response)}
                        return EmbeddingResult(
                            payload=payload,
                            pool=pool.name,
                            backend_used=member.backend,
                            model_served=resolved_model,
                            attempts=index + 1,
                            fallback_used=member.backend != first_backend,
                            status_code=status_code,
                        )
                    else:
                        payload = self._response_json(response)
                        error = self._dimension_error(payload, pool.dimensions)
                        if error is None:
                            await self._record_success(pool, candidate, start)
                            if isinstance(payload, dict):
                                payload["model"] = requested
                                payload["pool"] = pool.name
                                payload["backend_used"] = member.backend
                                payload["model_served"] = resolved_model
                                payload["attempts"] = index + 1
                                payload["fallback_used"] = member.backend != first_backend
                            else:
                                payload = {"data": [], "model": requested}
                            return EmbeddingResult(
                                payload=payload,
                                pool=pool.name,
                                backend_used=member.backend,
                                model_served=resolved_model,
                                attempts=index + 1,
                                fallback_used=member.backend != first_backend,
                                status_code=200,
                            )
                except httpx.TimeoutException as exc:
                    error = f"timeout: {type(exc).__name__}"
                except httpx.RequestError as exc:
                    error = f"network: {type(exc).__name__}: {exc}"
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"

                logger.warning(
                    "Pool '%s': member '%s' failed (%s); trying next candidate.",
                    pool.name,
                    member.backend,
                    error,
                )
                await self._record_failure(pool, candidate, error, start)
                errors.append({"backend": member.backend, "error": error})

            raise HTTPException(
                status_code=502,
                detail=self._error_detail("all_attempts_failed", pool, errors),
            )
        finally:
            # Any probe that was claimed but never resolved must be released.
            await self._release_probes(chain)

    async def _post_embeddings(self, url: str, body: Dict[str, Any], timeout: float) -> Any:
        async with self._client_factory(timeout) as client:
            return await client.post(f"{url}/v1/embeddings", json=body)

    @staticmethod
    def _response_json(response: Any) -> Any:
        try:
            return response.json()
        except Exception:
            return None

    @staticmethod
    def _response_text(response: Any) -> str:
        try:
            text = response.text
        except Exception:
            return ""
        return str(text)[:300]

    @staticmethod
    def _dimension_error(payload: Any, dimensions: int) -> Optional[str]:
        if not isinstance(payload, dict):
            return "invalid_response: not a JSON object"
        data = payload.get("data")
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            return "invalid_response: missing data[0]"
        embedding = data[0].get("embedding")
        if not isinstance(embedding, list):
            return "invalid_response: missing embedding vector"
        if len(embedding) != dimensions:
            return f"dimension_mismatch: expected {dimensions}, got {len(embedding)}"
        return None

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------
    async def _record_success(self, pool: PoolConfig, candidate: Candidate, now: float) -> None:
        async with self._lock:
            state = candidate.state
            state.consecutive_failures = 0
            state.open_until = 0.0
            state.probe_inflight = False
            state.total_requests += 1
            state.last_success_at = now

    async def _record_failure(
        self,
        pool: PoolConfig,
        candidate: Candidate,
        error: str,
        now: float,
    ) -> None:
        async with self._lock:
            state = candidate.state
            state.consecutive_failures += 1
            state.total_requests += 1
            state.total_failures += 1
            state.last_error = error
            state.last_failure_at = now
            state.probe_inflight = False
            if state.consecutive_failures >= pool.failure_threshold:
                state.open_until = now + pool.cooldown_seconds

    async def _record_member_request(
        self, pool: PoolConfig, candidate: Candidate, *_: Any, success: bool
    ) -> None:
        async with self._lock:
            candidate.state.total_requests += 1
            candidate.state.probe_inflight = False
            if success:
                candidate.state.last_success_at = time.monotonic()

    async def _release_candidate(self, candidate: Candidate) -> None:
        async with self._lock:
            candidate.state.probe_inflight = False

    async def _release_probes(self, candidates: Iterable[Candidate]) -> None:
        async with self._lock:
            for candidate in candidates:
                if candidate.is_probe:
                    candidate.state.probe_inflight = False

    async def reset_breaker(self, pool_name: str, backend: str) -> bool:
        """Close a breaker manually. Returns False when the member is unknown."""
        pool = self.get_pool(pool_name)
        if pool is None or pool.member(backend) is None:
            return False
        async with self._lock:
            state = self._state(pool.name, backend)
            state.consecutive_failures = 0
            state.open_until = 0.0
            state.probe_inflight = False
            state.last_error = None
        return True

    async def reset_all_breakers(self) -> None:
        async with self._lock:
            for state in self._states.values():
                state.consecutive_failures = 0
                state.open_until = 0.0
                state.probe_inflight = False
                state.last_error = None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def _breaker_state(self, state: MemberState, now: float) -> str:
        if state.open_until <= 0:
            if state.probe_inflight:
                return "half_open"
            return "closed"
        if now >= state.open_until:
            return "half_open"
        return "open"

    def status(
        self,
        backends: Optional[Dict[str, str]] = None,
        cache: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        now = time.monotonic()
        pools: List[Dict[str, Any]] = []
        for pool in self._pools.values():
            members: List[Dict[str, Any]] = []
            for member in pool.members:
                state = self._state(pool.name, member.backend)
                members.append(
                    {
                        "backend": member.backend,
                        "url": (backends or {}).get(member.backend),
                        "model": member.model,
                        "role": member.role,
                        "weight": member.weight,
                        "enabled": member.enabled,
                        "breaker": self._breaker_state(state, now),
                        "consecutive_failures": state.consecutive_failures,
                        "probe_inflight": state.probe_inflight,
                        "open_for_seconds": round(max(0.0, state.open_until - now), 3),
                        "total_requests": state.total_requests,
                        "total_failures": state.total_failures,
                        "last_error": state.last_error,
                        "cache_hint": self._cache_mismatch(member, cache),
                    }
                )
            healthy = sum(1 for item in members if item["breaker"] == "closed")
            pools.append(
                {
                    "name": pool.name,
                    "embedding_model": pool.embedding_model,
                    "dimensions": pool.dimensions,
                    "enabled": pool.enabled,
                    "disabled_reason": pool.disabled_reason,
                    "failure_threshold": pool.failure_threshold,
                    "cooldown_seconds": pool.cooldown_seconds,
                    "attempt_timeout_seconds": pool.attempt_timeout_seconds,
                    "max_total_seconds": pool.max_total_seconds,
                    "members_total": len(members),
                    "members_healthy": healthy,
                    "open_breakers": sum(1 for item in members if item["breaker"] == "open"),
                    "members": members,
                    "rejected": pool.rejected,
                }
            )
        return {
            "version": 1,
            "file": str(self.path),
            "loaded": self._loaded,
            "loaded_at": self._loaded_at,
            "last_load_error": self._last_load_error,
            "default_pool": self._default_pool,
            "defaults": dict(self._defaults),
            "pools": pools,
        }

    def health_summary(self, cache: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        now = time.monotonic()
        total_members = 0
        healthy_members = 0
        open_breakers = 0
        details: Dict[str, Any] = {}
        for pool in self._pools.values():
            pool_healthy = 0
            pool_open = 0
            for member in pool.members:
                state = self._state(pool.name, member.backend)
                total_members += 1
                if state.open_until <= 0 and not state.probe_inflight:
                    healthy_members += 1
                    pool_healthy += 1
                if self._breaker_state(state, now) == "open":
                    open_breakers += 1
                    pool_open += 1
            details[pool.name] = {
                "enabled": pool.enabled,
                "healthy": pool_healthy,
                "total": len(pool.members),
                "open_breakers": pool_open,
            }
        return {
            "pools": len(self._pools),
            "members_healthy": healthy_members,
            "members_total": total_members,
            "open_breakers": open_breakers,
            "default_pool": self._default_pool,
            "details": details,
        }

    def pool_model_entries(self) -> List[Dict[str, Any]]:
        """Entries for ``GET /v1/models`` describing pools as virtual models."""
        now = time.monotonic()
        entries: List[Dict[str, Any]] = []
        for pool in self._pools.values():
            entries.append(
                {
                    "id": pool.name,
                    "object": "model",
                    "pool": True,
                    "embedding_model": pool.embedding_model,
                    "dimensions": pool.dimensions,
                    "enabled": pool.enabled,
                    "members": [
                        {
                            "backend": member.backend,
                            "role": member.role,
                            "weight": member.weight,
                            "breaker": self._breaker_state(
                                self._state(pool.name, member.backend), now
                            ),
                        }
                        for member in pool.members
                    ],
                }
            )
        return entries