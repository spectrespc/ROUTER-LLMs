from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.templating import Jinja2Templates
import httpx
import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List, Any, Tuple
import asyncio
from contextlib import asynccontextmanager
import logging
import traceback

from embedding_pool import (
    EmbeddingPoolManager,
    MemberExistsError,
    MemberNotFoundError,
    PoolConfigError,
    PoolExistsError,
    PoolNotFoundError,
)

__version__ = "1.2.0"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

LLAMA_EMBEDDED_URL = os.getenv("LLAMA_EMBEDDED_URL", "http://127.0.0.1:8002")
BACKENDS_FILE = Path("backends.json")
_backends_lock = asyncio.Lock()

DEFAULT_BACKENDS = {
    "llama-coder-1": "http://llama-coder-1:8080",
    "llama-coder-2": "http://llama-coder-2:8080",
    "llama-embedded": LLAMA_EMBEDDED_URL
}

def load_backends() -> Dict[str, str]:
    """Загружает бэкенды из JSON-файла. Если файла нет или он повреждён — создаёт с дефолтными значениями."""
    if BACKENDS_FILE.is_file():
        try:
            with open(BACKENDS_FILE) as f:
                data = json.load(f)
            if isinstance(data, dict) and all(
                isinstance(k, str) and isinstance(v, str) for k, v in data.items()
            ):
                return data
            logger.error("backends.json has invalid shape. Using defaults.")
        except Exception as e:
            logger.error(f"Failed to load backends.json: {e}. Using defaults.")
        # Сохраняем повреждённый файл для разбирательства, не перезаписывая его молча.
        try:
            BACKENDS_FILE.replace(BACKENDS_FILE.with_name(BACKENDS_FILE.name + ".corrupt"))
        except OSError as e:
            logger.warning(f"Could not preserve corrupt backends.json: {e}")
    try:
        save_backends(DEFAULT_BACKENDS)
    except Exception as e:
        logger.error(f"Failed to persist default backends.json: {e}")
    return dict(DEFAULT_BACKENDS)

def save_backends(backends: Dict[str, str]) -> None:
    """Сохраняет бэкенды в JSON-файл атомарно (temp-файл + os.replace)."""
    data = json.dumps(backends, indent=2, ensure_ascii=False)
    tmp_file = BACKENDS_FILE.with_name(BACKENDS_FILE.name + ".tmp")
    try:
        with open(tmp_file, "w") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, BACKENDS_FILE)
    except OSError as e:
        # Например, single-file bind mount: rename поверх mountpoint возвращает EBUSY.
        logger.warning(f"Atomic write unavailable for {BACKENDS_FILE}: {e}. Writing in place.")
        try:
            tmp_file.unlink(missing_ok=True)
        except OSError:
            pass
        with open(BACKENDS_FILE, "w") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

# Глобальный словарь бэкендов — загружается при старте
BACKENDS: Dict[str, str] = load_backends()

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not Path("templates/admin.html").is_file():
        logger.error("templates/admin.html not found — the /admin panel will return HTTP 500.")
    await update_models_cache()
    await embedding_pool_manager.load(BACKENDS)
    yield

app = FastAPI(title="Heterogeneous LLM Router", lifespan=lifespan)
security = HTTPBearer()

API_KEY = os.getenv("API_KEY", "your-default-fallback-key")

templates = Jinja2Templates(directory="templates")

MODELS_CACHE: Dict[str, Any] = {}
CACHE_TTL = 60
LAST_CACHE_UPDATE = 0

# Пул embedding-моделей загружается в lifespan и перечитывается по mtime.
embedding_pool_manager = EmbeddingPoolManager(Path("embedding_pools.json"))

def clean_model_name(model_path: str, backend_name: str) -> str:
    try:
        name = str(model_path).replace("/models/", "").replace(".gguf", "")
        if "." in name:
            last_dot_index = name.rfind(".")
            name = name[:last_dot_index] + "-" + name[last_dot_index + 1:]
        return f"{name}@{backend_name}"
    except Exception as e:
        logger.error(f"Error cleaning model name {model_path}: {e}")
        return f"unknown_model@{backend_name}"

def parse_model_name(model_id: str) -> Tuple[str, str]:
    try:
        model_str = str(model_id).strip()
        if not model_str:
            return "", ""
        if "@" in model_str:
            parts = model_str.rsplit("@", 1)
            return parts[0].strip(), parts[1].strip()
        return model_str, ""
    except Exception as e:
        logger.error(f"Error parsing model name {model_id}: {e}")
        return str(model_id), ""

def match_cached_model(model_query: str, models: list) -> str | None:
    """
    Ищет модель в списке кэшированных моделей.
    Сначала точное совпадение, затем по префиксу.
    Возвращает cached_model_name или None.
    """
    model_query = str(model_query).strip()
    query_lower = model_query.lower()
    if not query_lower:
        return None

    # Точное совпадение
    for m in models:
        cached_name, _ = parse_model_name(m.get("id", ""))
        if cached_name.lower() == query_lower:
            return cached_name

    # Совпадение по префиксу
    matches = []
    for m in models:
        cached_name, _ = parse_model_name(m.get("id", ""))
        if cached_name.lower().startswith(query_lower):
            matches.append(cached_name)

    if matches:
        if len(matches) > 1:
            logger.warning(f"Multiple models match prefix '{model_query}': {matches}. Using first match.")
        return matches[0]

    return None

def find_backend_with_model(model_name: str) -> str | None:
    """
    Ищет модель среди всех online-бэкендов.
    Приоритет: точное совпадение по всем бэкендам, затем префикс.
    Возвращает имя бэкенда или None.
    """
    query_lower = str(model_name).strip().lower()
    if not query_lower:
        return None

    prefix_matches = []
    for backend_name, backend_info in MODELS_CACHE.items():
        if backend_info.get("status") != "online":
            continue
        for m in backend_info.get("models", []):
            cached_name, _ = parse_model_name(m.get("id", ""))
            cached_lower = cached_name.lower()
            if cached_lower == query_lower:
                return backend_name
            if cached_lower.startswith(query_lower):
                prefix_matches.append(backend_name)

    if prefix_matches:
        if len(prefix_matches) > 1:
            logger.warning(f"Model prefix '{model_name}' matches multiple backends: {prefix_matches}. Using first match '{prefix_matches[0]}'.")
        return prefix_matches[0]

    return None

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

async def fetch_backend_info(backend_name: str, backend_url: str) -> Dict[str, Any]:
    # УВЕЛИЧИВАЕМ ТАЙМАУТ с 5.0 до 15.0 секунд, чтобы занятый сервер успевал ответить
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            models_task = client.get(f"{backend_url}/v1/models")
            props_task = client.get(f"{backend_url}/props")
            health_task = client.get(f"{backend_url}/health")
            
            models_resp, props_resp, health_resp = await asyncio.gather(
                models_task, props_task, health_task, return_exceptions=True
            )
            
            models_data = models_resp.json() if not isinstance(models_resp, Exception) and models_resp.status_code == 200 else {}
            props_data = props_resp.json() if not isinstance(props_resp, Exception) and props_resp.status_code == 200 else {}
            health_data = health_resp.json() if not isinstance(health_resp, Exception) and health_resp.status_code == 200 else {}
            
            models_list = []
            for model in models_data.get("data", []):
                original_id = str(model.get("id", ""))
                clean_id = clean_model_name(original_id, backend_name)
                
                model["id"] = clean_id
                model["original_path"] = original_id
                model["backend"] = backend_name
                models_list.append(model)
            
            return {
                "backend": backend_name,
                "status": "online" if health_data.get("status") == "ok" else "offline",
                "models": models_list,
                "server_params": {
                    "n_ctx": props_data.get("n_ctx"),
                    "embedding": props_data.get("embedding", False),
                },
                "health": health_data
            }
        except Exception as e:
            logger.warning(f"Temporary check failed for {backend_name}: {e}. Marking as offline temporarily.")
            return {"backend": backend_name, "backend_url": backend_url, "status": "offline", "error": str(e), "models": []}

async def update_models_cache():
    global MODELS_CACHE, LAST_CACHE_UPDATE
    async with _backends_lock:
        backends_snapshot = dict(BACKENDS)
    tasks = [fetch_backend_info(name, url) for name, url in backends_snapshot.items()]
    results = await asyncio.gather(*tasks)
    async with _backends_lock:
        MODELS_CACHE = {r["backend"]: r for r in results if r["backend"] in BACKENDS}
        LAST_CACHE_UPDATE = asyncio.get_event_loop().time()
    total = sum(len(b.get('models', [])) for b in MODELS_CACHE.values())
    logger.info(f"Models cache updated. Found {total} models.")

async def get_models_cache() -> Dict[str, Any]:
    global LAST_CACHE_UPDATE
    current_time = asyncio.get_event_loop().time()
    if current_time - LAST_CACHE_UPDATE > CACHE_TTL or not MODELS_CACHE:
        await update_models_cache()
    return MODELS_CACHE

def _get_first_backend() -> Tuple[str, str]:
    """Возвращает (url, name) первого доступного бэкенда или поднимает 503."""
    if not BACKENDS:
        raise HTTPException(status_code=503, detail="No backends configured")
    first_name = next(iter(BACKENDS))
    return BACKENDS[first_name], first_name

def select_backend_by_model(model: str) -> Tuple[str, str]:
    model_str = str(model).strip()
    model_name, specified_backend = parse_model_name(model_str)
    model_lower = model_name.lower()

    if not model_name and not specified_backend:
        raise HTTPException(status_code=400, detail="Model name must not be empty.")

    # 0. Имя бэкенда как алиас модели, реально загруженной в этом бэкенде
    #    (например "llama-embedded" -> что бы ни было загружено в llama-embedded)
    for backend_name in BACKENDS:
        if backend_name.lower() == model_lower:
            return BACKENDS[backend_name], backend_name

    # 1. Если бэкенд указан явно через @
    if specified_backend:
        if specified_backend in BACKENDS:
            backend_info = MODELS_CACHE.get(specified_backend, {})
            if backend_info.get("status") == "online":
                if not model_name or match_cached_model(model_name, backend_info.get("models", [])):
                    return BACKENDS[specified_backend], specified_backend
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found on backend '{specified_backend}' or backend is offline.")
    
    # 2. Поиск по кэшу без явного указания бэкенда
    matched_backend = find_backend_with_model(model_name)
    if matched_backend and matched_backend in BACKENDS:
        return BACKENDS[matched_backend], matched_backend
    
    # 3. Fallback эвристика по имени модели
    if "qwen3" in model_lower or "30b" in model_lower or "coder" in model_lower:
        if "llama-coder-1" in BACKENDS:
            return BACKENDS["llama-coder-1"], "llama-coder-1"
    elif "nomic" in model_lower or "embed" in model_lower:
        if "llama-embedded" in BACKENDS:
            return BACKENDS["llama-embedded"], "llama-embedded"

    # Последний fallback: первый доступный бэкенд
    url, name = _get_first_backend()
    return url, name

# ==========================================
# ADMIN PANEL API
# ==========================================

@app.get("/admin")
async def admin_panel(request: Request):
    """Отдаёт HTML-страницу админ-панели."""
    return templates.TemplateResponse(request, "admin.html")

def _is_valid_backend_url(url: str) -> bool:
    """Проверяет, что URL — валидный http(s) без учётных данных во встраивании."""
    try:
        parsed = httpx.URL(url)
    except Exception:
        return False
    return parsed.scheme in ("http", "https") and bool(parsed.host) and not parsed.userinfo

@app.get("/admin/api/backends")
async def admin_list_backends(api_key: str = Depends(verify_api_key)):
    """Список всех бэкендов с их текущим статусом."""
    cache = await get_models_cache()
    result = []
    for name, url in list(BACKENDS.items()):
        info = cache.get(name, {})
        result.append({
            "name": name,
            "url": url,
            "status": info.get("status", "unknown"),
            "models": [m.get("id") for m in info.get("models", [])],
            "model_count": len(info.get("models", [])),
            "server_params": info.get("server_params", {}),
            "error": info.get("error"),
        })
    return {"backends": result}

@app.post("/admin/api/backends")
async def admin_add_backend(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """Добавляет новый бэкенд."""
    body = await request.json()
    name = str(body.get("name", "")).strip()
    url = str(body.get("url", "")).strip()

    if not name:
        raise HTTPException(status_code=400, detail="Backend name is required")
    if not url:
        raise HTTPException(status_code=400, detail="Backend URL is required")
    if not _is_valid_backend_url(url):
        raise HTTPException(
            status_code=400,
            detail="Backend URL must be a valid http(s) URL without embedded credentials"
        )
    if not re.match(r'^[a-zA-Z0-9_][a-zA-Z0-9_-]{0,62}$', name):
        raise HTTPException(
            status_code=400,
            detail="Backend name must be 1-63 chars, start with alphanumeric/underscore, "
                   "contain only [a-zA-Z0-9_-], and must not contain '@'"
        )

    async with _backends_lock:
        if name in BACKENDS:
            raise HTTPException(status_code=409, detail=f"Backend '{name}' already exists")
        BACKENDS[name] = url
        save_backends(dict(BACKENDS))

    await update_models_cache()
    logger.info(f"Backend '{name}' added: {url}")
    return {"status": "ok", "name": name, "url": url}

@app.delete("/admin/api/backends/{name}")
async def admin_remove_backend(
    name: str,
    api_key: str = Depends(verify_api_key)
):
    """Удаляет бэкенд по имени."""
    if name not in BACKENDS:
        raise HTTPException(status_code=404, detail=f"Backend '{name}' not found")

    async with _backends_lock:
        del BACKENDS[name]
        save_backends(dict(BACKENDS))
        if name in MODELS_CACHE:
            del MODELS_CACHE[name]

    logger.info(f"Backend '{name}' removed.")
    return {"status": "ok", "name": name}

@app.post("/admin/api/cache/refresh")
async def admin_refresh_cache(api_key: str = Depends(verify_api_key)):
    """Принудительно обновляет кэш моделей."""
    await update_models_cache()
    total = sum(len(b.get("models", [])) for b in MODELS_CACHE.values())
    return {"status": "ok", "total_models": total}

@app.get("/admin/api/embedding/pools")
async def admin_embedding_pools(api_key: str = Depends(verify_api_key)):
    """Полный статус embedding-пулов и их участников."""
    cache = await get_models_cache()
    return embedding_pool_manager.status(BACKENDS, cache)

@app.post("/admin/api/embedding/pools/reload")
async def admin_embedding_pools_reload(api_key: str = Depends(verify_api_key)):
    """Принудительно перечитывает embedding_pools.json."""
    reloaded = await embedding_pool_manager.load(BACKENDS)
    return {
        "status": "ok",
        "reloaded": reloaded,
        "pools": embedding_pool_manager.list_pools(),
        "last_load_error": embedding_pool_manager.status(BACKENDS)["last_load_error"],
    }

def _embedding_config_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, (PoolExistsError, MemberExistsError)):
        status = 409
    elif isinstance(exc, (PoolNotFoundError, MemberNotFoundError)):
        status = 404
    elif isinstance(exc, PoolConfigError):
        status = 400
    else:
        status = 500
    return HTTPException(status_code=status, detail=str(exc))


async def _read_admin_json(request: Request) -> Dict[str, Any]:
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON")
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object")
    return body


@app.post("/admin/api/embedding/pools")
async def admin_create_embedding_pool(
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    """Создаёт пул (опционально с первым участником)."""
    body = await _read_admin_json(request)
    try:
        await embedding_pool_manager.create_pool(
            name=body.get("name"),
            embedding_model=body.get("embedding_model"),
            dimensions=body.get("dimensions"),
            members=body.get("members") or [],
            backends=BACKENDS,
        )
    except PoolConfigError as exc:
        raise _embedding_config_http_error(exc) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail="embedding_pools.json not found"
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write embedding_pools.json: {exc}",
        ) from exc
    logger.info(f"Embedding pool '{body.get('name')}' created via admin API.")
    return {"status": "ok", "pools": embedding_pool_manager.list_pools()}


@app.delete("/admin/api/embedding/pools/{pool}")
async def admin_delete_embedding_pool(
    pool: str,
    api_key: str = Depends(verify_api_key),
):
    """Удаляет пул из embedding_pools.json."""
    try:
        await embedding_pool_manager.delete_pool(pool, backends=BACKENDS)
    except PoolConfigError as exc:
        raise _embedding_config_http_error(exc) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail="embedding_pools.json not found"
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write embedding_pools.json: {exc}",
        ) from exc
    logger.info(f"Embedding pool '{pool}' deleted via admin API.")
    return {"status": "ok", "pools": embedding_pool_manager.list_pools()}


@app.post("/admin/api/embedding/pools/{pool}/members")
async def admin_add_embedding_pool_member(
    pool: str,
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    """Добавляет участника (backend + model) в пул."""
    body = await _read_admin_json(request)
    try:
        await embedding_pool_manager.add_member(pool, body, backends=BACKENDS)
    except PoolConfigError as exc:
        raise _embedding_config_http_error(exc) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail="embedding_pools.json not found"
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write embedding_pools.json: {exc}",
        ) from exc
    logger.info(f"Member '{body.get('backend')}' added to pool '{pool}' via admin API.")
    return {"status": "ok", "pool": pool, "backend": body.get("backend")}


@app.put("/admin/api/embedding/pools/{pool}/members/{backend}")
async def admin_update_embedding_pool_member(
    pool: str,
    backend: str,
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    """Обновляет model/weight/role участника пула."""
    body = await _read_admin_json(request)
    try:
        await embedding_pool_manager.update_member(
            pool, backend, body, backends=BACKENDS
        )
    except PoolConfigError as exc:
        raise _embedding_config_http_error(exc) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail="embedding_pools.json not found"
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write embedding_pools.json: {exc}",
        ) from exc
    logger.info(f"Member '{backend}' of pool '{pool}' updated via admin API.")
    return {"status": "ok", "pool": pool, "backend": backend}


@app.delete("/admin/api/embedding/pools/{pool}/members/{backend}")
async def admin_remove_embedding_pool_member(
    pool: str,
    backend: str,
    api_key: str = Depends(verify_api_key),
):
    """Удаляет участника из пула."""
    try:
        await embedding_pool_manager.remove_member(pool, backend, backends=BACKENDS)
    except PoolConfigError as exc:
        raise _embedding_config_http_error(exc) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404, detail="embedding_pools.json not found"
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to write embedding_pools.json: {exc}",
        ) from exc
    logger.info(f"Member '{backend}' removed from pool '{pool}' via admin API.")
    return {"status": "ok", "pool": pool, "backend": backend}


@app.post("/admin/api/embedding/pools/{pool}/members/{backend}/reset")
async def admin_embedding_pool_reset(
    pool: str,
    backend: str,
    api_key: str = Depends(verify_api_key),
):
    """Закрывает circuit breaker участника вручную."""
    if not embedding_pool_manager.pool_exists(pool):
        raise HTTPException(status_code=404, detail=f"Pool '{pool}' not found")
    if not await embedding_pool_manager.reset_breaker(pool, backend):
        raise HTTPException(
            status_code=404,
            detail=f"Member '{backend}' not found in pool '{pool}'",
        )
    logger.info(f"Circuit breaker reset for pool '{pool}' member '{backend}'.")
    return {"status": "ok", "pool": pool, "backend": backend}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": __version__,
        "backends": list(BACKENDS.keys()),
        "cache_age_seconds": asyncio.get_event_loop().time() - LAST_CACHE_UPDATE,
        "embedding_pools": embedding_pool_manager.health_summary(MODELS_CACHE),
    }

@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    cache = await get_models_cache()
    all_models = []
    for backend_name, backend_info in cache.items():
        models = backend_info.get("models", [])
        for model in models:
            all_models.append({
                "id": model["id"],
                "object": model.get("object", "model"),
                "backend": model["backend"],
                "created": model.get("created", 0)
            })
        # Алиас: имя бэкенда (например "llama-embedded") адресует модель,
        # реально загруженную в этом бэкенде.
        if backend_info.get("status") == "online" and models:
            all_models.append({
                "id": backend_name,
                "object": "model",
                "backend": backend_name,
                "created": models[0].get("created", 0),
                "root": models[0]["id"],
            })
    # Пулы — виртуальные модели, адресуемые по имени пула.
    all_models.extend(embedding_pool_manager.pool_model_entries())
    return {"object": "list", "data": all_models}

async def stream_proxy(target_url: str, body: dict, backend_name: str):
    timeout = httpx.Timeout(connect=10.0, read=600.0, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async with client.stream("POST", f"{target_url}/v1/chat/completions", json=body, headers={"Accept": "text/event-stream"}) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        yield line + "\n\n"
        except Exception as e:
            logger.error(f"Stream error for {backend_name}: {e}")
            yield f'data: {{"error": "Stream failed: {str(e)}"}}\n\n'
            yield 'data: [DONE]\n\n'

def get_original_model_path(
    model: str,
    backend_name: str,
    cache: Dict[str, Any] | None = None,
) -> str:
    backend_info = (cache if cache is not None else MODELS_CACHE).get(backend_name, {})
    model_name_only, _ = parse_model_name(model)
    models = backend_info.get("models", [])

    matched_name = match_cached_model(model_name_only, models)
    if matched_name:
        for cached_model in models:
            cached_name_only, _ = parse_model_name(cached_model.get("id", ""))
            if cached_name_only.lower() == matched_name.lower():
                return cached_model.get("original_path", model)

    # Имя бэкенда или неизвестное имя: подставляем реально загруженную модель,
    # т.к. llama.cpp-сервер обслуживает одну модель и игнорирует поле "model".
    if models:
        loaded_model = models[0]
        logger.info(f"Model '{model_name_only}' not found on '{backend_name}'; forwarding loaded model '{loaded_model.get('id')}'.")
        return loaded_model.get("original_path", model)

    return model

# ==========================================
# ЭНДПОИНТЫ
# ==========================================

@app.post("/v1/chat/completions")
async def route_chat(request: Request, api_key: str = Depends(verify_api_key)):
    try:
        body = await request.json()
        model = str(body.get("model", "")).strip()
        is_stream = body.get("stream", False)
        
        await get_models_cache()
        target_url, backend_name = select_backend_by_model(model)
        body["model"] = get_original_model_path(model, backend_name)
        
        if is_stream:
            return StreamingResponse(
                stream_proxy(target_url, body, backend_name),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Backend-Used": backend_name}
            )
        else:
            timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{target_url}/v1/chat/completions", json=body)
                response.raise_for_status()
                result = response.json()
                result["backend_used"] = backend_name
                result["model"] = model
                return result
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unhandled error in route_chat: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal Router Error: {str(e)}")

@app.post("/v1/embeddings")
async def route_embeddings(request: Request, api_key: str = Depends(verify_api_key)):
    try:
        body = await request.json()
        model = str(body.get("model", "")).strip()

        cache = await get_models_cache()
        await embedding_pool_manager.maybe_reload(BACKENDS)

        # Пул адресуется точным именем в поле model и имеет приоритет над
        # алиасом бэкенда. При пустом model используется default_pool.
        pool = embedding_pool_manager.resolve_pool(model)
        if pool is None and not model and embedding_pool_manager.default_pool:
            pool = embedding_pool_manager.get_pool(embedding_pool_manager.default_pool)

        if pool is not None:
            result = await embedding_pool_manager.execute(
                pool,
                body,
                cache,
                lambda member_model, backend: get_original_model_path(
                    member_model, backend, cache
                ),
                backends=BACKENDS,
                requested_model=model or pool.name,
            )
            return JSONResponse(status_code=result.status_code, content=result.payload)

        target_url, backend_name = select_backend_by_model(model)
        body["model"] = get_original_model_path(model, backend_name, cache)
        
        timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{target_url}/v1/embeddings", json=body)
            response.raise_for_status()
            result = response.json()
            result["backend_used"] = backend_name
            result["model"] = model
            return result
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unhandled error in route_embeddings: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal Router Error: {str(e)}")

@app.post("/v1/completions")
async def route_completions(request: Request, api_key: str = Depends(verify_api_key)):
    try:
        body = await request.json()
        model = str(body.get("model", "")).strip()
        is_stream = body.get("stream", False)
        
        await get_models_cache()
        target_url, backend_name = select_backend_by_model(model)
        body["model"] = get_original_model_path(model, backend_name)
        
        if is_stream:
            return StreamingResponse(
                stream_proxy(target_url, body, backend_name),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Backend-Used": backend_name}
            )
        else:
            timeout = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(f"{target_url}/v1/completions", json=body)
                response.raise_for_status()
                result = response.json()
                result["backend_used"] = backend_name
                result["model"] = model
                return result
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unhandled error in route_completions: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal Router Error: {str(e)}")
