# ROUTER — гетерогенный роутер llama.cpp

## Русский

`ROUTER` — FastAPI-приложение, которое маршрутизирует запросы к нескольким
llama.cpp-серверам. Версия **1.1.0** добавляет пулы embedding-моделей с
балансировкой, circuit breaker и прозрачным fallback внутри пула.

### Архитектура

```
клиент
  │  POST /v1/embeddings  {model: "embed-768"}
  ▼
┌──────────────────────── ROUTER ────────────────────────┐
│ authenticate (API_KEY)                                  │
│ embedding_pool_manager.maybe_reload()  ← mtime          │
│                                                         │
│ pool "embed-768" (nomic-embed-text-v1.5, 768d)          │
│   ├─ active  llama-embedded        weight 3  ─┐         │
│   ├─ active  llama-embedded-2      weight 1  ─┤ SWRR    │
│   ├─ active  llama-embedded-3      weight 1  ─┘         │
│   └─ backup  llama-embedded-backup weight 1             │
│                                                         │
│ для каждого кандидата:                                  │
│   POST {backend}/v1/embeddings                          │
│   timeout = min(attempt_timeout, остаток max_total)     │
│   успех + len(vector) == dimensions → ответ клиенту     │
│   сеть/таймаут/429/5xx → failure, следующий кандидат    │
│   прочие 4xx → прозрачно клиенту без retry              │
└─────────────────────────────────────────────────────────┘
```

Каждый llama.cpp-бэкенд обслуживает одну загруженную модель и игнорирует поле
`model` в запросе. Поэтому размерность вектора проверяется на каждом успешном
ответе: если она не совпадает с `dimensions` пула, ответ не возвращается
клиенту, а участник считается сбойным.

Маршрутизация `/v1/chat/completions` и `/v1/completions` не изменилась.

### Запуск

```bash
cd ROUTER
docker compose up -d
curl -s http://127.0.0.1/health | python3 -m json.tool
```

Контейнер монтирует `./` как `/app`, поэтому новые файлы (`embedding_pool.py`,
`embedding_pools.json`) видны после перезапуска контейнера. Новых pip-зависимостей
нет. Версия приложения отображается в `GET /health` (`"version": "1.1.0"`).

### Файлы конфигурации

- `backends.json` — плоский словарь `имя → URL`, источник истины о бэкендах.
- `embedding_pools.json` — пулы; ссылается на имена из `backends.json`.
- `.env` — `API_KEY`, `LLAMA_EMBEDDED_URL`.

`backends.json` и `embedding_pools.json` пишутся атомарно (`.tmp` + `fsync` +
`os.replace`); повреждённый `embedding_pools.json` переименовывается в
`embedding_pools.json.corrupt`, а прежняя конфигурация продолжает работать.

### `embedding_pools.json`: справочник

```json
{
  "defaults": {
    "failure_threshold": 3,
    "cooldown_seconds": 30,
    "attempt_timeout_seconds": 30,
    "max_total_seconds": 90
  },
  "default_pool": null,
  "pools": {
    "embed-768": {
      "embedding_model": "nomic-embed-text-v1.5",
      "dimensions": 768,
      "members": [
        {"backend": "llama-embedded", "model": "nomic-embed-text-v1.5-Q8_0", "weight": 3, "role": "active"},
        {"backend": "llama-embedded-2", "model": "nomic-embed-text-v1.5-Q8_0", "weight": 1, "role": "active"},
        {"backend": "llama-embedded-backup", "model": "nomic-embed-text-v1.5-Q8_0", "weight": 1, "role": "backup"}
      ]
    }
  }
}
```

| Поле | Обязательно | Значение |
|---|---|---|
| `defaults.failure_threshold` | нет | Подряд ошибок до открытия breaker (по умолчанию 3) |
| `defaults.cooldown_seconds` | нет | Время открытого breaker до half-open (30) |
| `defaults.attempt_timeout_seconds` | нет | Таймаут одной попытки (30) |
| `defaults.max_total_seconds` | нет | Общий бюджет запроса (90); остаток < 2 с → 504 |
| `default_pool` | нет | Пул для запросов с пустым `model`; иначе прежний 400 |
| `pools.<name>` | да | Имя пула — виртуальная модель |
| `pools.<name>.embedding_model` | да | Одна модель на пул |
| `pools.<name>.dimensions` | да | Ожидаемая размерность вектора |
| `pools.<name>.members` | да | Участники пула |
| `pool.failure_threshold` и др. | нет | Переопределяют `defaults` для конкретного пула |
| `member.backend` | да | Ключ из `backends.json` |
| `member.model` | да | Логическое имя модели участника |
| `member.weight` | нет | Вес SWRR, целое > 0 (1) |
| `member.role` | нет | `active` или `backup` (active) |
| `member.enabled` | нет | `false` исключает участника (true) |

Правила валидации при загрузке:

- участник с неизвестным backend, несовпадающей моделью, дубликатом backend или
  `enabled: false` отбрасывается и попадает в `rejected` с причиной;
- пул без валидных участников получает `enabled: false` и
  `disabled_reason: "no_valid_members"`;
- совпадение имён моделей — префиксное, регистронезависимое, после нормализации
  (точки → дефис, `.gguf` и `@backend` отбрасываются). Разные кванты одной
  модели совместимы, но рекомендуется идентичный квант во всём пуле;
- если имя пула совпадает с именем бэкенда, в лог пишется warning; при
  маршрутизации побеждает пул.

**Правило: один пул — одна embedding-модель и одна размерность.** Смешивание
векторных пространств запрещено, escape-hatch отсутствует. При исчерпании пула
роутер возвращает явную ошибку, а не вектор чужой модели.

### Выбор участника

1. Healthy active в порядке smooth weighted round-robin (SWRR).
2. Half-open active probe (cooldown истёк, свободен probe-слот).
3. Healthy backup.
4. Half-open backup probe.

Half-open probe стоит в своём ярусе, чтобы восстановившийся primary мог снова
получить трафик при живом backup. На каждого участника одновременно допускается
ровно один probe.

Breaker: после `failure_threshold` подряд ошибок участник открывается на
`cooldown_seconds`; затем half-open, успех закрывает breaker, ошибка открывает
снова. Состояние участников (`(pool, backend)`) переживает reload конфигурации.

Reload выполняется по mtime перед каждым embedding-запросом, а также через
admin API. Сетевые запросы выполняются вне `asyncio.Lock`; под блокировкой
изменяются только веса SWRR и состояния breaker.

### Клиентские контракты

`POST /v1/embeddings`, `model` = точное имя пула:

```bash
curl -s http://127.0.0.1/v1/embeddings \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"embed-768","input":"hello"}'
```

Успешный ответ — исходное тело бэкенда плюс поля:

| Поле | Значение |
|---|---|
| `model` | имя из запроса (или имя default-пула) |
| `pool` | имя пула |
| `backend_used` | победивший бэкенд |
| `model_served` | модель, отправленная бэкенду |
| `attempts` | число попыток |
| `fallback_used` | `true`, если победил не первый кандидат |

Если `model` не совпадает с именем пула, работает прежняя логика (алиас
бэкенда, `@backend`, кэш моделей, эвристика). При пустом `model` используется
`default_pool`, иначе прежний 400.

Ошибки пула имеют вид:

```json
{
  "detail": {
    "type": "pool_error",
    "code": "all_attempts_failed",
    "pool": "embed-768",
    "members": [
      {"backend": "llama-embedded", "error": "network: ConnectError"},
      {"backend": "llama-embedded-2", "error": "HTTP 503"}
    ]
  }
}
```

| HTTP | `code` | Когда |
|---|---|---|
| 502 | `all_attempts_failed` | Все кандидаты исчерпаны |
| 503 | `no_healthy_members` | Нет кандидатов; есть `Retry-After` |
| 503 | `pool_disabled` | Пул без валидных участников |
| 504 | `deadline_exceeded` | Остаток `max_total_seconds` < 2 с |

Прочие 4xx бэкенда (кроме 429) возвращаются клиенту прозрачно, без retry и без
влияния на breaker.

`GET /v1/models` дополнительно содержит пулы:

```json
{
  "id": "embed-768",
  "object": "model",
  "pool": true,
  "embedding_model": "nomic-embed-text-v1.5",
  "dimensions": 768,
  "enabled": true,
  "members": [{"backend": "llama-embedded", "role": "active", "weight": 3, "breaker": "closed"}]
}
```

### Admin API

Все admin-эндпоинты требуют `Authorization: Bearer $API_KEY`.

```bash
# Полный статус пулов: участники, breaker, счётчики, rejected, cache_hint
curl -s http://127.0.0.1/admin/api/embedding/pools \
  -H "Authorization: Bearer $API_KEY" | python3 -m json.tool

# Принудительный reload embedding_pools.json
curl -s -X POST http://127.0.0.1/admin/api/embedding/pools/reload \
  -H "Authorization: Bearer $API_KEY"

# Закрыть breaker участника вручную
curl -s -X POST \
  http://127.0.0.1/admin/api/embedding/pools/embed-768/members/llama-embedded/reset \
  -H "Authorization: Bearer $API_KEY"

# Сводка пулов и версия в /health
curl -s http://127.0.0.1/health
```

Поля статуса участника: `breaker` (`closed`/`open`/`half_open`),
`consecutive_failures`, `open_for_seconds`, `probe_inflight`,
`total_requests`, `total_failures`, `last_error`, `cache_hint`,
`url`. `rejected` содержит отброшенных участников с причиной.

### Примеры

Примеры не пишут в рабочий `ROUTER/`: они копируют код в `/tmp/kilo/embed-demo`
и используют временные конфиги.

```bash
cd ROUTER

# Распределение 3:1:1 по трём healthy-бэкендам (mock 8127/8128/8129, router 8123)
/tmp/kilo/venv/bin/python examples/demo_load_balancing.py

# Падение primary, fallback на backup, открытие breaker, cooldown, восстановление
/tmp/kilo/venv/bin/python examples/demo_failover.py

# Отдельный mock-сервер
python examples/mock_embedding_server.py --port 8127 --dimensions 768 \
  --model nomic-embed-text-v1.5-Q8_0
```

Порты 8123/8127/8128/8129 должны быть свободны; примеры проверяют это заранее.

### Тесты

```bash
python3 -m venv /tmp/kilo/venv
/tmp/kilo/venv/bin/pip install fastapi uvicorn httpx python-dotenv jinja2 pytest
/tmp/kilo/venv/bin/python -m pytest ROUTER/tests -v
```

Тесты используют инъекцируемую фабрику `httpx.AsyncClient` и не требуют живых
серверов.

### Откат

Резервная копия создаётся снаружи `ROUTER/`:

```bash
tar czf ../ROUTER_backup_$(date +%Y%m%d-%H%M%S).tar.gz -C .. ROUTER
```

---

## English

`ROUTER` is a FastAPI application that routes requests to multiple llama.cpp
servers. Version **1.1.0** adds embedding model pools with load balancing,
circuit breakers and transparent in-pool fallback.

### Architecture

```
client
  │  POST /v1/embeddings  {model: "embed-768"}
  ▼
┌──────────────────────── ROUTER ────────────────────────┐
│ authenticate (API_KEY)                                  │
│ embedding_pool_manager.maybe_reload()  ← mtime          │
│                                                         │
│ pool "embed-768" (nomic-embed-text-v1.5, 768d)          │
│   ├─ active  llama-embedded        weight 3  ─┐         │
│   ├─ active  llama-embedded-2      weight 1  ─┤ SWRR    │
│   ├─ active  llama-embedded-3      weight 1  ─┘         │
│   └─ backup  llama-embedded-backup weight 1             │
│                                                         │
│ for each candidate:                                     │
│   POST {backend}/v1/embeddings                          │
│   timeout = min(attempt_timeout, remaining max_total)   │
│   success + len(vector) == dimensions → return          │
│   network/timeout/429/5xx → failure, next candidate     │
│   other 4xx → transparent passthrough, no retry         │
└─────────────────────────────────────────────────────────┘
```

Each llama.cpp backend serves exactly one loaded model and ignores the request
`model` field. The vector length is therefore checked on every successful
response: a mismatch with the pool `dimensions` is treated as a member failure
and is never returned to the client.

`/v1/chat/completions` and `/v1/completions` routing is unchanged.

### Running

```bash
cd ROUTER
docker compose up -d
curl -s http://127.0.0.1/health | python3 -m json.tool
```

The container mounts `./` as `/app`, so new files (`embedding_pool.py`,
`embedding_pools.json`) are visible after a container restart. No new pip
dependencies are required. The application version is exposed by
`GET /health` (`"version": "1.1.0"`).

### Configuration files

- `backends.json` — flat `name → URL` map, the source of truth for backends.
- `embedding_pools.json` — pools; references backend names.
- `.env` — `API_KEY`, `LLAMA_EMBEDDED_URL`.

`backends.json` and `embedding_pools.json` are written atomically (`.tmp` +
`fsync` + `os.replace`); a corrupt `embedding_pools.json` is renamed to
`embedding_pools.json.corrupt` and the previous configuration stays active.

### `embedding_pools.json` reference

The example above (Russian section) is the same schema. Key fields:

| Field | Required | Meaning |
|---|---|---|
| `defaults.failure_threshold` | no | Consecutive failures before opening a breaker (3) |
| `defaults.cooldown_seconds` | no | Open-breaker duration before half-open (30) |
| `defaults.attempt_timeout_seconds` | no | Single attempt timeout (30) |
| `defaults.max_total_seconds` | no | Total request budget (90); less than 2 s left → 504 |
| `default_pool` | no | Pool used when `model` is empty; otherwise the previous 400 |
| `pools.<name>` | yes | Pool name is a virtual model |
| `pools.<name>.embedding_model` | yes | Exactly one model per pool |
| `pools.<name>.dimensions` | yes | Expected vector length |
| `pools.<name>.members` | yes | Pool members |
| pool-level overrides | no | Override `defaults` for one pool |
| `member.backend` | yes | Key from `backends.json` |
| `member.model` | yes | Logical member model name |
| `member.weight` | no | SWRR weight, integer > 0 (1) |
| `member.role` | no | `active` or `backup` (active) |
| `member.enabled` | no | `false` excludes the member (true) |

Validation rules:

- a member with an unknown backend, mismatching model, duplicate backend or
  `enabled: false` is rejected and listed in `rejected` with a reason;
- a pool without valid members gets `enabled: false` and
  `disabled_reason: "no_valid_members"`;
- model matching is prefix-based, case-insensitive, after normalization (dots →
  dash, `.gguf` and `@backend` stripped). Different quants of one model are
  compatible, but an identical quant across the pool is recommended;
- a pool name that collides with a backend name logs a warning; the pool wins
  during routing.

**Rule: one pool = one embedding model and one dimension.** Mixing vector
spaces is forbidden and there is no escape hatch. When a pool is exhausted the
router returns an explicit error instead of a vector from another model.

### Member selection

1. Healthy actives in smooth weighted round-robin order.
2. Half-open active probe (cooldown expired, probe slot free).
3. Healthy backups.
4. Half-open backup probe.

The half-open probe stays in its own tier so a recovered primary can reclaim
traffic while a backup is still healthy. At most one probe per member is in
flight.

Breaker: after `failure_threshold` consecutive failures a member opens for
`cooldown_seconds`, then becomes half-open; success closes it, failure opens it
again. Member state for `(pool, backend)` survives config reloads.

Reload happens by mtime before every embedding request and via the admin API.
Network I/O runs outside the `asyncio.Lock`; only SWRR weights and breaker state
are mutated under the lock.

### Client contracts

`POST /v1/embeddings` with `model` set to the exact pool name (see the curl
example in the Russian section). A successful response contains the backend
body plus `model`, `pool`, `backend_used`, `model_served`, `attempts` and
`fallback_used`.

If `model` is not a pool name, the previous routing logic applies (backend
alias, `@backend`, model cache, heuristics). An empty `model` selects
`default_pool` when configured, otherwise it keeps returning 400.

Pool error responses use:

```json
{
  "detail": {
    "type": "pool_error",
    "code": "all_attempts_failed",
    "pool": "embed-768",
    "members": [
      {"backend": "llama-embedded", "error": "network: ConnectError"},
      {"backend": "llama-embedded-2", "error": "HTTP 503"}
    ]
  }
}
```

| HTTP | `code` | When |
|---|---|---|
| 502 | `all_attempts_failed` | All candidates exhausted |
| 503 | `no_healthy_members` | No candidates; `Retry-After` is included |
| 503 | `pool_disabled` | Pool has no valid members |
| 504 | `deadline_exceeded` | Less than 2 s of `max_total_seconds` left |

Other backend 4xx responses (except 429) are passed through transparently,
without retries and without affecting the breaker.

`GET /v1/models` additionally lists pools as virtual models with `pool: true`,
`embedding_model`, `dimensions`, `enabled` and member breaker states.

### Admin API

All admin endpoints require `Authorization: Bearer $API_KEY`. Curl examples are
in the Russian section:

- `GET /admin/api/embedding/pools` — full status (members, breakers, counters,
  rejected members, cache hints);
- `POST /admin/api/embedding/pools/reload` — force reload;
- `POST /admin/api/embedding/pools/{pool}/members/{backend}/reset` — close a
  breaker manually;
- `GET /health` — pool summary and `version`.

### Examples and tests

Examples copy the router into `/tmp/kilo/embed-demo` and never write into the
working `ROUTER/` directory:

```bash
cd ROUTER
/tmp/kilo/venv/bin/python examples/demo_load_balancing.py   # 3:1:1 distribution
/tmp/kilo/venv/bin/python examples/demo_failover.py         # fallback + breaker + recovery
python examples/mock_embedding_server.py --port 8127 --dimensions 768 \
  --model nomic-embed-text-v1.5-Q8_0
```

```bash
python3 -m venv /tmp/kilo/venv
/tmp/kilo/venv/bin/pip install fastapi uvicorn httpx python-dotenv jinja2 pytest
/tmp/kilo/venv/bin/python -m pytest ROUTER/tests -v
```

Tests inject a fake `httpx.AsyncClient` factory and need no live servers.

### Rollback

Backups are stored outside `ROUTER/`:

```bash
tar czf ../ROUTER_backup_$(date +%Y%m%d-%H%M%S).tar.gz -C .. ROUTER
```