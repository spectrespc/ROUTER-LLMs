# Changelog / История изменений

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/),
проект следует [Semantic Versioning](https://semver.org/lang/ru/).
The format is based on Keep a Changelog and this project adheres to Semantic
Versioning.

## Русский

### [1.1.0] — 2026-10-29

#### Добавлено

- Пул embedding-моделей `embedding_pools.json`: один пул = одна модель + одна
  размерность, участники ссылаются на имена из `backends.json`.
- Smooth weighted round-robin (SWRR, nginx-алгоритм) для healthy-участников.
- Circuit breaker: `failure_threshold`, `cooldown_seconds`, half-open probe
  (ровно один на участника), ручной reset через admin API.
- Прозрачный fallback внутри пула для сетевых ошибок, таймаутов, 429 и 5xx;
  прочие 4xx возвращаются клиенту без retry.
- Guard размерности на успешном ответе; несовместимый вектор не возвращается
  клиенту и считается сбоем участника.
- Hot-reload `embedding_pools.json` по mtime; повреждённый файл сохраняется как
  `.corrupt`, прежняя конфигурация продолжает работать.
- `default_pool` для запросов с пустым `model`.
- `POST /v1/embeddings` с `model` = имя пула; поля ответа `pool`,
  `backend_used`, `model_served`, `attempts`, `fallback_used`.
- `GET /v1/models` показывает пулы как виртуальные модели.
- Admin API: `GET /admin/api/embedding/pools`,
  `POST /admin/api/embedding/pools/reload`,
  `POST /admin/api/embedding/pools/{pool}/members/{backend}/reset`.
- `GET /health` содержит `version` и сводку пулов.
- Модуль `embedding_pool.py` с инъекцируемой фабрикой `httpx.AsyncClient`.
- Примеры: mock embedding-сервер, демонстрация балансировки 3:1:1 и
  демонстрация fallback/breaker/восстановления.
- Тесты `tests/test_embedding_pool.py` (23 теста, без живых серверов).

#### Изменено

- Версия приложения: `1.1.0`.
- Внутренняя функция `get_original_model_path` принимает кэш явным аргументом.
- `LLAMA_EMBEDDED_URL` остаётся в `.env`, но embedding-маршрут по умолчанию
  использует прежнюю логику, если `model` не является именем пула.

### [1.0.0] — 2026-10-17

#### Добавлено

- FastAPI-роутер к llama.cpp-бэкендам: `/v1/chat/completions`,
  `/v1/completions`, `/v1/embeddings`, `/v1/models`, `/health`.
- Маршрутизация по алиасу бэкенда, `model@backend`, кэшу `/v1/models`/`/props`
  с TTL 60 с и эвристике по имени модели.
- Admin API для `backends.json`: список, добавление, удаление, refresh кэша.
- Админ-панель `templates/admin.html`.
- Атомарное сохранение `backends.json` и обработка повреждённого файла.
- Docker Compose-развёртывание на `python:3.12-slim` с volume `./:/app`.

## English

### [1.1.0] — 2026-10-29

#### Added

- Embedding model pools in `embedding_pools.json`: one pool = one model + one
  dimension; members reference names from `backends.json`.
- Smooth weighted round-robin (SWRR, nginx algorithm) for healthy members.
- Circuit breaker: `failure_threshold`, `cooldown_seconds`, half-open probe
  (exactly one per member), manual reset via the admin API.
- Transparent in-pool fallback for network errors, timeouts, 429 and 5xx;
  other 4xx responses are returned to the client without retry.
- Dimension guard on successful responses; an incompatible vector is never
  returned and counts as a member failure.
- Hot reload of `embedding_pools.json` by mtime; a corrupt file is preserved as
  `.corrupt` and the previous configuration stays active.
- `default_pool` for requests with an empty `model`.
- `POST /v1/embeddings` with `model` set to a pool name; response fields
  `pool`, `backend_used`, `model_served`, `attempts`, `fallback_used`.
- `GET /v1/models` lists pools as virtual models.
- Admin API: `GET /admin/api/embedding/pools`,
  `POST /admin/api/embedding/pools/reload`,
  `POST /admin/api/embedding/pools/{pool}/members/{backend}/reset`.
- `GET /health` exposes `version` and a pool summary.
- `embedding_pool.py` with an injectable `httpx.AsyncClient` factory.
- Examples: mock embedding server, 3:1:1 load-balancing demo and a
  fallback/breaker/recovery demo.
- `tests/test_embedding_pool.py` suite (23 tests, no live servers).

#### Changed

- Application version: `1.1.0`.
- `get_original_model_path` now accepts the models cache explicitly.
- `LLAMA_EMBEDDED_URL` remains in `.env`, but the embedding route keeps the
  previous logic when `model` is not a pool name.

### [1.0.0] — 2026-10-17

#### Added

- FastAPI router for llama.cpp backends: `/v1/chat/completions`,
  `/v1/completions`, `/v1/embeddings`, `/v1/models`, `/health`.
- Routing by backend alias, `model@backend`, the `/v1/models`/`/props` cache
  with a 60 s TTL and model-name heuristics.
- Admin API for `backends.json`: list, add, remove, cache refresh.
- Admin panel `templates/admin.html`.
- Atomic `backends.json` writes and corrupt-file handling.
- Docker Compose deployment on `python:3.12-slim` with the `./:/app` volume.