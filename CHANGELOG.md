# Changelog / История изменений

Формат основан на [Keep a Changelog](https://keepachangelog.com/ru/1.1.0/),
проект следует [Semantic Versioning](https://semver.org/lang/ru/).
The format is based on Keep a Changelog and this project adheres to Semantic
Versioning.

## Русский

### [1.2.0] — 2026-10-29

#### Добавлено

- Управление пулами из админ-панели: форма «Добавить пул» (имя,
  `embedding_model`, `dimensions`, необязательный первый участник), форма
  «Добавить участника» в карточке пула (backend, model, weight, role), удаление
  участника и пула.
- Admin API: `POST /admin/api/embedding/pools`,
  `DELETE /admin/api/embedding/pools/{pool}`,
  `POST /admin/api/embedding/pools/{pool}/members`,
  `PUT /admin/api/embedding/pools/{pool}/members/{backend}`,
  `DELETE /admin/api/embedding/pools/{pool}/members/{backend}`.
- Атомарная запись `embedding_pools.json` (`.tmp` + `fsync` + `os.replace`) с
  fallback на запись in-place при `EBUSY` для single-file bind mount.
- Валидация перед записью: имя пула, существование бэкенда, совместимость
  модели участника с `embedding_model`, `weight` > 0, `role` = `active|backup`,
  запрет дубликатов.
- Тесты admin API для создания/удаления пулов и участников, валидации и очистки
  `default_pool` (всего 31 тест).

#### Изменено

- Версия приложения и образ: `1.2.0` / `llama-router:1.2.0`.
- `embedding_pools.json` в проде монтируется read-write (admin API пишет
  конфиг); у canary и test-сервисов монтирование остаётся read-only.
- Секция «Embedding-пулы» больше не read-only: автообновление приостанавливается,
  пока открыта форма редактирования.
- Версия `1.2.0` задеплоена на `:80` (2026-10-29).

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
- Read-only секция «Embedding-пулы» в админ-панели: состояние breaker'ов,
  счётчики, last_error, rejected-причины, кнопки «Перезагрузить конфиг» и
  «Сбросить», автообновление каждые 10 с.
- `GET /health` содержит `version` и сводку пулов.
- Модуль `embedding_pool.py` с инъекцируемой фабрикой `httpx.AsyncClient`.
- Примеры: mock embedding-сервер, демонстрация балансировки 3:1:1 и
  демонстрация fallback/breaker/восстановления.
- Тесты `tests/test_embedding_pool.py` (26 тестов, без живых серверов).

#### Изменено

- Версия приложения: `1.1.0`.
- Docker-развёртывание: собственный multi-stage образ `llama-router:1.1.0`
  (`Dockerfile`, `requirements.txt`) вместо `pip install` при каждом старте.
  Код запечён в образ, снаружи монтируются только `backends.json` и
  `embedding_pools.json`; `.env` передаётся через `env_file` и исключён из
  образа через `.dockerignore`.
- `docker-compose.yml`: сервисы `router` (прод, `:80`), `router-canary`
  (profile `canary`, `:8080`, read-only конфиги) и `router-test`
  (profile `test`, pytest на моках внутри финального образа). Конфиги
  монтируются long syntax с `create_host_path: false` (явная ошибка вместо
  каталога на месте отсутствующего файла); `embedding_pools.json` в проде
  read-only.
- Версия `1.1.0` задеплоена на `:80` (2026-10-29): canary-проверки, unit-тесты
  (26 passed) и переключение прода выполнены, наблюдение без ошибок.
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

### [1.2.0] — 2026-10-29

#### Added

- Pool management from the admin panel: an “Add pool” form (name,
  `embedding_model`, `dimensions`, optional first member), an “Add member” form
  in each pool card (backend, model, weight, role), plus member and pool
  deletion.
- Admin API: `POST /admin/api/embedding/pools`,
  `DELETE /admin/api/embedding/pools/{pool}`,
  `POST /admin/api/embedding/pools/{pool}/members`,
  `PUT /admin/api/embedding/pools/{pool}/members/{backend}`,
  `DELETE /admin/api/embedding/pools/{pool}/members/{backend}`.
- Atomic `embedding_pools.json` writes (`.tmp` + `fsync` + `os.replace`) with an
  in-place fallback on `EBUSY` for single-file bind mounts.
- Validation before writing: pool name, backend existence, member model
  compatibility with `embedding_model`, `weight` > 0, `role` =
  `active|backup`, and duplicate rejection.
- Admin API tests for pool/member creation and deletion, validation and
  `default_pool` cleanup (31 tests total).

#### Changed

- Application and image version: `1.2.0` / `llama-router:1.2.0`.
- `embedding_pools.json` is mounted read-write in production (the admin API
  writes it); canary and test services keep read-only mounts.
- The “Embedding pools” section is no longer read-only; auto-refresh pauses while
  an edit form is open.
- Version `1.2.0` deployed on `:80` (2026-10-29).

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
- Read-only “Embedding pools” section in the admin panel: breaker states,
  counters, last errors, rejected members, “Reload config” and “Reset” buttons,
  10-second auto-refresh.
- `GET /health` exposes `version` and a pool summary.
- `embedding_pool.py` with an injectable `httpx.AsyncClient` factory.
- Examples: mock embedding server, 3:1:1 load-balancing demo and a
  fallback/breaker/recovery demo.
- `tests/test_embedding_pool.py` suite (26 tests, no live servers).

#### Changed

- Application version: `1.1.0`.
- Docker deployment: a dedicated multi-stage `llama-router:1.1.0` image
  (`Dockerfile`, `requirements.txt`) instead of `pip install` on every start.
  Code is baked into the image; only `backends.json` and `embedding_pools.json`
  are mounted. `.env` is passed via `env_file` and excluded from the image by
  `.dockerignore`.
- `docker-compose.yml`: `router` (production, `:80`), `router-canary`
  (`canary` profile, `:8080`, read-only configs) and `router-test`
  (`test` profile, mocked pytest inside the final image). Configs use long
  syntax with `create_host_path: false` (an explicit error instead of a
  directory in place of a missing file); `embedding_pools.json` is read-only in
  production.
- Version `1.1.0` deployed on `:80` (2026-10-29): canary checks, unit tests
  (26 passed) and the production switch were completed; observation showed no
  errors.
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