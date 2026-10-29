FROM python:3.12-slim AS base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY router.py embedding_pool.py ./
COPY templates/ ./templates/

FROM base AS prod
EXPOSE 80
CMD ["uvicorn", "router:app", "--host", "0.0.0.0", "--port", "80"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:80/health',timeout=4)"

FROM base AS test
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt
COPY tests/ ./tests/
CMD ["python", "-m", "pytest", "-v", "tests"]