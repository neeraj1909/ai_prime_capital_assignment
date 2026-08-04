FROM python:3.12.3-slim-bookworm@sha256:afc139a0a640942491ec481ad8dda10f2c5b753f5c969393b12480155fe15a63

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN groupadd --system app \
    && useradd --system --gid app --home-dir /app --create-home app

# Keep the dependency layer stable when only application code changes.
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --disable-pip-version-check \
    --requirement requirements.txt

COPY --chown=app:app src ./src

USER app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=6 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2)"]

ENTRYPOINT ["python", "-m", "uvicorn", "ai_prime_capital.main:app", "--host", "0.0.0.0", "--port", "8000"]
