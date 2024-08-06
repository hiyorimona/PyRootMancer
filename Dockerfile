ARG PYTHON_VERSION=3.10.13
FROM python:${PYTHON_VERSION}-slim as base
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=docker_requirements.txt,target=docker_requirements.txt \
    python -m pip install -r docker_requirements.txt

USER appuser

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-root

COPY..

EXPOSE 8000

CMD gunicorn 'api:app' --bind=0.0.0.0:8000
