FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WEB_HOST=127.0.0.1 \
    WEB_PORT=8088 \
    STREAM_HOST=127.0.0.1 \
    STREAM_PORT=7777

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY app.py ./
COPY static ./static

RUN useradd --system --uid 10001 --home /app webaudio \
    && chown -R webaudio:webaudio /app
USER webaudio

EXPOSE 8088

CMD ["sh", "-c", "uvicorn app:app --host ${WEB_HOST} --port ${WEB_PORT} --no-access-log"]
