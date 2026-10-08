# syntax=docker/dockerfile:1

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY containerization/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV SERVICE_ROLE=server
ENV FED_SERVER_HOST=0.0.0.0
ENV FED_SERVER_PORT=8000
ENV FED_SERVER_URL=http://fed-server:8000
ENV CLIENT_ID=1
ENV MAIN_SCRIPT=main.py
ENV RUNTIME_DIR=/app/runtime

EXPOSE 8000

CMD ["python", "-m", "containerization.container_entrypoint"]
