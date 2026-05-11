#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${SONGGUO_POSTGRES_CONTAINER:-songguo-postgres}"
IMAGE="${SONGGUO_POSTGRES_IMAGE:-postgres:16}"
HOST="${SONGGUO_POSTGRES_HOST:-127.0.0.1}"
PORT="${SONGGUO_POSTGRES_PORT:-5432}"
USER_NAME="${SONGGUO_POSTGRES_USER:-songguo}"
PASSWORD="${SONGGUO_POSTGRES_PASSWORD:-password}"
DB_NAME="${SONGGUO_POSTGRES_DB:-songguo}"
VOLUME="${SONGGUO_POSTGRES_VOLUME:-songguo-postgres-data}"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  docker start "${CONTAINER_NAME}" >/dev/null
else
  docker run -d \
    --name "${CONTAINER_NAME}" \
    -e POSTGRES_USER="${USER_NAME}" \
    -e POSTGRES_PASSWORD="${PASSWORD}" \
    -e POSTGRES_DB="${DB_NAME}" \
    -p "${HOST}:${PORT}:5432" \
    -v "${VOLUME}:/var/lib/postgresql/data" \
    "${IMAGE}" >/dev/null
fi

for _ in $(seq 1 60); do
  if docker exec "${CONTAINER_NAME}" pg_isready -U "${USER_NAME}" -d "${DB_NAME}" >/dev/null 2>&1; then
    echo "postgres-ready ${HOST}:${PORT}/${DB_NAME}"
    exit 0
  fi
  sleep 1
done

echo "postgres-not-ready" >&2
docker logs --tail 80 "${CONTAINER_NAME}" >&2 || true
exit 1
