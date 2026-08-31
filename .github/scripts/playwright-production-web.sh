#!/usr/bin/env bash
set -euo pipefail

RUNNER_TMP="${RUNNER_TEMP:-/tmp}"
LOG_PATH="${RUNNER_TMP}/woof-playwright-web.log"
PID_PATH="${RUNNER_TMP}/woof-playwright-web.pid"
BASE_URL="http://127.0.0.1:3000"

start_server() {
  rm -f "$LOG_PATH" "$PID_PATH"

  nohup env NODE_ENV=production pnpm --filter @woof/web start >"$LOG_PATH" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" >"$PID_PATH"

  for _ in $(seq 1 45); do
    local status
    status="$(
      curl --silent --show-error --max-time 2 \
        --output /dev/null \
        --write-out '%{http_code}' \
        "${BASE_URL}/demo" || true
    )"
    if [[ "$status" == "200" ]]; then
      echo "Production Web server is ready on ${BASE_URL} (pid ${pid})."
      return 0
    fi

    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Production Web server exited before readiness." >&2
      cat "$LOG_PATH" >&2 || true
      return 1
    fi

    sleep 1
  done

  echo "Production Web server did not return HTTP 200 within 45 seconds." >&2
  cat "$LOG_PATH" >&2 || true
  return 1
}

stop_server() {
  if [[ ! -f "$PID_PATH" ]]; then
    return 0
  fi

  local pid
  pid="$(cat "$PID_PATH")"
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 10); do
      if ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    kill -9 "$pid" 2>/dev/null || true
  fi

  rm -f "$PID_PATH"
}

case "${1:-}" in
  start)
    start_server
    ;;
  stop)
    stop_server
    ;;
  log)
    cat "$LOG_PATH"
    ;;
  *)
    echo "usage: $0 {start|stop|log}" >&2
    exit 64
    ;;
esac
