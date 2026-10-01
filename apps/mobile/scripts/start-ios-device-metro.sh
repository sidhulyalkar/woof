#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MOBILE_DIR="$ROOT_DIR/apps/mobile"
ENV_FILE="$MOBILE_DIR/.env.local"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE." >&2
  echo "Run: pnpm --filter @woof/mobile ios:device:prepare" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if [[ -z "${EXPO_PUBLIC_API_URL:-}" ]]; then
  echo "EXPO_PUBLIC_API_URL is missing from $ENV_FILE" >&2
  exit 1
fi

echo "Starting Metro for Woof physical-device Debug builds"
echo "  API: ${EXPO_PUBLIC_API_URL}"
if [[ -n "${REACT_NATIVE_PACKAGER_HOSTNAME:-}" ]]; then
  echo "  Packager host: ${REACT_NATIVE_PACKAGER_HOSTNAME}"
fi

cd "$ROOT_DIR"
exec pnpm --filter @woof/mobile exec expo start --lan --clear
