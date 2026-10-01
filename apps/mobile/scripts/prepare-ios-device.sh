#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MOBILE_DIR="$ROOT_DIR/apps/mobile"
QUALIFIED_REMOTE_API="https://woof-api-prod.fly.dev/api/v1"
DEFAULT_BUNDLE_ID="com.sidhulyalkar.woof.dev"

API_URL="${EXPO_PUBLIC_API_URL:-}"
BUNDLE_ID="${WOOF_IOS_BUNDLE_IDENTIFIER:-$DEFAULT_BUNDLE_ID}"
USE_LOCAL_API=false
OPEN_XCODE=true
SKIP_API_CHECK=false

usage() {
  cat <<'EOF'
Prepare Woof for a physical iPhone Xcode run.

Usage:
  pnpm --filter @woof/mobile ios:device:prepare -- [options]

Options:
  --api-url URL       Explicit API origin, for example https://example.com/api/v1
  --local-api         Use this Mac's LAN address with port 4000 and /api/v1
  --bundle-id ID      Development-only bundle identifier for local signing
  --no-open           Prepare the workspace without opening Xcode
  --skip-api-check    Do not probe <api-url>/health before native generation
  -h, --help          Show this help
EOF
}

detect_lan_ip() {
  local default_interface
  default_interface="$(route -n get default 2>/dev/null | awk '/interface:/{print $2; exit}')"

  for interface in "$default_interface" en0 en1 en2; do
    [[ -n "$interface" ]] || continue
    local candidate
    candidate="$(ipconfig getifaddr "$interface" 2>/dev/null || true)"
    if [[ -n "$candidate" && "$candidate" != 127.* ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-url)
      [[ $# -ge 2 ]] || { echo "Missing value after --api-url" >&2; exit 2; }
      API_URL="$2"
      shift 2
      ;;
    --local-api)
      USE_LOCAL_API=true
      shift
      ;;
    --bundle-id)
      [[ $# -ge 2 ]] || { echo "Missing value after --bundle-id" >&2; exit 2; }
      BUNDLE_ID="$2"
      shift 2
      ;;
    --no-open)
      OPEN_XCODE=false
      shift
      ;;
    --skip-api-check)
      SKIP_API_CHECK=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This helper requires macOS because the final artifact is an Xcode workspace." >&2
  exit 1
fi

for command in pnpm node xcodebuild pod curl; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Missing required command: $command" >&2
    exit 1
  fi
done

if [[ "$USE_LOCAL_API" == true ]]; then
  LAN_IP="$(detect_lan_ip || true)"
  if [[ -z "$LAN_IP" ]]; then
    echo "Could not determine this Mac's LAN address. Re-run with --api-url URL." >&2
    exit 1
  fi
  API_URL="http://$LAN_IP:4000/api/v1"
fi

if [[ -z "$API_URL" ]]; then
  API_URL="$QUALIFIED_REMOTE_API"
fi

API_URL="${API_URL%/}"

if [[ ! "$API_URL" =~ ^https?:// ]]; then
  echo "API URL must be an absolute http(s) URL: $API_URL" >&2
  exit 1
fi

if [[ "$API_URL" =~ ^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])([:/]|$) ]]; then
  echo "A physical iPhone cannot reach your Mac through localhost. Use --local-api or --api-url." >&2
  exit 1
fi

if [[ ! "$BUNDLE_ID" =~ ^[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)+$ ]]; then
  echo "Invalid iOS bundle identifier: $BUNDLE_ID" >&2
  exit 1
fi

echo "Woof physical-device configuration"
echo "  API:       $API_URL"
echo "  Bundle ID: $BUNDLE_ID"
echo "  Xcode:     $(xcodebuild -version | tr '\n' ' ')"

if [[ "$SKIP_API_CHECK" != true ]]; then
  HEALTH_URL="$API_URL/health"
  if curl --silent --show-error --fail --max-time 5 "$HEALTH_URL" >/dev/null; then
    echo "  API health: reachable"
  else
    echo
    echo "WARNING: $HEALTH_URL is not reachable from this Mac yet."
    echo "The Xcode project can still be prepared, but authenticated app surfaces will not work"
    echo "until the API is reachable. For exact local-main testing, start the Woof API and use"
    echo "--local-api. macOS may also ask you to allow incoming connections for Node."
    echo
  fi
fi

cat > "$MOBILE_DIR/.env.local" <<EOF
EXPO_PUBLIC_API_URL=$API_URL
WOOF_BUILD_PROFILE=development
WOOF_IOS_BUNDLE_IDENTIFIER=$BUNDLE_ID
REACT_NATIVE_PACKAGER_HOSTNAME=${LAN_IP:-}
EOF

export EXPO_PUBLIC_API_URL="$API_URL"
export WOOF_BUILD_PROFILE=development
export WOOF_IOS_BUNDLE_IDENTIFIER="$BUNDLE_ID"
if [[ -n "${LAN_IP:-}" ]]; then
  export REACT_NATIVE_PACKAGER_HOSTNAME="$LAN_IP"
fi

cd "$ROOT_DIR"
pnpm install --frozen-lockfile

CONFIG_SNAPSHOT="$(mktemp)"
trap 'rm -f "$CONFIG_SNAPSHOT"' EXIT
pnpm --filter @woof/mobile exec expo config --type public > "$CONFIG_SNAPSHOT"
grep -F "$API_URL" "$CONFIG_SNAPSHOT" >/dev/null
grep -F "$BUNDLE_ID" "$CONFIG_SNAPSHOT" >/dev/null

pnpm --filter @woof/mobile exec expo prebuild --clean --platform ios --no-install

cd "$MOBILE_DIR/ios"
if ! pod install; then
  echo "Initial pod resolution failed; retrying with an updated Specs index."
  pod install --repo-update
fi

NODE_BINARY="$(command -v node)"
cat > .xcode.env.local <<EOF
export NODE_BINARY="$NODE_BINARY"
export EXPO_PUBLIC_API_URL="$API_URL"
export WOOF_BUILD_PROFILE="development"
export WOOF_IOS_BUNDLE_IDENTIFIER="$BUNDLE_ID"
EOF

test -d Woof.xcworkspace
echo
echo "Prepared: $MOBILE_DIR/ios/Woof.xcworkspace"
echo
echo "Next:"
echo "  1. Connect and unlock your iPhone. Enable Developer Mode if iOS asks for it."
echo "  2. In Xcode select the Woof target -> Signing & Capabilities -> your Apple Development team."
echo "  3. Select your iPhone as the run destination."
echo "  4. In another terminal run: pnpm --filter @woof/mobile ios:device:metro"
echo "  5. Press Cmd+R in Xcode."
echo
echo "If signing says the bundle identifier is unavailable, re-run this command with"
echo "  --bundle-id com.<your-name>.woof.dev"
echo

if [[ "$OPEN_XCODE" == true ]]; then
  open "$MOBILE_DIR/ios/Woof.xcworkspace"
fi
