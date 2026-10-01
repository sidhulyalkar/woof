#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
WORKSPACE="$ROOT_DIR/apps/mobile/ios/Woof.xcworkspace"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Xcode workspaces can only be opened on macOS." >&2
  exit 1
fi

if [[ ! -d "$WORKSPACE" ]]; then
  echo "Generated workspace is missing: $WORKSPACE" >&2
  echo "Run: pnpm --filter @woof/mobile ios:device:prepare" >&2
  exit 1
fi

open "$WORKSPACE"
