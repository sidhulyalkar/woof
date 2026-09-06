#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


app_json = json.loads(read("apps/mobile/app.json"))["expo"]
eas = json.loads(read("apps/mobile/eas.json"))
config = read("apps/mobile/app.config.ts")
client = read("apps/mobile/src/api/client.ts")
deploy = read(".github/workflows/deploy-production.yml")

static_extra = app_json.get("extra", {})
static_eas = static_extra.get("eas") or {}
if static_eas.get("projectId"):
    raise SystemExit("static mobile config must not carry placeholder or unqualified EAS project authority")
if "your-project-id" in read("apps/mobile/app.json"):
    raise SystemExit("placeholder EAS project id returned to mobile config")

for marker in [
    "EAS_BUILD_PROFILE",
    "WOOF_BUILD_PROFILE",
    "EXPO_PUBLIC_API_URL",
    "EAS_PROJECT_ID",
    "EAS_BUILD_PROJECT_ID",
    "must use HTTPS",
    "must not target a loopback host",
    "buildProfile",
]:
    if marker not in config:
        raise SystemExit(f"dynamic iOS config authority marker missing: {marker}")

for marker in [
    "BUILD_PROFILE",
    "NON_REMOTE_HOSTS",
    "parsed.protocol !== 'https:'",
    "must be a remote HTTPS endpoint",
]:
    if marker not in client:
        raise SystemExit(f"runtime API fail-closed marker missing: {marker}")

if eas.get("cli", {}).get("appVersionSource") != "remote":
    raise SystemExit("EAS appVersionSource must be remote")

expected_envs = {
    "development": "development",
    "preview": "preview",
    "production": "production",
}
for profile, environment in expected_envs.items():
    actual = eas.get("build", {}).get(profile, {}).get("environment")
    if actual != environment:
        raise SystemExit(f"EAS profile {profile} must bind environment {environment}, got {actual!r}")

if eas.get("build", {}).get("production", {}).get("autoIncrement") is not True:
    raise SystemExit("production EAS build must auto-increment developer-facing build versions")

if "https://woof-api-prod.fly.dev/api/v1" not in deploy:
    raise SystemExit("repository production API authority moved; native production config contract needs review")

print("iOS production config source contract: OK")
