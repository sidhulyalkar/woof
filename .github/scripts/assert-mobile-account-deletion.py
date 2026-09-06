#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


auth = read("apps/mobile/src/api/auth.ts")
context = read("apps/mobile/src/contexts/AuthContext.tsx")
profile = read("apps/mobile/src/screens/ProfileScreen.tsx")

endpoint = "await apiClient.delete<void>('/users/me');"
clear_token = "await SecureStore.deleteItemAsync(ACCESS_TOKEN_KEY);"
if endpoint not in auth:
    raise SystemExit("mobile account deletion must call canonical DELETE /users/me")

delete_method = auth.split("async deleteAccount(): Promise<void> {", 1)
if len(delete_method) != 2:
    raise SystemExit("authApi.deleteAccount method missing")
delete_body = delete_method[1].split("},", 1)[0]
if endpoint not in delete_body or clear_token not in delete_body:
    raise SystemExit("account deletion method must delete remotely then clear the access token")
if delete_body.index(endpoint) > delete_body.index(clear_token):
    raise SystemExit("local token must not be cleared before server confirms account deletion")

for marker in [
    "deleteAccount: () => Promise<void>;",
    "await authApi.deleteAccount();",
    "setUser(null);",
]:
    if marker not in context:
        raise SystemExit(f"AuthContext deletion marker missing: {marker}")

context_delete = context.split("const deleteAccount = async () => {", 1)
if len(context_delete) != 2:
    raise SystemExit("AuthContext deleteAccount implementation missing")
context_body = context_delete[1].split("};", 1)[0]
if context_body.index("await authApi.deleteAccount();") > context_body.index("setUser(null);"):
    raise SystemExit("AuthContext must retain the authenticated user until server deletion succeeds")

for marker in [
    "Delete account",
    "Delete your Woof account?",
    "Delete permanently?",
    "Delete permanently",
    "Account was not deleted",
    "Your account remains active so you can retry",
    "Woof-owned private Media Library files",
    "accessibilityLabel=\"Delete Woof account permanently\"",
]:
    if marker not in profile:
        raise SystemExit(f"Profile deletion UX marker missing: {marker}")

if profile.count("style: 'destructive'") < 2:
    raise SystemExit("native account deletion must retain two explicit destructive confirmations")

print("mobile account deletion source contract: OK")
