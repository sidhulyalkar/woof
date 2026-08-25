CREATE SCHEMA IF NOT EXISTS dogos_auth;

CREATE TABLE IF NOT EXISTS dogos_auth.sessions (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  expires_at TIMESTAMPTZ NOT NULL,
  revoked_at TIMESTAMPTZ,
  revocation_reason TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT auth_sessions_id_length CHECK (char_length(id) BETWEEN 8 AND 128),
  CONSTRAINT auth_sessions_expiry_after_creation CHECK (expires_at > created_at)
);

CREATE INDEX IF NOT EXISTS auth_sessions_user_active_idx
  ON dogos_auth.sessions (user_id, revoked_at, expires_at DESC);

CREATE INDEX IF NOT EXISTS auth_sessions_expiry_idx
  ON dogos_auth.sessions (expires_at);
