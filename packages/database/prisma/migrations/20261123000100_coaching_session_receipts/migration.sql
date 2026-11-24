CREATE SCHEMA IF NOT EXISTS dogos_coaching;

CREATE TABLE dogos_coaching.session_receipts (
  plan_id TEXT NOT NULL REFERENCES public.mutual_goals(id) ON DELETE CASCADE,
  session_key UUID NOT NULL,
  user_id TEXT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  activity_id TEXT NOT NULL UNIQUE REFERENCES public.activities(id) ON DELETE CASCADE,
  payload_hash TEXT NOT NULL CHECK (length(payload_hash) = 64),
  response JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (plan_id, session_key)
);
CREATE INDEX session_receipts_user_idx ON dogos_coaching.session_receipts(user_id);
