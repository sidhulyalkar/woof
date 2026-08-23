CREATE SCHEMA IF NOT EXISTS dogos_chat;

CREATE TABLE IF NOT EXISTS dogos_chat.message_receipts (
  user_id TEXT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  client_message_id TEXT NOT NULL,
  conversation_id TEXT NOT NULL REFERENCES public.conversations(id) ON DELETE CASCADE,
  message_id TEXT NOT NULL REFERENCES public.messages(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (user_id, client_message_id),
  UNIQUE (message_id),
  CONSTRAINT message_receipts_client_id_length
    CHECK (char_length(client_message_id) BETWEEN 8 AND 128)
);

CREATE INDEX IF NOT EXISTS message_receipts_conversation_created_idx
  ON dogos_chat.message_receipts (conversation_id, created_at DESC);
