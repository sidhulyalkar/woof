-- Expedition Authority v1
-- Cooperative progress is derived from canonical server evidence, never client-submitted points.

CREATE TABLE IF NOT EXISTS dogos_social.expedition_receipts (
  id TEXT PRIMARY KEY,
  expedition_key TEXT NOT NULL,
  expedition_version TEXT NOT NULL,
  season_key TEXT NOT NULL,
  policy_version TEXT NOT NULL,
  scope TEXT NOT NULL,
  pack_id TEXT REFERENCES dogos_social.packs(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  source_type TEXT NOT NULL,
  source_id TEXT NOT NULL,
  objective_key TEXT NOT NULL,
  category_key TEXT NOT NULL,
  pathway TEXT,
  source_fingerprint TEXT NOT NULL,
  evidence_at TIMESTAMPTZ NOT NULL,
  authorized_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  CONSTRAINT expedition_receipt_id_length CHECK (char_length(id) BETWEEN 12 AND 160),
  CONSTRAINT expedition_receipt_key_length CHECK (char_length(expedition_key) BETWEEN 2 AND 64),
  CONSTRAINT expedition_receipt_version_length CHECK (char_length(expedition_version) BETWEEN 1 AND 64),
  CONSTRAINT expedition_receipt_season_length CHECK (char_length(season_key) BETWEEN 8 AND 32),
  CONSTRAINT expedition_receipt_policy_length CHECK (char_length(policy_version) BETWEEN 1 AND 64),
  CONSTRAINT expedition_receipt_scope CHECK (scope IN ('GLOBAL', 'PACK')),
  CONSTRAINT expedition_receipt_scope_shape CHECK (
    (scope = 'GLOBAL' AND pack_id IS NULL)
    OR (scope = 'PACK' AND pack_id IS NOT NULL)
  ),
  CONSTRAINT expedition_receipt_source_type CHECK (
    source_type IN ('CARE_EVENT', 'HUMAN_SKILL_ATTEMPT')
  ),
  CONSTRAINT expedition_receipt_objective CHECK (
    objective_key IN ('SNIFF_EXPLORE', 'RECOVERY_COUNTS', 'READ_THE_ROOM')
  ),
  CONSTRAINT expedition_receipt_source_id_length CHECK (char_length(source_id) BETWEEN 1 AND 128),
  CONSTRAINT expedition_receipt_category_length CHECK (char_length(category_key) BETWEEN 2 AND 64),
  CONSTRAINT expedition_receipt_pathway CHECK (
    (source_type = 'CARE_EVENT' AND pathway IN ('EXPLORE', 'ENRICH', 'RECOVER'))
    OR (source_type = 'HUMAN_SKILL_ATTEMPT' AND pathway IS NULL)
  ),
  CONSTRAINT expedition_receipt_fingerprint CHECK (source_fingerprint ~ '^[0-9a-f]{64}$')
);

CREATE UNIQUE INDEX IF NOT EXISTS expedition_receipt_evidence_unique
  ON dogos_social.expedition_receipts (
    expedition_key,
    expedition_version,
    season_key,
    policy_version,
    scope,
    COALESCE(pack_id, ''),
    objective_key,
    source_type,
    source_id
  );

CREATE INDEX IF NOT EXISTS expedition_receipt_projection_idx
  ON dogos_social.expedition_receipts (
    expedition_key,
    expedition_version,
    season_key,
    policy_version,
    scope,
    pack_id,
    objective_key,
    authorized_at DESC
  );

CREATE INDEX IF NOT EXISTS expedition_receipt_user_season_idx
  ON dogos_social.expedition_receipts (user_id, season_key, authorized_at DESC, id);

CREATE OR REPLACE FUNCTION dogos_social.reject_expedition_receipt_update()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  IF NEW IS DISTINCT FROM OLD THEN
    RAISE EXCEPTION 'expedition contribution receipts are immutable'
      USING ERRCODE = '23514';
  END IF;
  RETURN OLD;
END;
$$;

DROP TRIGGER IF EXISTS expedition_receipts_immutable_after_issue
  ON dogos_social.expedition_receipts;
CREATE TRIGGER expedition_receipts_immutable_after_issue
BEFORE UPDATE ON dogos_social.expedition_receipts
FOR EACH ROW
EXECUTE FUNCTION dogos_social.reject_expedition_receipt_update();
