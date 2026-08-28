CREATE SCHEMA IF NOT EXISTS dogos_social;

CREATE TABLE IF NOT EXISTS dogos_social.preferences (
  user_id TEXT PRIMARY KEY REFERENCES public.users(id) ON DELETE CASCADE,
  global_leaderboard_opt_in BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dogos_social.shares (
  id TEXT PRIMARY KEY,
  post_id TEXT NOT NULL UNIQUE REFERENCES public.posts(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  pet_id TEXT REFERENCES public.pets(id) ON DELETE SET NULL,
  source_type TEXT NOT NULL,
  source_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  headline TEXT NOT NULL,
  summary TEXT NOT NULL,
  payload JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  CONSTRAINT social_share_source_type CHECK (
    source_type IN ('CARE_EVENT', 'HUMAN_SKILL_ATTEMPT')
  ),
  CONSTRAINT social_share_kind CHECK (
    kind IN ('ADVENTURE_MEMORY', 'DISCOVERY', 'SKILL_MOMENT', 'GOOD_READ', 'CHAPTER_MOMENT')
  ),
  CONSTRAINT social_share_headline_length CHECK (char_length(headline) BETWEEN 1 AND 120),
  CONSTRAINT social_share_summary_length CHECK (char_length(summary) BETWEEN 1 AND 320),
  CONSTRAINT social_share_source_id_length CHECK (char_length(source_id) BETWEEN 1 AND 128),
  CONSTRAINT social_share_payload_size CHECK (octet_length(payload::text) <= 4096),
  CONSTRAINT social_share_source_unique UNIQUE (user_id, source_type, source_id, kind)
);

CREATE INDEX IF NOT EXISTS social_shares_created_at_idx
  ON dogos_social.shares (created_at DESC, id);
CREATE INDEX IF NOT EXISTS social_shares_user_created_at_idx
  ON dogos_social.shares (user_id, created_at DESC, id);
CREATE INDEX IF NOT EXISTS social_shares_pet_created_at_idx
  ON dogos_social.shares (pet_id, created_at DESC, id)
  WHERE pet_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS dogos_social.reactions (
  id TEXT PRIMARY KEY,
  share_id TEXT NOT NULL REFERENCES dogos_social.shares(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  reaction TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  CONSTRAINT social_reaction_kind CHECK (
    reaction IN ('NICE_READ', 'GOOD_CALL', 'TRYING_THIS', 'ADVENTURE_INSPIRATION', 'CHEER')
  ),
  CONSTRAINT social_reaction_unique UNIQUE (share_id, user_id, reaction)
);

CREATE INDEX IF NOT EXISTS social_reactions_share_idx
  ON dogos_social.reactions (share_id, created_at DESC);
CREATE INDEX IF NOT EXISTS social_reactions_user_idx
  ON dogos_social.reactions (user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS dogos_social.human_skill_attempts (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  challenge_key TEXT NOT NULL,
  challenge_version TEXT NOT NULL,
  scenario_key TEXT NOT NULL,
  issued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL,
  completed_at TIMESTAMPTZ,
  response JSONB,
  score INTEGER,
  receipt JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  CONSTRAINT human_skill_attempt_id_length CHECK (char_length(id) BETWEEN 8 AND 128),
  CONSTRAINT human_skill_challenge_key CHECK (
    challenge_key IN ('MAKE_IT_EASIER', 'CATCH_THE_GOOD', 'PAIRING_LAB', 'MARKER_TIMING')
  ),
  CONSTRAINT human_skill_version_length CHECK (char_length(challenge_version) BETWEEN 1 AND 64),
  CONSTRAINT human_skill_scenario_length CHECK (char_length(scenario_key) BETWEEN 1 AND 96),
  CONSTRAINT human_skill_expiry_order CHECK (expires_at > issued_at),
  CONSTRAINT human_skill_score CHECK (score IS NULL OR score BETWEEN 0 AND 100),
  CONSTRAINT human_skill_completion_shape CHECK (
    (completed_at IS NULL AND response IS NULL AND score IS NULL AND receipt IS NULL)
    OR (completed_at IS NOT NULL AND response IS NOT NULL AND score IS NOT NULL AND receipt IS NOT NULL)
  ),
  CONSTRAINT human_skill_response_size CHECK (response IS NULL OR octet_length(response::text) <= 2048),
  CONSTRAINT human_skill_receipt_size CHECK (receipt IS NULL OR octet_length(receipt::text) <= 4096)
);

CREATE INDEX IF NOT EXISTS human_skill_attempts_user_time_idx
  ON dogos_social.human_skill_attempts (user_id, completed_at DESC, id)
  WHERE completed_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS human_skill_attempts_user_challenge_idx
  ON dogos_social.human_skill_attempts (user_id, challenge_key, completed_at DESC, id)
  WHERE completed_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS dogos_social.packs (
  id TEXT PRIMARY KEY,
  owner_user_id TEXT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  slug TEXT NOT NULL UNIQUE,
  scope TEXT NOT NULL DEFAULT 'LOCAL',
  region_key TEXT,
  visibility TEXT NOT NULL DEFAULT 'PUBLIC',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  CONSTRAINT social_pack_name_length CHECK (char_length(name) BETWEEN 2 AND 64),
  CONSTRAINT social_pack_slug_length CHECK (char_length(slug) BETWEEN 2 AND 72),
  CONSTRAINT social_pack_scope CHECK (scope IN ('LOCAL', 'FRIENDS')),
  CONSTRAINT social_pack_visibility CHECK (visibility IN ('PUBLIC', 'PRIVATE')),
  CONSTRAINT social_pack_region_shape CHECK (
    (scope = 'LOCAL' AND region_key IS NOT NULL AND char_length(region_key) BETWEEN 2 AND 64)
    OR (scope = 'FRIENDS' AND region_key IS NULL)
  )
);

CREATE INDEX IF NOT EXISTS social_packs_region_idx
  ON dogos_social.packs (region_key, created_at DESC)
  WHERE scope = 'LOCAL' AND visibility = 'PUBLIC';

CREATE TABLE IF NOT EXISTS dogos_social.pack_memberships (
  pack_id TEXT NOT NULL REFERENCES dogos_social.packs(id) ON DELETE CASCADE,
  user_id TEXT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  role TEXT NOT NULL DEFAULT 'MEMBER',
  status TEXT NOT NULL DEFAULT 'ACTIVE',
  joined_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  PRIMARY KEY (pack_id, user_id),
  CONSTRAINT social_pack_member_role CHECK (role IN ('OWNER', 'MEMBER')),
  CONSTRAINT social_pack_member_status CHECK (status IN ('ACTIVE', 'LEFT'))
);

CREATE INDEX IF NOT EXISTS social_pack_memberships_user_idx
  ON dogos_social.pack_memberships (user_id, status, joined_at DESC);

CREATE TABLE IF NOT EXISTS dogos_social.competition_receipts (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  season_key TEXT NOT NULL,
  policy_version TEXT NOT NULL,
  score INTEGER NOT NULL,
  components JSONB NOT NULL,
  source_hash TEXT NOT NULL,
  generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  CONSTRAINT social_competition_score CHECK (score BETWEEN 0 AND 600),
  CONSTRAINT social_competition_season_length CHECK (char_length(season_key) BETWEEN 8 AND 32),
  CONSTRAINT social_competition_policy_length CHECK (char_length(policy_version) BETWEEN 1 AND 64),
  CONSTRAINT social_competition_source_hash CHECK (source_hash ~ '^[0-9a-f]{64}$'),
  CONSTRAINT social_competition_components_size CHECK (octet_length(components::text) <= 4096),
  CONSTRAINT social_competition_snapshot_unique UNIQUE (user_id, season_key, policy_version, source_hash)
);

CREATE INDEX IF NOT EXISTS social_competition_user_season_idx
  ON dogos_social.competition_receipts (user_id, season_key, generated_at DESC, id);
CREATE INDEX IF NOT EXISTS social_competition_season_score_idx
  ON dogos_social.competition_receipts (season_key, score DESC, generated_at DESC);

CREATE OR REPLACE FUNCTION dogos_social.reject_completed_human_skill_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  IF OLD.completed_at IS NOT NULL AND NEW IS DISTINCT FROM OLD THEN
    RAISE EXCEPTION 'completed human skill attempts are immutable'
      USING ERRCODE = '23514';
  END IF;
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS human_skill_attempts_immutable_after_completion
  ON dogos_social.human_skill_attempts;
CREATE TRIGGER human_skill_attempts_immutable_after_completion
BEFORE UPDATE ON dogos_social.human_skill_attempts
FOR EACH ROW
EXECUTE FUNCTION dogos_social.reject_completed_human_skill_mutation();

CREATE OR REPLACE FUNCTION dogos_social.reject_competition_receipt_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  IF NEW IS DISTINCT FROM OLD THEN
    RAISE EXCEPTION 'social competition receipts are immutable'
      USING ERRCODE = '23514';
  END IF;
  RETURN OLD;
END;
$$;

DROP TRIGGER IF EXISTS social_competition_receipts_immutable
  ON dogos_social.competition_receipts;
CREATE TRIGGER social_competition_receipts_immutable
BEFORE UPDATE ON dogos_social.competition_receipts
FOR EACH ROW
EXECUTE FUNCTION dogos_social.reject_competition_receipt_mutation();
