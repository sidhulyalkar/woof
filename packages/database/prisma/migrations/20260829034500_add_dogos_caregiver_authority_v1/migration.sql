-- dogOS Caregiver Authority v1
--
-- Temporary caregiver authority is explicit, pet-scoped, accepted by the
-- recipient, time-bounded, and revocable. It is intentionally separate from
-- canonical household membership, pet ownership, reward authority, and
-- longitudinal intelligence evidence.

CREATE SCHEMA IF NOT EXISTS dogos_caregiver;

CREATE TABLE dogos_caregiver.grants (
  id TEXT PRIMARY KEY,
  pet_id TEXT NOT NULL REFERENCES public.pets(id) ON DELETE CASCADE,
  issuer_user_id TEXT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  recipient_user_id TEXT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  request_key TEXT NOT NULL,
  policy_version TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'PENDING_ACCEPTANCE',
  issued_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  accepted_at TIMESTAMPTZ,
  declined_at TIMESTAMPTZ,
  expires_at TIMESTAMPTZ NOT NULL,
  revoked_at TIMESTAMPTZ,
  revoked_by_user_id TEXT REFERENCES public.users(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  CONSTRAINT caregiver_grant_id_length CHECK (char_length(id) BETWEEN 8 AND 128),
  CONSTRAINT caregiver_grant_request_key_length CHECK (char_length(request_key) BETWEEN 8 AND 128),
  CONSTRAINT caregiver_grant_policy_version_length CHECK (char_length(policy_version) BETWEEN 1 AND 64),
  CONSTRAINT caregiver_grant_not_self CHECK (issuer_user_id <> recipient_user_id),
  CONSTRAINT caregiver_grant_status CHECK (
    status IN ('PENDING_ACCEPTANCE', 'ACTIVE', 'DECLINED', 'REVOKED')
  ),
  CONSTRAINT caregiver_grant_expiry_order CHECK (expires_at > issued_at),
  CONSTRAINT caregiver_grant_v1_min_duration CHECK (expires_at >= issued_at + INTERVAL '15 minutes'),
  CONSTRAINT caregiver_grant_v1_max_duration CHECK (expires_at <= issued_at + INTERVAL '31 days'),
  CONSTRAINT caregiver_grant_accepted_order CHECK (
    accepted_at IS NULL OR accepted_at >= issued_at
  ),
  CONSTRAINT caregiver_grant_declined_order CHECK (
    declined_at IS NULL OR declined_at >= issued_at
  ),
  CONSTRAINT caregiver_grant_revoked_order CHECK (
    revoked_at IS NULL OR revoked_at >= issued_at
  ),
  CONSTRAINT caregiver_grant_accept_before_revoke CHECK (
    accepted_at IS NULL OR revoked_at IS NULL OR accepted_at <= revoked_at
  ),
  CONSTRAINT caregiver_grant_state_shape CHECK (
    (
      status = 'PENDING_ACCEPTANCE'
      AND accepted_at IS NULL
      AND declined_at IS NULL
      AND revoked_at IS NULL
      AND revoked_by_user_id IS NULL
    )
    OR (
      status = 'ACTIVE'
      AND accepted_at IS NOT NULL
      AND declined_at IS NULL
      AND revoked_at IS NULL
      AND revoked_by_user_id IS NULL
    )
    OR (
      status = 'DECLINED'
      AND accepted_at IS NULL
      AND declined_at IS NOT NULL
      AND revoked_at IS NULL
      AND revoked_by_user_id IS NULL
    )
    OR (
      status = 'REVOKED'
      AND declined_at IS NULL
      AND revoked_at IS NOT NULL
      AND revoked_by_user_id IS NOT NULL
    )
  ),
  CONSTRAINT caregiver_grant_request_replay_unique UNIQUE (issuer_user_id, request_key),
  CONSTRAINT caregiver_grant_identity_unique UNIQUE (id, pet_id, recipient_user_id)
);

-- A recipient has at most one live invitation/authority relationship for a pet.
-- Changing the capability set or issuer requires the existing grant to leave the
-- live state first, preserving an auditable lifecycle rather than mutating scope.
CREATE UNIQUE INDEX caregiver_grant_one_live_pet_recipient_idx
  ON dogos_caregiver.grants (pet_id, recipient_user_id)
  WHERE status IN ('PENDING_ACCEPTANCE', 'ACTIVE');

CREATE INDEX caregiver_grant_recipient_state_idx
  ON dogos_caregiver.grants (recipient_user_id, status, expires_at DESC, id);
CREATE INDEX caregiver_grant_issuer_state_idx
  ON dogos_caregiver.grants (issuer_user_id, status, expires_at DESC, id);
CREATE INDEX caregiver_grant_pet_state_idx
  ON dogos_caregiver.grants (pet_id, status, expires_at DESC, id);

CREATE TABLE dogos_caregiver.grant_capabilities (
  grant_id TEXT NOT NULL REFERENCES dogos_caregiver.grants(id) ON DELETE CASCADE,
  capability TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  PRIMARY KEY (grant_id, capability),
  CONSTRAINT caregiver_capability_kind CHECK (
    capability IN ('VIEW_TODAY', 'LOG_OBSERVATION')
  )
);

CREATE TABLE dogos_caregiver.grant_receipts (
  id TEXT PRIMARY KEY,
  grant_id TEXT NOT NULL REFERENCES dogos_caregiver.grants(id) ON DELETE CASCADE,
  actor_user_id TEXT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  transition TEXT NOT NULL,
  status_after TEXT NOT NULL,
  capabilities TEXT[] NOT NULL,
  expires_at TIMESTAMPTZ NOT NULL,
  policy_version TEXT NOT NULL,
  source_hash TEXT NOT NULL,
  occurred_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  CONSTRAINT caregiver_receipt_id_length CHECK (char_length(id) BETWEEN 8 AND 128),
  CONSTRAINT caregiver_receipt_transition CHECK (
    transition IN ('ISSUED', 'ACCEPTED', 'DECLINED', 'REVOKED')
  ),
  CONSTRAINT caregiver_receipt_status_after CHECK (
    status_after IN ('PENDING_ACCEPTANCE', 'ACTIVE', 'DECLINED', 'REVOKED')
  ),
  CONSTRAINT caregiver_receipt_transition_shape CHECK (
    (transition = 'ISSUED' AND status_after = 'PENDING_ACCEPTANCE')
    OR (transition = 'ACCEPTED' AND status_after = 'ACTIVE')
    OR (transition = 'DECLINED' AND status_after = 'DECLINED')
    OR (transition = 'REVOKED' AND status_after = 'REVOKED')
  ),
  CONSTRAINT caregiver_receipt_capability_count CHECK (cardinality(capabilities) BETWEEN 1 AND 2),
  CONSTRAINT caregiver_receipt_capabilities CHECK (
    capabilities <@ ARRAY['VIEW_TODAY', 'LOG_OBSERVATION']::TEXT[]
  ),
  CONSTRAINT caregiver_receipt_policy_version_length CHECK (char_length(policy_version) BETWEEN 1 AND 64),
  CONSTRAINT caregiver_receipt_source_hash CHECK (source_hash ~ '^[0-9a-f]{64}$'),
  CONSTRAINT caregiver_receipt_transition_unique UNIQUE (grant_id, transition)
);

CREATE INDEX caregiver_receipt_grant_time_idx
  ON dogos_caregiver.grant_receipts (grant_id, occurred_at ASC, id);

CREATE TABLE dogos_caregiver.observations (
  id TEXT PRIMARY KEY,
  grant_id TEXT NOT NULL,
  pet_id TEXT NOT NULL,
  actor_user_id TEXT NOT NULL,
  authority_class TEXT NOT NULL DEFAULT 'CONTEXT_ONLY',
  kind TEXT NOT NULL,
  summary TEXT NOT NULL,
  note TEXT,
  context JSONB NOT NULL DEFAULT '{}'::jsonb,
  observed_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

  CONSTRAINT caregiver_observation_id_length CHECK (char_length(id) BETWEEN 8 AND 128),
  CONSTRAINT caregiver_observation_authority_class CHECK (authority_class = 'CONTEXT_ONLY'),
  CONSTRAINT caregiver_observation_kind CHECK (
    kind IN ('ROUTINE', 'ACTIVITY_RESPONSE', 'BEHAVIOR', 'HANDOFF_NOTE')
  ),
  CONSTRAINT caregiver_observation_summary_length CHECK (char_length(summary) BETWEEN 1 AND 240),
  CONSTRAINT caregiver_observation_note_length CHECK (note IS NULL OR char_length(note) <= 500),
  CONSTRAINT caregiver_observation_context_object CHECK (jsonb_typeof(context) = 'object'),
  CONSTRAINT caregiver_observation_context_size CHECK (octet_length(context::text) <= 2048),
  CONSTRAINT caregiver_observation_grant_identity_fkey
    FOREIGN KEY (grant_id, pet_id, actor_user_id)
    REFERENCES dogos_caregiver.grants(id, pet_id, recipient_user_id)
    ON DELETE CASCADE
);

CREATE INDEX caregiver_observation_actor_time_idx
  ON dogos_caregiver.observations (actor_user_id, observed_at DESC, id);
CREATE INDEX caregiver_observation_pet_time_idx
  ON dogos_caregiver.observations (pet_id, observed_at DESC, id);

-- Grant identity/scope does not morph after issuance. Lifecycle fields may move
-- through their constrained state machine, but a grant cannot become a grant for
-- another pet/person/expiry/policy under the same durable ID.
CREATE OR REPLACE FUNCTION dogos_caregiver.reject_grant_identity_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  IF NEW.pet_id IS DISTINCT FROM OLD.pet_id
     OR NEW.issuer_user_id IS DISTINCT FROM OLD.issuer_user_id
     OR NEW.recipient_user_id IS DISTINCT FROM OLD.recipient_user_id
     OR NEW.request_key IS DISTINCT FROM OLD.request_key
     OR NEW.policy_version IS DISTINCT FROM OLD.policy_version
     OR NEW.issued_at IS DISTINCT FROM OLD.issued_at
     OR NEW.expires_at IS DISTINCT FROM OLD.expires_at
  THEN
    RAISE EXCEPTION 'caregiver grant identity is immutable after issuance'
      USING ERRCODE = '23514';
  END IF;
  RETURN NEW;
END;
$$;

CREATE TRIGGER caregiver_grant_identity_immutable
BEFORE UPDATE ON dogos_caregiver.grants
FOR EACH ROW
EXECUTE FUNCTION dogos_caregiver.reject_grant_identity_mutation();

-- Capabilities are inserted before the ISSUED receipt in the issuance
-- transaction. After that receipt exists, scope can never be broadened or
-- narrowed in place.
CREATE OR REPLACE FUNCTION dogos_caregiver.reject_post_issue_capability_insert()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM dogos_caregiver.grant_receipts receipt
    WHERE receipt.grant_id = NEW.grant_id
      AND receipt.transition = 'ISSUED'
  ) THEN
    RAISE EXCEPTION 'caregiver grant capabilities cannot be broadened after issuance'
      USING ERRCODE = '23514';
  END IF;
  RETURN NEW;
END;
$$;

CREATE TRIGGER caregiver_capability_no_post_issue_insert
BEFORE INSERT ON dogos_caregiver.grant_capabilities
FOR EACH ROW
EXECUTE FUNCTION dogos_caregiver.reject_post_issue_capability_insert();

CREATE OR REPLACE FUNCTION dogos_caregiver.reject_capability_update()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  IF NEW IS DISTINCT FROM OLD THEN
    RAISE EXCEPTION 'caregiver grant capabilities are immutable'
      USING ERRCODE = '23514';
  END IF;
  RETURN OLD;
END;
$$;

CREATE TRIGGER caregiver_capability_immutable
BEFORE UPDATE ON dogos_caregiver.grant_capabilities
FOR EACH ROW
EXECUTE FUNCTION dogos_caregiver.reject_capability_update();

CREATE OR REPLACE FUNCTION dogos_caregiver.reject_direct_capability_delete()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  IF EXISTS (SELECT 1 FROM dogos_caregiver.grants WHERE id = OLD.grant_id) THEN
    RAISE EXCEPTION 'caregiver grant capabilities cannot be deleted after issuance'
      USING ERRCODE = '23514';
  END IF;
  RETURN OLD;
END;
$$;

CREATE TRIGGER caregiver_capability_delete_only_with_grant
BEFORE DELETE ON dogos_caregiver.grant_capabilities
FOR EACH ROW
EXECUTE FUNCTION dogos_caregiver.reject_direct_capability_delete();

-- Lifecycle receipts are immutable while their grant exists. Cascading privacy
-- deletion remains possible by deleting the parent grant/user/pet.
CREATE OR REPLACE FUNCTION dogos_caregiver.reject_receipt_update()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  IF NEW IS DISTINCT FROM OLD THEN
    RAISE EXCEPTION 'caregiver authority receipts are immutable'
      USING ERRCODE = '23514';
  END IF;
  RETURN OLD;
END;
$$;

CREATE TRIGGER caregiver_receipt_immutable
BEFORE UPDATE ON dogos_caregiver.grant_receipts
FOR EACH ROW
EXECUTE FUNCTION dogos_caregiver.reject_receipt_update();

CREATE OR REPLACE FUNCTION dogos_caregiver.reject_direct_receipt_delete()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  IF EXISTS (SELECT 1 FROM dogos_caregiver.grants WHERE id = OLD.grant_id) THEN
    RAISE EXCEPTION 'caregiver authority receipts cannot be deleted directly'
      USING ERRCODE = '23514';
  END IF;
  RETURN OLD;
END;
$$;

CREATE TRIGGER caregiver_receipt_delete_only_with_grant
BEFORE DELETE ON dogos_caregiver.grant_receipts
FOR EACH ROW
EXECUTE FUNCTION dogos_caregiver.reject_direct_receipt_delete();

-- Caregiver observations are source evidence, not mutable pet truth. Corrections
-- require a later explicit supersession policy rather than rewriting history.
CREATE OR REPLACE FUNCTION dogos_caregiver.reject_observation_update()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  IF NEW IS DISTINCT FROM OLD THEN
    RAISE EXCEPTION 'caregiver observations are immutable'
      USING ERRCODE = '23514';
  END IF;
  RETURN OLD;
END;
$$;

CREATE TRIGGER caregiver_observation_immutable
BEFORE UPDATE ON dogos_caregiver.observations
FOR EACH ROW
EXECUTE FUNCTION dogos_caregiver.reject_observation_update();

CREATE OR REPLACE FUNCTION dogos_caregiver.reject_direct_observation_delete()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  IF EXISTS (SELECT 1 FROM dogos_caregiver.grants WHERE id = OLD.grant_id) THEN
    RAISE EXCEPTION 'caregiver observations cannot be deleted directly'
      USING ERRCODE = '23514';
  END IF;
  RETURN OLD;
END;
$$;

CREATE TRIGGER caregiver_observation_delete_only_with_grant
BEFORE DELETE ON dogos_caregiver.observations
FOR EACH ROW
EXECUTE FUNCTION dogos_caregiver.reject_direct_observation_delete();

-- Direct database inserts of caregiver observations must satisfy current
-- persisted authority. A stale client or realtime session cannot write after
-- expiry/revocation, and LOG_OBSERVATION must have been present at issuance.
CREATE OR REPLACE FUNCTION dogos_caregiver.enforce_observation_authority()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
  grant_row RECORD;
BEGIN
  SELECT status, issued_at, accepted_at, expires_at, revoked_at
  INTO grant_row
  FROM dogos_caregiver.grants
  WHERE id = NEW.grant_id;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'caregiver grant not found'
      USING ERRCODE = '23503';
  END IF;

  IF grant_row.status <> 'ACTIVE'
     OR grant_row.accepted_at IS NULL
     OR grant_row.revoked_at IS NOT NULL
     OR CURRENT_TIMESTAMP >= grant_row.expires_at
  THEN
    RAISE EXCEPTION 'caregiver grant is not currently active'
      USING ERRCODE = '42501';
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM dogos_caregiver.grant_capabilities capability
    WHERE capability.grant_id = NEW.grant_id
      AND capability.capability = 'LOG_OBSERVATION'
  ) THEN
    RAISE EXCEPTION 'caregiver grant does not allow observations'
      USING ERRCODE = '42501';
  END IF;

  IF NEW.observed_at < grant_row.issued_at
     OR NEW.observed_at >= grant_row.expires_at
     OR NEW.observed_at > CURRENT_TIMESTAMP + INTERVAL '5 minutes'
  THEN
    RAISE EXCEPTION 'caregiver observation time is outside the authorized window'
      USING ERRCODE = '23514';
  END IF;

  RETURN NEW;
END;
$$;

CREATE TRIGGER caregiver_observation_authority
BEFORE INSERT ON dogos_caregiver.observations
FOR EACH ROW
EXECUTE FUNCTION dogos_caregiver.enforce_observation_authority();
