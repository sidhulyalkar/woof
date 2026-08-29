-- Pre-release hardening for dogOS Caregiver Authority v1.
-- This migration will be squashed into the feature migration before release.

DROP INDEX IF EXISTS dogos_caregiver.caregiver_grant_one_live_pet_recipient_idx;

-- Derived expiry means a stored ACTIVE/PENDING row may already be expired. A
-- status-only unique index would therefore block future grants forever. Serialize
-- issuance per pet/recipient and reject only overlapping pending/active windows.
CREATE OR REPLACE FUNCTION dogos_caregiver.enforce_grant_issuance()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  PERFORM pg_advisory_xact_lock(
    hashtextextended('dogos-caregiver:' || NEW.pet_id || ':' || NEW.recipient_user_id, 0)
  );

  IF EXISTS (
    SELECT 1
    FROM public.blocked_users blocked
    WHERE (blocked.user_id = NEW.issuer_user_id AND blocked.blocked_id = NEW.recipient_user_id)
       OR (blocked.user_id = NEW.recipient_user_id AND blocked.blocked_id = NEW.issuer_user_id)
  ) THEN
    RAISE EXCEPTION 'caregiver grant cannot be issued across a blocked relationship'
      USING ERRCODE = '42501';
  END IF;

  IF EXISTS (
    SELECT 1
    FROM dogos_caregiver.grants existing
    WHERE existing.pet_id = NEW.pet_id
      AND existing.recipient_user_id = NEW.recipient_user_id
      AND existing.status IN ('PENDING_ACCEPTANCE', 'ACTIVE')
      AND existing.issued_at < NEW.expires_at
      AND existing.expires_at > NEW.issued_at
  ) THEN
    RAISE EXCEPTION 'caregiver grant overlaps an existing live authority window'
      USING ERRCODE = '23505';
  END IF;

  RETURN NEW;
END;
$$;

CREATE TRIGGER caregiver_grant_issuance_guard
BEFORE INSERT ON dogos_caregiver.grants
FOR EACH ROW
EXECUTE FUNCTION dogos_caregiver.enforce_grant_issuance();

-- Acceptance must remain impossible after either participant blocks the other,
-- even if a caller bypasses the API service and tries a direct state update.
CREATE OR REPLACE FUNCTION dogos_caregiver.enforce_grant_activation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
  IF OLD.status = 'PENDING_ACCEPTANCE' AND NEW.status = 'ACTIVE' THEN
    IF EXISTS (
      SELECT 1
      FROM public.blocked_users blocked
      WHERE (blocked.user_id = NEW.issuer_user_id AND blocked.blocked_id = NEW.recipient_user_id)
         OR (blocked.user_id = NEW.recipient_user_id AND blocked.blocked_id = NEW.issuer_user_id)
    ) THEN
      RAISE EXCEPTION 'caregiver grant cannot activate across a blocked relationship'
        USING ERRCODE = '42501';
    END IF;
  END IF;
  RETURN NEW;
END;
$$;

CREATE TRIGGER caregiver_grant_activation_guard
BEFORE UPDATE ON dogos_caregiver.grants
FOR EACH ROW
EXECUTE FUNCTION dogos_caregiver.enforce_grant_activation();

-- Receipt rows must actually attest to the current immutable grant scope and the
-- correct lifecycle actor. Receipt capability arrays are canonicalized because
-- capability scope is set-semantic; presentation order must never decide whether
-- equivalent authority is accepted or rejected. This also makes
-- LOG_OBSERVATION imply VIEW_TODAY in v1.
CREATE OR REPLACE FUNCTION dogos_caregiver.validate_grant_receipt()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
  grant_row RECORD;
  actual_capabilities TEXT[];
BEGIN
  SELECT issuer_user_id, recipient_user_id, policy_version, expires_at, revoked_by_user_id
  INTO grant_row
  FROM dogos_caregiver.grants
  WHERE id = NEW.grant_id;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'caregiver receipt grant not found'
      USING ERRCODE = '23503';
  END IF;

  SELECT ARRAY_AGG(capability ORDER BY capability)
  INTO actual_capabilities
  FROM dogos_caregiver.grant_capabilities
  WHERE grant_id = NEW.grant_id;

  IF actual_capabilities IS NULL OR cardinality(actual_capabilities) = 0 THEN
    RAISE EXCEPTION 'caregiver receipt requires at least one grant capability'
      USING ERRCODE = '23514';
  END IF;

  SELECT ARRAY_AGG(receipt_capability ORDER BY receipt_capability)
  INTO NEW.capabilities
  FROM UNNEST(NEW.capabilities) AS receipt_capability;

  IF 'LOG_OBSERVATION' = ANY(actual_capabilities)
     AND NOT ('VIEW_TODAY' = ANY(actual_capabilities)) THEN
    RAISE EXCEPTION 'LOG_OBSERVATION requires VIEW_TODAY in caregiver v1'
      USING ERRCODE = '23514';
  END IF;

  IF NEW.capabilities <> actual_capabilities
     OR NEW.policy_version <> grant_row.policy_version
     OR NEW.expires_at <> grant_row.expires_at
  THEN
    RAISE EXCEPTION 'caregiver receipt snapshot does not match grant authority'
      USING ERRCODE = '23514';
  END IF;

  IF (NEW.transition = 'ISSUED' AND NEW.actor_user_id <> grant_row.issuer_user_id)
     OR (NEW.transition IN ('ACCEPTED', 'DECLINED') AND NEW.actor_user_id <> grant_row.recipient_user_id)
     OR (
       NEW.transition = 'REVOKED'
       AND (
         grant_row.revoked_by_user_id IS NULL
         OR NEW.actor_user_id <> grant_row.revoked_by_user_id
       )
     )
  THEN
    RAISE EXCEPTION 'caregiver receipt actor does not match lifecycle authority'
      USING ERRCODE = '23514';
  END IF;

  RETURN NEW;
END;
$$;

CREATE TRIGGER caregiver_receipt_snapshot_guard
BEFORE INSERT ON dogos_caregiver.grant_receipts
FOR EACH ROW
EXECUTE FUNCTION dogos_caregiver.validate_grant_receipt();

-- A block immediately disables direct caregiver evidence writes even if the
-- mutable grant row still says ACTIVE. Grant revocation can remain an explicit
-- audited lifecycle transition rather than making block handling mutate another
-- authority domain behind the user's back.
CREATE OR REPLACE FUNCTION dogos_caregiver.enforce_observation_authority()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
  grant_row RECORD;
BEGIN
  SELECT status, issuer_user_id, recipient_user_id, issued_at, accepted_at, expires_at, revoked_at
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

  IF EXISTS (
    SELECT 1
    FROM public.blocked_users blocked
    WHERE (blocked.user_id = grant_row.issuer_user_id AND blocked.blocked_id = grant_row.recipient_user_id)
       OR (blocked.user_id = grant_row.recipient_user_id AND blocked.blocked_id = grant_row.issuer_user_id)
  ) THEN
    RAISE EXCEPTION 'caregiver authority is disabled by relationship block'
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
