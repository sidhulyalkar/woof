-- dogOS Connectors operational metadata lives outside the canonical public schema.
-- No raw provider payloads or OAuth secrets are stored here.
CREATE SCHEMA IF NOT EXISTS dogos_connectors;

CREATE TABLE dogos_connectors.connections (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id TEXT NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
  provider TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'PARTNER_REQUIRED',
  external_account_id TEXT,
  display_label TEXT,
  granted_scopes TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
  connected_at TIMESTAMPTZ,
  last_sync_at TIMESTAMPTZ,
  revoked_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT connector_connections_user_provider_key UNIQUE (user_id, provider),
  CONSTRAINT connector_connections_provider_check CHECK (
    provider IN ('FI', 'TRACTIVE', 'VET_PARTNER', 'CHEWY', 'PETCO')
  ),
  CONSTRAINT connector_connections_status_check CHECK (
    status IN ('PARTNER_REQUIRED', 'CONNECTED', 'REAUTH_REQUIRED', 'REVOKED')
  )
);

CREATE INDEX connector_connections_user_status_idx
  ON dogos_connectors.connections(user_id, status);

CREATE TABLE dogos_connectors.pet_identities (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  connection_id UUID NOT NULL REFERENCES dogos_connectors.connections(id) ON DELETE CASCADE,
  pet_id TEXT NOT NULL REFERENCES public.pets(id) ON DELETE CASCADE,
  external_pet_id TEXT NOT NULL,
  external_pet_label TEXT,
  verified_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT connector_pet_identity_external_key UNIQUE (connection_id, external_pet_id),
  CONSTRAINT connector_pet_identity_pet_key UNIQUE (connection_id, pet_id)
);

CREATE INDEX connector_pet_identities_pet_idx
  ON dogos_connectors.pet_identities(pet_id);

CREATE FUNCTION dogos_connectors.enforce_pet_identity_owner()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM dogos_connectors.connections AS connection
    INNER JOIN public.pets AS pet ON pet.id = NEW.pet_id
    WHERE connection.id = NEW.connection_id
      AND pet.owner_id = connection.user_id
  ) THEN
    RAISE EXCEPTION 'connector pet identity must belong to the connection owner';
  END IF;
  RETURN NEW;
END;
$$;

CREATE TRIGGER connector_pet_identity_owner_guard
BEFORE INSERT OR UPDATE ON dogos_connectors.pet_identities
FOR EACH ROW
EXECUTE FUNCTION dogos_connectors.enforce_pet_identity_owner();

CREATE TABLE dogos_connectors.sync_cursors (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  connection_id UUID NOT NULL REFERENCES dogos_connectors.connections(id) ON DELETE CASCADE,
  resource_type TEXT NOT NULL,
  cursor_value TEXT,
  watermark_at TIMESTAMPTZ,
  last_successful_sync_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT connector_sync_cursor_resource_key UNIQUE (connection_id, resource_type)
);

CREATE TABLE dogos_connectors.import_receipts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  connection_id UUID NOT NULL REFERENCES dogos_connectors.connections(id) ON DELETE CASCADE,
  resource_type TEXT NOT NULL,
  external_object_id TEXT NOT NULL,
  payload_hash TEXT NOT NULL,
  disposition TEXT NOT NULL,
  canonical_ref_type TEXT,
  canonical_ref_id TEXT,
  occurred_at TIMESTAMPTZ,
  imported_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  detail_code TEXT,
  CONSTRAINT connector_import_receipt_object_key UNIQUE (
    connection_id,
    resource_type,
    external_object_id
  ),
  CONSTRAINT connector_import_receipt_hash_check CHECK (payload_hash ~ '^[0-9a-f]{64}$'),
  CONSTRAINT connector_import_receipt_disposition_check CHECK (
    disposition IN ('IMPORTED', 'SKIPPED', 'FAILED')
  )
);

CREATE INDEX connector_import_receipts_connection_time_idx
  ON dogos_connectors.import_receipts(connection_id, imported_at DESC);

CREATE TABLE dogos_connectors.revocation_receipts (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  connection_id UUID NOT NULL REFERENCES dogos_connectors.connections(id) ON DELETE CASCADE,
  mode TEXT NOT NULL,
  status TEXT NOT NULL,
  remote_receipt_ref TEXT,
  detail_code TEXT,
  attempted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  completed_at TIMESTAMPTZ,
  CONSTRAINT connector_revocation_mode_check CHECK (
    mode IN ('LOCAL_CREDENTIAL_DELETE', 'REMOTE_REVOKE')
  ),
  CONSTRAINT connector_revocation_status_check CHECK (
    status IN ('SUCCEEDED', 'UNAVAILABLE', 'FAILED')
  )
);

CREATE INDEX connector_revocations_connection_time_idx
  ON dogos_connectors.revocation_receipts(connection_id, attempted_at DESC);
