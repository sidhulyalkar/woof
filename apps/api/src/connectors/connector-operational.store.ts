import { Injectable } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import type { ConnectorProvider } from './connectors.types';

type ConnectionRow = {
  id: string;
  provider: ConnectorProvider;
  status: 'PARTNER_REQUIRED' | 'CONNECTED' | 'REAUTH_REQUIRED' | 'REVOKED';
  external_account_id: string | null;
  display_label: string | null;
  granted_scopes: string[];
  connected_at: Date | null;
  last_sync_at: Date | null;
  revoked_at: Date | null;
  created_at: Date;
  updated_at: Date;
};

type PetIdentityRow = {
  connection_id: string;
  pet_id: string;
  external_pet_id: string;
  external_pet_label: string | null;
  verified_at: Date | null;
};

type ImportReceiptRow = {
  id: string;
  connection_id: string;
  resource_type: string;
  external_object_id: string;
  payload_hash: string;
  disposition: 'IMPORTED' | 'SKIPPED' | 'FAILED';
  canonical_ref_type: string | null;
  canonical_ref_id: string | null;
  occurred_at: Date | null;
  imported_at: Date;
  detail_code: string | null;
};

type SyncCursorRow = {
  resource_type: string;
  cursor_value: string | null;
  watermark_at: Date | null;
  last_successful_sync_at: Date | null;
};

function textArray(values: string[]) {
  const unique = [...new Set(values)].sort();
  return unique.length > 0
    ? Prisma.sql`ARRAY[${Prisma.join(unique)}]::text[]`
    : Prisma.sql`ARRAY[]::text[]`;
}

function toConnection(row: ConnectionRow) {
  return {
    id: row.id,
    provider: row.provider,
    status: row.status,
    externalAccountId: row.external_account_id,
    displayLabel: row.display_label,
    grantedScopes: row.granted_scopes,
    connectedAt: row.connected_at?.toISOString() ?? null,
    lastSyncAt: row.last_sync_at?.toISOString() ?? null,
    revokedAt: row.revoked_at?.toISOString() ?? null,
    createdAt: row.created_at.toISOString(),
    updatedAt: row.updated_at.toISOString(),
  };
}

function toReceipt(row: ImportReceiptRow) {
  return {
    id: row.id,
    connectionId: row.connection_id,
    resourceType: row.resource_type,
    externalObjectId: row.external_object_id,
    payloadHash: row.payload_hash,
    disposition: row.disposition,
    canonicalRefType: row.canonical_ref_type,
    canonicalRefId: row.canonical_ref_id,
    occurredAt: row.occurred_at?.toISOString() ?? null,
    importedAt: row.imported_at.toISOString(),
    detailCode: row.detail_code,
  };
}

@Injectable()
export class ConnectorOperationalStore {
  constructor(private readonly prisma: PrismaService) {}

  async listConnections(userId: string) {
    const rows = await this.prisma.$queryRaw<ConnectionRow[]>(Prisma.sql`
      SELECT id, provider, status, external_account_id, display_label, granted_scopes,
             connected_at, last_sync_at, revoked_at, created_at, updated_at
      FROM dogos_connectors.connections
      WHERE user_id = ${userId}
      ORDER BY provider ASC
    `);
    return rows.map(toConnection);
  }

  async getConnection(userId: string, provider: ConnectorProvider) {
    const rows = await this.prisma.$queryRaw<ConnectionRow[]>(Prisma.sql`
      SELECT id, provider, status, external_account_id, display_label, granted_scopes,
             connected_at, last_sync_at, revoked_at, created_at, updated_at
      FROM dogos_connectors.connections
      WHERE user_id = ${userId} AND provider = ${provider}
      LIMIT 1
    `);
    return rows[0] ? toConnection(rows[0]) : null;
  }

  async markConnected(input: {
    userId: string;
    provider: ConnectorProvider;
    externalAccountId: string | null;
    displayLabel: string | null;
    grantedScopes: string[];
  }) {
    const scopes = textArray(input.grantedScopes);
    const rows = await this.prisma.$queryRaw<ConnectionRow[]>(Prisma.sql`
      INSERT INTO dogos_connectors.connections (
        user_id, provider, status, external_account_id, display_label, granted_scopes,
        connected_at, revoked_at, updated_at
      )
      VALUES (
        ${input.userId}, ${input.provider}, 'CONNECTED', ${input.externalAccountId},
        ${input.displayLabel}, ${scopes}, NOW(), NULL, NOW()
      )
      ON CONFLICT (user_id, provider) DO UPDATE SET
        status = 'CONNECTED',
        external_account_id = EXCLUDED.external_account_id,
        display_label = EXCLUDED.display_label,
        granted_scopes = EXCLUDED.granted_scopes,
        connected_at = COALESCE(dogos_connectors.connections.connected_at, NOW()),
        revoked_at = NULL,
        updated_at = NOW()
      RETURNING id, provider, status, external_account_id, display_label, granted_scopes,
                connected_at, last_sync_at, revoked_at, created_at, updated_at
    `);
    return toConnection(rows[0]!);
  }

  async markReauthRequired(userId: string, provider: ConnectorProvider) {
    await this.prisma.$executeRaw(Prisma.sql`
      UPDATE dogos_connectors.connections
      SET status = 'REAUTH_REQUIRED', updated_at = NOW()
      WHERE user_id = ${userId} AND provider = ${provider} AND status = 'CONNECTED'
    `);
  }

  async bindPetIdentity(input: {
    userId: string;
    provider: ConnectorProvider;
    petId: string;
    externalPetId: string;
    externalPetLabel?: string | null;
  }) {
    const rows = await this.prisma.$queryRaw<PetIdentityRow[]>(Prisma.sql`
      INSERT INTO dogos_connectors.pet_identities (
        connection_id, pet_id, external_pet_id, external_pet_label, verified_at, updated_at
      )
      SELECT id, ${input.petId}, ${input.externalPetId}, ${input.externalPetLabel ?? null}, NOW(), NOW()
      FROM dogos_connectors.connections
      WHERE user_id = ${input.userId}
        AND provider = ${input.provider}
        AND status = 'CONNECTED'
      ON CONFLICT (connection_id, pet_id) DO UPDATE SET
        external_pet_id = EXCLUDED.external_pet_id,
        external_pet_label = EXCLUDED.external_pet_label,
        verified_at = NOW(),
        updated_at = NOW()
      RETURNING connection_id, pet_id, external_pet_id, external_pet_label, verified_at
    `);
    return rows[0] ?? null;
  }

  async getPetIdentity(userId: string, provider: ConnectorProvider, externalPetId: string) {
    const rows = await this.prisma.$queryRaw<PetIdentityRow[]>(Prisma.sql`
      SELECT identity.connection_id, identity.pet_id, identity.external_pet_id,
             identity.external_pet_label, identity.verified_at
      FROM dogos_connectors.pet_identities AS identity
      INNER JOIN dogos_connectors.connections AS connection
        ON connection.id = identity.connection_id
      WHERE connection.user_id = ${userId}
        AND connection.provider = ${provider}
        AND connection.status = 'CONNECTED'
        AND identity.external_pet_id = ${externalPetId}
      LIMIT 1
    `);
    const row = rows[0];
    return row
      ? {
          connectionId: row.connection_id,
          petId: row.pet_id,
          externalPetId: row.external_pet_id,
          externalPetLabel: row.external_pet_label,
          verifiedAt: row.verified_at?.toISOString() ?? null,
        }
      : null;
  }

  async getSyncCursor(userId: string, provider: ConnectorProvider, resourceType: string) {
    const rows = await this.prisma.$queryRaw<SyncCursorRow[]>(Prisma.sql`
      SELECT cursor.resource_type, cursor.cursor_value, cursor.watermark_at,
             cursor.last_successful_sync_at
      FROM dogos_connectors.sync_cursors AS cursor
      INNER JOIN dogos_connectors.connections AS connection
        ON connection.id = cursor.connection_id
      WHERE connection.user_id = ${userId}
        AND connection.provider = ${provider}
        AND cursor.resource_type = ${resourceType}
      LIMIT 1
    `);
    const row = rows[0];
    return row
      ? {
          resourceType: row.resource_type,
          cursor: row.cursor_value,
          watermarkAt: row.watermark_at?.toISOString() ?? null,
          lastSuccessfulSyncAt: row.last_successful_sync_at?.toISOString() ?? null,
        }
      : null;
  }

  async advanceSyncCursor(input: {
    userId: string;
    provider: ConnectorProvider;
    resourceType: string;
    cursor: string | null;
    watermarkAt: Date | null;
  }) {
    await this.prisma.$executeRaw(Prisma.sql`
      INSERT INTO dogos_connectors.sync_cursors (
        connection_id, resource_type, cursor_value, watermark_at,
        last_successful_sync_at, updated_at
      )
      SELECT id, ${input.resourceType}, ${input.cursor}, ${input.watermarkAt}, NOW(), NOW()
      FROM dogos_connectors.connections
      WHERE user_id = ${input.userId}
        AND provider = ${input.provider}
        AND status = 'CONNECTED'
      ON CONFLICT (connection_id, resource_type) DO UPDATE SET
        cursor_value = EXCLUDED.cursor_value,
        watermark_at = EXCLUDED.watermark_at,
        last_successful_sync_at = NOW(),
        updated_at = NOW()
    `);
    await this.prisma.$executeRaw(Prisma.sql`
      UPDATE dogos_connectors.connections
      SET last_sync_at = NOW(), updated_at = NOW()
      WHERE user_id = ${input.userId} AND provider = ${input.provider}
    `);
  }

  async getImportReceipt(connectionId: string, resourceType: string, externalObjectId: string) {
    const rows = await this.prisma.$queryRaw<ImportReceiptRow[]>(Prisma.sql`
      SELECT id, connection_id, resource_type, external_object_id, payload_hash,
             disposition, canonical_ref_type, canonical_ref_id, occurred_at,
             imported_at, detail_code
      FROM dogos_connectors.import_receipts
      WHERE connection_id = CAST(${connectionId} AS uuid)
        AND resource_type = ${resourceType}
        AND external_object_id = ${externalObjectId}
      LIMIT 1
    `);
    return rows[0] ? toReceipt(rows[0]) : null;
  }

  async recordImportReceipt(input: {
    connectionId: string;
    resourceType: string;
    externalObjectId: string;
    payloadHash: string;
    disposition: 'IMPORTED' | 'SKIPPED' | 'FAILED';
    canonicalRefType?: string | null;
    canonicalRefId?: string | null;
    occurredAt?: Date | null;
    detailCode?: string | null;
  }) {
    const rows = await this.prisma.$queryRaw<ImportReceiptRow[]>(Prisma.sql`
      INSERT INTO dogos_connectors.import_receipts (
        connection_id, resource_type, external_object_id, payload_hash, disposition,
        canonical_ref_type, canonical_ref_id, occurred_at, detail_code
      ) VALUES (
        CAST(${input.connectionId} AS uuid), ${input.resourceType}, ${input.externalObjectId},
        ${input.payloadHash}, ${input.disposition}, ${input.canonicalRefType ?? null},
        ${input.canonicalRefId ?? null}, ${input.occurredAt ?? null}, ${input.detailCode ?? null}
      )
      ON CONFLICT (connection_id, resource_type, external_object_id) DO NOTHING
      RETURNING id, connection_id, resource_type, external_object_id, payload_hash,
                disposition, canonical_ref_type, canonical_ref_id, occurred_at,
                imported_at, detail_code
    `);
    if (rows[0]) return toReceipt(rows[0]);
    return this.getImportReceipt(input.connectionId, input.resourceType, input.externalObjectId);
  }

  async markLocallyRevoked(userId: string, provider: ConnectorProvider) {
    const rows = await this.prisma.$queryRaw<Array<{ id: string }>>(Prisma.sql`
      WITH revoked AS (
        UPDATE dogos_connectors.connections
        SET status = 'REVOKED', revoked_at = NOW(), updated_at = NOW()
        WHERE user_id = ${userId} AND provider = ${provider}
        RETURNING id
      )
      INSERT INTO dogos_connectors.revocation_receipts (
        connection_id, mode, status, detail_code, completed_at
      )
      SELECT id, 'LOCAL_CREDENTIAL_DELETE', 'SUCCEEDED', 'local_credentials_removed', NOW()
      FROM revoked
      RETURNING id
    `);
    return rows[0]?.id ?? null;
  }
}
