import { Injectable } from '@nestjs/common';
import { randomUUID } from 'node:crypto';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import {
  CAREGIVER_AUTHORITY_CLASS,
  CAREGIVER_POLICY_VERSION,
  caregiverReceiptHash,
  type CaregiverCapability,
  type CaregiverObservationKind,
  type CaregiverStoredStatus,
} from './caregiver.policy';

type GrantRow = {
  id: string;
  pet_id: string;
  issuer_user_id: string;
  recipient_user_id: string;
  request_key: string;
  policy_version: string;
  status: CaregiverStoredStatus;
  issued_at: Date;
  accepted_at: Date | null;
  declined_at: Date | null;
  expires_at: Date;
  revoked_at: Date | null;
  revoked_by_user_id: string | null;
  created_at: Date;
  updated_at: Date;
  capabilities: CaregiverCapability[];
  pet_name?: string;
  pet_species?: string;
  pet_breed?: string | null;
  pet_avatar_url?: string | null;
  issuer_handle?: string;
  recipient_handle?: string;
};

type MutationRow = Omit<GrantRow, 'created_at' | 'updated_at'>;

type ObservationRow = {
  id: string;
  grant_id: string;
  pet_id: string;
  actor_user_id: string;
  authority_class: typeof CAREGIVER_AUTHORITY_CLASS;
  kind: CaregiverObservationKind;
  summary: string;
  note: string | null;
  context: Record<string, unknown>;
  observed_at: Date;
  created_at: Date;
};

function capabilityArray(capabilities: readonly CaregiverCapability[]) {
  return Prisma.sql`ARRAY[${Prisma.join(capabilities)}]::text[]`;
}

function toGrant(row: GrantRow) {
  return {
    id: row.id,
    petId: row.pet_id,
    issuerUserId: row.issuer_user_id,
    recipientUserId: row.recipient_user_id,
    requestKey: row.request_key,
    policyVersion: row.policy_version,
    status: row.status,
    issuedAt: row.issued_at.toISOString(),
    acceptedAt: row.accepted_at?.toISOString() ?? null,
    declinedAt: row.declined_at?.toISOString() ?? null,
    expiresAt: row.expires_at.toISOString(),
    revokedAt: row.revoked_at?.toISOString() ?? null,
    revokedByUserId: row.revoked_by_user_id,
    createdAt: row.created_at.toISOString(),
    updatedAt: row.updated_at.toISOString(),
    capabilities: row.capabilities,
    ...(row.pet_name !== undefined
      ? {
          pet: {
            id: row.pet_id,
            name: row.pet_name,
            species: row.pet_species!,
            breed: row.pet_breed ?? null,
            avatarUrl: row.pet_avatar_url ?? null,
          },
        }
      : {}),
    ...(row.issuer_handle !== undefined ? { issuerHandle: row.issuer_handle } : {}),
    ...(row.recipient_handle !== undefined ? { recipientHandle: row.recipient_handle } : {}),
  };
}

function toObservation(row: ObservationRow) {
  return {
    id: row.id,
    grantId: row.grant_id,
    petId: row.pet_id,
    actorUserId: row.actor_user_id,
    authorityClass: row.authority_class,
    kind: row.kind,
    summary: row.summary,
    note: row.note,
    context: row.context,
    observedAt: row.observed_at.toISOString(),
    createdAt: row.created_at.toISOString(),
  };
}

const GRANT_COLUMNS = Prisma.sql`
  grant_row.id,
  grant_row.pet_id,
  grant_row.issuer_user_id,
  grant_row.recipient_user_id,
  grant_row.request_key,
  grant_row.policy_version,
  grant_row.status,
  grant_row.issued_at,
  grant_row.accepted_at,
  grant_row.declined_at,
  grant_row.expires_at,
  grant_row.revoked_at,
  grant_row.revoked_by_user_id,
  grant_row.created_at,
  grant_row.updated_at,
  ARRAY(
    SELECT capability.capability
    FROM dogos_caregiver.grant_capabilities capability
    WHERE capability.grant_id = grant_row.id
    ORDER BY capability.capability
  )::text[] AS capabilities
`;

@Injectable()
export class CaregiverOperationalStore {
  constructor(private readonly prisma: PrismaService) {}

  async issueGrant(input: {
    id: string;
    petId: string;
    issuerUserId: string;
    recipientUserId: string;
    requestKey: string;
    capabilities: CaregiverCapability[];
    issuedAt: Date;
    expiresAt: Date;
  }) {
    return this.prisma.$transaction(async (tx) => {
      const inserted = await tx.$queryRaw<Array<{ id: string }>>(Prisma.sql`
        INSERT INTO dogos_caregiver.grants (
          id, pet_id, issuer_user_id, recipient_user_id, request_key, policy_version,
          status, issued_at, expires_at, created_at, updated_at
        ) VALUES (
          ${input.id}, ${input.petId}, ${input.issuerUserId}, ${input.recipientUserId},
          ${input.requestKey}, ${CAREGIVER_POLICY_VERSION}, 'PENDING_ACCEPTANCE',
          ${input.issuedAt}, ${input.expiresAt}, ${input.issuedAt}, ${input.issuedAt}
        )
        ON CONFLICT DO NOTHING
        RETURNING id
      `);

      if (!inserted[0]) return false;

      for (const capability of input.capabilities) {
        await tx.$executeRaw(Prisma.sql`
          INSERT INTO dogos_caregiver.grant_capabilities (grant_id, capability)
          VALUES (${input.id}, ${capability})
        `);
      }

      await this.insertReceipt(tx, {
        grantId: input.id,
        petId: input.petId,
        issuerUserId: input.issuerUserId,
        recipientUserId: input.recipientUserId,
        transition: 'ISSUED',
        actorUserId: input.issuerUserId,
        statusAfter: 'PENDING_ACCEPTANCE',
        capabilities: input.capabilities,
        expiresAt: input.expiresAt,
        occurredAt: input.issuedAt,
      });

      return true;
    });
  }

  async getByIssuerRequestKey(issuerUserId: string, requestKey: string) {
    const rows = await this.prisma.$queryRaw<GrantRow[]>(Prisma.sql`
      SELECT ${GRANT_COLUMNS}
      FROM dogos_caregiver.grants grant_row
      WHERE grant_row.issuer_user_id = ${issuerUserId}
        AND grant_row.request_key = ${requestKey}
      LIMIT 1
    `);
    return rows[0] ? toGrant(rows[0]) : null;
  }

  async findLiveGrantForRecipientPet(recipientUserId: string, petId: string) {
    const rows = await this.prisma.$queryRaw<GrantRow[]>(Prisma.sql`
      SELECT ${GRANT_COLUMNS}
      FROM dogos_caregiver.grants grant_row
      WHERE grant_row.recipient_user_id = ${recipientUserId}
        AND grant_row.pet_id = ${petId}
        AND grant_row.status IN ('PENDING_ACCEPTANCE', 'ACTIVE')
      LIMIT 1
    `);
    return rows[0] ? toGrant(rows[0]) : null;
  }

  async getReceivedGrant(recipientUserId: string, grantId: string) {
    const rows = await this.prisma.$queryRaw<GrantRow[]>(Prisma.sql`
      SELECT ${GRANT_COLUMNS},
             pet.name AS pet_name,
             pet.species AS pet_species,
             pet.breed AS pet_breed,
             pet.avatar_url AS pet_avatar_url,
             issuer.handle AS issuer_handle
      FROM dogos_caregiver.grants grant_row
      INNER JOIN public.pets pet ON pet.id = grant_row.pet_id
      INNER JOIN public.users issuer ON issuer.id = grant_row.issuer_user_id
      WHERE grant_row.id = ${grantId}
        AND grant_row.recipient_user_id = ${recipientUserId}
      LIMIT 1
    `);
    return rows[0] ? toGrant(rows[0]) : null;
  }

  async getIssuedGrant(issuerUserId: string, grantId: string) {
    const rows = await this.prisma.$queryRaw<GrantRow[]>(Prisma.sql`
      SELECT ${GRANT_COLUMNS},
             pet.name AS pet_name,
             pet.species AS pet_species,
             pet.breed AS pet_breed,
             pet.avatar_url AS pet_avatar_url,
             recipient.handle AS recipient_handle
      FROM dogos_caregiver.grants grant_row
      INNER JOIN public.pets pet ON pet.id = grant_row.pet_id
      INNER JOIN public.users recipient ON recipient.id = grant_row.recipient_user_id
      WHERE grant_row.id = ${grantId}
        AND grant_row.issuer_user_id = ${issuerUserId}
      LIMIT 1
    `);
    return rows[0] ? toGrant(rows[0]) : null;
  }

  async listReceivedGrants(recipientUserId: string) {
    const rows = await this.prisma.$queryRaw<GrantRow[]>(Prisma.sql`
      SELECT ${GRANT_COLUMNS},
             pet.name AS pet_name,
             pet.species AS pet_species,
             pet.breed AS pet_breed,
             pet.avatar_url AS pet_avatar_url,
             issuer.handle AS issuer_handle
      FROM dogos_caregiver.grants grant_row
      INNER JOIN public.pets pet ON pet.id = grant_row.pet_id
      INNER JOIN public.users issuer ON issuer.id = grant_row.issuer_user_id
      WHERE grant_row.recipient_user_id = ${recipientUserId}
      ORDER BY grant_row.issued_at DESC, grant_row.id DESC
    `);
    return rows.map(toGrant);
  }

  async listIssuedGrants(issuerUserId: string) {
    const rows = await this.prisma.$queryRaw<GrantRow[]>(Prisma.sql`
      SELECT ${GRANT_COLUMNS},
             pet.name AS pet_name,
             pet.species AS pet_species,
             pet.breed AS pet_breed,
             pet.avatar_url AS pet_avatar_url,
             recipient.handle AS recipient_handle
      FROM dogos_caregiver.grants grant_row
      INNER JOIN public.pets pet ON pet.id = grant_row.pet_id
      INNER JOIN public.users recipient ON recipient.id = grant_row.recipient_user_id
      WHERE grant_row.issuer_user_id = ${issuerUserId}
      ORDER BY grant_row.issued_at DESC, grant_row.id DESC
    `);
    return rows.map(toGrant);
  }

  async findEffectiveGrantForCapability(input: {
    recipientUserId: string;
    petId: string;
    capability: CaregiverCapability;
    now: Date;
  }) {
    const rows = await this.prisma.$queryRaw<GrantRow[]>(Prisma.sql`
      SELECT ${GRANT_COLUMNS},
             pet.name AS pet_name,
             pet.species AS pet_species,
             pet.breed AS pet_breed,
             pet.avatar_url AS pet_avatar_url,
             issuer.handle AS issuer_handle
      FROM dogos_caregiver.grants grant_row
      INNER JOIN dogos_caregiver.grant_capabilities capability
        ON capability.grant_id = grant_row.id
       AND capability.capability = ${input.capability}
      INNER JOIN public.pets pet ON pet.id = grant_row.pet_id
      INNER JOIN public.users issuer ON issuer.id = grant_row.issuer_user_id
      WHERE grant_row.recipient_user_id = ${input.recipientUserId}
        AND grant_row.pet_id = ${input.petId}
        AND grant_row.status = 'ACTIVE'
        AND grant_row.accepted_at IS NOT NULL
        AND grant_row.revoked_at IS NULL
        AND grant_row.expires_at > ${input.now}
      ORDER BY grant_row.expires_at DESC, grant_row.id DESC
      LIMIT 1
    `);
    return rows[0] ? toGrant(rows[0]) : null;
  }

  async listActiveCaregiverPets(recipientUserId: string, now: Date) {
    const rows = await this.prisma.$queryRaw<GrantRow[]>(Prisma.sql`
      SELECT ${GRANT_COLUMNS},
             pet.name AS pet_name,
             pet.species AS pet_species,
             pet.breed AS pet_breed,
             pet.avatar_url AS pet_avatar_url,
             issuer.handle AS issuer_handle
      FROM dogos_caregiver.grants grant_row
      INNER JOIN dogos_caregiver.grant_capabilities capability
        ON capability.grant_id = grant_row.id
       AND capability.capability = 'VIEW_TODAY'
      INNER JOIN public.pets pet ON pet.id = grant_row.pet_id
      INNER JOIN public.users issuer ON issuer.id = grant_row.issuer_user_id
      WHERE grant_row.recipient_user_id = ${recipientUserId}
        AND grant_row.status = 'ACTIVE'
        AND grant_row.accepted_at IS NOT NULL
        AND grant_row.revoked_at IS NULL
        AND grant_row.expires_at > ${now}
      ORDER BY grant_row.expires_at ASC, pet.name ASC, grant_row.id ASC
    `);
    return rows.map(toGrant);
  }

  async acceptGrant(grantId: string, recipientUserId: string, now: Date) {
    return this.transitionGrant({
      grantId,
      actorUserId: recipientUserId,
      now,
      transition: 'ACCEPTED',
      statusAfter: 'ACTIVE',
      actorColumn: 'recipient_user_id',
    });
  }

  async declineGrant(grantId: string, recipientUserId: string, now: Date) {
    return this.transitionGrant({
      grantId,
      actorUserId: recipientUserId,
      now,
      transition: 'DECLINED',
      statusAfter: 'DECLINED',
      actorColumn: 'recipient_user_id',
    });
  }

  async revokeGrant(grantId: string, issuerUserId: string, now: Date) {
    return this.transitionGrant({
      grantId,
      actorUserId: issuerUserId,
      now,
      transition: 'REVOKED',
      statusAfter: 'REVOKED',
      actorColumn: 'issuer_user_id',
    });
  }

  async recordObservation(input: {
    grantId: string;
    petId: string;
    actorUserId: string;
    kind: CaregiverObservationKind;
    summary: string;
    note: string | null;
    observedAt: Date;
    context?: Record<string, unknown>;
  }) {
    const id = randomUUID();
    const context = JSON.stringify(input.context ?? {});
    const rows = await this.prisma.$queryRaw<ObservationRow[]>(Prisma.sql`
      INSERT INTO dogos_caregiver.observations (
        id, grant_id, pet_id, actor_user_id, authority_class, kind, summary, note,
        context, observed_at
      ) VALUES (
        ${id}, ${input.grantId}, ${input.petId}, ${input.actorUserId},
        ${CAREGIVER_AUTHORITY_CLASS}, ${input.kind}, ${input.summary}, ${input.note},
        CAST(${context} AS jsonb), ${input.observedAt}
      )
      RETURNING id, grant_id, pet_id, actor_user_id, authority_class, kind, summary,
                note, context, observed_at, created_at
    `);
    return toObservation(rows[0]!);
  }

  private async transitionGrant(input: {
    grantId: string;
    actorUserId: string;
    now: Date;
    transition: 'ACCEPTED' | 'DECLINED' | 'REVOKED';
    statusAfter: 'ACTIVE' | 'DECLINED' | 'REVOKED';
    actorColumn: 'recipient_user_id' | 'issuer_user_id';
  }) {
    return this.prisma.$transaction(async (tx) => {
      const actorPredicate =
        input.actorColumn === 'recipient_user_id'
          ? Prisma.sql`recipient_user_id = ${input.actorUserId}`
          : Prisma.sql`issuer_user_id = ${input.actorUserId}`;
      const allowedStatuses =
        input.transition === 'REVOKED'
          ? Prisma.sql`status IN ('PENDING_ACCEPTANCE', 'ACTIVE')`
          : Prisma.sql`status = 'PENDING_ACCEPTANCE'`;
      const stateUpdate =
        input.transition === 'ACCEPTED'
          ? Prisma.sql`status = 'ACTIVE', accepted_at = ${input.now}`
          : input.transition === 'DECLINED'
            ? Prisma.sql`status = 'DECLINED', declined_at = ${input.now}`
            : Prisma.sql`status = 'REVOKED', revoked_at = ${input.now}, revoked_by_user_id = ${input.actorUserId}`;

      const rows = await tx.$queryRaw<MutationRow[]>(Prisma.sql`
        WITH updated AS (
          UPDATE dogos_caregiver.grants
          SET ${stateUpdate}, updated_at = ${input.now}
          WHERE id = ${input.grantId}
            AND ${actorPredicate}
            AND ${allowedStatuses}
            AND expires_at > ${input.now}
          RETURNING id, pet_id, issuer_user_id, recipient_user_id, request_key,
                    policy_version, status, issued_at, accepted_at, declined_at,
                    expires_at, revoked_at, revoked_by_user_id
        )
        SELECT updated.*,
               ARRAY(
                 SELECT capability.capability
                 FROM dogos_caregiver.grant_capabilities capability
                 WHERE capability.grant_id = updated.id
                 ORDER BY capability.capability
               )::text[] AS capabilities
        FROM updated
      `);

      const row = rows[0];
      if (!row) return false;

      await this.insertReceipt(tx, {
        grantId: row.id,
        petId: row.pet_id,
        issuerUserId: row.issuer_user_id,
        recipientUserId: row.recipient_user_id,
        transition: input.transition,
        actorUserId: input.actorUserId,
        statusAfter: input.statusAfter,
        capabilities: row.capabilities,
        expiresAt: row.expires_at,
        occurredAt: input.now,
        policyVersion: row.policy_version,
      });
      return true;
    });
  }

  private async insertReceipt(
    tx: Prisma.TransactionClient,
    input: {
      grantId: string;
      petId: string;
      issuerUserId: string;
      recipientUserId: string;
      transition: 'ISSUED' | 'ACCEPTED' | 'DECLINED' | 'REVOKED';
      actorUserId: string;
      statusAfter: CaregiverStoredStatus;
      capabilities: CaregiverCapability[];
      expiresAt: Date;
      occurredAt: Date;
      policyVersion?: string;
    },
  ) {
    const policyVersion = input.policyVersion ?? CAREGIVER_POLICY_VERSION;
    const expiresAt = input.expiresAt.toISOString();
    const occurredAt = input.occurredAt.toISOString();
    const sourceHash = caregiverReceiptHash({
      grantId: input.grantId,
      petId: input.petId,
      issuerUserId: input.issuerUserId,
      recipientUserId: input.recipientUserId,
      transition: input.transition,
      actorUserId: input.actorUserId,
      statusAfter: input.statusAfter,
      capabilities: input.capabilities,
      expiresAt,
      occurredAt,
      policyVersion,
    });
    const capabilities = capabilityArray(input.capabilities);

    await tx.$executeRaw(Prisma.sql`
      INSERT INTO dogos_caregiver.grant_receipts (
        id, grant_id, actor_user_id, transition, status_after, capabilities,
        expires_at, policy_version, source_hash, occurred_at
      ) VALUES (
        ${randomUUID()}, ${input.grantId}, ${input.actorUserId}, ${input.transition},
        ${input.statusAfter}, ${capabilities}, ${input.expiresAt}, ${policyVersion},
        ${sourceHash}, ${input.occurredAt}
      )
    `);
  }
}
