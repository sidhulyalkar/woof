import { Injectable, ServiceUnavailableException } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { Prisma } from '@woof/database';
import { createHash } from 'crypto';
import { ConnectorCryptoService } from '../connectors/connector-crypto.service';
import type { ConnectorCredentialEnvelope } from '../connectors/connectors.types';
import { PrismaService } from '../prisma/prisma.service';

const PUSH_PROVIDER = 'push_subscription';
const PUSH_CONTEXT_VERSION = 'dogos-push-subscription-v1';

export type PushSubscriptionMaterial = {
  endpoint: string;
  expirationTime?: number | null;
  keys: {
    p256dh: string;
    auth: string;
  };
};

export type PushSubscriptionReadResult =
  | { state: 'MISSING' }
  | { state: 'USABLE'; subscription: PushSubscriptionMaterial; migratedLegacy: boolean }
  | { state: 'INVALID' }
  | { state: 'ENCRYPTION_UNAVAILABLE' }
  | { state: 'LEGACY_MIGRATION_REQUIRED' }
  | { state: 'CONCURRENT_CHANGE' };

export type PushSubscriptionMigrationReport = {
  scanned: number;
  migrated: number;
  alreadyEncrypted: number;
  invalid: number;
  concurrentChanges: number;
};

type StoredRow = {
  id: string;
  data: Prisma.JsonValue;
};

function contextKey(userId: string) {
  return `${PUSH_CONTEXT_VERSION}:${userId}`;
}

function canonicalSubscription(subscription: PushSubscriptionMaterial) {
  return JSON.stringify({
    endpoint: subscription.endpoint,
    expirationTime: subscription.expirationTime ?? null,
    keys: {
      p256dh: subscription.keys.p256dh,
      auth: subscription.keys.auth,
    },
  });
}

export function pushSubscriptionFingerprint(subscription: PushSubscriptionMaterial) {
  return createHash('sha256')
    .update(canonicalSubscription(subscription), 'utf8')
    .digest('base64url');
}

function asRecord(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  return value as Record<string, unknown>;
}

function readEnvelope(value: unknown): ConnectorCredentialEnvelope | null {
  const record = asRecord(value);
  if (!record) return null;
  if (
    record.v !== 1 ||
    record.alg !== 'A256GCM' ||
    typeof record.iv !== 'string' ||
    typeof record.tag !== 'string' ||
    typeof record.ciphertext !== 'string'
  ) {
    return null;
  }
  return {
    v: 1,
    alg: 'A256GCM',
    iv: record.iv,
    tag: record.tag,
    ciphertext: record.ciphertext,
  };
}

function looksLikeEnvelope(value: unknown) {
  const record = asRecord(value);
  if (!record) return false;
  return ['v', 'alg', 'iv', 'tag', 'ciphertext'].some((field) => field in record);
}

function readSubscription(value: unknown): PushSubscriptionMaterial | null {
  const record = asRecord(value);
  const keys = asRecord(record?.keys);
  if (
    !record ||
    typeof record.endpoint !== 'string' ||
    record.endpoint.length === 0 ||
    !keys ||
    typeof keys.p256dh !== 'string' ||
    keys.p256dh.length === 0 ||
    typeof keys.auth !== 'string' ||
    keys.auth.length === 0
  ) {
    return null;
  }

  const expirationTime = record.expirationTime;
  if (
    expirationTime !== undefined &&
    expirationTime !== null &&
    typeof expirationTime !== 'number'
  ) {
    return null;
  }

  return {
    endpoint: record.endpoint,
    expirationTime: expirationTime ?? null,
    keys: {
      p256dh: keys.p256dh,
      auth: keys.auth,
    },
  };
}

function toPlainJson(subscription: PushSubscriptionMaterial): Record<string, unknown> {
  return {
    endpoint: subscription.endpoint,
    expirationTime: subscription.expirationTime ?? null,
    keys: {
      p256dh: subscription.keys.p256dh,
      auth: subscription.keys.auth,
    },
  };
}

function toInputJson(value: ConnectorCredentialEnvelope): Prisma.InputJsonObject {
  return value as Prisma.InputJsonObject;
}

@Injectable()
export class PushSubscriptionStore {
  constructor(
    private readonly prisma: PrismaService,
    private readonly crypto: ConnectorCryptoService,
    private readonly config: ConfigService
  ) {}

  encryptionConfigured() {
    return this.crypto.isConfigured();
  }

  async put(userId: string, subscription: PushSubscriptionMaterial) {
    const encrypted = this.crypto.encrypt(toPlainJson(subscription), contextKey(userId));
    const data = toInputJson(encrypted);
    const expiresAt = subscription.expirationTime ? new Date(subscription.expirationTime) : null;

    await this.prisma.integrationToken.upsert({
      where: {
        userId_provider: {
          userId,
          provider: PUSH_PROVIDER,
        },
      },
      create: {
        userId,
        provider: PUSH_PROVIDER,
        data,
        scopes: ['notifications'],
        expiresAt,
      },
      update: {
        data,
        scopes: ['notifications'],
        expiresAt,
      },
    });
  }

  async get(userId: string): Promise<PushSubscriptionReadResult> {
    return this.readCurrentRow(userId, true);
  }

  async remove(userId: string) {
    await this.prisma.integrationToken.deleteMany({
      where: { userId, provider: PUSH_PROVIDER },
    });
  }

  async removeIfFingerprint(userId: string, expectedFingerprint: string) {
    const row = await this.prisma.integrationToken.findUnique({
      where: {
        userId_provider: {
          userId,
          provider: PUSH_PROVIDER,
        },
      },
      select: { id: true, data: true },
    });
    if (!row) return false;

    const subscription = this.readSubscriptionForConditionalRemoval(userId, row.data);
    if (!subscription) return false;
    if (pushSubscriptionFingerprint(subscription) !== expectedFingerprint) return false;

    return this.deleteExactRow(userId, row);
  }

  async removeInvalidCurrent(userId: string) {
    const row = await this.prisma.integrationToken.findUnique({
      where: {
        userId_provider: {
          userId,
          provider: PUSH_PROVIDER,
        },
      },
      select: { id: true, data: true },
    });
    if (!row) return false;
    if (!this.isInvalidSubscriptionRow(userId, row.data)) return false;

    return this.deleteExactRow(userId, row);
  }

  async migrateLegacyRows(batchSize = 100): Promise<PushSubscriptionMigrationReport> {
    if (!this.crypto.isConfigured()) {
      throw new ServiceUnavailableException('Push subscription encryption is not configured');
    }
    if (!Number.isInteger(batchSize) || batchSize < 1 || batchSize > 1000) {
      throw new RangeError('Push subscription migration batch size must be between 1 and 1000');
    }

    const report: PushSubscriptionMigrationReport = {
      scanned: 0,
      migrated: 0,
      alreadyEncrypted: 0,
      invalid: 0,
      concurrentChanges: 0,
    };
    let lastSeenId: string | undefined;

    while (true) {
      const rows = await this.prisma.integrationToken.findMany({
        where: {
          provider: PUSH_PROVIDER,
          ...(lastSeenId ? { id: { gt: lastSeenId } } : {}),
        },
        orderBy: { id: 'asc' },
        take: batchSize,
        select: { id: true, userId: true, data: true },
      });
      if (rows.length === 0) break;

      for (const row of rows) {
        report.scanned += 1;
        const envelope = readEnvelope(row.data);
        if (envelope) {
          try {
            const decrypted = this.crypto.decrypt(envelope, contextKey(row.userId));
            if (readSubscription(decrypted)) report.alreadyEncrypted += 1;
            else report.invalid += 1;
          } catch {
            report.invalid += 1;
          }
          continue;
        }

        if (looksLikeEnvelope(row.data)) {
          report.invalid += 1;
          continue;
        }

        const legacy = readSubscription(row.data);
        if (!legacy) {
          report.invalid += 1;
          continue;
        }

        const encrypted = this.crypto.encrypt(toPlainJson(legacy), contextKey(row.userId));
        const result = await this.compareAndSwap(row.id, row.data, encrypted);
        if (result) report.migrated += 1;
        else report.concurrentChanges += 1;
      }

      lastSeenId = rows[rows.length - 1]?.id;
      if (rows.length < batchSize) break;
    }

    return report;
  }

  private legacyPlaintextReadsEnabled() {
    const cutoff = this.config.get<string>('PUSH_LEGACY_PLAINTEXT_READS_UNTIL');
    if (!cutoff) return false;
    const cutoffMillis = Date.parse(cutoff);
    return Number.isFinite(cutoffMillis) && Date.now() <= cutoffMillis;
  }

  private readSubscriptionForConditionalRemoval(userId: string, data: Prisma.JsonValue) {
    const envelope = readEnvelope(data);
    if (envelope) {
      if (!this.crypto.isConfigured()) return null;
      try {
        return readSubscription(this.crypto.decrypt(envelope, contextKey(userId)));
      } catch {
        return null;
      }
    }
    if (looksLikeEnvelope(data) || !this.legacyPlaintextReadsEnabled()) return null;
    return readSubscription(data);
  }

  private isInvalidSubscriptionRow(userId: string, data: Prisma.JsonValue) {
    const envelope = readEnvelope(data);
    if (envelope) {
      if (!this.crypto.isConfigured()) return false;
      try {
        return readSubscription(this.crypto.decrypt(envelope, contextKey(userId))) === null;
      } catch {
        return true;
      }
    }
    if (looksLikeEnvelope(data)) return true;
    return readSubscription(data) === null;
  }

  private async deleteExactRow(userId: string, row: StoredRow) {
    const result = await this.prisma.integrationToken.deleteMany({
      where: {
        id: row.id,
        userId,
        provider: PUSH_PROVIDER,
        data: { equals: row.data as Prisma.InputJsonValue },
      },
    });
    return result.count === 1;
  }

  private async readCurrentRow(
    userId: string,
    migrateLegacy: boolean
  ): Promise<PushSubscriptionReadResult> {
    const row = await this.prisma.integrationToken.findUnique({
      where: {
        userId_provider: {
          userId,
          provider: PUSH_PROVIDER,
        },
      },
      select: { id: true, data: true },
    });
    if (!row) return { state: 'MISSING' };

    return this.decodeRow(userId, row, migrateLegacy);
  }

  private async decodeRow(
    userId: string,
    row: StoredRow,
    migrateLegacy: boolean
  ): Promise<PushSubscriptionReadResult> {
    const envelope = readEnvelope(row.data);
    if (envelope) {
      if (!this.crypto.isConfigured()) return { state: 'ENCRYPTION_UNAVAILABLE' };
      try {
        const subscription = readSubscription(this.crypto.decrypt(envelope, contextKey(userId)));
        return subscription
          ? { state: 'USABLE', subscription, migratedLegacy: false }
          : { state: 'INVALID' };
      } catch {
        return { state: 'INVALID' };
      }
    }

    if (looksLikeEnvelope(row.data)) return { state: 'INVALID' };
    const legacy = readSubscription(row.data);
    if (!legacy) return { state: 'INVALID' };
    if (!this.legacyPlaintextReadsEnabled()) return { state: 'LEGACY_MIGRATION_REQUIRED' };
    if (!this.crypto.isConfigured()) return { state: 'ENCRYPTION_UNAVAILABLE' };
    if (!migrateLegacy) return { state: 'CONCURRENT_CHANGE' };

    const encrypted = this.crypto.encrypt(toPlainJson(legacy), contextKey(userId));
    const migrated = await this.compareAndSwap(row.id, row.data, encrypted);
    if (migrated) {
      return { state: 'USABLE', subscription: legacy, migratedLegacy: true };
    }

    return this.readCurrentRow(userId, false);
  }

  private async compareAndSwap(
    id: string,
    expectedData: Prisma.JsonValue,
    encrypted: ConnectorCredentialEnvelope
  ) {
    const result = await this.prisma.integrationToken.updateMany({
      where: {
        id,
        provider: PUSH_PROVIDER,
        data: { equals: expectedData as Prisma.InputJsonValue },
      },
      data: { data: toInputJson(encrypted) },
    });
    return result.count === 1;
  }
}
