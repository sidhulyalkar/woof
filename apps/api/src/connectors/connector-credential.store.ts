import { Injectable } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import { ConnectorCryptoService } from './connector-crypto.service';
import type {
  ConnectorCredentialEnvelope,
  ConnectorCredentialState,
  ConnectorProvider,
} from './connectors.types';

function providerKey(provider: ConnectorProvider) {
  return `dogos_connector:${provider}`;
}

function contextKey(userId: string, provider: ConnectorProvider) {
  return `dogos-connector-credential-v1:${userId}:${provider}`;
}

function toInputJson(envelope: ConnectorCredentialEnvelope): Prisma.InputJsonObject {
  return envelope as unknown as Prisma.InputJsonObject;
}

function readEnvelope(value: Prisma.JsonValue): ConnectorCredentialEnvelope | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  if (
    value.v !== 1 ||
    value.alg !== 'A256GCM' ||
    typeof value.iv !== 'string' ||
    typeof value.tag !== 'string' ||
    typeof value.ciphertext !== 'string'
  ) {
    return null;
  }
  return {
    v: 1,
    alg: 'A256GCM',
    iv: value.iv,
    tag: value.tag,
    ciphertext: value.ciphertext,
  };
}

@Injectable()
export class ConnectorCredentialStore {
  constructor(
    private readonly prisma: PrismaService,
    private readonly crypto: ConnectorCryptoService
  ) {}

  encryptionConfigured() {
    return this.crypto.isConfigured();
  }

  async listMetadata(userId: string) {
    const providers = ['FI', 'TRACTIVE', 'VET_PARTNER', 'CHEWY', 'PETCO'] as const;
    const rows = await this.prisma.integrationToken.findMany({
      where: {
        userId,
        provider: { in: providers.map((provider) => providerKey(provider)) },
      },
      select: { provider: true, scopes: true, expiresAt: true, createdAt: true },
    });

    return rows.map((row) => ({
      provider: row.provider.slice('dogos_connector:'.length) as ConnectorProvider,
      scopes: row.scopes,
      expiresAt: row.expiresAt?.toISOString() ?? null,
      createdAt: row.createdAt.toISOString(),
    }));
  }

  async state(userId: string, provider: ConnectorProvider): Promise<ConnectorCredentialState> {
    const row = await this.prisma.integrationToken.findUnique({
      where: {
        userId_provider: {
          userId,
          provider: providerKey(provider),
        },
      },
      select: { data: true, expiresAt: true },
    });
    if (!row) return 'MISSING';
    if (row.expiresAt && row.expiresAt.getTime() <= Date.now()) return 'EXPIRED';

    const envelope = readEnvelope(row.data);
    if (!envelope || !this.crypto.isConfigured()) return 'INVALID';
    try {
      this.crypto.decrypt(envelope, contextKey(userId, provider));
      return 'USABLE';
    } catch {
      return 'INVALID';
    }
  }

  async has(userId: string, provider: ConnectorProvider) {
    return (await this.state(userId, provider)) === 'USABLE';
  }

  async put(
    userId: string,
    provider: ConnectorProvider,
    credentials: Record<string, unknown>,
    scopes: string[],
    expiresAt: Date | null
  ) {
    const encrypted = this.crypto.encrypt(credentials, contextKey(userId, provider));
    await this.prisma.integrationToken.upsert({
      where: {
        userId_provider: {
          userId,
          provider: providerKey(provider),
        },
      },
      create: {
        userId,
        provider: providerKey(provider),
        scopes: [...new Set(scopes)].sort(),
        expiresAt,
        data: toInputJson(encrypted),
      },
      update: {
        scopes: [...new Set(scopes)].sort(),
        expiresAt,
        data: toInputJson(encrypted),
      },
    });
  }

  async get(userId: string, provider: ConnectorProvider) {
    const row = await this.prisma.integrationToken.findUnique({
      where: {
        userId_provider: {
          userId,
          provider: providerKey(provider),
        },
      },
      select: { data: true, scopes: true, expiresAt: true },
    });
    if (!row || (row.expiresAt && row.expiresAt.getTime() <= Date.now())) return null;
    const envelope = readEnvelope(row.data);
    if (!envelope) return null;
    try {
      return {
        credentials: this.crypto.decrypt(envelope, contextKey(userId, provider)),
        scopes: row.scopes,
        expiresAt: row.expiresAt,
      };
    } catch {
      return null;
    }
  }

  async remove(userId: string, provider: ConnectorProvider) {
    await this.prisma.integrationToken.deleteMany({
      where: { userId, provider: providerKey(provider) },
    });
  }
}
