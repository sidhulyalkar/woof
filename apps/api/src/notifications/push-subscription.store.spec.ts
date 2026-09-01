import { ConfigService } from '@nestjs/config';
import { PrismaService } from '../prisma/prisma.service';
import { ConnectorCryptoService } from '../connectors/connector-crypto.service';
import {
  pushSubscriptionFingerprint,
  PushSubscriptionMaterial,
  PushSubscriptionStore,
} from './push-subscription.store';

const userId = 'user-a';
const otherUserId = 'user-b';
const key = Buffer.alloc(32, 7).toString('base64');
const activeLegacyCutoff = '2099-01-01T00:00:00Z';
const expiredLegacyCutoff = '2000-01-01T00:00:00Z';
const subscription: PushSubscriptionMaterial = {
  endpoint: 'https://push.example.com/private-endpoint',
  expirationTime: null,
  keys: {
    p256dh: 'private-p256dh',
    auth: 'private-auth',
  },
};

function configuration(options: { configured?: boolean; legacyReadsUntil?: string | null } = {}) {
  const legacyReadsUntil =
    options.legacyReadsUntil === null
      ? undefined
      : (options.legacyReadsUntil ?? activeLegacyCutoff);
  return new ConfigService({
    CONNECTOR_CREDENTIALS_KEY: options.configured === false ? undefined : key,
    PUSH_LEGACY_PLAINTEXT_READS_UNTIL: legacyReadsUntil,
  });
}

function crypto(configured = true) {
  return new ConnectorCryptoService(configuration({ configured }));
}

function envelopeFor(targetUserId = userId, value = subscription) {
  return crypto().encrypt(
    value as unknown as Record<string, unknown>,
    `dogos-push-subscription-v1:${targetUserId}`
  );
}

function database() {
  return {
    integrationToken: {
      upsert: jest.fn().mockResolvedValue({ id: 'row-1' }),
      findUnique: jest.fn(),
      findMany: jest.fn(),
      updateMany: jest.fn().mockResolvedValue({ count: 1 }),
      deleteMany: jest.fn().mockResolvedValue({ count: 1 }),
    },
  };
}

function store(
  options: {
    configured?: boolean;
    database?: ReturnType<typeof database>;
    legacyReadsUntil?: string | null;
  } = {}
) {
  const db = options.database ?? database();
  const config = configuration({
    configured: options.configured,
    legacyReadsUntil: options.legacyReadsUntil,
  });
  return {
    db,
    store: new PushSubscriptionStore(
      db as unknown as PrismaService,
      new ConnectorCryptoService(config),
      config
    ),
  };
}

describe('PushSubscriptionStore encrypted persistence', () => {
  it('writes only an authenticated encryption envelope for new subscriptions', async () => {
    const { db, store: pushStore } = store();

    await pushStore.put(userId, subscription);

    expect(db.integrationToken.upsert).toHaveBeenCalledTimes(1);
    const call = db.integrationToken.upsert.mock.calls[0]?.[0];
    const serialized = JSON.stringify(call);
    expect(call.create.provider).toBe('push_subscription');
    expect(call.create.scopes).toEqual(['notifications']);
    expect(call.create.data).toMatchObject({ v: 1, alg: 'A256GCM' });
    expect(call.update.data).toMatchObject({ v: 1, alg: 'A256GCM' });
    for (const privateValue of [
      subscription.endpoint,
      subscription.keys.p256dh,
      subscription.keys.auth,
    ]) {
      expect(serialized).not.toContain(privateValue);
    }
  });

  it('fingerprints the complete subscription so rotated keys at one endpoint are distinct', () => {
    const rotated = {
      ...subscription,
      keys: { ...subscription.keys, auth: 'rotated-private-auth' },
    };

    expect(pushSubscriptionFingerprint(rotated)).not.toBe(
      pushSubscriptionFingerprint(subscription)
    );
  });

  it('decrypts a correctly bound encrypted subscription without rewriting it', async () => {
    const db = database();
    db.integrationToken.findUnique.mockResolvedValue({ id: 'row-1', data: envelopeFor() });
    const { store: pushStore } = store({ database: db });

    await expect(pushStore.get(userId)).resolves.toEqual({
      state: 'USABLE',
      subscription,
      migratedLegacy: false,
    });
    expect(db.integrationToken.updateMany).not.toHaveBeenCalled();
  });

  it('rejects an encrypted subscription copied into the wrong user context', async () => {
    const db = database();
    db.integrationToken.findUnique.mockResolvedValue({
      id: 'row-1',
      data: envelopeFor(otherUserId),
    });
    const { store: pushStore } = store({ database: db });

    await expect(pushStore.get(userId)).resolves.toEqual({ state: 'INVALID' });
    expect(db.integrationToken.updateMany).not.toHaveBeenCalled();
  });

  it('rejects tampered ciphertext and never falls back to plaintext interpretation', async () => {
    const encrypted = envelopeFor();
    const db = database();
    db.integrationToken.findUnique.mockResolvedValue({
      id: 'row-1',
      data: { ...encrypted, ciphertext: `${encrypted.ciphertext}tampered` },
    });
    const { store: pushStore } = store({ database: db });

    await expect(pushStore.get(userId)).resolves.toEqual({ state: 'INVALID' });
  });

  it('rejects partial envelope-shaped data instead of downgrading to legacy plaintext', async () => {
    const db = database();
    db.integrationToken.findUnique.mockResolvedValue({
      id: 'row-1',
      data: {
        ...subscription,
        v: 1,
        alg: 'A256GCM',
        iv: 'partial-envelope-marker',
      },
    });
    const { store: pushStore } = store({ database: db });

    await expect(pushStore.get(userId)).resolves.toEqual({ state: 'INVALID' });
    expect(db.integrationToken.updateMany).not.toHaveBeenCalled();
  });

  it('fails closed when encryption material is unavailable', async () => {
    const db = database();
    db.integrationToken.findUnique.mockResolvedValue({ id: 'row-1', data: envelopeFor() });
    const { store: pushStore } = store({ configured: false, database: db });

    await expect(pushStore.get(userId)).resolves.toEqual({ state: 'ENCRYPTION_UNAVAILABLE' });
    await expect(pushStore.put(userId, subscription)).rejects.toThrow(
      /encryption is not configured/i
    );
  });

  it('lazily migrates a valid legacy plaintext row only inside the configured compatibility window', async () => {
    const db = database();
    db.integrationToken.findUnique.mockResolvedValue({ id: 'row-1', data: subscription });
    const { store: pushStore } = store({ database: db });

    await expect(pushStore.get(userId)).resolves.toEqual({
      state: 'USABLE',
      subscription,
      migratedLegacy: true,
    });

    const call = db.integrationToken.updateMany.mock.calls[0]?.[0];
    expect(call.where.id).toBe('row-1');
    expect(call.where.provider).toBe('push_subscription');
    expect(call.where.data.equals).toEqual(subscription);
    expect(call.data.data).toMatchObject({ v: 1, alg: 'A256GCM' });
  });

  it('fails closed after the legacy plaintext compatibility window without deleting the row', async () => {
    const db = database();
    db.integrationToken.findUnique.mockResolvedValue({ id: 'row-1', data: subscription });
    const { store: pushStore } = store({
      database: db,
      legacyReadsUntil: expiredLegacyCutoff,
    });

    await expect(pushStore.get(userId)).resolves.toEqual({ state: 'LEGACY_MIGRATION_REQUIRED' });
    await expect(
      pushStore.removeIfFingerprint(userId, pushSubscriptionFingerprint(subscription))
    ).resolves.toBe(false);
    await expect(pushStore.removeInvalidCurrent(userId)).resolves.toBe(false);
    expect(db.integrationToken.updateMany).not.toHaveBeenCalled();
    expect(db.integrationToken.deleteMany).not.toHaveBeenCalled();
  });

  it('treats an absent legacy plaintext cutoff as compatibility disabled', async () => {
    const db = database();
    db.integrationToken.findUnique.mockResolvedValue({ id: 'row-1', data: subscription });
    const { store: pushStore } = store({ database: db, legacyReadsUntil: null });

    await expect(pushStore.get(userId)).resolves.toEqual({ state: 'LEGACY_MIGRATION_REQUIRED' });
    expect(db.integrationToken.updateMany).not.toHaveBeenCalled();
    expect(db.integrationToken.deleteMany).not.toHaveBeenCalled();
  });

  it('does not resurrect stale legacy material when a concurrent subscription update wins', async () => {
    const freshSubscription = {
      ...subscription,
      endpoint: 'https://push.example.com/fresh-endpoint',
    };
    const db = database();
    db.integrationToken.findUnique
      .mockResolvedValueOnce({ id: 'row-1', data: subscription })
      .mockResolvedValueOnce({ id: 'row-1', data: envelopeFor(userId, freshSubscription) });
    db.integrationToken.updateMany.mockResolvedValueOnce({ count: 0 });
    const { store: pushStore } = store({ database: db });

    await expect(pushStore.get(userId)).resolves.toEqual({
      state: 'USABLE',
      subscription: freshSubscription,
      migratedLegacy: false,
    });
  });

  it('deletes by authenticated owner without needing to read or decrypt the row', async () => {
    const db = database();
    const { store: pushStore } = store({ configured: false, database: db });

    await pushStore.remove(userId);

    expect(db.integrationToken.deleteMany).toHaveBeenCalledWith({
      where: { userId, provider: 'push_subscription' },
    });
    expect(db.integrationToken.findUnique).not.toHaveBeenCalled();
  });

  it('conditionally deletes only the exact encrypted row matching the browser fingerprint', async () => {
    const db = database();
    const encrypted = envelopeFor();
    db.integrationToken.findUnique.mockResolvedValue({ id: 'row-1', data: encrypted });
    const { store: pushStore } = store({ database: db });

    await expect(
      pushStore.removeIfFingerprint(userId, pushSubscriptionFingerprint(subscription))
    ).resolves.toBe(true);

    expect(db.integrationToken.deleteMany).toHaveBeenCalledWith({
      where: {
        id: 'row-1',
        userId,
        provider: 'push_subscription',
        data: { equals: encrypted },
      },
    });
  });

  it('does not delete rotated keys at the same endpoint when the fingerprint is stale', async () => {
    const rotated = {
      ...subscription,
      keys: { ...subscription.keys, auth: 'rotated-private-auth' },
    };
    const db = database();
    db.integrationToken.findUnique.mockResolvedValue({
      id: 'row-1',
      data: envelopeFor(userId, rotated),
    });
    const { store: pushStore } = store({ database: db });

    await expect(
      pushStore.removeIfFingerprint(userId, pushSubscriptionFingerprint(subscription))
    ).resolves.toBe(false);
    expect(db.integrationToken.deleteMany).not.toHaveBeenCalled();
  });

  it('does not delete another browser row when the fingerprint does not match', async () => {
    const db = database();
    db.integrationToken.findUnique.mockResolvedValue({ id: 'row-1', data: envelopeFor() });
    const { store: pushStore } = store({ database: db });

    await expect(pushStore.removeIfFingerprint(userId, 'not-this-browser')).resolves.toBe(false);
    expect(db.integrationToken.deleteMany).not.toHaveBeenCalled();
  });

  it('does not delete a concurrently replaced row after fingerprint verification', async () => {
    const db = database();
    db.integrationToken.findUnique.mockResolvedValue({ id: 'row-1', data: envelopeFor() });
    db.integrationToken.deleteMany.mockResolvedValueOnce({ count: 0 });
    const { store: pushStore } = store({ database: db });

    await expect(
      pushStore.removeIfFingerprint(userId, pushSubscriptionFingerprint(subscription))
    ).resolves.toBe(false);
    expect(db.integrationToken.deleteMany).toHaveBeenCalledTimes(1);
  });

  it('fails conditional deletion closed for encrypted rows when the key is unavailable', async () => {
    const db = database();
    db.integrationToken.findUnique.mockResolvedValue({ id: 'row-1', data: envelopeFor() });
    const { store: pushStore } = store({ configured: false, database: db });

    await expect(
      pushStore.removeIfFingerprint(userId, pushSubscriptionFingerprint(subscription))
    ).resolves.toBe(false);
    expect(db.integrationToken.deleteMany).not.toHaveBeenCalled();
  });

  it('removes only the exact invalid row snapshot it inspected', async () => {
    const encrypted = envelopeFor();
    const invalid = { ...encrypted, ciphertext: `${encrypted.ciphertext}tampered` };
    const db = database();
    db.integrationToken.findUnique.mockResolvedValue({ id: 'row-1', data: invalid });
    const { store: pushStore } = store({ database: db });

    await expect(pushStore.removeInvalidCurrent(userId)).resolves.toBe(true);
    expect(db.integrationToken.deleteMany).toHaveBeenCalledWith({
      where: {
        id: 'row-1',
        userId,
        provider: 'push_subscription',
        data: { equals: invalid },
      },
    });
  });

  it('does not remove a valid replacement while cleaning previously observed invalid state', async () => {
    const db = database();
    db.integrationToken.findUnique.mockResolvedValue({ id: 'row-1', data: envelopeFor() });
    const { store: pushStore } = store({ database: db });

    await expect(pushStore.removeInvalidCurrent(userId)).resolves.toBe(false);
    expect(db.integrationToken.deleteMany).not.toHaveBeenCalled();
  });

  it('does not remove a concurrently replaced row after invalid-state verification', async () => {
    const encrypted = envelopeFor();
    const invalid = { ...encrypted, ciphertext: `${encrypted.ciphertext}tampered` };
    const db = database();
    db.integrationToken.findUnique.mockResolvedValue({ id: 'row-1', data: invalid });
    db.integrationToken.deleteMany.mockResolvedValueOnce({ count: 0 });
    const { store: pushStore } = store({ database: db });

    await expect(pushStore.removeInvalidCurrent(userId)).resolves.toBe(false);
    expect(db.integrationToken.deleteMany).toHaveBeenCalledTimes(1);
  });

  it('does not classify encrypted rows as invalid when the encryption key is unavailable', async () => {
    const db = database();
    db.integrationToken.findUnique.mockResolvedValue({ id: 'row-1', data: envelopeFor() });
    const { store: pushStore } = store({ configured: false, database: db });

    await expect(pushStore.removeInvalidCurrent(userId)).resolves.toBe(false);
    expect(db.integrationToken.deleteMany).not.toHaveBeenCalled();
  });
});

describe('PushSubscriptionStore explicit migration', () => {
  it('reports only bounded counts while migrating legacy rows and preserving encrypted rows', async () => {
    const db = database();
    const encrypted = envelopeFor();
    db.integrationToken.findMany
      .mockResolvedValueOnce([
        { id: 'row-1', userId, data: subscription },
        { id: 'row-2', userId, data: encrypted },
        { id: 'row-3', userId, data: { endpoint: 'broken' } },
      ])
      .mockResolvedValueOnce([]);
    db.integrationToken.updateMany.mockResolvedValueOnce({ count: 1 });
    const { store: pushStore } = store({ database: db });

    await expect(pushStore.migrateLegacyRows(3)).resolves.toEqual({
      scanned: 3,
      migrated: 1,
      alreadyEncrypted: 1,
      invalid: 1,
      concurrentChanges: 0,
    });
  });

  it('migrates legacy rows explicitly even after runtime compatibility is closed', async () => {
    const db = database();
    db.integrationToken.findMany.mockResolvedValueOnce([
      { id: 'row-1', userId, data: subscription },
    ]);
    db.integrationToken.updateMany.mockResolvedValueOnce({ count: 1 });
    const { store: pushStore } = store({
      database: db,
      legacyReadsUntil: expiredLegacyCutoff,
    });

    await expect(pushStore.migrateLegacyRows(10)).resolves.toEqual({
      scanned: 1,
      migrated: 1,
      alreadyEncrypted: 0,
      invalid: 0,
      concurrentChanges: 0,
    });
    expect(db.integrationToken.updateMany).toHaveBeenCalledTimes(1);
  });

  it('advances by id range so a deleted previous page tail cannot invalidate the next scan', async () => {
    const db = database();
    db.integrationToken.findMany
      .mockResolvedValueOnce([
        { id: 'row-1', userId, data: envelopeFor() },
        { id: 'row-2', userId, data: envelopeFor() },
      ])
      .mockResolvedValueOnce([{ id: 'row-3', userId, data: envelopeFor() }]);
    const { store: pushStore } = store({ database: db });

    await expect(pushStore.migrateLegacyRows(2)).resolves.toEqual({
      scanned: 3,
      migrated: 0,
      alreadyEncrypted: 3,
      invalid: 0,
      concurrentChanges: 0,
    });

    expect(db.integrationToken.findMany.mock.calls[0]?.[0]).toMatchObject({
      where: { provider: 'push_subscription' },
      orderBy: { id: 'asc' },
      take: 2,
    });
    expect(db.integrationToken.findMany.mock.calls[1]?.[0]).toMatchObject({
      where: { provider: 'push_subscription', id: { gt: 'row-2' } },
      orderBy: { id: 'asc' },
      take: 2,
    });
    expect(db.integrationToken.findMany.mock.calls[1]?.[0]).not.toHaveProperty('cursor');
  });

  it('counts compare-and-swap losses instead of overwriting concurrent changes', async () => {
    const db = database();
    db.integrationToken.findMany.mockResolvedValueOnce([
      { id: 'row-1', userId, data: subscription },
    ]);
    db.integrationToken.updateMany.mockResolvedValueOnce({ count: 0 });
    const { store: pushStore } = store({ database: db });

    await expect(pushStore.migrateLegacyRows(10)).resolves.toEqual({
      scanned: 1,
      migrated: 0,
      alreadyEncrypted: 0,
      invalid: 0,
      concurrentChanges: 1,
    });
  });

  it('requires encryption before scanning legacy rows', async () => {
    const db = database();
    const { store: pushStore } = store({ configured: false, database: db });

    await expect(pushStore.migrateLegacyRows()).rejects.toThrow(/encryption is not configured/i);
    expect(db.integrationToken.findMany).not.toHaveBeenCalled();
  });
});
