import { randomUUID } from 'node:crypto';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import { ConnectorOperationalStore } from './connector-operational.store';

describe('ConnectorOperationalStore integration', () => {
  const prisma = new PrismaService();
  const store = new ConnectorOperationalStore(prisma);
  const usersToDelete: string[] = [];

  beforeAll(async () => {
    await prisma.$connect();
  });

  afterAll(async () => {
    if (usersToDelete.length > 0) {
      await prisma.user.deleteMany({ where: { id: { in: usersToDelete } } });
    }
    await prisma.$disconnect();
  });

  async function fixture(label: string) {
    const suffix = randomUUID().slice(0, 8);
    const user = await prisma.user.create({
      data: {
        handle: `connector-${label}-${suffix}`,
        email: `connector-${label}-${suffix}@example.test`,
      },
      select: { id: true },
    });
    usersToDelete.push(user.id);
    const pet = await prisma.pet.create({
      data: { ownerId: user.id, name: `Connector ${label}`, species: 'DOG' },
      select: { id: true },
    });
    return { userId: user.id, petId: pet.id };
  }

  it('keeps connection metadata, identity, cursor, and receipt data relational', async () => {
    const { userId, petId } = await fixture('relational');
    const connection = await store.markConnected({
      userId,
      provider: 'FI',
      externalAccountId: 'fi-account-1',
      displayLabel: 'Home collar',
      grantedScopes: ['activity', 'profile', 'activity'],
    });

    expect(connection.status).toBe('CONNECTED');
    expect(connection.grantedScopes).toEqual(['activity', 'profile']);

    const identity = await store.bindPetIdentity({
      userId,
      provider: 'FI',
      petId,
      externalPetId: 'fi-pet-1',
      externalPetLabel: 'Scout',
    });
    expect(identity).toEqual(
      expect.objectContaining({ pet_id: petId, external_pet_id: 'fi-pet-1' })
    );

    await store.advanceSyncCursor({
      userId,
      provider: 'FI',
      resourceType: 'DAILY_ACTIVITY',
      cursor: 'cursor-2',
      watermarkAt: new Date('2026-08-22T12:00:00.000Z'),
    });
    await expect(store.getSyncCursor(userId, 'FI', 'DAILY_ACTIVITY')).resolves.toEqual({
      resourceType: 'DAILY_ACTIVITY',
      cursor: 'cursor-2',
      watermarkAt: '2026-08-22T12:00:00.000Z',
      lastSuccessfulSyncAt: expect.any(String),
    });

    const payloadHash = 'a'.repeat(64);
    const first = await store.recordImportReceipt({
      connectionId: connection.id,
      resourceType: 'WEARABLE_DAILY_ACTIVITY',
      externalObjectId: 'day-1',
      payloadHash,
      disposition: 'IMPORTED',
      canonicalRefType: 'CARE_EVENT',
      canonicalRefId: 'care-event-1',
      occurredAt: new Date('2026-08-22T12:00:00.000Z'),
    });
    const replay = await store.recordImportReceipt({
      connectionId: connection.id,
      resourceType: 'WEARABLE_DAILY_ACTIVITY',
      externalObjectId: 'day-1',
      payloadHash: 'b'.repeat(64),
      disposition: 'FAILED',
      detailCode: 'should_not_replace',
    });

    expect(replay).toEqual(first);
    const count = await prisma.$queryRaw<Array<{ count: number }>>(Prisma.sql`
      SELECT COUNT(*)::int AS count
      FROM dogos_connectors.import_receipts
      WHERE connection_id = ${connection.id}
        AND resource_type = 'WEARABLE_DAILY_ACTIVITY'
        AND external_object_id = 'day-1'
    `);
    expect(count[0]?.count).toBe(1);
  });

  it('rejects mapping a provider animal to a pet owned by someone else', async () => {
    const owner = await fixture('owner');
    const other = await fixture('other');
    await store.markConnected({
      userId: owner.userId,
      provider: 'TRACTIVE',
      externalAccountId: 'tractive-account-1',
      displayLabel: null,
      grantedScopes: ['activity'],
    });

    await expect(
      store.bindPetIdentity({
        userId: owner.userId,
        provider: 'TRACTIVE',
        petId: other.petId,
        externalPetId: 'foreign-pet',
      })
    ).rejects.toThrow();
  });

  it('records local revocation evidence while leaving provider revocation unclaimed', async () => {
    const { userId } = await fixture('revoke');
    const connection = await store.markConnected({
      userId,
      provider: 'PETCO',
      externalAccountId: 'petco-account-1',
      displayLabel: null,
      grantedScopes: ['catalog'],
    });

    const receiptId = await store.markLocallyRevoked(userId, 'PETCO');
    expect(receiptId).toEqual(expect.any(String));
    await expect(store.getConnection(userId, 'PETCO')).resolves.toEqual(
      expect.objectContaining({ status: 'REVOKED', revokedAt: expect.any(String) })
    );

    const receipt = await prisma.$queryRaw<
      Array<{ mode: string; status: string; remote_receipt_ref: string | null }>
    >(Prisma.sql`
      SELECT mode, status, remote_receipt_ref
      FROM dogos_connectors.revocation_receipts
      WHERE connection_id = ${connection.id}
      ORDER BY attempted_at DESC
      LIMIT 1
    `);
    expect(receipt[0]).toEqual({
      mode: 'LOCAL_CREDENTIAL_DELETE',
      status: 'SUCCEEDED',
      remote_receipt_ref: null,
    });
  });
});
