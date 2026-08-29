import { randomUUID } from 'node:crypto';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import { CaregiverOperationalStore } from './caregiver-operational.store';

describe('CaregiverOperationalStore integration', () => {
  const prisma = new PrismaService();
  const store = new CaregiverOperationalStore(prisma);
  const usersToDelete: string[] = [];

  beforeAll(async () => {
    await prisma.$connect();
  });

  afterEach(async () => {
    if (usersToDelete.length === 0) return;
    await prisma.blockedUser.deleteMany({
      where: {
        OR: [{ userId: { in: usersToDelete } }, { blockedId: { in: usersToDelete } }],
      },
    });
    await prisma.user.deleteMany({ where: { id: { in: usersToDelete } } });
    usersToDelete.length = 0;
  });

  afterAll(async () => {
    await prisma.$disconnect();
  });

  async function fixture(label: string) {
    const suffix = randomUUID().slice(0, 8);
    const issuer = await prisma.user.create({
      data: {
        handle: `caregiver-issuer-${label}-${suffix}`,
        email: `caregiver-issuer-${label}-${suffix}@example.test`,
      },
      select: { id: true },
    });
    const recipient = await prisma.user.create({
      data: {
        handle: `caregiver-recipient-${label}-${suffix}`,
        email: `caregiver-recipient-${label}-${suffix}@example.test`,
      },
      select: { id: true },
    });
    usersToDelete.push(issuer.id, recipient.id);
    const pet = await prisma.pet.create({
      data: { ownerId: issuer.id, name: `Caregiver ${label}`, species: 'DOG' },
      select: { id: true },
    });
    return { issuerUserId: issuer.id, recipientUserId: recipient.id, petId: pet.id };
  }

  async function issue(input: {
    issuerUserId: string;
    recipientUserId: string;
    petId: string;
    issuedAt: Date;
    expiresAt: Date;
    capabilities?: Array<'VIEW_TODAY' | 'LOG_OBSERVATION'>;
    requestKey?: string;
  }) {
    const id = randomUUID();
    const created = await store.issueGrant({
      id,
      petId: input.petId,
      issuerUserId: input.issuerUserId,
      recipientUserId: input.recipientUserId,
      requestKey: input.requestKey ?? `request-${randomUUID()}`,
      capabilities: input.capabilities ?? ['VIEW_TODAY'],
      issuedAt: input.issuedAt,
      expiresAt: input.expiresAt,
    });
    expect(created).toBe(true);
    return id;
  }

  it('grants zero authority before acceptance, then fails closed immediately after revocation', async () => {
    const { issuerUserId, recipientUserId, petId } = await fixture('lifecycle');
    const issuedAt = new Date();
    const expiresAt = new Date(issuedAt.getTime() + 60 * 60 * 1000);
    const grantId = await issue({
      issuerUserId,
      recipientUserId,
      petId,
      issuedAt,
      expiresAt,
      capabilities: ['VIEW_TODAY', 'LOG_OBSERVATION'],
    });

    await expect(
      store.findEffectiveGrantForCapability({
        recipientUserId,
        petId,
        capability: 'VIEW_TODAY',
        now: new Date(issuedAt.getTime() + 1000),
      })
    ).resolves.toBeNull();

    const acceptedAt = new Date(issuedAt.getTime() + 2000);
    await expect(store.acceptGrant(grantId, recipientUserId, acceptedAt)).resolves.toBe(true);
    await expect(
      store.findEffectiveGrantForCapability({
        recipientUserId,
        petId,
        capability: 'VIEW_TODAY',
        now: new Date(acceptedAt.getTime() + 1000),
      })
    ).resolves.toEqual(expect.objectContaining({ id: grantId, status: 'ACTIVE' }));

    const observation = await store.recordObservation({
      grantId,
      petId,
      actorUserId: recipientUserId,
      kind: 'ROUTINE',
      summary: 'Settled after the evening routine.',
      note: null,
      observedAt: new Date(acceptedAt.getTime() + 1500),
    });
    expect(observation).toEqual(
      expect.objectContaining({ grantId, petId, actorUserId: recipientUserId })
    );

    const revokedAt = new Date(acceptedAt.getTime() + 3000);
    await expect(store.revokeGrant(grantId, issuerUserId, revokedAt)).resolves.toBe(true);
    await expect(
      store.findEffectiveGrantForCapability({
        recipientUserId,
        petId,
        capability: 'VIEW_TODAY',
        now: new Date(revokedAt.getTime() + 1),
      })
    ).resolves.toBeNull();
    await expect(
      store.recordObservation({
        grantId,
        petId,
        actorUserId: recipientUserId,
        kind: 'ROUTINE',
        summary: 'This must not be accepted after revocation.',
        note: null,
        observedAt: new Date(revokedAt.getTime() + 1),
      })
    ).rejects.toThrow();

    const receipts = await prisma.$queryRaw<Array<{ transition: string }>>(Prisma.sql`
      SELECT transition
      FROM dogos_caregiver.grant_receipts
      WHERE grant_id = ${grantId}
      ORDER BY occurred_at ASC
    `);
    expect(receipts.map((receipt) => receipt.transition)).toEqual([
      'ISSUED',
      'ACCEPTED',
      'REVOKED',
    ]);

    await expect(
      prisma.$executeRaw(Prisma.sql`
        UPDATE dogos_caregiver.grant_receipts
        SET source_hash = ${'0'.repeat(64)}
        WHERE grant_id = ${grantId} AND transition = 'ISSUED'
      `)
    ).rejects.toThrow();
  });

  it('serializes concurrent overlapping issuance so at most one authority window is created', async () => {
    const { issuerUserId, recipientUserId, petId } = await fixture('concurrent');
    const issuedAt = new Date();
    const expiresAt = new Date(issuedAt.getTime() + 60 * 60 * 1000);

    const results = await Promise.allSettled(
      [0, 1].map((index) =>
        store.issueGrant({
          id: randomUUID(),
          petId,
          issuerUserId,
          recipientUserId,
          requestKey: `concurrent-${index}-${randomUUID()}`,
          capabilities: ['VIEW_TODAY'],
          issuedAt,
          expiresAt,
        })
      )
    );

    expect(results.filter((result) => result.status === 'fulfilled')).toHaveLength(1);
    expect(results.filter((result) => result.status === 'rejected')).toHaveLength(1);

    const rows = await prisma.$queryRaw<Array<{ count: number }>>(Prisma.sql`
      SELECT COUNT(*)::int AS count
      FROM dogos_caregiver.grants
      WHERE pet_id = ${petId}
        AND recipient_user_id = ${recipientUserId}
        AND status IN ('PENDING_ACCEPTANCE', 'ACTIVE')
        AND expires_at > ${issuedAt}
    `);
    expect(rows[0]?.count).toBe(1);
  });

  it('treats expiry as present-tense authority and allows a later non-overlapping grant', async () => {
    const { issuerUserId, recipientUserId, petId } = await fixture('expiry');
    const now = new Date();
    const oldIssuedAt = new Date(now.getTime() - 2 * 60 * 60 * 1000);
    const oldExpiresAt = new Date(now.getTime() - 60 * 60 * 1000);
    await issue({
      issuerUserId,
      recipientUserId,
      petId,
      issuedAt: oldIssuedAt,
      expiresAt: oldExpiresAt,
    });

    await expect(
      store.findEffectiveGrantForCapability({
        recipientUserId,
        petId,
        capability: 'VIEW_TODAY',
        now,
      })
    ).resolves.toBeNull();

    await expect(
      store.issueGrant({
        id: randomUUID(),
        petId,
        issuerUserId,
        recipientUserId,
        requestKey: `renew-${randomUUID()}`,
        capabilities: ['VIEW_TODAY'],
        issuedAt: now,
        expiresAt: new Date(now.getTime() + 60 * 60 * 1000),
      })
    ).resolves.toBe(true);
  });

  it('requires LOG_OBSERVATION explicitly and blocks direct evidence after a relationship block', async () => {
    const { issuerUserId, recipientUserId, petId } = await fixture('evidence');
    const issuedAt = new Date();
    const expiresAt = new Date(issuedAt.getTime() + 60 * 60 * 1000);
    const viewOnlyGrant = await issue({
      issuerUserId,
      recipientUserId,
      petId,
      issuedAt,
      expiresAt,
    });
    const acceptedAt = new Date(issuedAt.getTime() + 1000);
    await expect(store.acceptGrant(viewOnlyGrant, recipientUserId, acceptedAt)).resolves.toBe(true);
    await expect(
      store.recordObservation({
        grantId: viewOnlyGrant,
        petId,
        actorUserId: recipientUserId,
        kind: 'BEHAVIOR',
        summary: 'View-only authority must not permit evidence writes.',
        note: null,
        observedAt: new Date(acceptedAt.getTime() + 1000),
      })
    ).rejects.toThrow();

    await expect(
      store.revokeGrant(viewOnlyGrant, issuerUserId, new Date(acceptedAt.getTime() + 2000))
    ).resolves.toBe(true);

    const fullIssuedAt = new Date(acceptedAt.getTime() + 3000);
    const fullGrant = await issue({
      issuerUserId,
      recipientUserId,
      petId,
      issuedAt: fullIssuedAt,
      expiresAt: new Date(fullIssuedAt.getTime() + 60 * 60 * 1000),
      capabilities: ['VIEW_TODAY', 'LOG_OBSERVATION'],
    });
    const fullAcceptedAt = new Date(fullIssuedAt.getTime() + 1000);
    await expect(store.acceptGrant(fullGrant, recipientUserId, fullAcceptedAt)).resolves.toBe(true);

    await prisma.blockedUser.create({
      data: { userId: issuerUserId, blockedId: recipientUserId, reason: 'test boundary' },
    });

    await expect(
      store.recordObservation({
        grantId: fullGrant,
        petId,
        actorUserId: recipientUserId,
        kind: 'BEHAVIOR',
        summary: 'A stale session must not write across a block.',
        note: null,
        observedAt: new Date(fullAcceptedAt.getTime() + 1000),
      })
    ).rejects.toThrow();
  });

  it('keeps issued capability scope immutable while allowing parent privacy deletion to cascade', async () => {
    const { issuerUserId, recipientUserId, petId } = await fixture('privacy');
    const issuedAt = new Date();
    const expiresAt = new Date(issuedAt.getTime() + 60 * 60 * 1000);
    const grantId = await issue({
      issuerUserId,
      recipientUserId,
      petId,
      issuedAt,
      expiresAt,
      capabilities: ['VIEW_TODAY', 'LOG_OBSERVATION'],
    });

    await expect(
      prisma.$executeRaw(Prisma.sql`
        DELETE FROM dogos_caregiver.grant_capabilities
        WHERE grant_id = ${grantId} AND capability = 'LOG_OBSERVATION'
      `)
    ).rejects.toThrow();

    await prisma.user.delete({ where: { id: issuerUserId } });
    usersToDelete.splice(usersToDelete.indexOf(issuerUserId), 1);

    const counts = await prisma.$queryRaw<
      Array<{ grants: number; capabilities: number; receipts: number; observations: number }>
    >(Prisma.sql`
      SELECT
        (SELECT COUNT(*)::int FROM dogos_caregiver.grants WHERE id = ${grantId}) AS grants,
        (SELECT COUNT(*)::int FROM dogos_caregiver.grant_capabilities WHERE grant_id = ${grantId}) AS capabilities,
        (SELECT COUNT(*)::int FROM dogos_caregiver.grant_receipts WHERE grant_id = ${grantId}) AS receipts,
        (SELECT COUNT(*)::int FROM dogos_caregiver.observations WHERE grant_id = ${grantId}) AS observations
    `);
    expect(counts[0]).toEqual({ grants: 0, capabilities: 0, receipts: 0, observations: 0 });
  });
});
