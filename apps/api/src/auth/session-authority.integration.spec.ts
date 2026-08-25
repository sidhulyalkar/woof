import { randomUUID } from 'node:crypto';
import { ChatSecurityService } from '../chat/chat-security.service';
import { PrismaService } from '../prisma/prisma.service';
import { SessionAuthorityService } from './session-authority.service';

describe('SessionAuthorityService integration', () => {
  const prisma = new PrismaService();
  const service = new SessionAuthorityService(prisma);
  const usersToDelete: string[] = [];
  const conversationsToDelete: string[] = [];

  beforeAll(async () => {
    await prisma.$connect();
  });

  afterAll(async () => {
    if (conversationsToDelete.length > 0) {
      await prisma.conversation.deleteMany({ where: { id: { in: conversationsToDelete } } });
    }
    if (usersToDelete.length > 0) {
      await prisma.user.deleteMany({ where: { id: { in: usersToDelete } } });
    }
    await prisma.$disconnect();
  });

  async function userFixture(label: string) {
    const suffix = randomUUID().slice(0, 8);
    const user = await prisma.user.create({
      data: {
        handle: `session-${label}-${suffix}`,
        email: `session-${label}-${suffix}@example.test`,
      },
      select: { id: true },
    });
    usersToDelete.push(user.id);
    return user;
  }

  async function sessionFixture(userId: string, seconds = 300) {
    const id = randomUUID();
    await service.createSession({
      id,
      userId,
      expiresAt: new Date(Date.now() + seconds * 1_000),
    });
    return id;
  }

  it('accepts an active session and rejects it after explicit revocation', async () => {
    const user = await userFixture('revoke');
    const sessionId = await sessionFixture(user.id);

    await expect(service.assertActive(sessionId, user.id)).resolves.toMatchObject({
      id: sessionId,
      userId: user.id,
      revokedAt: null,
    });

    await expect(service.revokeSession(user.id, sessionId)).resolves.toEqual({ revoked: true });
    await expect(service.assertActive(sessionId, user.id)).rejects.toThrow(
      'Session is unavailable'
    );
  });

  it('revokes all active sessions for one user without touching another user', async () => {
    const userA = await userFixture('all-a');
    const userB = await userFixture('all-b');
    const [sessionA1, sessionA2, sessionB] = await Promise.all([
      sessionFixture(userA.id),
      sessionFixture(userA.id),
      sessionFixture(userB.id),
    ]);

    await expect(service.revokeAllSessions(userA.id)).resolves.toEqual({ revokedCount: 2 });
    await expect(service.assertActive(sessionA1, userA.id)).rejects.toThrow(
      'Session is unavailable'
    );
    await expect(service.assertActive(sessionA2, userA.id)).rejects.toThrow(
      'Session is unavailable'
    );
    await expect(service.assertActive(sessionB, userB.id)).resolves.toMatchObject({ id: sessionB });
  });

  it('allows concurrent shared authority admissions for the same active session', async () => {
    const user = await userFixture('shared-admission');
    const sessionId = await sessionFixture(user.id);

    let firstEnteredResolve!: () => void;
    const firstEntered = new Promise<void>((resolve) => {
      firstEnteredResolve = resolve;
    });
    let releaseFirstResolve!: () => void;
    const releaseFirst = new Promise<void>((resolve) => {
      releaseFirstResolve = resolve;
    });

    const first = service.withActiveSession(sessionId, user.id, async () => {
      firstEnteredResolve();
      await releaseFirst;
      return 'first';
    });
    await firstEntered;

    let secondEnteredResolve!: () => void;
    const secondEntered = new Promise<void>((resolve) => {
      secondEnteredResolve = resolve;
    });
    const second = service.withActiveSession(sessionId, user.id, async () => {
      secondEnteredResolve();
      return 'second';
    });

    await Promise.race([
      secondEntered,
      new Promise<never>((_, reject) => {
        setTimeout(() => reject(new Error('concurrent FOR SHARE admission was blocked')), 1_000);
      }),
    ]);

    await expect(second).resolves.toEqual({ authorized: true, result: 'second' });
    releaseFirstResolve();
    await expect(first).resolves.toEqual({ authorized: true, result: 'first' });
  });

  it('reuses the authority transaction for canonical message work with a one-connection pool', async () => {
    const originalDatabaseUrl = process.env.DATABASE_URL;
    if (!originalDatabaseUrl) throw new Error('DATABASE_URL is required for integration tests');

    const constrainedUrl = new URL(originalDatabaseUrl);
    constrainedUrl.searchParams.set('connection_limit', '1');
    constrainedUrl.searchParams.set('pool_timeout', '2');

    process.env.DATABASE_URL = constrainedUrl.toString();
    const constrainedPrisma = new PrismaService();
    try {
      await constrainedPrisma.$connect();
    } finally {
      process.env.DATABASE_URL = originalDatabaseUrl;
    }

    const constrainedAuthority = new SessionAuthorityService(constrainedPrisma);
    const constrainedChat = new ChatSecurityService(constrainedPrisma);

    try {
      const suffix = randomUUID().slice(0, 8);
      const [owner, peer] = await Promise.all([
        constrainedPrisma.user.create({
          data: {
            handle: `session-pool-owner-${suffix}`,
            email: `session-pool-owner-${suffix}@example.test`,
          },
          select: { id: true },
        }),
        constrainedPrisma.user.create({
          data: {
            handle: `session-pool-peer-${suffix}`,
            email: `session-pool-peer-${suffix}@example.test`,
          },
          select: { id: true },
        }),
      ]);
      usersToDelete.push(owner.id, peer.id);

      const conversation = await constrainedPrisma.conversation.create({
        data: {
          participants: {
            create: [{ userId: owner.id }, { userId: peer.id }],
          },
        },
        select: { id: true },
      });
      conversationsToDelete.push(conversation.id);

      const sessionId = randomUUID();
      await constrainedAuthority.createSession({
        id: sessionId,
        userId: owner.id,
        expiresAt: new Date(Date.now() + 300_000),
      });

      const actions = Array.from({ length: 3 }, (_, index) =>
        constrainedAuthority.withActiveSession(sessionId, owner.id, (tx) =>
          constrainedChat.persistMessageInTransaction(tx, {
            userId: owner.id,
            conversationId: conversation.id,
            clientMessageId: `pool-message-${index}-${randomUUID()}`,
            text: `pool-safe message ${index}`,
          })
        )
      );

      const results = await Promise.race([
        Promise.all(actions),
        new Promise<never>((_, reject) => {
          setTimeout(
            () =>
              reject(new Error('transaction-bound message work exhausted the one-connection pool')),
            4_000
          );
        }),
      ]);

      expect(results).toHaveLength(3);
      expect(
        results.every((result) => result.authorized && result.result.duplicate === false)
      ).toBe(true);
      await expect(
        constrainedPrisma.message.count({ where: { conversationId: conversation.id } })
      ).resolves.toBe(3);
    } finally {
      await constrainedPrisma.$disconnect();
    }
  });

  it('orders an admitted realtime action against current-session revocation', async () => {
    const user = await userFixture('action-ordering');
    const sessionId = await sessionFixture(user.id);

    let actionEnteredResolve!: () => void;
    const actionEntered = new Promise<void>((resolve) => {
      actionEnteredResolve = resolve;
    });
    let releaseActionResolve!: () => void;
    const releaseAction = new Promise<void>((resolve) => {
      releaseActionResolve = resolve;
    });

    const action = service.withActiveSession(sessionId, user.id, async () => {
      actionEnteredResolve();
      await releaseAction;
      return 'persisted';
    });
    await actionEntered;

    let revokeSettled = false;
    const revoke = service.revokeSession(user.id, sessionId).finally(() => {
      revokeSettled = true;
    });

    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(revokeSettled).toBe(false);

    releaseActionResolve();
    await expect(action).resolves.toEqual({ authorized: true, result: 'persisted' });
    await expect(revoke).resolves.toEqual({ revoked: true });

    const work = jest.fn();
    await expect(service.withActiveSession(sessionId, user.id, work)).resolves.toEqual({
      authorized: false,
    });
    expect(work).not.toHaveBeenCalled();
  });

  it('serializes passive delivery against revocation of the same session row', async () => {
    const user = await userFixture('ordering');
    const sessionId = await sessionFixture(user.id);

    let deliveryEnteredResolve!: () => void;
    const deliveryEntered = new Promise<void>((resolve) => {
      deliveryEnteredResolve = resolve;
    });
    let releaseDeliveryResolve!: () => void;
    const releaseDelivery = new Promise<void>((resolve) => {
      releaseDeliveryResolve = resolve;
    });

    const delivery = service.withActiveSessions([sessionId], async (activeSessionIds) => {
      expect(activeSessionIds.has(sessionId)).toBe(true);
      deliveryEnteredResolve();
      await releaseDelivery;
    });

    await deliveryEntered;

    let revokeSettled = false;
    const revoke = service.revokeSession(user.id, sessionId).finally(() => {
      revokeSettled = true;
    });

    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(revokeSettled).toBe(false);

    releaseDeliveryResolve();
    await delivery;
    await expect(revoke).resolves.toEqual({ revoked: true });

    let deliveredAfterRevocation = false;
    await service.withActiveSessions([sessionId], (activeSessionIds) => {
      deliveredAfterRevocation = activeSessionIds.has(sessionId);
    });
    expect(deliveredAfterRevocation).toBe(false);
  });

  it('orders multi-session logout-all behind active delivery locks and revokes the whole user set', async () => {
    const user = await userFixture('ordering-all');
    const sessionIds = await Promise.all([sessionFixture(user.id), sessionFixture(user.id)]);

    let deliveryEnteredResolve!: () => void;
    const deliveryEntered = new Promise<void>((resolve) => {
      deliveryEnteredResolve = resolve;
    });
    let releaseDeliveryResolve!: () => void;
    const releaseDelivery = new Promise<void>((resolve) => {
      releaseDeliveryResolve = resolve;
    });

    const delivery = service.withActiveSessions(
      [...sessionIds].reverse(),
      async (activeSessionIds) => {
        expect([...activeSessionIds].sort()).toEqual([...sessionIds].sort());
        deliveryEnteredResolve();
        await releaseDelivery;
      }
    );
    await deliveryEntered;

    let revokeAllSettled = false;
    const revokeAll = service.revokeAllSessions(user.id).finally(() => {
      revokeAllSettled = true;
    });

    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(revokeAllSettled).toBe(false);

    releaseDeliveryResolve();
    await delivery;
    await expect(revokeAll).resolves.toEqual({ revokedCount: 2 });

    for (const sessionId of sessionIds) {
      await expect(service.assertActive(sessionId, user.id)).rejects.toThrow(
        'Session is unavailable'
      );
    }
  });
});
