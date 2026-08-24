import { randomUUID } from 'node:crypto';
import { PrismaService } from '../prisma/prisma.service';
import { acquireRelationshipLocks } from '../trust-safety/relationship-lock';
import { ChatSecurityService } from './chat-security.service';

describe('chat realtime authorization integration', () => {
  const prisma = new PrismaService();
  const service = new ChatSecurityService(prisma);
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
        handle: `realtime-${label}-${suffix}`,
        email: `realtime-${label}-${suffix}@example.test`,
        visibility: 'PUBLIC',
      },
      select: { id: true },
    });
    usersToDelete.push(user.id);
    return user;
  }

  async function groupFixture(label: string) {
    const [userA, userB, userC] = await Promise.all([
      userFixture(`${label}-a`),
      userFixture(`${label}-b`),
      userFixture(`${label}-c`),
    ]);
    const conversation = await prisma.conversation.create({
      data: {
        participants: {
          create: [{ userId: userA.id }, { userId: userB.id }, { userId: userC.id }],
        },
      },
      select: { id: true },
    });
    conversationsToDelete.push(conversation.id);
    return { userA, userB, userC, conversation };
  }

  it('waits for a relationship block and excludes both blocked endpoints when the block commits first', async () => {
    const { userA, userB, userC, conversation } = await groupFixture('block-first');

    let blockWrittenResolve!: () => void;
    const blockWritten = new Promise<void>((resolve) => {
      blockWrittenResolve = resolve;
    });
    let releaseBlockResolve!: () => void;
    const releaseBlock = new Promise<void>((resolve) => {
      releaseBlockResolve = resolve;
    });

    const blockTransaction = prisma.$transaction(async (tx) => {
      await acquireRelationshipLocks(tx, userA.id, [userB.id]);
      await tx.blockedUser.create({
        data: { userId: userA.id, blockedId: userB.id, reason: 'realtime-block-first' },
      });
      blockWrittenResolve();
      await releaseBlock;
    });

    await blockWritten;

    let delivered = false;
    const deliveryPromise = service.withAuthorizedRealtimeRecipients(
      conversation.id,
      (authorizedUserIds) => {
        delivered = true;
        expect(authorizedUserIds).toEqual([userC.id]);
      }
    );

    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(delivered).toBe(false);

    releaseBlockResolve();
    await blockTransaction;

    await expect(deliveryPromise).resolves.toEqual({ authorizedUserIds: [userC.id] });
    expect(delivered).toBe(true);
  });

  it('holds block commit behind realtime delivery when delivery acquires the relationship graph first', async () => {
    const { userA, userB, userC, conversation } = await groupFixture('delivery-first');

    let deliveryEnteredResolve!: () => void;
    const deliveryEntered = new Promise<void>((resolve) => {
      deliveryEnteredResolve = resolve;
    });
    let releaseDeliveryResolve!: () => void;
    const releaseDelivery = new Promise<void>((resolve) => {
      releaseDeliveryResolve = resolve;
    });
    let deliveredUserIds: string[] = [];

    const deliveryPromise = service.withAuthorizedRealtimeRecipients(
      conversation.id,
      async (authorizedUserIds) => {
        deliveredUserIds = authorizedUserIds;
        deliveryEnteredResolve();
        await releaseDelivery;
      }
    );

    await deliveryEntered;

    let blockSettled = false;
    const blockPromise = prisma
      .$transaction(async (tx) => {
        await acquireRelationshipLocks(tx, userA.id, [userB.id]);
        await tx.blockedUser.create({
          data: { userId: userA.id, blockedId: userB.id, reason: 'realtime-delivery-first' },
        });
      })
      .finally(() => {
        blockSettled = true;
      });

    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(blockSettled).toBe(false);

    releaseDeliveryResolve();
    await expect(deliveryPromise).resolves.toEqual({
      authorizedUserIds: [...deliveredUserIds],
    });
    await blockPromise;

    expect([...deliveredUserIds].sort()).toEqual([userA.id, userB.id, userC.id].sort());
    await expect(
      prisma.blockedUser.findUnique({
        where: { userId_blockedId: { userId: userA.id, blockedId: userB.id } },
        select: { id: true },
      })
    ).resolves.toBeTruthy();
  });
});
