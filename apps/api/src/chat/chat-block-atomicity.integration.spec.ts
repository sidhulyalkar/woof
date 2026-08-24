import { randomUUID } from 'node:crypto';
import { ForbiddenException } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import { acquireRelationshipLocks } from '../trust-safety/relationship-lock';
import { ChatSecurityService } from './chat-security.service';

describe('chat block atomicity integration', () => {
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
        handle: `atomic-${label}-${suffix}`,
        email: `atomic-${label}-${suffix}@example.test`,
        visibility: 'PUBLIC',
      },
      select: { id: true },
    });
    usersToDelete.push(user.id);
    return user;
  }

  it('rejects a message that passed its outer access check when a waiting block commits first', async () => {
    const [sender, blocker] = await Promise.all([userFixture('sender'), userFixture('blocker')]);
    const conversation = await prisma.conversation.create({
      data: {
        participants: {
          create: [{ userId: sender.id }, { userId: blocker.id }],
        },
      },
      select: { id: true },
    });
    conversationsToDelete.push(conversation.id);

    let blockWrittenResolve!: () => void;
    const blockWritten = new Promise<void>((resolve) => {
      blockWrittenResolve = resolve;
    });
    let releaseBlockResolve!: () => void;
    const releaseBlock = new Promise<void>((resolve) => {
      releaseBlockResolve = resolve;
    });

    const blockTransaction = prisma.$transaction(async (tx) => {
      await acquireRelationshipLocks(tx, blocker.id, [sender.id]);
      await tx.blockedUser.create({
        data: { userId: blocker.id, blockedId: sender.id, reason: 'atomicity-test' },
      });
      blockWrittenResolve();
      await releaseBlock;
    });

    await blockWritten;

    let outerAccessPassedResolve!: () => void;
    const outerAccessPassed = new Promise<void>((resolve) => {
      outerAccessPassedResolve = resolve;
    });
    const originalAssertAccess = service.assertConversationAccess.bind(service);
    const accessSpy = jest
      .spyOn(service, 'assertConversationAccess')
      .mockImplementation(async (userId, conversationId) => {
        const result = await originalAssertAccess(userId, conversationId);
        outerAccessPassedResolve();
        return result;
      });

    let messageSettled = false;
    const clientMessageId = `atomic-${randomUUID()}`;
    const messagePromise = service
      .persistMessage({
        userId: sender.id,
        conversationId: conversation.id,
        clientMessageId,
        text: 'must not cross a committed block',
      })
      .finally(() => {
        messageSettled = true;
      });

    await outerAccessPassed;
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(messageSettled).toBe(false);

    releaseBlockResolve();
    await blockTransaction;

    await expect(messagePromise).rejects.toBeInstanceOf(ForbiddenException);
    accessSpy.mockRestore();

    await expect(
      prisma.message.count({ where: { conversationId: conversation.id, senderId: sender.id } })
    ).resolves.toBe(0);

    const receipts = await prisma.$queryRaw<Array<{ count: bigint }>>(Prisma.sql`
      SELECT COUNT(*)::bigint AS count
      FROM dogos_chat.message_receipts
      WHERE user_id = CAST(${sender.id} AS text)
        AND client_message_id = ${clientMessageId}
    `);
    expect(Number(receipts[0]?.count ?? 0)).toBe(0);
  });
});
