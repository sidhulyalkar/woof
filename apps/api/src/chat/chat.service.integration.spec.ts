import { randomUUID } from 'node:crypto';
import { ChatSecurityService } from './chat-security.service';
import { ChatService } from './chat.service';
import { PrismaService } from '../prisma/prisma.service';

describe('ChatService integration', () => {
  const prisma = new PrismaService();
  const security = { assertConversationAccess: jest.fn() } as unknown as ChatSecurityService;
  const service = new ChatService(prisma, security);
  const usersToDelete: string[] = [];

  beforeAll(async () => {
    await prisma.$connect();
  });

  afterAll(async () => {
    if (usersToDelete.length > 0) {
      await prisma.telemetry.deleteMany({ where: { userId: { in: usersToDelete } } });
      await prisma.user.deleteMany({ where: { id: { in: usersToDelete } } });
    }
    await prisma.$disconnect();
  });

  async function userFixture(label: string) {
    const suffix = randomUUID().slice(0, 8);
    const user = await prisma.user.create({
      data: {
        handle: `chat-${label}-${suffix}`,
        email: `chat-${label}-${suffix}@example.test`,
        visibility: 'PUBLIC',
      },
      select: { id: true },
    });
    usersToDelete.push(user.id);
    return user;
  }

  it('creates exactly one direct thread under concurrent opposite-direction requests', async () => {
    const [left, right] = await Promise.all([userFixture('left'), userFixture('right')]);

    const attempts = await Promise.all(
      Array.from({ length: 8 }, (_, index) =>
        index % 2 === 0
          ? service.createDirectConversation(left.id, right.id)
          : service.createDirectConversation(right.id, left.id)
      )
    );

    expect(attempts.filter((attempt) => attempt.created)).toHaveLength(1);
    expect(new Set(attempts.map((attempt) => attempt.id)).size).toBe(1);

    const conversations = await prisma.conversation.findMany({
      where: {
        AND: [
          { participants: { some: { userId: left.id } } },
          { participants: { some: { userId: right.id } } },
        ],
      },
      select: {
        id: true,
        participants: { select: { userId: true } },
      },
    });

    const direct = conversations.filter(
      (conversation) =>
        conversation.participants.length === 2 &&
        conversation.participants.some((participant) => participant.userId === left.id) &&
        conversation.participants.some((participant) => participant.userId === right.id)
    );
    expect(direct).toHaveLength(1);

    await expect(
      prisma.telemetry.count({
        where: {
          userId: { in: [left.id, right.id] },
          source: 'chat',
          event: 'CONVERSATION_STARTED',
        },
      })
    ).resolves.toBe(1);
  });

  it('computes unread direct-message counts in one batched query using lastReadAt', async () => {
    const [owner, other] = await Promise.all([userFixture('owner'), userFixture('other')]);
    const lastReadAt = new Date('2026-08-23T20:00:00.000Z');
    const conversation = await prisma.conversation.create({
      data: {
        participants: {
          create: [
            { userId: owner.id, lastReadAt },
            { userId: other.id },
          ],
        },
      },
      select: { id: true },
    });

    await prisma.message.createMany({
      data: [
        {
          conversationId: conversation.id,
          senderId: other.id,
          text: 'old message',
          createdAt: new Date('2026-08-23T19:59:00.000Z'),
        },
        {
          conversationId: conversation.id,
          senderId: other.id,
          text: 'new unread message',
          createdAt: new Date('2026-08-23T20:01:00.000Z'),
        },
        {
          conversationId: conversation.id,
          senderId: owner.id,
          text: 'my reply',
          createdAt: new Date('2026-08-23T20:02:00.000Z'),
        },
      ],
    });

    await expect(service.listConversations(owner.id)).resolves.toEqual([
      expect.objectContaining({
        id: conversation.id,
        participant: expect.objectContaining({ id: other.id }),
        unreadCount: 1,
        lastMessage: expect.objectContaining({ content: 'my reply' }),
      }),
    ]);
  });
});
