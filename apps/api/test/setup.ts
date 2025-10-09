import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

// Clear test database before tests
beforeAll(async () => {
  // Only run in test environment
  if (process.env.NODE_ENV !== 'test') {
    throw new Error('Tests can only run in test environment');
  }

  // Clean up database
  await prisma.$transaction([
    prisma.message.deleteMany(),
    prisma.conversation.deleteMany(),
    prisma.comment.deleteMany(),
    prisma.like.deleteMany(),
    prisma.post.deleteMany(),
    prisma.meetupInvite.deleteMany(),
    prisma.meetup.deleteMany(),
    prisma.activity.deleteMany(),
    prisma.petEdge.deleteMany(),
    prisma.pet.deleteMany(),
    prisma.user.deleteMany(),
  ]);
});

// Disconnect after all tests
afterAll(async () => {
  await prisma.$disconnect();
});

export { prisma };
