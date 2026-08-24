const prismaClient = require('@prisma/client');

const { PrismaClient } = prismaClient;

// Reuse one client during non-production hot reloads to avoid exhausting the
// database connection limit. Production processes own their client normally.
const globalForPrisma = global;
const prisma =
  globalForPrisma.prisma ||
  new PrismaClient({
    log: process.env.NODE_ENV === 'development' ? ['query', 'error', 'warn'] : ['error'],
  });

if (process.env.NODE_ENV !== 'production') {
  globalForPrisma.prisma = prisma;
}

module.exports = {
  ...prismaClient,
  prisma,
  default: prisma,
};
