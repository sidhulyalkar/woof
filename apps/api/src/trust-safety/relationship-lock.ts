import { Prisma } from '@woof/database';

type RelationshipLockClient = Pick<Prisma.TransactionClient, '$queryRaw'>;

export function relationshipLockKey(userAId: string, userBId: string) {
  return `woof:relationship:${[userAId, userBId].sort().join(':')}`;
}

export function relationshipLockKeys(userId: string, otherUserIds: string[]) {
  return [
    ...new Set(
      otherUserIds
        .filter((otherUserId) => otherUserId && otherUserId !== userId)
        .map((otherUserId) => relationshipLockKey(userId, otherUserId))
    ),
  ].sort();
}

export async function acquireRelationshipLocks(
  tx: RelationshipLockClient,
  userId: string,
  otherUserIds: string[]
) {
  const keys = relationshipLockKeys(userId, otherUserIds);
  for (const key of keys) {
    await tx.$queryRaw<Array<{ locked: number }>>(Prisma.sql`
      SELECT 1 AS locked
      FROM (
        SELECT pg_advisory_xact_lock(hashtextextended(${key}, 0))
      ) AS relationship_lock
    `);
  }
  return keys;
}
