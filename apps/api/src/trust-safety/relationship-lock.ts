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

export function relationshipLockKeysForParticipants(userIds: string[]) {
  const uniqueUserIds = [...new Set(userIds.filter(Boolean))].sort();
  const keys: string[] = [];

  for (let index = 0; index < uniqueUserIds.length; index += 1) {
    for (let otherIndex = index + 1; otherIndex < uniqueUserIds.length; otherIndex += 1) {
      keys.push(relationshipLockKey(uniqueUserIds[index]!, uniqueUserIds[otherIndex]!));
    }
  }

  return keys.sort();
}

async function acquireRelationshipLockKeys(tx: RelationshipLockClient, keys: string[]) {
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

export async function acquireRelationshipLocks(
  tx: RelationshipLockClient,
  userId: string,
  otherUserIds: string[]
) {
  return acquireRelationshipLockKeys(tx, relationshipLockKeys(userId, otherUserIds));
}

export async function acquireParticipantRelationshipLocks(
  tx: RelationshipLockClient,
  userIds: string[]
) {
  return acquireRelationshipLockKeys(tx, relationshipLockKeysForParticipants(userIds));
}
