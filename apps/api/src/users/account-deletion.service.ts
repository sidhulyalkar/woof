import { Injectable, NotFoundException, ServiceUnavailableException } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';
import { StorageService } from '../storage/storage.service';

@Injectable()
export class AccountDeletionService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly storage: StorageService,
  ) {}

  async deleteCurrentAccount(userId: string): Promise<void> {
    const user = await this.prisma.user.findUnique({
      where: { id: userId },
      select: {
        id: true,
        pets: { select: { id: true } },
        householdMemberships: { select: { householdId: true } },
        conversationParticipants: { select: { conversationId: true } },
      },
    });

    if (!user) {
      throw new NotFoundException('User not found');
    }

    const petIds = user.pets.map((pet) => pet.id);
    const householdIds = [...new Set(user.householdMemberships.map((item) => item.householdId))];
    const conversationIds = [
      ...new Set(user.conversationParticipants.map((item) => item.conversationId)),
    ];

    const mediaAssets = await this.prisma.mediaAsset.findMany({
      where: {
        OR: [
          { ownerId: userId },
          ...(petIds.length > 0 ? [{ petId: { in: petIds } }] : []),
        ],
      },
      select: {
        storageKey: true,
        derivatives: { select: { storageKey: true } },
      },
    });

    const storageKeys = [
      ...new Set(
        mediaAssets.flatMap((asset) => [
          asset.storageKey,
          ...asset.derivatives.map((derivative) => derivative.storageKey),
        ]),
      ),
    ];

    // Private object deletion happens before the database transaction. If the object
    // store is unavailable, fail closed and leave relational account state intact so
    // the user can retry instead of creating durable orphaned private media. S3-style
    // DeleteObject is idempotent, so retry remains safe after a partial provider pass.
    try {
      for (const key of storageKeys) {
        await this.storage.deleteFile(key);
      }
    } catch {
      throw new ServiceUnavailableException('Account deletion could not remove private media');
    }

    await this.prisma.$transaction(async (tx) => {
      // These historical tables intentionally have no user/pet foreign keys. Delete
      // their identifier-bearing rows explicitly before the canonical user cascade.
      await tx.telemetry.deleteMany({
        where: {
          OR: [
            { userId },
            ...(petIds.length > 0 ? [{ petId: { in: petIds } }] : []),
          ],
        },
      });

      await tx.meetupProposal.deleteMany({
        where: { OR: [{ proposerId: userId }, { recipientId: userId }] },
      });

      await tx.coActivitySegment.deleteMany({
        where: {
          OR: [
            { userId },
            { otherUserId: userId },
            ...(petIds.length > 0
              ? [{ petId: { in: petIds } }, { otherPetId: { in: petIds } }]
              : []),
          ],
        },
      });

      await tx.serviceIntent.deleteMany({ where: { userId } });
      await tx.gamification.deleteMany({ where: { userId } });
      await tx.pointTransaction.deleteMany({ where: { userId } });
      await tx.badgeAward.deleteMany({ where: { userId } });
      await tx.weeklyStreak.deleteMany({ where: { userId } });
      await tx.proactiveNudge.deleteMany({
        where: { OR: [{ userId }, { targetUserId: userId }] },
      });
      await tx.nudgeCooldown.deleteMany({
        where: { OR: [{ userId }, { targetUserId: userId }] },
      });

      // Historical safety/moderation tables also use raw user identifiers. Identity
      // involving the deleting account is removed; unrelated moderation history is
      // retained without the deleted account acting as an administrator identity.
      await tx.safetyVerification.updateMany({
        where: { verifiedBy: userId, NOT: { userId } },
        data: { verifiedBy: null },
      });
      await tx.safetyVerification.deleteMany({ where: { userId } });

      await tx.reportFlag.updateMany({
        where: {
          reviewedBy: userId,
          NOT: { OR: [{ reporterId: userId }, { reportedId: userId }] },
        },
        data: { reviewedBy: null },
      });
      await tx.reportFlag.deleteMany({
        where: { OR: [{ reporterId: userId }, { reportedId: userId }] },
      });
      await tx.blockedUser.deleteMany({
        where: { OR: [{ userId }, { blockedId: userId }] },
      });

      // These live foreign keys do not currently declare ON DELETE behavior. Remove
      // or detach their current ownership before deleting the user so the database
      // remains the final referential-integrity authority.
      await tx.reward.updateMany({
        where: { redeemedBy: userId },
        data: { redeemedBy: null, redeemedAt: null },
      });
      await tx.meetup.deleteMany({ where: { creatorUserId: userId } });
      await tx.communityEvent.deleteMany({ where: { hostUserId: userId } });

      // Legacy training exports embed identity inside JSON rather than foreign keys.
      // Delete only the known versioned MLTrainingDataPoint identity locations. Keep
      // the zero-pet case separate so the query never depends on array parameter casts.
      if (petIds.length > 0) {
        await tx.$executeRaw(
          Prisma.sql`
            DELETE FROM "ml_training_data"
            WHERE "dataPoint"->'userFeatures'->>'userId' = ${userId}
               OR "dataPoint"->'candidateFeatures'->>'userId' = ${userId}
               OR "dataPoint"->'userFeatures'->>'petId' IN (${Prisma.join(petIds)})
               OR "dataPoint"->'candidateFeatures'->>'petId' IN (${Prisma.join(petIds)})
          `,
        );
      } else {
        await tx.$executeRaw(
          Prisma.sql`
            DELETE FROM "ml_training_data"
            WHERE "dataPoint"->'userFeatures'->>'userId' = ${userId}
               OR "dataPoint"->'candidateFeatures'->>'userId' = ${userId}
          `,
        );
      }

      // The canonical delete now owns modern cascades: pets, sessions, household
      // memberships, activity/social rows, integrations, media metadata, dogOS
      // operational schemas, CareEvents, rewards, Story evidence, and projections.
      await tx.user.delete({ where: { id: userId } });

      // Relationship containers are not historical tombstones. Remove only containers
      // this user participated in that became truly empty after the canonical cascade.
      if (householdIds.length > 0) {
        await tx.household.deleteMany({
          where: {
            id: { in: householdIds },
            members: { none: {} },
            pets: { none: {} },
          },
        });
      }

      if (conversationIds.length > 0) {
        await tx.conversation.deleteMany({
          where: {
            id: { in: conversationIds },
            participants: { none: {} },
          },
        });
      }
    });
  }
}
