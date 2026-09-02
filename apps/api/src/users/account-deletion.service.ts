import { Injectable, NotFoundException, ServiceUnavailableException } from '@nestjs/common';
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
      },
    });

    if (!user) {
      throw new NotFoundException('User not found');
    }

    const petIds = user.pets.map((pet) => pet.id);
    const householdIds = [...new Set(user.householdMemberships.map((item) => item.householdId))];

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
    // the user can retry instead of creating durable orphaned private media.
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
      // Delete only the known versioned MLTrainingDataPoint identity locations.
      await tx.$executeRaw`
        DELETE FROM "ml_training_data"
        WHERE "dataPoint"->'userFeatures'->>'userId' = ${userId}
           OR "dataPoint"->'candidateFeatures'->>'userId' = ${userId}
           OR "dataPoint"->'userFeatures'->>'petId' = ANY(${petIds}::text[])
           OR "dataPoint"->'candidateFeatures'->>'petId' = ANY(${petIds}::text[])
      `;

      // The canonical delete now owns the modern cascades: pets, sessions, household
      // memberships, activity/social rows, integrations, media metadata, dogOS
      // operational schemas, CareEvents, rewards, Story evidence, and projections.
      await tx.user.delete({ where: { id: userId } });

      // Households are relationship containers, not historical tombstones. Remove
      // containers that became empty only because this account and its owned pets
      // disappeared; shared households remain intact.
      if (householdIds.length > 0) {
        await tx.household.deleteMany({
          where: {
            id: { in: householdIds },
            members: { none: {} },
            pets: { none: {} },
          },
        });
      }
    });
  }
}
