import { BadRequestException, Injectable, NotFoundException } from '@nestjs/common';
import { PrismaService } from '../prisma/prisma.service';
import { BlockUserDto, ReportUserDto } from './dto/trust-safety.dto';
import { acquireRelationshipLocks } from './relationship-lock';

@Injectable()
export class TrustSafetyService {
  constructor(private readonly prisma: PrismaService) {}

  async blockUser(userId: string, dto: BlockUserDto) {
    if (userId === dto.blockedUserId) {
      throw new BadRequestException('You cannot block yourself');
    }
    const target = await this.prisma.user.findUnique({
      where: { id: dto.blockedUserId },
      select: { id: true },
    });
    if (!target) throw new NotFoundException('Member not found');

    const block = await this.prisma.$transaction(async (tx) => {
      await acquireRelationshipLocks(tx, userId, [dto.blockedUserId]);
      return tx.blockedUser.upsert({
        where: {
          userId_blockedId: { userId, blockedId: dto.blockedUserId },
        },
        create: {
          userId,
          blockedId: dto.blockedUserId,
          reason: dto.reason?.trim() || null,
        },
        update: { reason: dto.reason?.trim() || null },
      });
    });

    await Promise.all([
      this.prisma.meetupProposal.updateMany({
        where: {
          OR: [
            { proposerId: userId, recipientId: dto.blockedUserId },
            { proposerId: dto.blockedUserId, recipientId: userId },
          ],
          status: { in: ['pending', 'accepted'] },
        },
        data: { status: 'cancelled' },
      }),
      this.markPetRelationshipsAvoid(userId, dto.blockedUserId),
      this.prisma.telemetry.create({
        data: {
          userId,
          source: 'trust_safety',
          event: 'USER_BLOCKED',
          data: { blockedUserId: dto.blockedUserId },
        },
      }),
    ]);

    return { id: block.id, blockedUserId: dto.blockedUserId, createdAt: block.createdAt };
  }

  async unblockUser(userId: string, blockedUserId: string) {
    const result = await this.prisma.$transaction(async (tx) => {
      await acquireRelationshipLocks(tx, userId, [blockedUserId]);
      return tx.blockedUser.deleteMany({
        where: { userId, blockedId: blockedUserId },
      });
    });
    return { unblocked: result.count > 0 };
  }

  async getBlockedUsers(userId: string) {
    const blocks = await this.prisma.blockedUser.findMany({
      where: { userId },
      orderBy: { createdAt: 'desc' },
    });
    const ids = blocks.map((block) => block.blockedId);
    const users = ids.length
      ? await this.prisma.user.findMany({
          where: { id: { in: ids } },
          select: { id: true, handle: true, avatarUrl: true },
        })
      : [];
    const byId = new Map(users.map((user) => [user.id, user]));
    return blocks.map((block) => ({
      ...byId.get(block.blockedId),
      blockedUserId: block.blockedId,
      blockedAt: block.createdAt,
    }));
  }

  async reportUser(userId: string, dto: ReportUserDto) {
    if (userId === dto.reportedUserId) {
      throw new BadRequestException('You cannot report yourself');
    }
    const target = await this.prisma.user.findUnique({
      where: { id: dto.reportedUserId },
      select: { id: true },
    });
    if (!target) throw new NotFoundException('Member not found');

    const report = await this.prisma.reportFlag.create({
      data: {
        reporterId: userId,
        reportedId: dto.reportedUserId,
        reason: dto.reason,
        description: dto.description?.trim() || null,
        evidence: dto.evidence ?? [],
      },
      select: { id: true, reason: true, status: true, createdAt: true },
    });

    await this.prisma.telemetry.create({
      data: {
        userId,
        source: 'trust_safety',
        event: 'SAFETY_REPORT_SUBMITTED',
        data: {
          reportId: report.id,
          reportedUserId: dto.reportedUserId,
          reason: dto.reason,
        },
      },
    });
    return report;
  }

  async getMyReports(userId: string) {
    return this.prisma.reportFlag.findMany({
      where: { reporterId: userId },
      select: {
        id: true,
        reportedId: true,
        reason: true,
        description: true,
        status: true,
        actionTaken: true,
        createdAt: true,
        reviewedAt: true,
      },
      orderBy: { createdAt: 'desc' },
      take: 100,
    });
  }

  async isBlockedEitherDirection(userAId: string, userBId: string) {
    return Boolean(
      await this.prisma.blockedUser.findFirst({
        where: {
          OR: [
            { userId: userAId, blockedId: userBId },
            { userId: userBId, blockedId: userAId },
          ],
        },
        select: { id: true },
      })
    );
  }

  private async markPetRelationshipsAvoid(userAId: string, userBId: string) {
    const [petsA, petsB] = await Promise.all([
      this.prisma.pet.findMany({ where: { ownerId: userAId }, select: { id: true } }),
      this.prisma.pet.findMany({ where: { ownerId: userBId }, select: { id: true } }),
    ]);
    const idsA = petsA.map((pet) => pet.id);
    const idsB = petsB.map((pet) => pet.id);
    if (idsA.length === 0 || idsB.length === 0) return;
    await this.prisma.petEdge.updateMany({
      where: {
        OR: [
          { petAId: { in: idsA }, petBId: { in: idsB } },
          { petAId: { in: idsB }, petBId: { in: idsA } },
        ],
      },
      data: { status: 'AVOID' },
    });
  }
}
