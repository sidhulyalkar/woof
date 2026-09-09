import { Injectable, NotFoundException } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';

export type PackAccess = {
  id: string;
  name: string;
  scope: string;
  visibility: string;
  memberCount: number;
  viewerJoined: boolean;
};

@Injectable()
export class PackAccessService {
  constructor(private readonly prisma: PrismaService) {}

  async getAccess(userId: string, packId: string): Promise<PackAccess | null> {
    const rows = await this.prisma.$queryRaw<PackAccess[]>(Prisma.sql`
      SELECT
        pack.id,
        pack.name,
        pack.scope,
        pack.visibility,
        COUNT(member.user_id)::int AS "memberCount",
        COALESCE(
          BOOL_OR(member.user_id = ${userId} AND member.status = 'ACTIVE'),
          FALSE
        ) AS "viewerJoined"
      FROM dogos_social.packs pack
      LEFT JOIN dogos_social.pack_memberships member
        ON member.pack_id = pack.id AND member.status = 'ACTIVE'
      WHERE pack.id = ${packId}
      GROUP BY pack.id, pack.name, pack.scope, pack.visibility
    `);
    return rows[0] ?? null;
  }

  async requireViewable(userId: string, packId: string): Promise<PackAccess> {
    const pack = await this.getAccess(userId, packId);
    if (!pack || (pack.visibility !== 'PUBLIC' && !pack.viewerJoined)) {
      throw new NotFoundException('Pack not found');
    }
    return pack;
  }

  async requireActiveMembership(userId: string, packId: string): Promise<PackAccess> {
    const pack = await this.getAccess(userId, packId);
    if (!pack || !pack.viewerJoined) {
      // Keep membership state non-enumerable through this authority boundary.
      throw new NotFoundException('Pack not found');
    }
    return pack;
  }
}
