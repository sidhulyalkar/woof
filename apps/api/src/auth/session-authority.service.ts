import { Injectable, UnauthorizedException } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { PrismaService } from '../prisma/prisma.service';

type SessionRow = {
  id: string;
  userId: string;
  expiresAt: Date;
  revokedAt: Date | null;
  revocationReason: string | null;
  createdAt: Date;
};

type SessionIdRow = { id: string };
type ActiveSessionDelivery = (activeSessionIds: ReadonlySet<string>) => void | Promise<void>;
type ActiveSessionWork<T> = (tx: Prisma.TransactionClient) => T | Promise<T>;

@Injectable()
export class SessionAuthorityService {
  constructor(private readonly prisma: PrismaService) {}

  async createSession(input: { id: string; userId: string; expiresAt: Date }) {
    if (!input.id || !input.userId || input.expiresAt.getTime() <= Date.now()) {
      throw new UnauthorizedException('Session is unavailable');
    }

    await this.prisma.$executeRaw(Prisma.sql`
      INSERT INTO dogos_auth.sessions (id, user_id, expires_at)
      VALUES (
        CAST(${input.id} AS text),
        CAST(${input.userId} AS text),
        ${input.expiresAt}
      )
    `);

    return { id: input.id, userId: input.userId, expiresAt: input.expiresAt };
  }

  async assertActive(sessionId: string, userId: string): Promise<SessionRow> {
    if (!sessionId || !userId) throw new UnauthorizedException('Session is unavailable');

    const rows = await this.prisma.$queryRaw<SessionRow[]>(Prisma.sql`
      SELECT
        id,
        user_id AS "userId",
        expires_at AS "expiresAt",
        revoked_at AS "revokedAt",
        revocation_reason AS "revocationReason",
        created_at AS "createdAt"
      FROM dogos_auth.sessions
      WHERE id = CAST(${sessionId} AS text)
        AND user_id = CAST(${userId} AS text)
        AND revoked_at IS NULL
        AND expires_at > NOW()
      LIMIT 1
    `);

    const session = rows[0];
    if (!session) throw new UnauthorizedException('Session is unavailable');
    return session;
  }

  async withActiveSession<T>(sessionId: string, userId: string, work: ActiveSessionWork<T>) {
    if (!sessionId || !userId) return { authorized: false as const };

    return this.prisma.$transaction(async (tx) => {
      const rows = await tx.$queryRaw<SessionIdRow[]>(Prisma.sql`
        SELECT id
        FROM dogos_auth.sessions
        WHERE id = CAST(${sessionId} AS text)
          AND user_id = CAST(${userId} AS text)
          AND revoked_at IS NULL
          AND expires_at > NOW()
        LIMIT 1
        FOR SHARE
      `);
      if (!rows[0]) return { authorized: false as const };

      const result = await work(tx);
      return { authorized: true as const, result };
    });
  }

  async revokeSession(userId: string, sessionId: string, reason = 'LOGOUT') {
    const rows = await this.prisma.$queryRaw<SessionIdRow[]>(Prisma.sql`
      UPDATE dogos_auth.sessions
      SET
        revoked_at = COALESCE(revoked_at, NOW()),
        revocation_reason = COALESCE(revocation_reason, ${reason})
      WHERE id = CAST(${sessionId} AS text)
        AND user_id = CAST(${userId} AS text)
      RETURNING id
    `);

    return { revoked: rows.length > 0 };
  }

  async revokeAllSessions(userId: string, reason = 'LOGOUT_ALL') {
    return this.prisma.$transaction(async (tx) => {
      const activeRows = await tx.$queryRaw<SessionIdRow[]>(Prisma.sql`
        SELECT id
        FROM dogos_auth.sessions
        WHERE user_id = CAST(${userId} AS text)
          AND revoked_at IS NULL
          AND expires_at > NOW()
        ORDER BY id
        FOR UPDATE
      `);
      const sessionIds = activeRows.map((row) => row.id);
      if (sessionIds.length === 0) return { revokedCount: 0 };

      const revokedRows = await tx.$queryRaw<SessionIdRow[]>(Prisma.sql`
        UPDATE dogos_auth.sessions
        SET
          revoked_at = NOW(),
          revocation_reason = ${reason}
        WHERE id IN (${Prisma.join(sessionIds)})
          AND user_id = CAST(${userId} AS text)
        RETURNING id
      `);

      return { revokedCount: revokedRows.length };
    });
  }

  async withActiveSessions(sessionIds: string[], deliver: ActiveSessionDelivery) {
    return this.prisma.$transaction((tx) =>
      this.withActiveSessionsInTransaction(tx, sessionIds, deliver)
    );
  }

  async withActiveSessionsInTransaction(
    tx: Prisma.TransactionClient,
    sessionIds: string[],
    deliver: ActiveSessionDelivery
  ) {
    const uniqueIds = [...new Set(sessionIds.filter(Boolean))];
    if (uniqueIds.length === 0) {
      const empty = new Set<string>();
      await deliver(empty);
      return { activeSessionIds: empty };
    }

    const rows = await tx.$queryRaw<SessionIdRow[]>(Prisma.sql`
      SELECT id
      FROM dogos_auth.sessions
      WHERE id IN (${Prisma.join(uniqueIds)})
        AND revoked_at IS NULL
        AND expires_at > NOW()
      ORDER BY id
      FOR SHARE
    `);
    const activeSessionIds = new Set(rows.map((row) => row.id));
    await deliver(activeSessionIds);
    return { activeSessionIds };
  }
}
