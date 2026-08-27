import { BadRequestException, Injectable, NotFoundException } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { createHash, randomUUID } from 'node:crypto';
import { PrismaService } from '../prisma/prisma.service';
import {
  publicArcadeCatalog,
  scenarioByKey,
  scenarioForChallenge,
  scoreArcadeResponse,
} from './social-adventure.arcade';
import {
  currentUtcSeason,
  deriveSocialAdventureScore,
  HUMAN_SKILL_CHALLENGES,
  HUMAN_SKILL_CHALLENGE_VERSION,
  LOCAL_LEAGUE_MINIMUM_COHORT,
  SOCIAL_ADVENTURE_POLICY_VERSION,
  type HumanSkillChallenge,
} from './social-adventure.policy';
import type {
  CompleteHumanSkillAttemptDto,
  CreatePackDto,
  CreateSocialShareDto,
  SocialReactionDto,
  UpdateSocialAdventurePreferencesDto,
} from './dto/social-adventure.dto';

type PreferenceRow = {
  globalLeaderboardOptIn: boolean;
};

type SkillAttemptRow = {
  id: string;
  challengeKey: string;
  challengeVersion: string;
  scenarioKey: string;
  issuedAt: Date;
  expiresAt: Date;
  completedAt: Date | null;
  score: number | null;
  receipt: unknown;
};

type SkillScoreRow = {
  id: string;
  challengeKey: string;
  score: number;
};

type AdventureEvidenceRow = {
  id: string;
  pathway: string;
};

type LeaderboardUserRow = {
  id: string;
  handle: string;
  avatarUrl: string | null;
};

type ShareSource = {
  sourceType: 'CARE_EVENT' | 'HUMAN_SKILL_ATTEMPT';
  sourceId: string;
  petId: string | null;
  kind: 'ADVENTURE_MEMORY' | 'DISCOVERY' | 'SKILL_MOMENT' | 'GOOD_READ';
  headline: string;
  summary: string;
  payload: Record<string, unknown>;
};

type ExistingShareRow = {
  id: string;
  postId: string;
};

type FeedRow = {
  shareId: string;
  postId: string;
  kind: string;
  headline: string;
  summary: string;
  payload: unknown;
  caption: string | null;
  visibility: string;
  createdAt: Date;
  authorUserId: string;
  handle: string;
  avatarUrl: string | null;
  petId: string | null;
  petName: string | null;
  petAvatarUrl: string | null;
  likesCount: number;
  commentsCount: number;
};

type ReactionRow = {
  shareId: string;
  reaction: string;
  count: number;
  mine: boolean;
};

type PackRow = {
  id: string;
  name: string;
  slug: string;
  scope: string;
  regionKey: string | null;
  visibility: string;
  memberCount: number;
  joined: boolean;
  role: string | null;
};

type PackAccessRow = {
  id: string;
  name: string;
  scope: string;
  visibility: string;
  memberCount: number;
  viewerJoined: boolean;
};

const REACTION_TYPES = [
  'NICE_READ',
  'GOOD_CALL',
  'TRYING_THIS',
  'ADVENTURE_INSPIRATION',
  'CHEER',
] as const;

const ATTEMPT_TTL_MS = 10 * 60 * 1000;

@Injectable()
export class SocialAdventureService {
  constructor(private readonly prisma: PrismaService) {}

  async getMine(userId: string) {
    const [preferences, score, bestScores] = await Promise.all([
      this.getPreferences(userId),
      this.computeScore(userId),
      this.getBestHumanSkillScores(userId),
    ]);

    return {
      preferences,
      season: score.season,
      score: score.score,
      maxScore: score.maxScore,
      components: score.components,
      humanSkillBestScores: bestScores,
      policyVersion: SOCIAL_ADVENTURE_POLICY_VERSION,
      principles: [
        'human-skill-over-pet-performance',
        'variety-over-volume',
        'no-streak-loss',
        'no-health-competition',
        'no-posting-or-popularity-points',
      ],
    };
  }

  async updatePreferences(userId: string, dto: UpdateSocialAdventurePreferencesDto) {
    await this.prisma.$executeRaw(Prisma.sql`
      INSERT INTO dogos_social.preferences (user_id, global_leaderboard_opt_in, updated_at)
      VALUES (${userId}, ${dto.globalLeaderboardOptIn}, NOW())
      ON CONFLICT (user_id)
      DO UPDATE SET
        global_leaderboard_opt_in = EXCLUDED.global_leaderboard_opt_in,
        updated_at = NOW()
    `);
    return this.getPreferences(userId);
  }

  async getGlobalLeaderboard(userId: string, limit = 30) {
    const safeLimit = Math.max(1, Math.min(Number(limit) || 30, 50));
    const candidates = await this.prisma.$queryRaw<LeaderboardUserRow[]>(Prisma.sql`
      SELECT
        u.id,
        u.handle,
        u.avatar_url AS "avatarUrl"
      FROM dogos_social.preferences pref
      JOIN public.users u ON u.id = pref.user_id
      WHERE pref.global_leaderboard_opt_in = TRUE
        AND u.visibility = 'PUBLIC'
        AND NOT EXISTS (
          SELECT 1
          FROM public.blocked_users blocked
          WHERE (blocked.user_id = ${userId} AND blocked.blocked_id = u.id)
             OR (blocked.user_id = u.id AND blocked.blocked_id = ${userId})
        )
      ORDER BY u.id ASC
      LIMIT 100
    `);

    const rows = await this.scoreLeaderboardUsers(candidates);
    const entries = rows.slice(0, safeLimit).map((row, index) => ({ ...row, rank: index + 1 }));
    const me = await this.computeScore(userId);
    const myPublicRank = rows.findIndex((row) => row.userId === userId);

    return {
      scope: 'GLOBAL' as const,
      season: me.season,
      entries,
      me: {
        score: me.score,
        maxScore: me.maxScore,
        rank: myPublicRank >= 0 ? myPublicRank + 1 : null,
        public: myPublicRank >= 0,
      },
      policyVersion: SOCIAL_ADVENTURE_POLICY_VERSION,
      disclaimer:
        'This league scores human learning and bounded Adventure variety. It does not rank pet obedience, health, exercise volume, symptoms, mileage, likes, or streaks.',
    };
  }

  async getPackLeaderboard(userId: string, packId: string, limit = 30) {
    const safeLimit = Math.max(1, Math.min(Number(limit) || 30, 50));
    const accessRows = await this.prisma.$queryRaw<PackAccessRow[]>(Prisma.sql`
      SELECT
        pack.id,
        pack.name,
        pack.scope,
        pack.visibility,
        COUNT(member.user_id)::int AS "memberCount",
        BOOL_OR(member.user_id = ${userId} AND member.status = 'ACTIVE') AS "viewerJoined"
      FROM dogos_social.packs pack
      LEFT JOIN dogos_social.pack_memberships member
        ON member.pack_id = pack.id AND member.status = 'ACTIVE'
      WHERE pack.id = ${packId}
      GROUP BY pack.id, pack.name, pack.scope, pack.visibility
    `);
    const pack = accessRows[0];
    if (!pack || (pack.visibility !== 'PUBLIC' && !pack.viewerJoined)) {
      throw new NotFoundException('Pack not found');
    }

    if (pack.scope === 'LOCAL' && pack.memberCount < LOCAL_LEAGUE_MINIMUM_COHORT) {
      return {
        scope: 'PACK' as const,
        pack: { id: pack.id, name: pack.name, memberCount: pack.memberCount },
        cohortReady: false,
        minimumCohort: LOCAL_LEAGUE_MINIMUM_COHORT,
        entries: [],
        policyVersion: SOCIAL_ADVENTURE_POLICY_VERSION,
        message: `Local ranks appear after ${LOCAL_LEAGUE_MINIMUM_COHORT} active members so a small leaderboard cannot reveal too much about a locality.`,
      };
    }

    const candidates = await this.prisma.$queryRaw<LeaderboardUserRow[]>(Prisma.sql`
      SELECT u.id, u.handle, u.avatar_url AS "avatarUrl"
      FROM dogos_social.pack_memberships member
      JOIN public.users u ON u.id = member.user_id
      WHERE member.pack_id = ${packId}
        AND member.status = 'ACTIVE'
        AND u.visibility = 'PUBLIC'
        AND NOT EXISTS (
          SELECT 1
          FROM public.blocked_users blocked
          WHERE (blocked.user_id = ${userId} AND blocked.blocked_id = u.id)
             OR (blocked.user_id = u.id AND blocked.blocked_id = ${userId})
        )
      ORDER BY u.id ASC
      LIMIT 100
    `);
    const rows = await this.scoreLeaderboardUsers(candidates);

    return {
      scope: 'PACK' as const,
      pack: { id: pack.id, name: pack.name, memberCount: pack.memberCount },
      cohortReady: true,
      minimumCohort: LOCAL_LEAGUE_MINIMUM_COHORT,
      entries: rows.slice(0, safeLimit).map((row, index) => ({ ...row, rank: index + 1 })),
      policyVersion: SOCIAL_ADVENTURE_POLICY_VERSION,
    };
  }

  async listPacks(userId: string) {
    const rows = await this.prisma.$queryRaw<PackRow[]>(Prisma.sql`
      SELECT
        pack.id,
        pack.name,
        pack.slug,
        pack.scope,
        pack.region_key AS "regionKey",
        pack.visibility,
        COUNT(active_member.user_id)::int AS "memberCount",
        BOOL_OR(my_member.user_id IS NOT NULL AND my_member.status = 'ACTIVE') AS joined,
        MAX(CASE WHEN my_member.status = 'ACTIVE' THEN my_member.role ELSE NULL END) AS role
      FROM dogos_social.packs pack
      LEFT JOIN dogos_social.pack_memberships active_member
        ON active_member.pack_id = pack.id AND active_member.status = 'ACTIVE'
      LEFT JOIN dogos_social.pack_memberships my_member
        ON my_member.pack_id = pack.id AND my_member.user_id = ${userId}
      WHERE pack.visibility = 'PUBLIC' OR my_member.user_id IS NOT NULL
      GROUP BY pack.id, pack.name, pack.slug, pack.scope, pack.region_key, pack.visibility
      ORDER BY joined DESC, "memberCount" DESC, pack.created_at DESC
      LIMIT 50
    `);
    return {
      packs: rows,
      localMinimumCohort: LOCAL_LEAGUE_MINIMUM_COHORT,
      locationContract: 'coarse-user-chosen-region-only',
    };
  }

  async createPack(userId: string, dto: CreatePackDto) {
    const id = randomUUID();
    const slugBase = this.slugify(dto.name);
    const slug = `${slugBase}-${createHash('sha256').update(`${userId}:${id}`).digest('hex').slice(0, 7)}`;

    await this.prisma.$transaction(async (tx) => {
      await tx.$executeRaw(Prisma.sql`
        INSERT INTO dogos_social.packs
          (id, owner_user_id, name, slug, scope, region_key, visibility)
        VALUES
          (${id}, ${userId}, ${dto.name.trim()}, ${slug}, 'LOCAL', ${dto.regionKey}, 'PUBLIC')
      `);
      await tx.$executeRaw(Prisma.sql`
        INSERT INTO dogos_social.pack_memberships (pack_id, user_id, role, status)
        VALUES (${id}, ${userId}, 'OWNER', 'ACTIVE')
      `);
    });

    return { id, name: dto.name.trim(), slug, regionKey: dto.regionKey, joined: true };
  }

  async joinPack(userId: string, packId: string) {
    const rows = await this.prisma.$queryRaw<Array<{ id: string }>>(Prisma.sql`
      SELECT id FROM dogos_social.packs WHERE id = ${packId} AND visibility = 'PUBLIC' LIMIT 1
    `);
    if (!rows[0]) throw new NotFoundException('Pack not found');

    await this.prisma.$executeRaw(Prisma.sql`
      INSERT INTO dogos_social.pack_memberships (pack_id, user_id, role, status, joined_at)
      VALUES (${packId}, ${userId}, 'MEMBER', 'ACTIVE', NOW())
      ON CONFLICT (pack_id, user_id)
      DO UPDATE SET status = 'ACTIVE', joined_at = NOW()
    `);
    return { ok: true };
  }

  async leavePack(userId: string, packId: string) {
    const rows = await this.prisma.$queryRaw<Array<{ role: string }>>(Prisma.sql`
      SELECT role
      FROM dogos_social.pack_memberships
      WHERE pack_id = ${packId} AND user_id = ${userId} AND status = 'ACTIVE'
      LIMIT 1
    `);
    const membership = rows[0];
    if (!membership) return { ok: true };
    if (membership.role === 'OWNER') {
      throw new BadRequestException('A Pack owner cannot leave without transferring or retiring the Pack');
    }

    await this.prisma.$executeRaw(Prisma.sql`
      UPDATE dogos_social.pack_memberships
      SET status = 'LEFT'
      WHERE pack_id = ${packId} AND user_id = ${userId}
    `);
    return { ok: true };
  }

  async getArcade(userId: string) {
    const bestScores = await this.getBestHumanSkillScores(userId);
    return {
      challengeVersion: HUMAN_SKILL_CHALLENGE_VERSION,
      challenges: publicArcadeCatalog().map((scenario) => ({
        ...scenario,
        bestScore: bestScores[scenario.challengeKey] ?? null,
      })),
      scoring:
        'Only your best score for each Human Skill game contributes to the weekly Social Adventure score. Repetition volume adds nothing.',
    };
  }

  async startHumanSkillAttempt(userId: string, challengeKey: string) {
    if (!this.isHumanSkillChallenge(challengeKey)) {
      throw new BadRequestException('Unknown Human Skill challenge');
    }
    const scenario = scenarioForChallenge(challengeKey);
    const publicScenario = publicArcadeCatalog().find((item) => item.challengeKey === challengeKey);
    if (!scenario || !publicScenario) throw new NotFoundException('Human Skill challenge unavailable');

    const id = randomUUID();
    const issuedAt = new Date();
    const expiresAt = new Date(issuedAt.getTime() + ATTEMPT_TTL_MS);
    await this.prisma.$executeRaw(Prisma.sql`
      INSERT INTO dogos_social.human_skill_attempts
        (id, user_id, challenge_key, challenge_version, scenario_key, issued_at, expires_at)
      VALUES
        (${id}, ${userId}, ${challengeKey}, ${HUMAN_SKILL_CHALLENGE_VERSION}, ${scenario.scenarioKey}, ${issuedAt}, ${expiresAt})
    `);

    return {
      attemptId: id,
      issuedAt: issuedAt.toISOString(),
      expiresAt: expiresAt.toISOString(),
      scenario: publicScenario,
    };
  }

  async completeHumanSkillAttempt(
    userId: string,
    attemptId: string,
    dto: CompleteHumanSkillAttemptDto
  ) {
    const attempt = await this.getAttempt(userId, attemptId);
    if (attempt.completedAt) return attempt.receipt;
    if (attempt.expiresAt.getTime() < Date.now()) {
      throw new BadRequestException('This Human Skill attempt expired. Start a fresh round.');
    }

    const scenario = scenarioByKey(attempt.scenarioKey);
    if (!scenario || scenario.challengeVersion !== attempt.challengeVersion) {
      throw new BadRequestException('This challenge version is no longer available');
    }
    const scored = scoreArcadeResponse(scenario, dto.response);
    const completedAt = new Date();
    const receipt = {
      attemptId,
      challengeKey: scenario.challengeKey,
      challengeVersion: scenario.challengeVersion,
      score: scored.score,
      correct: scored.correct,
      ...(scored.timingErrorMs === undefined ? {} : { timingErrorMs: scored.timingErrorMs }),
      explanation: scored.explanation,
      completedAt: completedAt.toISOString(),
    };

    const updated = await this.prisma.$queryRaw<Array<{ receipt: unknown }>>(Prisma.sql`
      UPDATE dogos_social.human_skill_attempts
      SET
        completed_at = ${completedAt},
        response = ${JSON.stringify(dto.response)}::jsonb,
        score = ${scored.score},
        receipt = ${JSON.stringify(receipt)}::jsonb
      WHERE id = ${attemptId}
        AND user_id = ${userId}
        AND completed_at IS NULL
      RETURNING receipt
    `);

    if (updated[0]) return updated[0].receipt;
    return (await this.getAttempt(userId, attemptId)).receipt;
  }

  async createShare(userId: string, dto: CreateSocialShareDto) {
    const source = await this.resolveShareSource(userId, dto);
    const identity = `social-share:${userId}:${source.sourceType}:${source.sourceId}:${source.kind}`;

    const shareId = await this.prisma.$transaction(async (tx) => {
      await tx.$queryRaw<Array<{ acquired: number }>>(Prisma.sql`
        WITH lock_row AS MATERIALIZED (
          SELECT pg_advisory_xact_lock(hashtextextended(${identity}, 0))
        )
        SELECT 1::int AS acquired FROM lock_row
      `);

      const existing = await tx.$queryRaw<ExistingShareRow[]>(Prisma.sql`
        SELECT id, post_id AS "postId"
        FROM dogos_social.shares
        WHERE user_id = ${userId}
          AND source_type = ${source.sourceType}
          AND source_id = ${source.sourceId}
          AND kind = ${source.kind}
        LIMIT 1
      `);
      if (existing[0]) return existing[0].id;

      const post = await tx.post.create({
        data: {
          authorUserId: userId,
          petId: source.petId,
          text: dto.caption?.trim() || source.summary,
          mediaUrls: [],
          visibility: dto.visibility ?? 'PRIVATE',
        },
        select: { id: true },
      });
      const id = randomUUID();
      await tx.$executeRaw(Prisma.sql`
        INSERT INTO dogos_social.shares
          (id, post_id, user_id, pet_id, source_type, source_id, kind, headline, summary, payload)
        VALUES
          (${id}, ${post.id}, ${userId}, ${source.petId}, ${source.sourceType}, ${source.sourceId}, ${source.kind}, ${source.headline}, ${source.summary}, ${JSON.stringify(source.payload)}::jsonb)
      `);
      return id;
    });

    const created = await this.getShare(userId, shareId);
    if (!created) throw new NotFoundException('Shared moment not found');
    return created;
  }

  async getFeed(userId: string, take = 20) {
    const safeTake = Math.max(1, Math.min(Number(take) || 20, 50));
    const rows = await this.prisma.$queryRaw<FeedRow[]>(Prisma.sql`
      SELECT
        share.id AS "shareId",
        post.id AS "postId",
        share.kind,
        share.headline,
        share.summary,
        share.payload,
        post.text AS caption,
        post.visibility,
        post.created_at AS "createdAt",
        author.id AS "authorUserId",
        author.handle,
        author.avatar_url AS "avatarUrl",
        pet.id AS "petId",
        pet.name AS "petName",
        pet.avatar_url AS "petAvatarUrl",
        (SELECT COUNT(*)::int FROM public.likes liked WHERE liked.post_id = post.id) AS "likesCount",
        (SELECT COUNT(*)::int FROM public.comments comment WHERE comment.post_id = post.id) AS "commentsCount"
      FROM dogos_social.shares share
      JOIN public.posts post ON post.id = share.post_id
      JOIN public.users author ON author.id = post.author_user_id
      LEFT JOIN public.pets pet ON pet.id = share.pet_id
      WHERE (post.author_user_id = ${userId} OR post.visibility = 'PUBLIC')
        AND NOT EXISTS (
          SELECT 1
          FROM public.blocked_users blocked
          WHERE (blocked.user_id = ${userId} AND blocked.blocked_id = post.author_user_id)
             OR (blocked.user_id = post.author_user_id AND blocked.blocked_id = ${userId})
        )
      ORDER BY post.created_at DESC, share.id DESC
      LIMIT ${safeTake}
    `);

    const reactions = rows.length
      ? await this.prisma.$queryRaw<ReactionRow[]>(Prisma.sql`
          SELECT
            reaction.share_id AS "shareId",
            reaction.reaction,
            COUNT(*)::int AS count,
            BOOL_OR(reaction.user_id = ${userId}) AS mine
          FROM dogos_social.reactions reaction
          WHERE reaction.share_id IN (${Prisma.join(rows.map((row) => row.shareId))})
          GROUP BY reaction.share_id, reaction.reaction
        `)
      : [];

    return {
      posts: rows.map((row) => ({
        ...row,
        createdAt: row.createdAt.toISOString(),
        reactions: REACTION_TYPES.map((reaction) => {
          const aggregate = reactions.find(
            (item) => item.shareId === row.shareId && item.reaction === reaction
          );
          return { reaction, count: aggregate?.count ?? 0, mine: aggregate?.mine ?? false };
        }),
      })),
      privacy:
        'Only PUBLIC Social Adventure shares and your own private shares appear here. FRIENDS_ONLY legacy posts remain hidden until a modern friend-authority contract exists.',
    };
  }

  async addReaction(userId: string, shareId: string, dto: SocialReactionDto) {
    await this.assertShareViewable(userId, shareId);
    await this.prisma.$executeRaw(Prisma.sql`
      INSERT INTO dogos_social.reactions (id, share_id, user_id, reaction)
      VALUES (${randomUUID()}, ${shareId}, ${userId}, ${dto.reaction})
      ON CONFLICT (share_id, user_id, reaction) DO NOTHING
    `);
    return { ok: true };
  }

  async removeReaction(userId: string, shareId: string, reaction: string) {
    if (!REACTION_TYPES.includes(reaction as (typeof REACTION_TYPES)[number])) {
      throw new BadRequestException('Unknown Social Adventure reaction');
    }
    await this.prisma.$executeRaw(Prisma.sql`
      DELETE FROM dogos_social.reactions
      WHERE share_id = ${shareId} AND user_id = ${userId} AND reaction = ${reaction}
    `);
    return { ok: true };
  }

  private async getPreferences(userId: string) {
    const rows = await this.prisma.$queryRaw<PreferenceRow[]>(Prisma.sql`
      SELECT global_leaderboard_opt_in AS "globalLeaderboardOptIn"
      FROM dogos_social.preferences
      WHERE user_id = ${userId}
      LIMIT 1
    `);
    return rows[0] ?? { globalLeaderboardOptIn: false };
  }

  private async getAttempt(userId: string, attemptId: string) {
    const rows = await this.prisma.$queryRaw<SkillAttemptRow[]>(Prisma.sql`
      SELECT
        id,
        challenge_key AS "challengeKey",
        challenge_version AS "challengeVersion",
        scenario_key AS "scenarioKey",
        issued_at AS "issuedAt",
        expires_at AS "expiresAt",
        completed_at AS "completedAt",
        score,
        receipt
      FROM dogos_social.human_skill_attempts
      WHERE id = ${attemptId} AND user_id = ${userId}
      LIMIT 1
    `);
    if (!rows[0]) throw new NotFoundException('Human Skill attempt not found');
    return rows[0];
  }

  private async getBestHumanSkillScores(userId: string) {
    const season = currentUtcSeason();
    const rows = await this.prisma.$queryRaw<Array<{ challengeKey: string; bestScore: number }>>(
      Prisma.sql`
        SELECT challenge_key AS "challengeKey", MAX(score)::int AS "bestScore"
        FROM dogos_social.human_skill_attempts
        WHERE user_id = ${userId}
          AND completed_at >= ${season.startsAt}
          AND completed_at < ${season.endsAt}
        GROUP BY challenge_key
      `
    );
    return Object.fromEntries(rows.map((row) => [row.challengeKey, row.bestScore])) as Partial<
      Record<HumanSkillChallenge, number>
    >;
  }

  private async computeScore(userId: string) {
    const season = currentUtcSeason();
    const [adventureRows, skillRows] = await Promise.all([
      this.prisma.$queryRaw<AdventureEvidenceRow[]>(Prisma.sql`
        SELECT id, pathway
        FROM public.care_events
        WHERE user_id = ${userId}
          AND source = 'QUEST_ENGINE'
          AND event_type LIKE 'QUEST_%'
          AND occurred_at >= ${season.startsAt}
          AND occurred_at < ${season.endsAt}
        ORDER BY id ASC
      `),
      this.prisma.$queryRaw<SkillScoreRow[]>(Prisma.sql`
        SELECT id, challenge_key AS "challengeKey", score
        FROM dogos_social.human_skill_attempts
        WHERE user_id = ${userId}
          AND completed_at >= ${season.startsAt}
          AND completed_at < ${season.endsAt}
          AND score IS NOT NULL
        ORDER BY id ASC
      `),
    ]);

    const bestScores: Partial<Record<HumanSkillChallenge, number>> = {};
    for (const row of skillRows) {
      if (!this.isHumanSkillChallenge(row.challengeKey)) continue;
      bestScores[row.challengeKey] = Math.max(bestScores[row.challengeKey] ?? 0, row.score);
    }

    const score = deriveSocialAdventureScore({
      adventurePathways: adventureRows.map((row) => row.pathway),
      humanSkillBestScores: bestScores,
    });
    const sourceHash = createHash('sha256')
      .update(
        JSON.stringify({
          adventure: adventureRows.map((row) => [row.id, row.pathway]),
          humanSkill: skillRows.map((row) => [row.id, row.challengeKey, row.score]),
        })
      )
      .digest('hex');
    const receiptId = createHash('sha256')
      .update(`${userId}:${season.key}:${SOCIAL_ADVENTURE_POLICY_VERSION}:${sourceHash}`)
      .digest('hex')
      .slice(0, 40);

    await this.prisma.$executeRaw(Prisma.sql`
      INSERT INTO dogos_social.competition_receipts
        (id, user_id, season_key, policy_version, score, components, source_hash)
      VALUES
        (${receiptId}, ${userId}, ${season.key}, ${SOCIAL_ADVENTURE_POLICY_VERSION}, ${score.score}, ${JSON.stringify(score.components)}::jsonb, ${sourceHash})
      ON CONFLICT (user_id, season_key, policy_version, source_hash) DO NOTHING
    `);

    return {
      ...score,
      season: {
        key: season.key,
        startsAt: season.startsAt.toISOString(),
        endsAt: season.endsAt.toISOString(),
      },
    };
  }

  private async scoreLeaderboardUsers(users: LeaderboardUserRow[]) {
    const rows = await Promise.all(
      users.map(async (user) => {
        const score = await this.computeScore(user.id);
        return {
          userId: user.id,
          handle: user.handle,
          avatarUrl: user.avatarUrl,
          score: score.score,
          maxScore: score.maxScore,
          components: score.components,
        };
      })
    );
    return rows.sort((a, b) => b.score - a.score || a.handle.localeCompare(b.handle));
  }

  private async resolveShareSource(userId: string, dto: CreateSocialShareDto): Promise<ShareSource> {
    if (dto.sourceType === 'CARE_EVENT') {
      const event = await this.prisma.careEvent.findFirst({
        where: { id: dto.sourceId, userId, source: 'QUEST_ENGINE' },
        include: { pet: { select: { id: true, name: true } } },
      });
      if (!event) throw new NotFoundException('Adventure moment not found');

      const context = this.asRecord(event.context);
      const outcome = this.asRecord(event.outcome);
      const safeOptOut = outcome.safeOptOut === true;
      const dogExperience = typeof outcome.dogExperience === 'string' ? outcome.dogExperience : null;
      const questTitle =
        typeof context.questTitle === 'string' && context.questTitle.trim()
          ? context.questTitle.trim().slice(0, 100)
          : 'Shared Adventure';
      const petName = event.pet?.name ?? 'your dog';
      const kind = safeOptOut
        ? 'GOOD_READ'
        : dogExperience === 'not_their_thing'
          ? 'DISCOVERY'
          : 'ADVENTURE_MEMORY';
      const summary = safeOptOut
        ? `We listened to ${petName} and changed course. Stopping appropriately counted as a good read.`
        : kind === 'DISCOVERY'
          ? `We learned something useful about what fits ${petName}. Discovery counts even when an activity is not a favorite.`
          : `A ${event.pathway.toLowerCase()} moment with ${petName}, saved from a real Adventure outcome.`;

      return {
        sourceType: 'CARE_EVENT',
        sourceId: event.id,
        petId: event.petId,
        kind,
        headline: questTitle,
        summary,
        payload: {
          pathway: event.pathway,
          eventType: event.eventType,
          safeOptOut,
          dogExperience,
          petName,
        },
      };
    }

    const rows = await this.prisma.$queryRaw<SkillAttemptRow[]>(Prisma.sql`
      SELECT
        id,
        challenge_key AS "challengeKey",
        challenge_version AS "challengeVersion",
        scenario_key AS "scenarioKey",
        issued_at AS "issuedAt",
        expires_at AS "expiresAt",
        completed_at AS "completedAt",
        score,
        receipt
      FROM dogos_social.human_skill_attempts
      WHERE id = ${dto.sourceId} AND user_id = ${userId} AND completed_at IS NOT NULL
      LIMIT 1
    `);
    const attempt = rows[0];
    if (!attempt || attempt.score === null) throw new NotFoundException('Human Skill result not found');
    const scenario = scenarioByKey(attempt.scenarioKey);
    const title = scenario?.title ?? 'Human Skill Arcade';

    return {
      sourceType: 'HUMAN_SKILL_ATTEMPT',
      sourceId: attempt.id,
      petId: null,
      kind: 'SKILL_MOMENT',
      headline: title,
      summary: `Practiced ${this.humanize(attempt.challengeKey)} and scored ${attempt.score}/100. The game measures human learning, not pet performance.`,
      payload: {
        challengeKey: attempt.challengeKey,
        challengeVersion: attempt.challengeVersion,
        score: attempt.score,
      },
    };
  }

  private async getShare(userId: string, shareId: string) {
    const rows = await this.prisma.$queryRaw<FeedRow[]>(Prisma.sql`
      SELECT
        share.id AS "shareId",
        post.id AS "postId",
        share.kind,
        share.headline,
        share.summary,
        share.payload,
        post.text AS caption,
        post.visibility,
        post.created_at AS "createdAt",
        author.id AS "authorUserId",
        author.handle,
        author.avatar_url AS "avatarUrl",
        pet.id AS "petId",
        pet.name AS "petName",
        pet.avatar_url AS "petAvatarUrl",
        (SELECT COUNT(*)::int FROM public.likes liked WHERE liked.post_id = post.id) AS "likesCount",
        (SELECT COUNT(*)::int FROM public.comments comment WHERE comment.post_id = post.id) AS "commentsCount"
      FROM dogos_social.shares share
      JOIN public.posts post ON post.id = share.post_id
      JOIN public.users author ON author.id = post.author_user_id
      LEFT JOIN public.pets pet ON pet.id = share.pet_id
      WHERE share.id = ${shareId}
        AND (post.author_user_id = ${userId} OR post.visibility = 'PUBLIC')
        AND NOT EXISTS (
          SELECT 1
          FROM public.blocked_users blocked
          WHERE (blocked.user_id = ${userId} AND blocked.blocked_id = post.author_user_id)
             OR (blocked.user_id = post.author_user_id AND blocked.blocked_id = ${userId})
        )
      LIMIT 1
    `);
    const row = rows[0];
    return row ? { ...row, createdAt: row.createdAt.toISOString() } : null;
  }

  private async assertShareViewable(userId: string, shareId: string) {
    const share = await this.getShare(userId, shareId);
    if (!share) throw new NotFoundException('Shared moment not found');
    return share;
  }

  private isHumanSkillChallenge(value: string): value is HumanSkillChallenge {
    return HUMAN_SKILL_CHALLENGES.includes(value as HumanSkillChallenge);
  }

  private asRecord(value: Prisma.JsonValue | null): Record<string, unknown> {
    return value && typeof value === 'object' && !Array.isArray(value)
      ? (value as Record<string, unknown>)
      : {};
  }

  private slugify(value: string) {
    const normalized = value
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '')
      .slice(0, 56);
    return normalized || 'pack';
  }

  private humanize(value: string) {
    return value
      .toLowerCase()
      .split('_')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  }
}
