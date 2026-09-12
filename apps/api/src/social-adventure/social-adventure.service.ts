import { BadRequestException, ConflictException, Injectable, NotFoundException } from '@nestjs/common';
import { Prisma } from '@woof/database';
import { createHash, randomUUID } from 'crypto';
import { PrismaService } from '../prisma/prisma.service';
import type {
  CompleteHumanSkillAttemptDto,
  CreatePackDto,
  CreateSocialShareDto,
  SocialReactionDto,
} from './dto/social-adventure.dto';
import { PackAccessService } from './pack-access.service';
import {
  ADVENTURE_VARIETY_POINTS,
  HUMAN_SKILL_BREADTH_POINTS,
  HUMAN_SKILL_CHALLENGE_VERSION,
  HUMAN_SKILL_CHALLENGES,
  HUMAN_SKILL_SCENARIOS,
  LOCAL_LEAGUE_MINIMUM_COHORT,
  SOCIAL_ADVENTURE_PATHWAYS,
  SOCIAL_ADVENTURE_POLICY_VERSION,
  deriveSocialAdventureScore,
  getCurrentSocialSeason,
  scoreHumanSkillAttempt,
  type HumanSkillAttemptResponse,
  type HumanSkillChallengeKey,
} from './social-adventure.policy';

type PreferenceRow = { globalLeaderboardOptIn: boolean };
type LeaderboardUserRow = { id: string; handle: string; avatarUrl: string | null };
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
type HumanSkillRow = {
  id: string;
  challengeKey: string;
  challengeVersion: string;
  score: number;
};
type HumanSkillBestScoreRow = { challengeKey: string; score: number };
type HumanSkillAttemptRow = {
  id: string;
  userId: string;
  challengeKey: string;
  challengeVersion: string;
  scenarioKey: string;
  issuedAt: Date;
  expiresAt: Date;
  completedAt: Date | null;
};
type ShareRow = {
  id: string;
  postId: string;
  sourceType: string;
  sourceId: string;
  caption: string | null;
  visibility: string;
  createdAt: Date;
};

type CareShareSourceRow = {
  id: string;
  petId: string;
  createdBy: string;
  eventType: string;
  title: string;
  note: string | null;
  occurredAt: Date;
  petName: string;
};

type SkillShareSourceRow = {
  id: string;
  userId: string;
  challengeKey: string;
  challengeVersion: string;
  score: number;
  receipt: Prisma.JsonValue;
  completedAt: Date;
};

type FeedRow = {
  shareId: string;
  postId: string;
  kind: string;
  headline: string;
  summary: string;
  payload: Prisma.JsonValue;
  caption: string | null;
  visibility: string;
  createdAt: Date;
  authorUserId: string;
  handle: string;
  avatarUrl: string | null;
  petName: string | null;
};
type ReactionRow = { shareId: string; reaction: string; count: number; mine: boolean };
type PathwayRow = { pathway: string };

@Injectable()
export class SocialAdventureService {
  constructor(
    private readonly prisma: PrismaService,
    private readonly packAccess: PackAccessService
  ) {}

  private async getPreferences(userId: string): Promise<PreferenceRow> {
    const rows = await this.prisma.$queryRaw<PreferenceRow[]>(Prisma.sql`
      SELECT global_leaderboard_opt_in AS "globalLeaderboardOptIn"
      FROM dogos_social.user_preferences
      WHERE user_id = ${userId}
      LIMIT 1
    `);
    return rows[0] ?? { globalLeaderboardOptIn: false };
  }

  async getMine(userId: string) {
    const [preferences, score, humanSkillBestScores] = await Promise.all([
      this.getPreferences(userId),
      this.computeScore(userId),
      this.getHumanSkillBestScores(userId),
    ]);
    return {
      preferences,
      ...score,
      humanSkillBestScores,
      policyVersion: SOCIAL_ADVENTURE_POLICY_VERSION,
      principles: [
        'You compete. Your dog does not.',
        'Health, symptoms, exercise volume, pet performance, and missed days never add league points.',
        'Reactions build culture, not rank.',
      ],
    };
  }

  async updatePreferences(userId: string, dto: { globalLeaderboardOptIn: boolean }) {
    const rows = await this.prisma.$queryRaw<PreferenceRow[]>(Prisma.sql`
      INSERT INTO dogos_social.user_preferences (user_id, global_leaderboard_opt_in, updated_at)
      VALUES (${userId}, ${dto.globalLeaderboardOptIn}, NOW())
      ON CONFLICT (user_id)
      DO UPDATE SET global_leaderboard_opt_in = EXCLUDED.global_leaderboard_opt_in, updated_at = NOW()
      RETURNING global_leaderboard_opt_in AS "globalLeaderboardOptIn"
    `);
    return rows[0] ?? { globalLeaderboardOptIn: dto.globalLeaderboardOptIn };
  }

  async getGlobalLeaderboard(userId: string, limit = 30) {
    const safeLimit = Math.max(1, Math.min(Number(limit) || 30, 50));
    const candidates = await this.prisma.$queryRaw<LeaderboardUserRow[]>(Prisma.sql`
      SELECT u.id, u.handle, u.avatar_url AS "avatarUrl"
      FROM dogos_social.user_preferences pref
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
        'This league scores human learning breadth and bounded Adventure variety. It does not rank pet obedience, health, exercise volume, symptoms, mileage, likes, streaks, or Arcade practice-score magnitude.',
    };
  }

  async getPackLeaderboard(userId: string, packId: string, limit = 30) {
    const safeLimit = Math.max(1, Math.min(Number(limit) || 30, 50));
    const pack = await this.packAccess.requireViewable(userId, packId);

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
      locationContract: 'server-approved-coarse-region-only',
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

    return {
      id,
      name: dto.name.trim(),
      slug,
      scope: 'LOCAL',
      regionKey: dto.regionKey,
      visibility: 'PUBLIC',
      memberCount: 1,
      joined: true,
      role: 'OWNER',
    };
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
      throw new ConflictException('Transfer or retire this Pack before leaving');
    }

    await this.prisma.$executeRaw(Prisma.sql`
      UPDATE dogos_social.pack_memberships
      SET status = 'LEFT', left_at = NOW()
      WHERE pack_id = ${packId} AND user_id = ${userId}
    `);
    return { ok: true };
  }

  async getArcade(userId: string) {
    const bestScores = await this.getHumanSkillBestScores(userId);
    const challenges = HUMAN_SKILL_CHALLENGES.map((challengeKey) => {
      const scenario = HUMAN_SKILL_SCENARIOS[challengeKey][0];
      return {
        challengeKey,
        challengeVersion: HUMAN_SKILL_CHALLENGE_VERSION,
        scenarioKey: scenario.scenarioKey,
        title: scenario.title,
        skill: scenario.skill,
        prompt: scenario.prompt,
        options: 'options' in scenario ? scenario.options : undefined,
        timing: 'timing' in scenario ? scenario.timing : undefined,
        bestScore: bestScores[challengeKey] ?? null,
      };
    });
    return {
      challengeVersion: HUMAN_SKILL_CHALLENGE_VERSION,
      challenges,
      scoring:
        'Practice scores are private feedback. Social Adventure league credit comes from completing distinct Human Skill rooms, not from score magnitude or browser timing.',
    };
  }

  async startHumanSkillAttempt(userId: string, challengeKeyRaw: string) {
    const challengeKey = challengeKeyRaw as HumanSkillChallengeKey;
    if (!HUMAN_SKILL_CHALLENGES.includes(challengeKey)) {
      throw new BadRequestException('Unknown Human Skill challenge');
    }
    const scenarios = HUMAN_SKILL_SCENARIOS[challengeKey];
    const scenarioIndex = this.deterministicScenarioIndex(userId, challengeKey, scenarios.length);
    const scenario = scenarios[scenarioIndex];
    const id = randomUUID();
    const issuedAt = new Date();
    const expiresAt = new Date(issuedAt.getTime() + 10 * 60 * 1000);

    await this.prisma.$executeRaw(Prisma.sql`
      INSERT INTO dogos_social.human_skill_attempts (
        id,
        user_id,
        challenge_key,
        challenge_version,
        scenario_key,
        issued_at,
        expires_at
      ) VALUES (
        ${id},
        ${userId},
        ${challengeKey},
        ${HUMAN_SKILL_CHALLENGE_VERSION},
        ${scenario.scenarioKey},
        ${issuedAt},
        ${expiresAt}
      )
    `);

    return {
      attemptId: id,
      issuedAt: issuedAt.toISOString(),
      expiresAt: expiresAt.toISOString(),
      scenario: {
        challengeKey,
        challengeVersion: HUMAN_SKILL_CHALLENGE_VERSION,
        scenarioKey: scenario.scenarioKey,
        title: scenario.title,
        skill: scenario.skill,
        prompt: scenario.prompt,
        options: 'options' in scenario ? scenario.options : undefined,
        timing: 'timing' in scenario ? scenario.timing : undefined,
      },
    };
  }

  async completeHumanSkillAttempt(
    userId: string,
    attemptId: string,
    dto: CompleteHumanSkillAttemptDto
  ) {
    const rows = await this.prisma.$queryRaw<HumanSkillAttemptRow[]>(Prisma.sql`
      SELECT
        id,
        user_id AS "userId",
        challenge_key AS "challengeKey",
        challenge_version AS "challengeVersion",
        scenario_key AS "scenarioKey",
        issued_at AS "issuedAt",
        expires_at AS "expiresAt",
        completed_at AS "completedAt"
      FROM dogos_social.human_skill_attempts
      WHERE id = ${attemptId} AND user_id = ${userId}
      LIMIT 1
    `);
    const attempt = rows[0];
    if (!attempt) throw new NotFoundException('Human Skill attempt not found');
    if (attempt.completedAt) throw new ConflictException('Human Skill attempt already completed');
    if (attempt.expiresAt.getTime() <= Date.now()) {
      throw new ConflictException('Human Skill attempt expired');
    }
    if (attempt.challengeVersion !== HUMAN_SKILL_CHALLENGE_VERSION) {
      throw new ConflictException('Human Skill attempt version is no longer supported');
    }

    const challengeKey = attempt.challengeKey as HumanSkillChallengeKey;
    const scenario = HUMAN_SKILL_SCENARIOS[challengeKey].find(
      (candidate) => candidate.scenarioKey === attempt.scenarioKey
    );
    if (!scenario) throw new ConflictException('Human Skill scenario is unavailable');

    const result = scoreHumanSkillAttempt(
      challengeKey,
      attempt.scenarioKey,
      dto.response as HumanSkillAttemptResponse
    );
    const completedAt = new Date();
    const receipt = {
      attemptId,
      challengeKey,
      challengeVersion: attempt.challengeVersion,
      score: result.score,
      correct: result.correct,
      timingErrorMs: 'timingErrorMs' in result ? result.timingErrorMs : undefined,
      explanation: result.explanation,
      completedAt: completedAt.toISOString(),
    };

    await this.prisma.$executeRaw(Prisma.sql`
      UPDATE dogos_social.human_skill_attempts
      SET completed_at = ${completedAt},
          response = ${JSON.stringify(dto.response)}::jsonb,
          score = ${result.score},
          receipt = ${JSON.stringify(receipt)}::jsonb
      WHERE id = ${attemptId} AND completed_at IS NULL
    `);
    return receipt;
  }

  async getFeed(userId: string, take = 30) {
    const safeTake = Math.max(1, Math.min(Number(take) || 30, 50));
    const rows = await this.prisma.$queryRaw<FeedRow[]>(Prisma.sql`
      SELECT
        share.id AS "shareId",
        share.post_id AS "postId",
        post.kind,
        post.headline,
        post.summary,
        post.payload,
        share.caption,
        share.visibility,
        share.created_at AS "createdAt",
        share.user_id AS "authorUserId",
        author.handle,
        author.avatar_url AS "avatarUrl",
        pet.name AS "petName"
      FROM dogos_social.shares share
      JOIN public.posts post ON post.id = share.post_id
      JOIN public.users author ON author.id = share.user_id
      LEFT JOIN public.pets pet ON pet.id = post.pet_id
      WHERE share.visibility = 'PUBLIC'
        AND NOT EXISTS (
          SELECT 1
          FROM public.blocked_users blocked
          WHERE (blocked.user_id = ${userId} AND blocked.blocked_id = share.user_id)
             OR (blocked.user_id = share.user_id AND blocked.blocked_id = ${userId})
        )
      ORDER BY share.created_at DESC
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
        reactions: reactions.filter((reaction) => reaction.shareId === row.shareId),
      })),
      privacy: 'public-opt-in-feed-only',
    };
  }

  async createShare(userId: string, dto: CreateSocialShareDto) {
    if ((dto.visibility ?? 'PRIVATE') !== 'PUBLIC') {
      throw new BadRequestException('Social Adventure sharing is explicit public opt-in only');
    }

    const source = await this.resolveShareSource(userId, dto);
    const existing = await this.prisma.$queryRaw<ShareRow[]>(Prisma.sql`
      SELECT
        id,
        post_id AS "postId",
        source_type AS "sourceType",
        source_id AS "sourceId",
        caption,
        visibility,
        created_at AS "createdAt"
      FROM dogos_social.shares
      WHERE user_id = ${userId}
        AND source_type = ${dto.sourceType}
        AND source_id = ${dto.sourceId}
      LIMIT 1
    `);
    if (existing[0]) {
      return {
        shareId: existing[0].id,
        postId: existing[0].postId,
        headline: source.headline,
        summary: source.summary,
        visibility: existing[0].visibility,
      };
    }

    const shareId = randomUUID();
    const postId = randomUUID();
    const caption = dto.caption?.trim() || null;

    await this.prisma.$transaction(async (tx) => {
      await tx.$executeRaw(Prisma.sql`
        INSERT INTO public.posts (
          id,
          user_id,
          pet_id,
          kind,
          headline,
          summary,
          payload,
          visibility,
          source_type,
          source_id,
          occurred_at,
          created_at
        ) VALUES (
          ${postId},
          ${userId},
          ${source.petId},
          ${source.kind},
          ${source.headline},
          ${source.summary},
          ${JSON.stringify(source.payload)}::jsonb,
          'PUBLIC',
          ${dto.sourceType},
          ${dto.sourceId},
          ${source.occurredAt},
          NOW()
        )
      `);
      await tx.$executeRaw(Prisma.sql`
        INSERT INTO dogos_social.shares (
          id,
          user_id,
          post_id,
          source_type,
          source_id,
          caption,
          visibility,
          created_at
        ) VALUES (
          ${shareId},
          ${userId},
          ${postId},
          ${dto.sourceType},
          ${dto.sourceId},
          ${caption},
          'PUBLIC',
          NOW()
        )
      `);
    });

    return { shareId, postId, headline: source.headline, summary: source.summary, visibility: 'PUBLIC' };
  }

  async addReaction(userId: string, shareId: string, dto: SocialReactionDto) {
    await this.requireVisibleShare(userId, shareId);
    await this.prisma.$executeRaw(Prisma.sql`
      INSERT INTO dogos_social.reactions (share_id, user_id, reaction, created_at)
      VALUES (${shareId}, ${userId}, ${dto.reaction}, NOW())
      ON CONFLICT (share_id, user_id, reaction) DO NOTHING
    `);
    return { ok: true };
  }

  async removeReaction(userId: string, shareId: string, reaction: string) {
    await this.requireVisibleShare(userId, shareId);
    await this.prisma.$executeRaw(Prisma.sql`
      DELETE FROM dogos_social.reactions
      WHERE share_id = ${shareId} AND user_id = ${userId} AND reaction = ${reaction}
    `);
    return { ok: true };
  }

  private async requireVisibleShare(userId: string, shareId: string) {
    const rows = await this.prisma.$queryRaw<Array<{ id: string }>>(Prisma.sql`
      SELECT share.id
      FROM dogos_social.shares share
      WHERE share.id = ${shareId}
        AND share.visibility = 'PUBLIC'
        AND NOT EXISTS (
          SELECT 1
          FROM public.blocked_users blocked
          WHERE (blocked.user_id = ${userId} AND blocked.blocked_id = share.user_id)
             OR (blocked.user_id = share.user_id AND blocked.blocked_id = ${userId})
        )
      LIMIT 1
    `);
    if (!rows[0]) throw new NotFoundException('Share not found');
  }

  private async resolveShareSource(userId: string, dto: CreateSocialShareDto) {
    if (dto.sourceType === 'HUMAN_SKILL_ATTEMPT') {
      const rows = await this.prisma.$queryRaw<SkillShareSourceRow[]>(Prisma.sql`
        SELECT
          id,
          user_id AS "userId",
          challenge_key AS "challengeKey",
          challenge_version AS "challengeVersion",
          score,
          receipt,
          completed_at AS "completedAt"
        FROM dogos_social.human_skill_attempts
        WHERE id = ${dto.sourceId}
          AND user_id = ${userId}
          AND completed_at IS NOT NULL
        LIMIT 1
      `);
      const attempt = rows[0];
      if (!attempt) throw new NotFoundException('Human Skill attempt not found');
      const challengeTitle = HUMAN_SKILL_CHALLENGES.includes(
        attempt.challengeKey as HumanSkillChallengeKey
      )
        ? HUMAN_SKILL_SCENARIOS[attempt.challengeKey as HumanSkillChallengeKey][0].title
        : 'Human Skill';
      return {
        petId: null,
        kind: 'HUMAN_SKILL',
        headline: `Practiced ${challengeTitle}`,
        summary: 'Worked on the human side of the relationship.',
        occurredAt: attempt.completedAt,
        payload: {
          sourceType: 'HUMAN_SKILL_ATTEMPT',
          challengeKey: attempt.challengeKey,
          challengeVersion: attempt.challengeVersion,
        },
      };
    }

    const rows = await this.prisma.$queryRaw<CareShareSourceRow[]>(Prisma.sql`
      SELECT
        event.id,
        event.pet_id AS "petId",
        event.created_by AS "createdBy",
        event.event_type AS "eventType",
        event.title,
        event.note,
        event.occurred_at AS "occurredAt",
        pet.name AS "petName"
      FROM public.care_events event
      JOIN public.pets pet ON pet.id = event.pet_id
      JOIN dogos_household.pet_households ph ON ph.pet_id = pet.id
      JOIN dogos_household.household_membership hm ON hm.household_id = ph.household_id
      WHERE event.id = ${dto.sourceId}
        AND hm.user_id = ${userId}
        AND hm.status = 'ACTIVE'
      LIMIT 1
    `);
    const event = rows[0];
    if (!event) throw new NotFoundException('CARE event not found');
    return {
      petId: event.petId,
      kind: 'CARE',
      headline: event.title || `A ${event.eventType.toLowerCase()} moment`,
      summary: event.note?.trim() || `Shared a ${event.eventType.toLowerCase()} moment with ${event.petName}.`,
      occurredAt: event.occurredAt,
      payload: { sourceType: 'CARE_EVENT', eventType: event.eventType },
    };
  }

  private async scoreLeaderboardUsers(users: LeaderboardUserRow[]) {
    const scored = await Promise.all(
      users.map(async (user) => ({
        userId: user.id,
        handle: user.handle,
        avatarUrl: user.avatarUrl,
        ...(await this.computeScore(user.id)),
      }))
    );
    return scored.sort((a, b) => b.score - a.score || a.handle.localeCompare(b.handle));
  }

  private async computeScore(userId: string) {
    const season = getCurrentSocialSeason();
    const [skillRows, pathwayRows] = await Promise.all([
      this.prisma.$queryRaw<HumanSkillRow[]>(Prisma.sql`
        SELECT DISTINCT ON (challenge_key)
          id,
          challenge_key AS "challengeKey",
          challenge_version AS "challengeVersion",
          score
        FROM dogos_social.human_skill_attempts
        WHERE user_id = ${userId}
          AND completed_at IS NOT NULL
          AND completed_at >= ${season.startsAt}
          AND completed_at < ${season.endsAt}
          AND challenge_version = ${HUMAN_SKILL_CHALLENGE_VERSION}
        ORDER BY challenge_key, completed_at DESC, id DESC
      `),
      this.prisma.$queryRaw<PathwayRow[]>(Prisma.sql`
        SELECT DISTINCT ON (pathway)
          pathway
        FROM public.pet_adventure_outcomes outcome
        WHERE outcome.user_id = ${userId}
          AND outcome.completed_at IS NOT NULL
          AND outcome.completed_at >= ${season.startsAt}
          AND outcome.completed_at < ${season.endsAt}
          AND outcome.pathway IN (${Prisma.join([...SOCIAL_ADVENTURE_PATHWAYS])})
        ORDER BY pathway, completed_at DESC, id DESC
      `),
    ]);

    const sourceIdentity = {
      policyVersion: SOCIAL_ADVENTURE_POLICY_VERSION,
      seasonKey: season.key,
      humanSkill: skillRows.map((row) => [row.id, row.challengeKey]),
      adventureVariety: pathwayRows.map((row) => row.pathway),
    };
    const sourceHash = createHash('sha256')
      .update(JSON.stringify(sourceIdentity))
      .digest('hex');
    const components = deriveSocialAdventureScore({
      completedHumanSkills: skillRows.map((row) => row.challengeKey),
      completedAdventurePathways: pathwayRows.map((row) => row.pathway),
    });
    const score = components.humanSkill.score + components.adventureVariety.score;

    await this.prisma.$executeRaw(Prisma.sql`
      INSERT INTO dogos_social.competition_receipts (
        id,
        user_id,
        season_key,
        policy_version,
        score,
        components,
        source_hash,
        calculated_at
      ) VALUES (
        ${randomUUID()},
        ${userId},
        ${season.key},
        ${SOCIAL_ADVENTURE_POLICY_VERSION},
        ${score},
        ${JSON.stringify(components)}::jsonb,
        ${sourceHash},
        NOW()
      )
      ON CONFLICT (user_id, season_key, policy_version, source_hash) DO NOTHING
    `);

    return {
      season,
      score,
      maxScore: HUMAN_SKILL_BREADTH_POINTS + ADVENTURE_VARIETY_POINTS,
      components,
    };
  }

  private async getHumanSkillBestScores(userId: string) {
    const rows = await this.prisma.$queryRaw<HumanSkillBestScoreRow[]>(Prisma.sql`
      SELECT challenge_key AS "challengeKey", MAX(score)::int AS score
      FROM dogos_social.human_skill_attempts
      WHERE user_id = ${userId}
        AND completed_at IS NOT NULL
        AND challenge_version = ${HUMAN_SKILL_CHALLENGE_VERSION}
      GROUP BY challenge_key
    `);
    return Object.fromEntries(rows.map((row) => [row.challengeKey, row.score]));
  }

  private deterministicScenarioIndex(userId: string, challengeKey: string, scenarioCount: number) {
    const hash = createHash('sha256').update(`${userId}:${challengeKey}`).digest();
    return hash.readUInt32BE(0) % scenarioCount;
  }

  private slugify(value: string) {
    const slug = value
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '')
      .slice(0, 48);
    return slug || 'pack';
  }
}
