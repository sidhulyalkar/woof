export const SOCIAL_ADVENTURE_POLICY_VERSION = 'social-adventure-score-v1';
export const HUMAN_SKILL_CHALLENGE_VERSION = 'human-skill-arcade-v1';
export const LOCAL_LEAGUE_MINIMUM_COHORT = 5;

export const SOCIAL_ADVENTURE_PATHWAYS = [
  'MOVE',
  'EXPLORE',
  'ENRICH',
  'LEARN',
  'CONNECT',
  'RECOVER',
  'BOND',
] as const;

export type SocialAdventurePathway = (typeof SOCIAL_ADVENTURE_PATHWAYS)[number];

export const HUMAN_SKILL_CHALLENGES = [
  'MAKE_IT_EASIER',
  'CATCH_THE_GOOD',
  'PAIRING_LAB',
  'MARKER_TIMING',
] as const;

export type HumanSkillChallenge = (typeof HUMAN_SKILL_CHALLENGES)[number];

export type SocialAdventureScoreInput = {
  adventurePathways: string[];
  humanSkillBestScores: Partial<Record<HumanSkillChallenge, number>>;
};

export type SocialAdventureScore = {
  score: number;
  maxScore: number;
  policyVersion: string;
  components: {
    humanSkill: {
      score: number;
      maxScore: number;
      bestByChallenge: Partial<Record<HumanSkillChallenge, number>>;
    };
    adventureVariety: {
      score: number;
      maxScore: number;
      pathways: SocialAdventurePathway[];
    };
  };
};

const PATHWAY_POINTS = 25;
const HUMAN_SKILL_MAX = 400;
const ADVENTURE_VARIETY_MAX = SOCIAL_ADVENTURE_PATHWAYS.length * PATHWAY_POINTS;
const MAX_SCORE = HUMAN_SKILL_MAX + ADVENTURE_VARIETY_MAX;

export function deriveSocialAdventureScore(input: SocialAdventureScoreInput): SocialAdventureScore {
  const allowedPathways = new Set<string>(SOCIAL_ADVENTURE_PATHWAYS);
  const distinctPathways = [...new Set(input.adventurePathways)]
    .filter((pathway): pathway is SocialAdventurePathway => allowedPathways.has(pathway))
    .sort((a, b) => a.localeCompare(b));

  const bestByChallenge: Partial<Record<HumanSkillChallenge, number>> = {};
  let humanSkillScore = 0;

  for (const challenge of HUMAN_SKILL_CHALLENGES) {
    const raw = input.humanSkillBestScores[challenge];
    if (raw === undefined || !Number.isFinite(raw)) continue;
    const bounded = Math.max(0, Math.min(100, Math.round(raw)));
    bestByChallenge[challenge] = bounded;
    humanSkillScore += bounded;
  }

  const adventureVarietyScore = distinctPathways.length * PATHWAY_POINTS;

  return {
    score: humanSkillScore + adventureVarietyScore,
    maxScore: MAX_SCORE,
    policyVersion: SOCIAL_ADVENTURE_POLICY_VERSION,
    components: {
      humanSkill: {
        score: humanSkillScore,
        maxScore: HUMAN_SKILL_MAX,
        bestByChallenge,
      },
      adventureVariety: {
        score: adventureVarietyScore,
        maxScore: ADVENTURE_VARIETY_MAX,
        pathways: distinctPathways,
      },
    },
  };
}

export function currentUtcSeason(now = new Date()) {
  const start = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()));
  const weekday = start.getUTCDay();
  const daysSinceMonday = (weekday + 6) % 7;
  start.setUTCDate(start.getUTCDate() - daysSinceMonday);
  const end = new Date(start.getTime() + 7 * 24 * 60 * 60 * 1000);

  return {
    key: `week:${start.toISOString().slice(0, 10)}`,
    startsAt: start,
    endsAt: end,
  };
}
