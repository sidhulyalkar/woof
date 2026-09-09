export const EXPEDITION_POLICY_VERSION = 'expedition-authority-v1';
export const EXPEDITION_KEY = 'shared-world';
export const EXPEDITION_VERSION = 'v1';

export const EXPEDITION_ELIGIBLE_PATHWAYS = ['EXPLORE', 'ENRICH', 'RECOVER'] as const;
export type ExpeditionEligiblePathway = (typeof EXPEDITION_ELIGIBLE_PATHWAYS)[number];

export const EXPEDITION_HUMAN_SKILL_CATEGORIES = [
  'MAKE_IT_EASIER',
  'CATCH_THE_GOOD',
  'PAIRING_LAB',
  'MARKER_TIMING',
] as const;

export const EXPEDITION_OBJECTIVES = [
  {
    key: 'SNIFF_EXPLORE',
    title: 'Sniff & Explore',
    description:
      'Collect varied exploration and enrichment moments without rewarding distance, duration, or repetition volume.',
    sourceType: 'CARE_EVENT',
    categories: ['EXPLORE', 'ENRICH'],
    perContributorCap: 4,
    perCategoryCap: 2,
    legacyChallengeId: 'sniff-explore-week',
    legacyTitle: 'Sniff & Explore Week',
    legacyTarget: 250,
    legacyUnit: 'shared adventures',
  },
  {
    key: 'RECOVERY_COUNTS',
    title: 'Recovery Counts',
    description:
      'Make decompression and recovery visible as legitimate participation without streak pressure.',
    sourceType: 'CARE_EVENT',
    categories: ['RECOVER'],
    perContributorCap: 2,
    perCategoryCap: 2,
    legacyChallengeId: 'recovery-counts-week',
    legacyTitle: 'Recovery Counts',
    legacyTarget: 100,
    legacyUnit: 'recovery moments',
  },
  {
    key: 'READ_THE_ROOM',
    title: 'Read the Room',
    description:
      'Practice distinct Human Skill rooms. Completion breadth counts once per room; score magnitude does not.',
    sourceType: 'HUMAN_SKILL_ATTEMPT',
    categories: [...EXPEDITION_HUMAN_SKILL_CATEGORIES],
    perContributorCap: 4,
    perCategoryCap: 1,
    legacyChallengeId: null,
    legacyTitle: null,
    legacyTarget: null,
    legacyUnit: null,
  },
] as const;

export type ExpeditionObjective = (typeof EXPEDITION_OBJECTIVES)[number];
export type ExpeditionObjectiveKey = ExpeditionObjective['key'];
export type ExpeditionScope = 'GLOBAL' | 'PACK';

export function currentExpeditionSeason(now = new Date()) {
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
