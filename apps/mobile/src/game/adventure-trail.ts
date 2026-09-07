import type { AdventureDashboard, WellbeingPathway } from '../api/adventure';

export const ADVENTURE_TRAIL_POLICY_VERSION = 'adventure-trail-presentation-v1';

export const TRAIL_PATHWAYS = [
  'MOVE',
  'EXPLORE',
  'ENRICH',
  'LEARN',
  'CONNECT',
  'RECOVER',
  'BOND',
] as const satisfies readonly WellbeingPathway[];

export type TrailPathway = (typeof TRAIL_PATHWAYS)[number];

type AdventureTrailChapterId =
  | 'FIRST_PAWPRINTS'
  | 'FINDING_RHYTHM'
  | 'READING_EACH_OTHER'
  | 'WIDER_WORLD'
  | 'FAMILIAR_TRAIL'
  | 'SHARED_LANGUAGE';

export type AdventureTrailChapter = {
  id: AdventureTrailChapterId;
  label: string;
  minTrailXp: number;
  description: string;
};

export const ADVENTURE_TRAIL_CHAPTERS = [
  {
    id: 'FIRST_PAWPRINTS',
    label: 'First Pawprints',
    minTrailXp: 0,
    description: 'Start noticing what fits instead of chasing a perfect routine.',
  },
  {
    id: 'FINDING_RHYTHM',
    label: 'Finding Rhythm',
    minTrailXp: 100,
    description: 'Useful moments are beginning to form a pattern you can return to.',
  },
  {
    id: 'READING_EACH_OTHER',
    label: 'Reading Each Other',
    minTrailXp: 250,
    description: 'Outcomes are becoming clues for making the next shared choice easier.',
  },
  {
    id: 'WIDER_WORLD',
    label: 'A Wider World',
    minTrailXp: 500,
    description: 'Your shared trail now holds more than one kind of good day.',
  },
  {
    id: 'FAMILIAR_TRAIL',
    label: 'The Familiar Trail',
    minTrailXp: 900,
    description: 'You have a growing library of what works, what does not, and when to stop.',
  },
  {
    id: 'SHARED_LANGUAGE',
    label: 'Shared Language',
    minTrailXp: 1500,
    description: 'The story is richer because listening, adapting, and recovery all count.',
  },
] as const satisfies readonly AdventureTrailChapter[];

export type AdventureTrailState = {
  policyVersion: typeof ADVENTURE_TRAIL_POLICY_VERSION;
  trailXp: number;
  chapter: AdventureTrailChapter;
  nextChapter: AdventureTrailChapter | null;
  chapterProgress: number;
  xpToNextChapter: number;
  discoveredPathways: TrailPathway[];
  discoveryCount: number;
  discoveryTotal: number;
  activeWeeks: number;
  windowWeeks: number;
};

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
}

export function deriveAdventureTrail(dashboard: AdventureDashboard): AdventureTrailState {
  const xpByPathway = new Map(
    dashboard.compass.map((item) => [item.pathway, Math.max(0, Math.floor(item.xp))] as const)
  );
  const trailXp = TRAIL_PATHWAYS.reduce(
    (total, pathway) => total + (xpByPathway.get(pathway) ?? 0),
    0
  );
  let chapterIndex = 0;

  for (let index = 1; index < ADVENTURE_TRAIL_CHAPTERS.length; index += 1) {
    if (trailXp < ADVENTURE_TRAIL_CHAPTERS[index].minTrailXp) break;
    chapterIndex = index;
  }

  const chapter = ADVENTURE_TRAIL_CHAPTERS[chapterIndex];
  const nextChapter = ADVENTURE_TRAIL_CHAPTERS[chapterIndex + 1] ?? null;
  const chapterProgress = nextChapter
    ? clamp01((trailXp - chapter.minTrailXp) / (nextChapter.minTrailXp - chapter.minTrailXp))
    : 1;
  const discoveredPathways = TRAIL_PATHWAYS.filter(
    (pathway) => (xpByPathway.get(pathway) ?? 0) > 0
  );
  const windowWeeks = Math.max(0, Math.floor(dashboard.rhythm.windowWeeks));
  const activeWeeks = Math.min(windowWeeks, Math.max(0, Math.floor(dashboard.rhythm.activeWeeks)));

  return {
    policyVersion: ADVENTURE_TRAIL_POLICY_VERSION,
    trailXp,
    chapter,
    nextChapter,
    chapterProgress,
    xpToNextChapter: nextChapter ? Math.max(0, nextChapter.minTrailXp - trailXp) : 0,
    discoveredPathways,
    discoveryCount: discoveredPathways.length,
    discoveryTotal: TRAIL_PATHWAYS.length,
    activeWeeks,
    windowWeeks,
  };
}
