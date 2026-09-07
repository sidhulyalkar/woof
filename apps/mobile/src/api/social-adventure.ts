import apiClient from './client';

export type SocialAdventureReaction =
  | 'NICE_READ'
  | 'GOOD_CALL'
  | 'TRYING_THIS'
  | 'ADVENTURE_INSPIRATION'
  | 'CHEER';

export type SocialAdventureScoreComponents = {
  humanSkill: {
    score: number;
    maxScore: number;
    completedChallenges: string[];
  };
  adventureVariety: {
    score: number;
    maxScore: number;
    pathways: string[];
  };
};

export type SocialAdventureMe = {
  preferences: { globalLeaderboardOptIn: boolean };
  season: { key: string; startsAt: string; endsAt: string };
  score: number;
  maxScore: number;
  components: SocialAdventureScoreComponents;
  humanSkillBestScores: Record<string, number>;
  policyVersion: string;
  principles: string[];
};

export type LeagueEntry = {
  rank: number;
  userId: string;
  handle: string;
  avatarUrl: string | null;
  score: number;
  maxScore: number;
  components: SocialAdventureScoreComponents;
};

export type GlobalLeaderboard = {
  scope: 'GLOBAL';
  season: { key: string; startsAt: string; endsAt: string };
  entries: LeagueEntry[];
  me: { score: number; maxScore: number; rank: number | null; public: boolean };
  policyVersion: string;
  disclaimer: string;
};

export type PackLeaderboard = {
  scope: 'PACK';
  pack: { id: string; name: string; memberCount: number };
  cohortReady: boolean;
  minimumCohort: number;
  entries: LeagueEntry[];
  policyVersion: string;
  message?: string;
};

export type SocialAdventurePost = {
  shareId: string;
  postId: string;
  kind: string;
  headline: string;
  summary: string;
  payload: Record<string, unknown>;
  caption: string | null;
  visibility: string;
  createdAt: string;
  authorUserId: string;
  handle: string;
  avatarUrl: string | null;
  petName: string | null;
  reactions: Array<{ reaction: SocialAdventureReaction; count: number; mine: boolean }>;
};

export type SocialAdventureFeed = {
  posts: SocialAdventurePost[];
  privacy: string;
};

export type SocialPack = {
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

export type PacksCatalog = {
  packs: SocialPack[];
  localMinimumCohort: number;
  locationContract: string;
};

export type ArcadeChallengeKey =
  | 'MAKE_IT_EASIER'
  | 'CATCH_THE_GOOD'
  | 'PAIRING_LAB'
  | 'MARKER_TIMING';

export type ArcadeOption = {
  id: string;
  label: string;
};

export type ArcadeScenario = {
  challengeKey: ArcadeChallengeKey;
  challengeVersion: string;
  scenarioKey: string;
  title: string;
  skill: string;
  prompt: string;
  options?: ArcadeOption[];
  timing?: {
    durationMs: number;
    targetAtMs: number;
    targetLabel: string;
  };
  bestScore: number | null;
};

export type ArcadeCatalog = {
  challengeVersion: string;
  challenges: ArcadeScenario[];
  scoring: string;
};

export type ArcadeAttempt = {
  attemptId: string;
  issuedAt: string;
  expiresAt: string;
  scenario: Omit<ArcadeScenario, 'bestScore'>;
};

export type ArcadeReceipt = {
  attemptId: string;
  challengeKey: ArcadeChallengeKey;
  challengeVersion: string;
  score: number;
  correct: boolean;
  timingErrorMs?: number;
  explanation: string;
  completedAt: string;
};

export type SocialAdventureShare = {
  shareId: string;
  postId: string;
  headline: string;
  summary: string;
  visibility: string;
};

export const socialAdventureApi = {
  getMine: () => apiClient.get<SocialAdventureMe>('/social-adventure/me'),

  updatePreferences: (globalLeaderboardOptIn: boolean) =>
    apiClient.put<{ globalLeaderboardOptIn: boolean }>('/social-adventure/preferences', {
      globalLeaderboardOptIn,
    }),

  globalLeaderboard: () =>
    apiClient.get<GlobalLeaderboard>('/social-adventure/leaderboard/global'),

  feed: () => apiClient.get<SocialAdventureFeed>('/social-adventure/feed'),

  addReaction: (shareId: string, reaction: SocialAdventureReaction) =>
    apiClient.post<{ ok: true }>(`/social-adventure/shares/${shareId}/reactions`, { reaction }),

  removeReaction: (shareId: string, reaction: SocialAdventureReaction) =>
    apiClient.delete<{ ok: true }>(`/social-adventure/shares/${shareId}/reactions/${reaction}`),

  packs: () => apiClient.get<PacksCatalog>('/social-adventure/packs'),

  createPack: (input: { name: string; regionKey: string }) =>
    apiClient.post<SocialPack>('/social-adventure/packs', input),

  joinPack: (packId: string) =>
    apiClient.post<{ ok: true }>(`/social-adventure/packs/${packId}/join`, {}),

  leavePack: (packId: string) =>
    apiClient.delete<{ ok: true }>(`/social-adventure/packs/${packId}/membership`),

  packLeaderboard: (packId: string) =>
    apiClient.get<PackLeaderboard>(`/social-adventure/packs/${packId}/leaderboard`),

  arcade: () => apiClient.get<ArcadeCatalog>('/social-adventure/arcade'),

  startArcadeAttempt: (challengeKey: ArcadeChallengeKey) =>
    apiClient.post<ArcadeAttempt>(`/social-adventure/arcade/${challengeKey}/attempts`, {}),

  completeArcadeAttempt: (attemptId: string, response: Record<string, unknown>) =>
    apiClient.post<ArcadeReceipt>(`/social-adventure/arcade/attempts/${attemptId}/complete`, {
      response,
    }),

  shareSkillAttempt: (attemptId: string) =>
    apiClient.post<SocialAdventureShare>('/social-adventure/shares', {
      sourceType: 'HUMAN_SKILL_ATTEMPT',
      sourceId: attemptId,
      visibility: 'PUBLIC',
    }),
};
