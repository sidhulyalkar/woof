import apiClient from './client';

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
