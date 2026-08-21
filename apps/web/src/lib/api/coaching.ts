import { apiClient } from './client';

export type CoachingProgression = {
  action: 'start' | 'hold' | 'increase' | 'decrease';
  nextLevel: number;
  headline: string;
  reason: string;
};

export type CoachingTemplate = {
  id: string;
  species: 'DOG' | 'CAT' | 'ANY';
  title: string;
  skill: string;
  objective: string;
  cue: string;
  handlerFocus: string;
  steps: string[];
  rewardExamples: string[];
  safety: string;
};

export type CoachingPlan = {
  id: string;
  petId: string;
  status: string;
  title: string;
  skill: string;
  objective: string;
  cue: string;
  handlerFocus: string;
  steps: string[];
  rewardExamples: string[];
  safety: string;
  level: number;
  levelLabel: string;
  targetSuccessRate: number;
  sessionsCompleted: number;
  practiceCoverage: number;
  recentSuccessRate: number | null;
  lastPracticedAt: string | null;
  nextPractice: CoachingProgression;
  support: { recommended: boolean; message: string };
  updatedAt: string;
};

export type CoachingDashboard = {
  pet: {
    id: string;
    name: string;
    species: string;
    avatarUrl?: string | null;
  } | null;
  activePlan: CoachingPlan | null;
  pausedPlans: CoachingPlan[];
  templates: CoachingTemplate[];
  weeklyRhythm: { sessions: number; minutes: number };
  methodology: {
    version: string;
    principles: string[];
    progressionPolicy: string;
    sources: Array<{ label: string; url: string }>;
  };
  onboarding?: string;
};

export type TrainingSessionInput = {
  attempts: number;
  successes: number;
  durationSeconds: number;
  distractionLevel?: number;
  rewardType: 'food' | 'play' | 'praise' | 'access' | 'environmental' | 'other';
  stressSignals?: string[];
  stoppedEarly?: boolean;
  notes?: string;
};

export const coachingApi = {
  getMine: async (petId?: string) =>
    (apiClient.get<CoachingDashboard>('/coaching/me', {
      params: petId ? { petId } : undefined,
    }) as unknown as Promise<CoachingDashboard>),

  startPlan: async (petId: string, templateId: string) =>
    (apiClient.post('/coaching/plans', { petId, templateId }) as unknown as Promise<CoachingPlan>),

  setPlanStatus: async (planId: string, status: 'ACTIVE' | 'PAUSED') =>
    (apiClient.patch(`/coaching/plans/${planId}/status`, { status }) as unknown as Promise<unknown>),

  recordSession: async (planId: string, input: TrainingSessionInput) =>
    (apiClient.post(`/coaching/plans/${planId}/sessions`, input) as unknown as Promise<{
      activityId: string;
      plan: CoachingPlan;
      decision: CoachingProgression;
      support: { recommended: boolean; message: string };
    }>),
};
