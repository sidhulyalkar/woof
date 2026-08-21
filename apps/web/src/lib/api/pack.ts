import { apiClient } from './client';

export type PackChallenge = {
  id: string;
  title: string;
  description: string;
  pathways: string[];
  target: number;
  unit: string;
  total: number;
  contributors: number;
  myContribution: number;
  progress: number;
  completed: boolean;
};

export type PackChallengesResponse = {
  generatedAt: string;
  windowDays: number;
  challenges: PackChallenge[];
  principles: string[];
};

export const packApi = {
  challenges: async () =>
    (apiClient.get<PackChallengesResponse>('/pack/challenges') as unknown as Promise<PackChallengesResponse>),
};
