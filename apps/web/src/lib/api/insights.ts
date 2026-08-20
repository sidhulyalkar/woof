import { apiClient } from './client';

export type InsightRecommendation = {
  id: string;
  category: 'activity' | 'enrichment' | 'social' | 'recovery' | 'reflection' | 'goal';
  title: string;
  reason: string;
  actionLabel: string;
  href: string;
  score: number;
  confidence: number;
  evidence: string[];
};

export type RelationshipSignal = {
  key: string;
  label: string;
  value: number;
  explanation: string;
};

export type InsightsResponse = {
  pet: {
    id: string;
    name: string;
    species: string;
    avatarUrl?: string | null;
  };
  generatedAt: string;
  algorithm: {
    recommendations: string;
    relationshipSignals: string;
    confidence: number;
    principles: string[];
  };
  recommendations: InsightRecommendation[];
  relationshipSignals: RelationshipSignal[];
  learningSummary: string[];
  disclaimer: string;
};

export const insightsApi = {
  getMine: async (petId?: string) =>
    (apiClient.get<InsightsResponse>('/insights/me', {
      params: petId ? { petId } : undefined,
    }) as unknown as Promise<InsightsResponse>),

  feedback: async (
    petId: string,
    recommendation: Pick<InsightRecommendation, 'id' | 'category'>,
    outcome: 'shown' | 'accepted' | 'dismissed' | 'completed',
  ) =>
    (apiClient.post(`/insights/pets/${petId}/recommendation-feedback`, {
      recommendationId: recommendation.id,
      category: recommendation.category,
      outcome,
    }) as unknown as Promise<{ id: string }>),
};
