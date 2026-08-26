import { apiClient } from './client';

export type ProfileQuestionOutcome = 'ANSWERED' | 'NOT_SURE' | 'SKIPPED';

export type AdaptiveProfileQuestionResponseInput = {
  responseId: string;
  questionId: string;
  outcome: ProfileQuestionOutcome;
  answers?: string[];
};

export type AdaptiveProfileDimensionState = {
  dimension: string;
  subject: 'DOG' | 'OWNER' | 'PAIR';
  state: 'UNKNOWN' | 'LEARNING' | 'KNOWN';
  value: unknown;
  confidence: number;
  provenance: string | null;
  updatedAt: string | null;
};

export type AdaptiveProfileState = {
  schemaVersion: string;
  householdId: string;
  petId: string;
  dimensions: AdaptiveProfileDimensionState[];
  coverage: {
    known: string[];
    learning: string[];
    unknown: string[];
  };
};

export type AdaptiveProfileWriteReceipt = {
  duplicate: boolean;
  profile: AdaptiveProfileState;
};

function pairPath(householdId: string, petId: string) {
  return `/adventure/profile/${encodeURIComponent(householdId)}/${encodeURIComponent(petId)}`;
}

export const adaptiveProfileApi = {
  getState: (householdId: string, petId: string) =>
    apiClient.get<AdaptiveProfileState>(pairPath(householdId, petId)),

  recordQuestionResponse: (
    householdId: string,
    petId: string,
    input: AdaptiveProfileQuestionResponseInput
  ) =>
    apiClient.post<AdaptiveProfileWriteReceipt, AdaptiveProfileQuestionResponseInput>(
      `${pairPath(householdId, petId)}/questions/respond`,
      input
    ),
};
