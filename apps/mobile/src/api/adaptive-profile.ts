import apiClient from './client';

export type ProfileQuestionOutcome = 'ANSWERED' | 'NOT_SURE' | 'SKIPPED';

export type AdaptiveProfileQuestionResponseInput = {
  responseId: string;
  questionId: string;
  outcome: ProfileQuestionOutcome;
  answers?: string[];
};

export type AdaptiveProfileWriteReceipt = {
  duplicate: boolean;
};

function pairPath(householdId: string, petId: string) {
  return `/adventure/profile/${encodeURIComponent(householdId)}/${encodeURIComponent(petId)}`;
}

export const adaptiveProfileApi = {
  recordQuestionResponse: (
    householdId: string,
    petId: string,
    input: AdaptiveProfileQuestionResponseInput
  ) =>
    apiClient.post<AdaptiveProfileWriteReceipt>(
      `${pairPath(householdId, petId)}/questions/respond`,
      input
    ),
};
