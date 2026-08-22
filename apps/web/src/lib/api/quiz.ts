import { apiClient } from './client';

export type QuizAnswers = Record<string, string | string[]>;

export type SavedQuizPreferenceSession = {
  sessionId: string;
  petId?: string;
  responses: QuizAnswers;
  completedAt?: string;
};

export const quizApi = {
  saveResponses: (data: { sessionId: string; petId?: string; responses: QuizAnswers }) =>
    apiClient.post<SavedQuizPreferenceSession>('/quiz/responses', data),
  latest: () => apiClient.get<SavedQuizPreferenceSession | null>('/quiz/responses/latest'),
};
