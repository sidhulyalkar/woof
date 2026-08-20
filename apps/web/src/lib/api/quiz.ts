import { apiClient } from './client';

export type QuizAnswers = Record<string, string | string[]>;

export const quizApi = {
  saveResponses: (data: { sessionId: string; petId?: string; responses: QuizAnswers }) =>
    apiClient.post('/quiz/responses', data),
  latest: () => apiClient.get('/quiz/responses/latest'),
};
