import { apiClient } from './client';

export type MeetupProposalStatus = 'pending' | 'accepted' | 'declined' | 'completed' | 'cancelled';

export type MeetupProposal = {
  id: string;
  proposerId: string;
  recipientId: string;
  suggestedTime: string;
  suggestedVenue: {
    name?: string;
    type?: string;
    area?: string;
  };
  status: MeetupProposalStatus;
  occurredAt?: string | null;
  rating?: number | null;
  feedbackTags?: string[];
  checklistOk?: boolean;
  notes?: string | null;
  createdAt: string;
  updatedAt: string;
};

export type MeetupOutcome = {
  occurred: boolean;
  dogExperience?: 'loved_it' | 'comfortable' | 'not_their_thing';
  ownerExperience?: 'great' | 'fine' | 'a_lot_today';
  meetAgain?: 'yes' | 'maybe' | 'no';
  checklistOk?: boolean;
};

export const meetupProposalsApi = {
  getMine: () =>
    apiClient.get<{ sent: MeetupProposal[]; received: MeetupProposal[] }>('/meetup-proposals'),
  create: (input: {
    recipientId: string;
    suggestedTime: string;
    suggestedVenue: { name: string; type: string; area?: string };
    notes?: string;
  }) => apiClient.post<MeetupProposal>('/meetup-proposals', input),
  updateStatus: (id: string, status: 'accepted' | 'declined') =>
    apiClient.put<MeetupProposal>(`/meetup-proposals/${id}/status`, { status }),
  complete: (id: string, outcome: MeetupOutcome) =>
    apiClient.put<{ proposal: MeetupProposal; feedbackRecorded: true; reportSuggested: boolean }>(
      `/meetup-proposals/${id}/complete`,
      outcome,
    ),
  cancel: (id: string) => apiClient.delete<MeetupProposal>(`/meetup-proposals/${id}`),
};
