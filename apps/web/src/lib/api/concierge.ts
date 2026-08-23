import { apiClient } from './client';

export type ConciergeEvidence = {
  source: 'ADVENTURE' | 'CARE_EVENT' | 'AUTOPILOT' | 'CONNECTOR';
  label: string;
  occurredAt?: string;
  referenceId?: string;
};

export type ConciergeSuggestion = {
  id: string;
  kind: 'CARE_PREP' | 'CHECK_IN' | 'RECOVERY_PACE' | 'CONNECTION_ATTENTION';
  priority: 'ATTENTION' | 'GENTLE' | 'INFO';
  title: string;
  body: string;
  reason: string;
  evidence: ConciergeEvidence[];
  action?: { label: string; href: string };
  suggestionOnly: true;
};

export type ConciergeToday = {
  generatedAt: string;
  pet: { id: string; name: string; species: string };
  briefing: {
    title: string;
    summary: string;
    topQuest: {
      title: string;
      reason: string;
      action: { label: string; href: string };
      evidence: ConciergeEvidence[];
    } | null;
  };
  context: {
    weather: { status: 'NOT_CONFIGURED'; live: false; detail: string };
    pace: {
      mode: 'NORMAL' | 'GENTLE';
      reason: string;
      evidence: ConciergeEvidence[];
    };
  };
  suggestions: ConciergeSuggestion[];
  connectorSummary: { connected: number; needsReauthorization: number };
  boundaries: {
    suggestionOnly: true;
    liveWeatherUsed: false;
    diagnosticInferenceAllowed: false;
    prescriptionOrDoseCalculationAllowed: false;
    persistentStateMutationAllowed: false;
    autonomousPurchaseAllowed: false;
  };
};

export const conciergeApi = {
  getToday: (petId?: string) =>
    apiClient.get<ConciergeToday>('/concierge/today', {
      params: petId ? { petId } : undefined,
    }),
};
