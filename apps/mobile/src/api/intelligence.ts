import apiClient from './client';

export type DailySignalChoice = 'LESS' | 'USUAL' | 'MORE' | 'UNSURE';

export type DailySignalsAnswers = {
  appetite?: DailySignalChoice;
  energy?: DailySignalChoice;
  bathroomRoutine?: DailySignalChoice;
  mobilityComfort?: DailySignalChoice;
  engagementSocialComfort?: DailySignalChoice;
  sleepRest?: DailySignalChoice;
};

export type CreateDailySignalsInput = {
  householdId: string;
  petId: string;
  observedAt?: string;
  signals: DailySignalsAnswers;
  note?: string;
};

export type DailySignalsCaptureReceipt = {
  careEventId: string;
  householdId: string;
  petId: string;
  localDate: string;
  timezone: string;
  duplicate: boolean;
  futureTimestampNormalized: boolean;
  projectedDimensions: string[];
};

export type DailySignalsEffectiveState = {
  rootCareEventId: string;
  currentCareEventId: string;
  correctionSequence: number;
  householdId: string;
  petId: string;
  localDate: string;
  timezone: string;
  status: 'ORIGINAL' | 'CORRECTED';
  signals: DailySignalsAnswers;
};

export type CorrectDailySignalsInput = {
  householdId: string;
  petId: string;
  expectedCurrentCareEventId: string;
  signals: DailySignalsAnswers;
};

export type DailySignalsCorrectionReceipt = {
  correctionCareEventId: string;
  duplicate: boolean;
  currentAdvanced: boolean;
  state: DailySignalsEffectiveState;
  projectedDimensions: string[];
  retractedDimensions: string[];
};

export const intelligenceApi = {
  currentDailySignals: (householdId: string, petId: string) =>
    apiClient.get<DailySignalsEffectiveState | null>('/intelligence/daily-signals/current', {
      params: { householdId, petId },
    }),
  correctDailySignals: (input: CorrectDailySignalsInput) =>
    apiClient.post<DailySignalsCorrectionReceipt>('/intelligence/daily-signals/corrections', input),
  captureDailySignals: (input: CreateDailySignalsInput) =>
    apiClient.post<DailySignalsCaptureReceipt>('/intelligence/daily-signals', input),
};
