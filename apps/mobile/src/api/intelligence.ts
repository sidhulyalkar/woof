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

export const intelligenceApi = {
  captureDailySignals: (input: CreateDailySignalsInput) =>
    apiClient.post<DailySignalsCaptureReceipt>('/intelligence/daily-signals', input),
};
