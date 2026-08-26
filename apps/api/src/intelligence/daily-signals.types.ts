import type { SignalDimension } from './baseline-policy-v1.types';

export const DAILY_SIGNAL_CHOICES = ['LESS', 'USUAL', 'MORE', 'UNSURE'] as const;
export type DailySignalChoice = (typeof DAILY_SIGNAL_CHOICES)[number];

export const DAILY_SIGNAL_FIELDS = [
  'appetite',
  'energy',
  'bathroomRoutine',
  'mobilityComfort',
  'engagementSocialComfort',
  'sleepRest',
] as const;
export type DailySignalField = (typeof DAILY_SIGNAL_FIELDS)[number];

export const DAILY_SIGNAL_DIMENSION_BY_FIELD: Record<DailySignalField, SignalDimension> = {
  appetite: 'APPETITE',
  energy: 'ENERGY',
  bathroomRoutine: 'BATHROOM_ROUTINE',
  mobilityComfort: 'MOBILITY_COMFORT',
  engagementSocialComfort: 'ENGAGEMENT_SOCIAL_COMFORT',
  sleepRest: 'SLEEP_REST',
};

export type DailySignalsAnswers = Partial<Record<DailySignalField, DailySignalChoice>>;

export type DailySignalsCaptureReceipt = {
  careEventId: string;
  householdId: string;
  petId: string;
  localDate: string;
  timezone: string;
  duplicate: boolean;
  futureTimestampNormalized: boolean;
  projectedDimensions: SignalDimension[];
};
