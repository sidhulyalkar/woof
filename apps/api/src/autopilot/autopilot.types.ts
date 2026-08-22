export const AUTOPILOT_PROVIDERS = ['FI', 'TRACTIVE'] as const;
export type AutopilotProvider = (typeof AUTOPILOT_PROVIDERS)[number];

export const AUTOPILOT_OBSERVATION_KINDS = ['DAILY_ACTIVITY', 'DEVICE_STATUS'] as const;
export type AutopilotObservationKind = (typeof AUTOPILOT_OBSERVATION_KINDS)[number];

export type NormalizedTrackerObservation = {
  provider: AutopilotProvider;
  externalEventId: string;
  kind: AutopilotObservationKind;
  observedAt: Date;
  metrics: {
    activityMinutes?: number;
    distanceMeters?: number;
    steps?: number;
    batteryPercent?: number;
    deviceState?: 'ONLINE' | 'OFFLINE' | 'UNKNOWN';
  };
};

export const CARE_REMINDER_KINDS = [
  'VET_APPOINTMENT',
  'MEDICATION',
  'GROOMING',
  'GENERAL_CARE',
] as const;
export type CareReminderKind = (typeof CARE_REMINDER_KINDS)[number];

export type CareReminderPayload = {
  schemaVersion: 'dogos-care-reminder-v1';
  kind: CareReminderKind;
  title: string;
  detail?: string;
  petId?: string;
  dueAt: string;
  repeatEveryDays?: number;
  status: 'SCHEDULED' | 'COMPLETED' | 'CANCELLED';
  lastAttemptAt?: string;
  lastDeliveredAt?: string;
};

export type AutopilotSignalPayload = {
  schemaVersion: 'dogos-autopilot-signal-v1';
  petId: string;
  sourceCareEventId: string;
  signalType: 'ACTIVITY_BELOW_RECENT_BASELINE' | 'TRACKER_BATTERY_LOW';
  level: 'CHECK_IN' | 'INFO';
  title: string;
  body: string;
  observedAt: string;
  evidence: Record<string, string | number | boolean>;
  nonDiagnostic: true;
};
