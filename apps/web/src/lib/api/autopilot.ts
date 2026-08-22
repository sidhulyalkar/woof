import { apiClient } from './client';

export type AutopilotProviderCapability = {
  provider: 'FI' | 'TRACTIVE';
  status: 'STUB_READY';
  accepts: Array<'DAILY_ACTIVITY' | 'DEVICE_STATUS'>;
};

export type CareReminderKind = 'VET_APPOINTMENT' | 'MEDICATION' | 'GROOMING' | 'GENERAL_CARE';

export type CareReminder = {
  id: string;
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

export type AutopilotSignal = {
  id: string;
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

export type AutopilotDashboard = {
  providers: AutopilotProviderCapability[];
  reminders: CareReminder[];
  signals: AutopilotSignal[];
  boundaries: {
    locationTelemetryStored: false;
    canonicalPetMutationAllowed: false;
    trackerObservationsRewardEligible: false;
    signalsDiagnostic: false;
  };
};

export type CreateCareReminderInput = {
  petId?: string;
  kind: CareReminderKind;
  title: string;
  detail?: string;
  dueAt: string;
  repeatEveryDays?: number;
};

export const autopilotApi = {
  getDashboard: () => apiClient.get<AutopilotDashboard>('/autopilot'),
  createReminder: (input: CreateCareReminderInput) =>
    apiClient.post<CareReminder, CreateCareReminderInput>('/autopilot/reminders', input),
  cancelReminder: (id: string) =>
    apiClient.delete<{ success: boolean }>(`/autopilot/reminders/${id}`),
  acknowledgeSignal: (id: string) =>
    apiClient.post<{ success: boolean }>(`/autopilot/signals/${id}/acknowledge`),
};
