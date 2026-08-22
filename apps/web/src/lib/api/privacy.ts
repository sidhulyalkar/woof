import { apiClient } from './client';

export type MeetupLocationSharing = 'NEVER' | 'AFTER_CONFIRMATION';

export type PrivacyPreferences = {
  preciseLocation: boolean;
  proximitySuggestions: boolean;
  shareActivityRoutes: boolean;
  meetupLocationSharing: MeetupLocationSharing;
  locationRetentionHours: number;
};

export type PrivacyUpdateResponse = {
  preferences: PrivacyPreferences;
  updatedAt: string;
};

export type LocationSummary = {
  preferences: PrivacyPreferences;
  storedLocationPings: number;
  oldestStoredAt: string | null;
  newestStoredAt: string | null;
  maxRetentionHours: number;
};

export type ClearLocationHistoryResponse = {
  deleted: number;
};

export const privacyApi = {
  preferences: () =>
    apiClient.get('/privacy/preferences') as unknown as Promise<PrivacyPreferences>,

  updatePreferences: (patch: Partial<PrivacyPreferences>) =>
    apiClient.put('/privacy/preferences', patch) as unknown as Promise<PrivacyUpdateResponse>,

  locationSummary: () =>
    apiClient.get('/privacy/location-summary') as unknown as Promise<LocationSummary>,

  clearLocationHistory: () =>
    apiClient.delete(
      '/privacy/location-history'
    ) as unknown as Promise<ClearLocationHistoryResponse>,
};
