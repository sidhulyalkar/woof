export const LOCATION_SHARING_MODES = ['NEVER', 'AFTER_CONFIRMATION'] as const;
export type MeetupLocationSharing = (typeof LOCATION_SHARING_MODES)[number];

export type PrivacyPreferences = {
  preciseLocation: boolean;
  proximitySuggestions: boolean;
  shareActivityRoutes: boolean;
  meetupLocationSharing: MeetupLocationSharing;
  locationRetentionHours: number;
};

export const DEFAULT_PRIVACY_PREFERENCES: PrivacyPreferences = {
  preciseLocation: false,
  proximitySuggestions: false,
  shareActivityRoutes: false,
  meetupLocationSharing: 'AFTER_CONFIRMATION',
  locationRetentionHours: 12,
};
