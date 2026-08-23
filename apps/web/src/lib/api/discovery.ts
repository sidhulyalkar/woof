import { apiClient } from './client';

export type DiscoveryLocationStatus = {
  status: 'NOT_CONFIGURED' | 'DISABLED' | 'STALE' | 'OPTED_IN';
  exactLocationStored: false;
  precisionMeters: number;
  expiresAt?: string;
  updatedAt?: string;
};

export type DiscoveryDistanceBand = 'WITHIN_2_5_KM' | 'WITHIN_5_KM' | 'WITHIN_10_KM';

export type DiscoveryCandidate = {
  petId: string;
  ownerId: string;
  petName: string;
  species: string;
  breed?: string | null;
  avatarUrl?: string | null;
  owner: {
    id: string;
    handle: string;
    avatarUrl?: string | null;
    isVerified: boolean;
  };
  distanceBand: DiscoveryDistanceBand;
};

export type NearbyDiscoveryResponse = {
  petId: string;
  locationStatus: DiscoveryLocationStatus['status'];
  candidates: DiscoveryCandidate[];
  boundaries: {
    exactCoordinatesStored: false;
    exactCoordinatesReturned: false;
    homeLocationExposed: false;
    blockedUsersExcluded: true;
    publicProfilesOnly: true;
    maxRadiusKm: number;
  };
};

export const discoveryApi = {
  getLocationStatus: () => apiClient.get<DiscoveryLocationStatus>('/discovery/location'),
  enableLocation: (latitude: number, longitude: number) =>
    apiClient.put<DiscoveryLocationStatus>('/discovery/location', { latitude, longitude }),
  disableLocation: () => apiClient.delete<DiscoveryLocationStatus>('/discovery/location'),
  getNearby: (petId: string, radiusKm = 5, limit = 30) =>
    apiClient.get<NearbyDiscoveryResponse>(`/discovery/nearby/${petId}`, {
      params: { radiusKm, limit },
    }),
};
