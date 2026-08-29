import { apiClient } from './client';

export type CaregiverCapability = 'VIEW_TODAY' | 'LOG_OBSERVATION';
export type CaregiverStoredStatus = 'PENDING_ACCEPTANCE' | 'ACTIVE' | 'DECLINED' | 'REVOKED';
export type CaregiverEffectiveStatus = CaregiverStoredStatus | 'EXPIRED';
export type CaregiverObservationKind = 'ROUTINE' | 'ACTIVITY_RESPONSE' | 'BEHAVIOR' | 'HANDOFF_NOTE';

export type CaregiverPet = {
  id: string;
  name: string;
  species: string;
  breed: string | null;
  avatarUrl: string | null;
};

export type CaregiverGrant = {
  id: string;
  petId: string;
  issuerUserId: string;
  recipientUserId: string;
  requestKey: string;
  policyVersion: string;
  status: CaregiverStoredStatus;
  effectiveStatus: CaregiverEffectiveStatus;
  issuedAt: string;
  acceptedAt: string | null;
  declinedAt: string | null;
  expiresAt: string;
  revokedAt: string | null;
  revokedByUserId: string | null;
  createdAt: string;
  updatedAt: string;
  capabilities: CaregiverCapability[];
  pet: CaregiverPet;
  issuerHandle?: string;
  recipientHandle?: string;
  relationshipBlocked?: boolean;
  replayed?: boolean;
};

export type CaregiverToday = {
  pet: CaregiverPet;
  relationship: {
    grantId: string;
    issuerUserId: string;
    issuerHandle?: string;
    expiresAt: string;
    capabilities: CaregiverCapability[];
    effectiveStatus: CaregiverEffectiveStatus;
  };
  available: {
    viewToday: boolean;
    logObservation: boolean;
  };
  boundaries: {
    householdHistory: false;
    siblingPets: false;
    medicalAuthority: false;
    profileCorrection: false;
    connectorAdmin: false;
    bondXpAuthority: false;
    recommendationEvidenceAuthority: false;
  };
};

export type CaregiverObservation = {
  id: string;
  grantId: string;
  petId: string;
  actorUserId: string;
  authorityClass: 'CONTEXT_ONLY';
  kind: CaregiverObservationKind;
  summary: string;
  note: string | null;
  context: Record<string, unknown>;
  observedAt: string;
  createdAt: string;
};

export const caregiverApi = {
  received: () => apiClient.get<CaregiverGrant[]>('/caregiver/grants/received'),
  activePets: () => apiClient.get<CaregiverGrant[]>('/caregiver/pets'),
  accept: (grantId: string) =>
    apiClient.post<CaregiverGrant, Record<string, never>>(`/caregiver/grants/${grantId}/accept`, {}),
  decline: (grantId: string) =>
    apiClient.post<CaregiverGrant, Record<string, never>>(`/caregiver/grants/${grantId}/decline`, {}),
  today: (petId: string) => apiClient.get<CaregiverToday>(`/caregiver/pets/${petId}/today`),
  observe: (
    petId: string,
    input: {
      kind: CaregiverObservationKind;
      summary: string;
      note?: string;
      observedAt?: string;
    }
  ) =>
    apiClient.post<CaregiverObservation, typeof input>(`/caregiver/pets/${petId}/observations`, input),
};
