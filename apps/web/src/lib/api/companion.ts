import { apiClient } from './client';

export type CompanionMode = 'PET_GUARDIAN' | 'ANIMAL_ALLY' | 'FOSTER_CAREGIVER';
export type CompanionLanding = 'NEEDS_MODE' | 'NEEDS_PET_SETUP' | 'PET_TODAY' | 'COMPANION_TODAY';
export type ReadinessStatus = 'NOT_SURE' | 'WORKING_ON_IT' | 'READY_TO_DISCUSS';
export type ReadinessDimension =
  'housing' | 'householdAlignment' | 'timeCapacity' | 'financialPlan' | 'supportPlan' | 'carePlan';

export type CompanionState = {
  mode: CompanionMode | null;
  modeSource: 'PERSISTED' | 'PET_AUTHORITY_COMPAT' | 'UNSET';
  hasAuthorizedPet: boolean;
  landing: CompanionLanding;
  authority: {
    modeControlsPresentation: boolean;
    petAccessComesFromRelationships: boolean;
    modeNeverCreatesPetAuthority: boolean;
  };
};

export type ReadinessReflection = {
  dimensions: Record<ReadinessDimension, ReadinessStatus | null>;
  updatedAt: string | null;
  statuses: readonly ReadinessStatus[];
  disclaimer: string;
};

export const companionApi = {
  state: () => apiClient.get<CompanionState>('/companion/state'),
  updateMode: (mode: CompanionMode) =>
    apiClient.put<CompanionState, { mode: CompanionMode }>('/companion/mode', { mode }),
  readiness: () => apiClient.get<ReadinessReflection>('/companion/readiness'),
  updateReadiness: (patch: Partial<Record<ReadinessDimension, ReadinessStatus>>) =>
    apiClient.put<ReadinessReflection, Partial<Record<ReadinessDimension, ReadinessStatus>>>(
      '/companion/readiness',
      patch
    ),
};
